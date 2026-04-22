import json
import os
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from nerf_triplane.network import AudioEncoder
from nerf_triplane.utils import AudDataset

from .config import ModelConfig
from .rich_utils import rich_track
from .utils.audio_utils import get_audio_features
from .utils.camera_utils import cameraList_from_camInfos
from .utils.graphics_utils import BasicPointCloud, focal2fov, getWorld2View2
from .utils.sh_utils import SH2RGB
from .utils.system_utils import searchForMaxIteration


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def _get_df_column(dataframe: pd.DataFrame, name: str) -> pd.Series:
    normalized = {column.strip(): column for column in dataframe.columns}
    if name not in normalized:
        raise KeyError(f"Missing OpenFace column {name!r} in au.csv")
    return dataframe[normalized[name]]


def _get_ave_features(root_path: str, audio_file: str = "", cache_audio: bool = True) -> torch.Tensor:
    if audio_file.endswith(".npy"):
        features = np.load(audio_file)
    else:
        cache_path = os.path.join(root_path, "aud_ave.npy")
        wav_path = audio_file or os.path.join(root_path, "aud.wav")
        if cache_audio and os.path.exists(cache_path) and audio_file == "":
            features = np.load(cache_path)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AudioEncoder().to(device).eval()
            ckpt = torch.load(
                "./nerf_triplane/checkpoints/audio_visual_encoder.pth",
                map_location=device,
            )
            model.load_state_dict({f"audio_encoder.{k}": v for k, v in ckpt.items()})
            dataset = AudDataset(wav_path)
            data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            outputs = []
            for mel in data_loader:
                mel = mel.to(device)
                with torch.no_grad():
                    outputs.append(model(mel))
            outputs = torch.cat(outputs, dim=0).cpu()
            first_frame, last_frame = outputs[:1], outputs[-1:]
            features = torch.cat(
                [first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                dim=0,
            ).numpy()
            if cache_audio and audio_file == "":
                np.save(cache_path, features)
    features = torch.from_numpy(features).float()
    if features.ndim == 2:
        features = features.unsqueeze(1)
    return features


def _get_audio_features(config: ModelConfig) -> torch.Tensor:
    if config.audio_extractor == "ave":
        return _get_ave_features(config.path, config.audio, config.cache_audio)

    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}
    feature_path = config.audio or os.path.join(
        config.path, f"aud_{postfix_dict[config.audio_extractor]}.npy"
    )
    aud_features = torch.from_numpy(np.load(feature_path))
    return aud_features.float().permute(0, 2, 1)


def _load_masks(root_path: str, img_id: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    teeth_mask_path = os.path.join(root_path, "teeth_mask", f"{img_id}.npy")
    if os.path.exists(teeth_mask_path):
        teeth_mask = np.load(teeth_mask_path).astype(bool)
    else:
        teeth_mask = np.zeros(shape, dtype=bool)

    mask_path = os.path.join(root_path, "parsing", f"{img_id}.png")
    mask = np.array(Image.open(mask_path).convert("RGB"))
    face_mask = ((mask[:, :, 2] > 254) & (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0)) ^ teeth_mask
    hair_mask = (mask[:, :, 0] < 1) & (mask[:, :, 1] < 1) & (mask[:, :, 2] < 1)
    mouth_mask = ((mask[:, :, 0] == 100) & (mask[:, :, 1] == 100) & (mask[:, :, 2] == 100)) | teeth_mask
    return face_mask, hair_mask, mouth_mask


def _build_frame_metadata(root_path: str, frames: list[dict], preload: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ldmks_lips = []
    ldmks_mouth = []
    ldmks_lhalf = []
    for frame in rich_track(frames, description="Parsing landmarks"):
        lms = np.loadtxt(os.path.join(root_path, "ori_imgs", f"{frame['img_id']}.lms"))
        lips = slice(48, 60)
        mouth = slice(60, 68)
        xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
        ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
        ldmks_lips.append([xmin, xmax, ymin, ymax])
        ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])
        lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())
        ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
        ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
    return np.array(ldmks_lips), np.array(ldmks_mouth), np.array(ldmks_lhalf)


def _get_nerfpp_norm(cam_infos: list[CameraInfo]) -> dict:
    cam_centers = []
    for cam in cam_infos:
        world_to_camera = getWorld2View2(cam.R, cam.T)
        camera_to_world = np.linalg.inv(world_to_camera)
        cam_centers.append(camera_to_world[:3, 3:4])

    cam_centers = np.hstack(cam_centers)
    center = np.mean(cam_centers, axis=1, keepdims=True)
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return {"translate": -center.flatten(), "radius": diagonal * 1.1}


def _read_cameras_from_transforms(config: ModelConfig, transforms_file: str) -> list[CameraInfo]:
    with open(os.path.join(config.path, transforms_file), "r", encoding="utf-8") as handle:
        contents = json.load(handle)

    focal_len = contents["focal_len"]
    bg_img = np.array(Image.open(os.path.join(config.path, "bc.jpg")).convert("RGB"))
    frames = contents["frames"]
    aud_features = _get_audio_features(config)
    au_info = pd.read_csv(os.path.join(config.path, "au.csv"))

    au_blink = _get_df_column(au_info, "AU45_r").values
    au25 = _get_df_column(au_info, "AU25_r").values
    au25 = np.clip(au25, 0, np.percentile(au25, 95))
    au25_25 = np.percentile(au25, 25)
    au25_50 = np.percentile(au25, 50)
    au25_75 = np.percentile(au25, 75)
    au25_100 = au25.max()

    au_exp = []
    for index in [1, 4, 5, 6, 7, 45]:
        values = _get_df_column(au_info, f"AU{index:02d}_r").values
        if index == 45:
            values = values.clip(0, 2)
        au_exp.append(values[:, None])
    au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

    ldmks_lips, ldmks_mouth, ldmks_lhalf = _build_frame_metadata(
        config.path, frames, config.preload
    )
    mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
    mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()

    cam_infos = []
    for idx, frame in enumerate(rich_track(frames, description=f"Loading {transforms_file}")):
        image_path = os.path.join(config.path, "gt_imgs", f"{frame['img_id']}.jpg")
        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_pil = Image.open(image_path)
        width, height = image_pil.size
        image = np.array(image_pil.convert("RGB")) if config.preload else None

        torso_img_path = os.path.join(config.path, "torso_imgs", f"{frame['img_id']}.png")
        if config.preload:
            torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
            background = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (
                1 - torso_img[..., 3:] / 255.0
            )
            background = background.astype(np.uint8)
        else:
            background = None

        if config.audio:
            audio_index = idx
        else:
            audio_index = frame["img_id"]
        talking_dict = {
            "img_id": frame["img_id"],
            "auds": get_audio_features(aud_features, 2, audio_index),
            "blink": torch.as_tensor(np.clip(au_blink[frame["img_id"]], 0, 2) / 2),
            "au25": [au25[frame["img_id"]], au25_25, au25_50, au25_75, au25_100],
            "au_exp": torch.as_tensor(au_exp[frame["img_id"]]),
            "lhalf_rect": ldmks_lhalf[idx],
            "mouth_bound": [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]],
        }

        xmin, xmax, ymin, ymax = ldmks_lips[idx].tolist()
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        side = max(xmax - xmin, ymax - ymin) // 2
        talking_dict["lips_rect"] = [cx - side, cx + side, cy - side, cy + side]

        if config.preload:
            face_mask, hair_mask, mouth_mask = _load_masks(
                config.path, frame["img_id"], (height, width)
            )
            talking_dict["face_mask"] = face_mask
            talking_dict["hair_mask"] = hair_mask
            talking_dict["mouth_mask"] = mouth_mask

        fov_x = focal2fov(focal_len, width)
        fov_y = focal2fov(focal_len, height)
        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=fov_y,
                FovX=fov_x,
                image=image,
                image_path=image_path,
                image_name=Path(image_path).stem,
                width=width,
                height=height,
                background=background,
                talking_dict=talking_dict,
            )
        )

    return cam_infos


def load_mytalk_scene(config: ModelConfig) -> SceneInfo:
    train_cam_infos = [] if config.eval else _read_cameras_from_transforms(config, "transforms_train.json")
    test_cam_infos = _read_cameras_from_transforms(config, "transforms_val.json")
    if config.eval:
        train_cam_infos = test_cam_infos

    normalization = _get_nerfpp_norm(train_cam_infos)
    ply_path = os.path.join(config.path, "points3d.ply")
    num_points = config.init_num
    xyz = np.random.random((num_points, 3)) * 0.2 - 0.1
    shs = np.random.random((num_points, 3)) / 255.0
    point_cloud = BasicPointCloud(
        points=xyz,
        colors=SH2RGB(shs),
        normals=np.zeros((num_points, 3)),
    )

    return SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=normalization,
        ply_path=ply_path,
    )


class GaussianScene:
    def __init__(self, config: ModelConfig, gaussians, load_iteration=None, shuffle=True):
        self.model_path = config.workspace
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        scene_info = load_mytalk_scene(config)
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, config)
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, config)

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
