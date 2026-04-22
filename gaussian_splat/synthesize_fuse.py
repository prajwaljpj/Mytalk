import copy
import os
import subprocess
import sys
from argparse import ArgumentParser
from os import makedirs

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from .gaussian_model import GaussianModel
from .motion_net import MotionNetwork, MouthMotionNetwork
from .provider import GaussianScene
from .renderer import render_motion, render_motion_mouth
from .train_utils import add_model_args, add_optimization_args, add_pipeline_args, extract_configs
from .utils.camera_utils import loadCamOnTheFly
from .utils.general_utils import safe_state, torch_load_unsafe


def dilate_fn(bin_img, ksize=13):
    pad = (ksize - 1) // 2
    return F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)


def mux_audio(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"[WARNING] ffmpeg audio mux failed:\n{result.stderr.decode()}")
    else:
        print(f"Saved with audio: {output_path}")


def render_set(model_path, name, views, gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate, audio_path=""):
    render_path = os.path.join(model_path, name, "renders")
    gts_path = os.path.join(model_path, name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    all_preds = []
    all_gts = []
    all_preds_face = []
    all_preds_mouth = []

    for idx, view in enumerate(views):
        print(f"\rRendering {idx + 1}/{len(views)}", end="", flush=True)
        if view.original_image is None:
            view = loadCamOnTheFly(copy.deepcopy(view))

        with torch.no_grad():
            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background)
            render_pkg_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, pipeline, background)

        if dilate:
            alpha_mouth = dilate_fn(render_pkg_mouth["alpha"][None])[0]
        else:
            alpha_mouth = render_pkg_mouth["alpha"]

        mouth_image = render_pkg_mouth["render"] + view.background.cuda() / 255.0 * (1.0 - alpha_mouth)
        alpha = render_pkg["alpha"]
        image = render_pkg["render"] + mouth_image * (1.0 - alpha)

        pred = (image[0:3, ...].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        all_preds.append(pred)

        if not fast:
            all_preds_face.append((render_pkg["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            all_preds_mouth.append((render_pkg_mouth["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            if view.original_image is not None:
                all_gts.append(view.original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    print()
    silent_path = os.path.join(render_path, "out.mp4")
    imageio.mimwrite(silent_path, all_preds, fps=25, quality=8, macro_block_size=1)
    print(f"Saved: {silent_path}")

    wav = audio_path if audio_path and not audio_path.endswith(".npy") else ""
    if not wav:
        wav = os.path.join(model_path, "..", "aud.wav")  # fallback to subject aud.wav
        if not os.path.exists(wav):
            wav = ""
    if wav:
        mux_audio(silent_path, wav, os.path.join(render_path, "out_audio.mp4"))

    if not fast:
        if all_gts:
            imageio.mimwrite(os.path.join(gts_path, "out.mp4"), all_gts, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(render_path, "out_face.mp4"), all_preds_face, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(render_path, "out_mouth.mp4"), all_preds_mouth, fps=25, quality=8, macro_block_size=1)


def render_sets(dataset, pipeline, use_train, fast, dilate):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians_mouth = GaussianModel(dataset.sh_degree)
        scene = GaussianScene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda()
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()

        ckpt_path = os.path.join(scene.model_path, "chkpnt_fuse_latest.pth")
        (model_params, motion_params, model_mouth_params, motion_mouth_params) = torch_load_unsafe(ckpt_path)
        gaussians.restore(model_params, None)
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians_mouth.restore(model_mouth_params, None)
        motion_net_mouth.load_state_dict(motion_mouth_params, strict=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        split = "train" if use_train else "test"
        cameras = scene.getTrainCameras() if use_train else scene.getTestCameras()
        render_set(scene.model_path, split, cameras, gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate, audio_path=dataset.audio)


if __name__ == "__main__":
    parser = ArgumentParser(description="Synthesize talking head video from trained Gaussian model")
    add_model_args(parser)
    add_optimization_args(parser)
    add_pipeline_args(parser)
    parser.add_argument("--use_train", action="store_true", help="Render training cameras instead of test cameras")
    parser.add_argument("--fast", action="store_true", help="Skip per-component videos and GT, only save final composite")
    parser.add_argument("--dilate", action="store_true", help="Dilate mouth alpha mask to soften boundary")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)

    dataset, _, pipeline = extract_configs(args)
    print(f"Synthesizing from: {dataset.model_path}")
    render_sets(dataset, pipeline, args.use_train, args.fast, args.dilate)
    print("Done.")
