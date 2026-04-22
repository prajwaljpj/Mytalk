"""
SegFace face parsing wrapper for SyncTalk preprocessing

SegFace CelebAMask-HQ classes (19 total):
    0: background
    1: neck
    2: skin (face)
    3: cloth
    4: l_ear, 5: r_ear
    6: l_brow, 7: r_brow
    8: l_eye, 9: r_eye
    10: nose
    11: mouth
    12: l_lip, 13: u_lip
    14: hair
    15: eye_g (eyeglasses)
    16: hat
    17: ear_r (earrings)
    18: neck_l (necklace)

Mapping to SyncTalk format:
- Head (RED: 255,0,0): classes 2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 (all facial features)
- Neck (GREEN: 0,255,0): class 1 (neck)
- Torso (BLUE: 0,0,255): class 3 (cloth)
- Background (WHITE: 255,255,255): class 0 (background)

Face-only mask (for optical flow):
- Face (RED: 255,0,0): classes 2,6,7,8,9,10,11,12,13 (skin, brows, eyes, nose, mouth, lips)
"""

import sys
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.console import Console

# Add SegFace to path
segface_path = os.path.join(os.path.dirname(__file__), "../SegFace")
sys.path.insert(0, segface_path)

from network import get_model

console = Console()


def create_parsing_maps(parsing_anno, img_size):
    """
    Create both full body segmentation and face-specific masks from SegFace output

    Args:
        parsing_anno: SegFace segmentation mask (19 classes, 0-18)
        img_size: (width, height) tuple for output size

    Returns:
        full_mask: Full segmentation (head/neck/torso/bg)
        face_mask: Face-only mask (for optical flow)
    """
    vis_parsing_anno = parsing_anno.astype(np.uint8)

    # Initialize with white background (255, 255, 255)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)
    ) + np.array([255, 255, 255])
    vis_parsing_anno_color_face = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)
    ) + np.array([255, 255, 255])

    # Full body segmentation mapping
    # Head parts (RED): skin, ears, brows, eyes, nose, lips, glasses, earrings, necklace
    # Excludes mouth(11), hair(14), hat(16) which need distinct colors for Gaussian training
    head_classes = [2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 18]
    for cls in head_classes:
        index = np.where(vis_parsing_anno == cls)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

    # Mouth interior (GREY: 100,100,100) — required by Gaussian training mouth_mask detection
    mouth_index = np.where(vis_parsing_anno == 11)
    vis_parsing_anno_color[mouth_index[0], mouth_index[1], :] = np.array([100, 100, 100])

    # Hair + Hat (BLACK: 0,0,0) — required by Gaussian training hair_mask detection
    for cls in [14, 16]:
        index = np.where(vis_parsing_anno == cls)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 0])

    # Neck (GREEN)
    neck_index = np.where(vis_parsing_anno == 1)
    vis_parsing_anno_color[neck_index[0], neck_index[1], :] = np.array([0, 255, 0])

    # Torso/Cloth (BLUE)
    torso_index = np.where(vis_parsing_anno == 3)
    vis_parsing_anno_color[torso_index[0], torso_index[1], :] = np.array([0, 0, 255])

    # Resize to original image size
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    full_mask = cv2.resize(
        vis_parsing_anno_color, img_size, interpolation=cv2.INTER_NEAREST
    )

    # Face-specific mask (for optical flow)
    # Core face parts only: skin, brows, eyes, nose, mouth, lips
    face_classes = [2, 6, 7, 8, 9, 10, 11, 12, 13]
    for cls in face_classes:
        index = np.where(vis_parsing_anno == cls)
        vis_parsing_anno_color_face[index[0], index[1], :] = np.array([255, 0, 0])

    # Add padding below face boundary (similar to original implementation)
    pad = 5
    vis_parsing_anno_color_face = vis_parsing_anno_color_face.astype(np.uint8)
    face_part = (
        (vis_parsing_anno_color_face[..., 0] == 255)
        & (vis_parsing_anno_color_face[..., 1] == 0)
        & (vis_parsing_anno_color_face[..., 2] == 0)
    )

    if np.any(face_part):
        face_coords = np.stack(np.nonzero(face_part), axis=-1)
        sorted_inds = np.lexsort((-face_coords[:, 0], face_coords[:, 1]))
        sorted_face_coords = face_coords[sorted_inds]
        u, uid, ucnt = np.unique(
            sorted_face_coords[:, 1], return_index=True, return_counts=True
        )
        bottom_face_coords = sorted_face_coords[uid] + np.array([pad, 0])
        rows, cols, _ = vis_parsing_anno_color_face.shape

        # Clip coordinates to image bounds
        bottom_face_coords[:, 0] = np.clip(bottom_face_coords[:, 0], 0, rows - 1)

        y_min = np.min(bottom_face_coords[:, 1])
        y_max = np.max(bottom_face_coords[:, 1])

        # Add padding in middle sections (2nd and 3rd quarters)
        y_range = y_max - y_min
        height_per_part = y_range // 4

        start_y_part1 = y_min + height_per_part
        end_y_part1 = start_y_part1 + height_per_part

        start_y_part2 = end_y_part1
        end_y_part2 = start_y_part2 + height_per_part

        for coord in bottom_face_coords:
            x, y = coord
            start_x = max(x - pad, 0)
            end_x = min(x + pad, rows)
            if start_y_part1 <= y <= end_y_part1 or start_y_part2 <= y <= end_y_part2:
                vis_parsing_anno_color_face[start_x:end_x, y] = [255, 0, 0]

        # Apply Gaussian blur for smooth transitions
        vis_parsing_anno_color_face = cv2.GaussianBlur(
            vis_parsing_anno_color_face, (9, 9), cv2.BORDER_DEFAULT
        )

    face_mask = cv2.resize(
        vis_parsing_anno_color_face, img_size, interpolation=cv2.INTER_NEAREST
    )

    return full_mask, face_mask


def preprocess_image(image, resolution=512):
    """Preprocess image for SegFace model"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Store original size
    original_size = image_rgb.shape[:2]  # H, W

    # Resize to model resolution
    image_resized = cv2.resize(image_rgb, (resolution, resolution))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std

    # Convert to tensor: HWC -> CHW
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()

    return image_tensor.unsqueeze(0), original_size


def evaluate(
    respth="./res/test_res", dspth="./data", model_name="convnext_base", resolution=512
):
    """
    Process images with SegFace model and create parsing masks

    Args:
        respth: Output directory for parsing results
        dspth: Input directory containing images
        model_name: SegFace model architecture (convnext_base, swin_base, mobilenet, etc.)
        resolution: Model input resolution (default: 512)
    """
    Path(respth).mkdir(parents=True, exist_ok=True)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(
        f"[bold blue][INFO][/bold blue] Loading SegFace model ([yellow]{model_name}[/yellow]) on [yellow]{device}[/yellow]"
    )

    # Map model names to weight directory names
    weight_dir_mapping = {
        "convnext_base": "convnext_celeba_512",
        "swin_base": "swinb_celeba_512",
        "mobilenet": "mobilenet_celeba_512",
        "efficientnet": "efficientnet_celeba_512",
    }

    weight_dir = weight_dir_mapping.get(model_name, f"{model_name}_celeba_512")
    model_path = os.path.join(segface_path, f"weights/{weight_dir}/model_299.pt")

    if not os.path.exists(model_path):
        console.print(
            f"[bold red][ERROR][/bold red] SegFace model not found at {model_path}"
        )
        console.print(f"[yellow]Please download the model first:[/yellow]")
        console.print(f"  cd data_utils/SegFace")
        console.print(f"  uv run python download_model.py")
        sys.exit(1)

    # Initialize model
    model = get_model("segface_celeb", resolution, model_name).to(device)
    model.eval()

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict_backbone"])

    console.print(
        f"[bold green][INFO][/bold green] ✓ SegFace model loaded successfully"
    )

    # Get list of images
    image_paths = sorted(
        glob.glob(os.path.join(dspth, "*.jpg"))
        + glob.glob(os.path.join(dspth, "*.png")),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing with SegFace", total=len(image_paths)
        )

        with torch.no_grad():
            for image_path in image_paths:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    console.print(
                        f"[yellow]Warning: Failed to read {image_path}, skipping[/yellow]"
                    )
                    progress.advance(task)
                    continue

                ori_h, ori_w = image.shape[:2]
                ori_size = (ori_w, ori_h)  # (width, height)

                # Preprocess
                image_tensor, _ = preprocess_image(image, resolution)
                image_tensor = image_tensor.to(device)

                # Create dummy inputs (required by SegFace model signature)
                dummy_labels = {
                    "segmentation": torch.zeros(1, resolution, resolution).to(device)
                }
                dummy_dataset = torch.zeros(1).to(device)

                # Run inference
                output = model(image_tensor, dummy_labels, dummy_dataset)
                output = F.interpolate(
                    output,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                output = output.softmax(dim=1)
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

                # Resize mask to original image size
                pred_mask_resized = cv2.resize(
                    pred_mask, ori_size, interpolation=cv2.INTER_NEAREST
                )

                # Create both full and face masks
                full_mask, face_mask = create_parsing_maps(pred_mask_resized, ori_size)

                # Save masks with consistent naming (matching original pipeline)
                frame_num = int(os.path.basename(image_path).split(".")[0])
                output_name = f"{frame_num}.png"

                cv2.imwrite(os.path.join(respth, output_name), full_mask)
                cv2.imwrite(
                    os.path.join(respth, output_name.replace(".png", "_face.png")),
                    face_mask,
                )

                progress.advance(task)

    console.print(
        f"[bold green][INFO][/bold green] ✓ SegFace processing completed successfully"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="SegFace wrapper for face parsing")
    parser.add_argument(
        "--respath", type=str, default="./result/", help="result path for label"
    )
    parser.add_argument(
        "--imgpath", type=str, default="./imgs/", help="path for input images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnext",
        help="SegFace model architecture (convnext, swin_base, mobilenet, efficientnet)",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="model input resolution"
    )

    args = parser.parse_args()

    # Map short names to full model names
    model_mapping = {
        "convnext": "convnext_base",
        "swin": "swin_base",
        "swin_base": "swin_base",
        "mobilenet": "mobilenet",
        "efficientnet": "efficientnet",
        "convnext_base": "convnext_base",
    }

    model_name = model_mapping.get(args.model, args.model)

    evaluate(
        respth=args.respath,
        dspth=args.imgpath,
        model_name=model_name,
        resolution=args.resolution,
    )
