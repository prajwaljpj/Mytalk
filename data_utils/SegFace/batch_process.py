#!/usr/bin/env python3
"""
Batch process all frames in a SyncTalk dataset with SegFace
Usage: python batch_process.py --data_dir ../../data/YourID
"""

import os
import sys
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from network import get_model


def preprocess_image(image_path, resolution=512):
    """Load and preprocess image for SegFace"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]  # H, W
    image_resized = cv2.resize(image_rgb, (resolution, resolution))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std

    # Convert to tensor: HWC -> CHW
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    return image_tensor.unsqueeze(0), original_size


@torch.no_grad()
def batch_process(args):
    """Process all images in a directory"""

    # Find all images
    image_patterns = [
        os.path.join(args.data_dir, args.image_subdir, "*.jpg"),
        os.path.join(args.data_dir, args.image_subdir, "*.png"),
        os.path.join(args.data_dir, args.image_subdir, "*.jpeg"),
    ]

    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"❌ No images found in {os.path.join(args.data_dir, args.image_subdir)}")
        return

    print(f"Found {len(image_files)} images to process")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_model(args.backbone, args.resolution, args.model).cuda()
    model.eval()

    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict_backbone'])
    print("✓ Model loaded successfully\n")

    # Create output directory
    output_dir = os.path.join(args.data_dir, args.output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    print(f"Processing images...")
    print(f"Output directory: {output_dir}")

    for img_path in tqdm(image_files, desc="Segmenting faces"):
        # Preprocess
        image_tensor, original_size = preprocess_image(img_path, args.resolution)
        if image_tensor is None:
            print(f"⚠️  Failed to read: {img_path}")
            continue

        # Inference
        image_tensor = image_tensor.cuda()
        dummy_labels = {'segmentation': torch.zeros(1, args.resolution, args.resolution).cuda()}
        dummy_dataset = torch.zeros(1).cuda()

        output = model(image_tensor, dummy_labels, dummy_dataset)
        output = F.interpolate(output, size=(args.resolution, args.resolution),
                              mode='bilinear', align_corners=False)
        output = output.softmax(dim=1)
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

        # Resize mask back to original size if needed
        if args.resize_to_original:
            pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Save mask
        basename = os.path.basename(img_path)
        output_name = os.path.splitext(basename)[0] + '.png'
        output_path = os.path.join(output_dir, output_name)

        cv2.imwrite(output_path, pred_mask.astype(np.uint8))

    print(f"\n✓ Processed {len(image_files)} images")
    print(f"✓ Masks saved to: {output_dir}")

    # Print statistics
    print("\n📊 Segmentation format:")
    print("  - Mask values: 0-18 (19 classes)")
    print("  - 0: background, 1: neck, 2: skin, 3: cloth, 4: l_ear, 5: r_ear")
    print("  - 6: l_brow, 7: r_brow, 8: l_eye, 9: r_eye, 10: nose, 11: mouth")
    print("  - 12: l_lip, 13: u_lip, 14: hair, 15: eye_g, 16: hat, 17: ear_r, 18: neck_l")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process SyncTalk dataset with SegFace")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to SyncTalk data directory (e.g., ../../data/MyID)")
    parser.add_argument("--image_subdir", type=str, default="ori_imgs",
                        help="Subdirectory containing images (default: ori_imgs)")
    parser.add_argument("--output_subdir", type=str, default="segface_masks",
                        help="Output subdirectory for masks (default: segface_masks)")
    parser.add_argument("--model_path", type=str,
                        default="weights/convnext_celeba_512/model_299.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="segface_celeb",
                        help="Model backbone (segface_celeb, segface_lapa, segface_helen)")
    parser.add_argument("--model", type=str, default="convnext_base",
                        help="Model architecture (convnext_base, swin_base, mobilenet, etc.)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Processing resolution (default: 512)")
    parser.add_argument("--resize_to_original", action="store_true",
                        help="Resize masks back to original image size")

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found: {args.model_path}")
        print("Please download the model first using download_model.py")
        sys.exit(1)

    batch_process(args)
