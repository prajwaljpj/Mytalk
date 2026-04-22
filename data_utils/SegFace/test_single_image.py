#!/usr/bin/env python3
"""
Simple test script for SegFace inference on a single image
Usage: python test_single_image.py --image path/to/face.jpg
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add SegFace to path
sys.path.insert(0, os.path.dirname(__file__))

from network import get_model

# Color mapping for visualization (CelebAMask-HQ dataset)
COLOR_MAPPING = np.array([
    [0, 0, 0],        # 0: background
    [128, 64, 0],     # 1: neck
    [200, 80, 80],    # 2: skin
    [0, 192, 0],      # 3: cloth
    [64, 0, 0],       # 4: l_ear
    [192, 0, 0],      # 5: r_ear
    [0, 128, 128],    # 6: l_brow
    [128, 128, 128],  # 7: r_brow
    [0, 0, 128],      # 8: l_eye
    [128, 0, 128],    # 9: r_eye
    [0, 128, 0],      # 10: nose
    [64, 128, 0],     # 11: mouth
    [64, 0, 128],     # 12: l_lip
    [192, 128, 0],    # 13: u_lip
    [192, 0, 128],    # 14: hair
    [128, 128, 0],    # 15: eye_g (eyeglasses)
    [64, 128, 128],   # 16: hat
    [192, 128, 128],  # 17: ear_r (earring)
    [0, 64, 0]        # 18: neck_l (necklace)
], dtype=np.uint8)

LABELS = ['background', 'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
          'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair', 'eye_g',
          'hat', 'ear_r', 'neck_l']


def preprocess_image(image_path, resolution=512):
    """Load and preprocess image for SegFace"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model resolution
    image_resized = cv2.resize(image_rgb, (resolution, resolution))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std

    # Convert to tensor: HWC -> CHW
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()

    return image_tensor.unsqueeze(0), image  # Add batch dimension


def visualize_mask(mask, alpha=0.6):
    """Convert mask to colored visualization"""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for index, color in enumerate(COLOR_MAPPING):
        color_mask[mask == index] = color

    return color_mask


def run_inference(args):
    """Run inference on a single image"""
    print(f"Loading model from {args.model_path}...")

    # Load model
    model = get_model(args.backbone, args.resolution, args.model).cuda()
    model.eval()

    # Load weights
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict_backbone'])
    print("✓ Model loaded successfully")

    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor, original_image = preprocess_image(args.image, args.resolution)

    # Run inference
    with torch.no_grad():
        # Create dummy labels and dataset tensors (not used in inference)
        dummy_labels = {'segmentation': torch.zeros(1, args.resolution, args.resolution).cuda()}
        dummy_dataset = torch.zeros(1).cuda()

        # Forward pass
        image_tensor = image_tensor.cuda()
        output = model(image_tensor, dummy_labels, dummy_dataset)

        # Get prediction
        output = F.interpolate(output, size=(args.resolution, args.resolution),
                              mode='bilinear', align_corners=False)
        output = output.softmax(dim=1)
        pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    print("✓ Inference completed")

    # Visualize results
    color_mask = visualize_mask(pred_mask)

    # Resize original image to match resolution
    original_resized = cv2.resize(original_image, (args.resolution, args.resolution))

    # Create overlay
    overlay = cv2.addWeighted(original_resized, 1-args.alpha, color_mask, args.alpha, 0)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.image))[0])

    cv2.imwrite(f"{output_prefix}_original.jpg", original_resized)
    cv2.imwrite(f"{output_prefix}_mask.jpg", color_mask)
    cv2.imwrite(f"{output_prefix}_overlay.jpg", overlay)

    print(f"\n✓ Results saved:")
    print(f"  - Original: {output_prefix}_original.jpg")
    print(f"  - Mask: {output_prefix}_mask.jpg")
    print(f"  - Overlay: {output_prefix}_overlay.jpg")

    # Print class statistics
    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"\n📊 Detected facial regions:")
    for idx, count in zip(unique, counts):
        percentage = (count / pred_mask.size) * 100
        if percentage > 0.5:  # Only show regions > 0.5%
            print(f"  - {LABELS[idx]}: {percentage:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SegFace on a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input face image")
    parser.add_argument("--model_path", type=str,
                        default="weights/convnext_celeba_512/model_299.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="segface_celeb",
                        help="Model backbone (segface_celeb, segface_lapa, segface_helen)")
    parser.add_argument("--model", type=str, default="convnext_base",
                        help="Model architecture (convnext_base, swin_base, mobilenet, efficientnet, etc.)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Input resolution (default: 512)")
    parser.add_argument("--output_dir", type=str, default="test_output",
                        help="Output directory for results")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Overlay transparency (0-1, default: 0.6)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found: {args.model_path}")
        print("Please download the model first using download_model.py")
        sys.exit(1)

    run_inference(args)
