#!/usr/bin/env python3
"""Download SegFace pre-trained model from HuggingFace"""

from huggingface_hub import hf_hub_download
import os

# Create weights directory
os.makedirs("weights", exist_ok=True)

# Download the best model (ConvNext - highest accuracy)
print("Downloading ConvNext model (best quality, 89.22 F1 score)...")
hf_hub_download(
    repo_id="kartiknarayan/SegFace",
    filename="convnext_celeba_512/model_299.pt",
    local_dir="./weights"
)
print("✓ Model downloaded to weights/convnext_celeba_512/model_299.pt")

# Optionally download faster model (MobileNet - for quick testing)
# Uncomment below if you need a faster model for experimentation
# print("\nDownloading MobileNet model (fastest, 87.91 F1 score)...")
# hf_hub_download(
#     repo_id="kartiknarayan/SegFace",
#     filename="mobilenet_celeba_512/model_299.pt",
#     local_dir="./weights"
# )
# print("✓ Model downloaded to weights/mobilenet_celeba_512/model_299.pt")
