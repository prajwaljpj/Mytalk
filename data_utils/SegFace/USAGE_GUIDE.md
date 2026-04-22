# SegFace Usage Guide for SyncTalk

## ✅ Installation Complete

All dependencies are installed and working correctly!

## 📦 Available Models

**✅ Recommended for preprocessing: ConvNext**

You currently have downloaded:
- **ConvNext (512px)**: `weights/convnext_celeba_512/model_299.pt` ⭐
  - Best quality (89.22 F1 score)
  - Excellent long-tail class detection (earrings, necklaces, hats)
  - **Recommended for SyncTalk preprocessing**

- **MobileNet (512px)**: `weights/mobilenet_celeba_512/model_299.pt`
  - Fastest model (95.96 FPS)
  - Mean F1: 87.91
  - Best for: Quick testing only

### Download Additional Models

```bash
# Best quality model (recommended for final processing)
uv run python download_model.py  # Edit to uncomment ConvNext model

# Or manually download specific models:
from huggingface_hub import hf_hub_download

# ConvNext - Best quality (89.22 F1)
hf_hub_download(repo_id="kartiknarayan/SegFace",
                filename="convnext_celeba_512/model_299.pt",
                local_dir="./weights")

# Swin Base - Good balance (88.96 F1)
hf_hub_download(repo_id="kartiknarayan/SegFace",
                filename="swinb_celeba_512/model_299.pt",
                local_dir="./weights")
```

## 🚀 Quick Test

```bash
# Test on a single image (uses ConvNext by default)
cd data_utils/SegFace
uv run python test_single_image.py --image /path/to/face.jpg

# With custom options:
uv run python test_single_image.py \
    --image /path/to/face.jpg \
    --model_path weights/convnext_celeba_512/model_299.pt \
    --backbone segface_celeb \
    --model convnext_base \
    --resolution 512 \
    --output_dir test_output \
    --alpha 0.6
```

## 🔄 Batch Process SyncTalk Dataset

Process all frames in your SyncTalk training data:

```bash
cd data_utils/SegFace

# Process all frames (ConvNext - best quality)
uv run python batch_process.py --data_dir ../../data/YourID

# Custom options:
uv run python batch_process.py \
    --data_dir ../../data/Prasad \
    --image_subdir ori_imgs \
    --output_subdir segface_masks \
    --model_path weights/convnext_celeba_512/model_299.pt \
    --model convnext_base \
    --resolution 512 \
    --resize_to_original  # Resize masks back to original image size

# Output: data/YourID/segface_masks/*.png (19-class masks)
```

## 📊 Facial Regions Detected (CelebAMask-HQ)

The model segments **19 facial regions**:
1. **background** - Background/non-face
2. **neck** - Neck area
3. **skin** - Facial skin
4. **cloth** - Clothing
5. **l_ear, r_ear** - Left/right ears
6. **l_brow, r_brow** - Left/right eyebrows
7. **l_eye, r_eye** - Left/right eyes
8. **nose** - Nose
9. **mouth** - Mouth cavity
10. **l_lip, u_lip** - Lower/upper lips
11. **hair** - Hair
12. **eye_g** - Eyeglasses (long-tail class)
13. **hat** - Hat (long-tail class)
14. **ear_r** - Earrings (long-tail class)
15. **neck_l** - Necklace (long-tail class)

## 🔧 Integration with SyncTalk Preprocessing

To use SegFace in your preprocessing pipeline:

```python
import sys
sys.path.insert(0, 'data_utils/SegFace')

from network import get_model
import torch
import cv2
import numpy as np

# Load model
model = get_model("segface_celeb", 512, "mobilenet").cuda()
checkpoint = torch.load("data_utils/SegFace/weights/mobilenet_celeba_512/model_299.pt")
model.load_state_dict(checkpoint['state_dict_backbone'])
model.eval()

# Process image
def segment_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std

    # To tensor
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    image_tensor = image_tensor.unsqueeze(0).cuda()

    # Inference
    with torch.no_grad():
        dummy_labels = {'segmentation': torch.zeros(1, 512, 512).cuda()}
        dummy_dataset = torch.zeros(1).cuda()
        output = model(image_tensor, dummy_labels, dummy_dataset)
        output = output.softmax(dim=1)
        mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    return mask  # Shape: [512, 512], values 0-18

# Use in preprocessing
mask = segment_face("data/MyID/imgs/0001.jpg")
```

## 🎯 Model Selection Guide

| Model | Resolution | F1 Score | Speed (FPS) | Use Case |
|-------|-----------|----------|-------------|----------|
| **MobileNet** | 512 | 87.91 | 95.96 | Real-time, preprocessing |
| **EfficientNet** | 512 | 88.94 | ~50 | Balanced |
| **Swin Base** | 512 | 88.96 | ~30 | Good quality |
| **ConvNext** | 512 | 89.22 | ~25 | Best quality |

## 🔍 Compatibility Notes

✅ **Working with your setup:**
- PyTorch 2.x ✓
- CUDA 12.8 ✓
- UV package manager ✓
- Latest versions of mmsegmentation, timm, accelerate ✓

## 📝 Next Steps

1. **Test on your SyncTalk dataset:**
   ```bash
   cd data_utils/SegFace
   uv run python test_single_image.py \
       --image ../../data/YourID/imgs/0001.jpg
   ```

2. **Batch process all frames:**
   Create a loop to process all training frames

3. **Compare with existing segmentation:**
   Compare SegFace results vs. current face parsing model

4. **Integrate into preprocessing:**
   Replace `data_utils/face_parsing` with SegFace if better results

## 🐛 Troubleshooting

**Out of memory:**
```bash
# Use smaller resolution
--resolution 256  # or 224

# Use MobileNet instead of Swin/ConvNext
--model mobilenet
```

**Wrong segmentation:**
- Ensure face is clearly visible
- Try different model backbones
- Check if image is 25 FPS aligned with your video

**Import errors:**
```bash
# Make sure you're in the SegFace directory
cd data_utils/SegFace

# Or add to path in Python
import sys
sys.path.insert(0, 'data_utils/SegFace')
```
