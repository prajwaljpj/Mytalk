# Gaussian Splatting In MyTalk

This package hosts a MyTalk-native Gaussian-splatting training path built from
TalkingGaussian's renderer, Gaussian model, and motion networks.

## Expected Subject Layout

The subject directory passed with `--path` must already contain MyTalk
preprocessing outputs:

- `transforms_train.json`
- `transforms_val.json`
- `gt_imgs/*.jpg`
- `torso_imgs/*.png`
- `parsing/*.png`
- `ori_imgs/*.lms`
- `bc.jpg`
- `aud.wav`

It also expects TalkingGaussian-style control assets:

- `au.csv` from OpenFace
- optional `teeth_mask/*.npy`

Audio features are loaded as follows:

- `--audio_extractor ave`: computes or reuses `aud_ave.npy`
- `--audio_extractor deepspeech`: loads `aud_ds.npy`
- `--audio_extractor hubert`: loads `aud_hu.npy`
- `--audio_extractor esperanto`: loads `aud_eo.npy`

## Entry Points

Train the face model:

```bash
python train_gaussian_face.py --path data/<subject> --workspace output/<subject> --audio_extractor ave
```

Train the mouth model:

```bash
python train_gaussian_mouth.py --path data/<subject> --workspace output/<subject> --audio_extractor ave
```

Fuse the face and mouth checkpoints:

```bash
python train_gaussian_fuse.py --path data/<subject> --workspace output/<subject> --audio_extractor ave
```

## Runtime Requirements

- PyTorch must be available in the selected Python environment.
- `plyfile` is required for point-cloud checkpoints.
- The CUDA extensions from:
  - `submodules/diff-gaussian-rasterization`
  - `submodules/simple-knn`
  - `gridencoder`
  - `freqencoder`
  - `shencoder`
  must be installed/built before training.
