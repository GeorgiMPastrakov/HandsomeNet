# FreiHAND Dataset Setup

## Locked v1 dataset contract

- Dataset: FreiHAND only
- Image source: FreiHAND training RGB images
- Raw supervision source: FreiHAND `training_xyz.json` and `training_K.json`
- Input space: final `224x224` model-input image
- Target space: normalized `(21, 2)` coordinates in `[0, 1]`

## Expected local dataset location

Place the downloaded FreiHAND dataset under:

- `data/raw/freihand/`

Expected top-level contents should include:

- `training/rgb/`
- `training_xyz.json`
- `training_K.json`
- optionally `evaluation/rgb/` and evaluation metadata

If the downloaded archive initially extracts into an extra nested folder, normalize it so the files above live directly under `data/raw/freihand/`.

## Minimum required files for the v1 checkpoint

- `training/rgb/`
- `training_xyz.json`
- `training_K.json`

## Observed FreiHAND layout used by HandsomeNet v1

The local FreiHAND training set used for this project contains:

- `32,560` unique supervision entries in `training_xyz.json`
- `32,560` camera matrices in `training_K.json`
- `130,240` training RGB images under `training/rgb/`
- `3,960` evaluation RGB images under `evaluation/rgb/`

## What gets verified first

Before model implementation proceeds, the repository is meant to verify:

1. The exact FreiHAND annotation load path.
2. Projection of 3D joints into image space using `K`.
3. The image/label preprocessing contract.
4. Final normalized `(21, 2)` supervision targets in `224x224` space.
5. Inverse mapping back to original image space.
6. Visual overlays confirming alignment.

## Notes on geometry

For native FreiHAND training images, preprocessing may be identity or trivial because the dataset images are already `224x224`. The geometry contract still remains explicit: any transform applied to the image must also be applied to the projected 2D joints, and sufficient metadata must be preserved for exact inversion.
