# HandsomeNet

HandsomeNet is a lightweight, heatmap-free hand pose estimation project built around a custom architecture:

`224x224 RGB -> CNN backbone -> joint tokens -> graph attention -> (21, 2)`

This repository is set up to support the locked v1 scope:

- Dataset: FreiHAND only
- Task: single-image 2D hand pose estimation
- Input: `224x224` RGB
- Output: `21 x 2` normalized coordinates in final model-input space
- First checkpoint: verify the FreiHAND geometry and inverse-mapping contract before serious model work

## Current status

The FreiHAND dataset integration checkpoint is now implemented:

- canonical FreiHAND local layout under `data/raw/freihand/`
- 3D-to-2D projection using `training_xyz.json` and `training_K.json`
- geometry metadata for exact inverse mapping
- normalized `(21, 2)` target generation
- verification script for visual overlays and inversion checks

Model and training work remain downstream of this verified data contract.

## Repository layout

- `configs/`: project and experiment configuration files
- `docs/`: locked specifications and setup documentation
- `scripts/`: thin entrypoints for later project tasks
- `src/handsomenet/`: source package
- `tests/`: basic repo-level tests
- `data/`: local dataset storage and processed outputs
- `artifacts/`: checkpoints and run outputs

## Dataset location

Set the FreiHAND download directory in `.env` using:

- `HANDSOMENET_DATA_DIR=./data/raw/freihand`

Detailed dataset placement and required files are documented in [docs/dataset_setup.md](/Users/georgipastrakov/Personal%20projects/HandsomeNet/docs/dataset_setup.md).

## Development

This repo uses:

- Python `3.11+`
- Python packaging via `pyproject.toml`
- `pytest` for tests
- `ruff` for linting and formatting

The authoritative v1 specification lives in [docs/specs/handsomenet_v1_locked_spec.md](/Users/georgipastrakov/Personal%20projects/HandsomeNet/docs/specs/handsomenet_v1_locked_spec.md).

## Local environment

Use a local virtual environment with Python `3.11+`. The repository includes a `.python-version` file set to `3.11` so the expected interpreter version stays explicit.

## Dataset verification

Run the FreiHAND dataset checkpoint with:

- `.venv/bin/python scripts/verify_freihand.py`

This writes sample overlay images under `artifacts/runs/freihand_verification/` and reports the maximum inversion error for the selected samples.
