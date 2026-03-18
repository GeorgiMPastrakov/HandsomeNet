# Scripts

This directory holds thin project entrypoints.

The repository is structured so that scripts should stay minimal and delegate real logic into `src/handsomenet/`.

- `train.py` runs train/validation experiments and writes checkpoints plus overlays.
- `evaluate.py` loads a saved checkpoint and reports validation loss plus pixel error.
- `run_training_plan.py` runs the staged baseline and HandsomeNet training milestones.
- `verify_freihand.py` checks the FreiHAND projection and inverse-mapping contract.
- `webcam_demo.py` runs live webcam inference from a saved checkpoint with MediaPipe ROI
  detection and HandsomeNet pose overlay.
