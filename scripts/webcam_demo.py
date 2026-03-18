"""Run real-time webcam inference for a trained baseline or HandsomeNet checkpoint."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import cv2

from handsomenet.inference.webcam import (
    FpsTracker,
    LandmarkSmoother,
    annotate_frame,
    draw_landmarks,
    load_checkpoint_model,
    predict_frame_landmarks,
    prepare_frame,
)
from handsomenet.training.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "handsomenet"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--window-name", type=str, default="HandsomeNet Webcam Demo")
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts/webcam"))
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_checkpoint_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(_camera_open_error(args.camera_index))

    fps_tracker = FpsTracker()
    smoother = LandmarkSmoother(alpha=args.smoothing_alpha)
    save_index = 0

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                raise RuntimeError("Failed to read a frame from the webcam.")

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            prepared_frame = prepare_frame(frame_bgr, device=device)
            landmarks_xy = predict_frame_landmarks(model, prepared_frame)
            smoothed_landmarks_xy = smoother.update(landmarks_xy)

            fps = fps_tracker.tick()
            rendered_frame = draw_landmarks(frame_bgr, smoothed_landmarks_xy)
            rendered_frame = annotate_frame(rendered_frame, args.model, device, fps)
            cv2.imshow(args.window_name, rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.save_dir / f"{args.model}_frame_{save_index:03d}.jpg"
                cv2.imwrite(str(output_path), rendered_frame)
                save_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

def _camera_open_error(camera_index: int) -> str:
    message = f"Could not open camera index {camera_index}."
    if platform.system() == "Darwin":
        message += (
            " macOS camera access is likely blocked for the app hosting Python. "
            "Open System Settings -> Privacy & Security -> Camera and enable access for "
            "Terminal, iTerm, Visual Studio Code, or Codex, depending on where "
            "you ran the command. "
            "Then fully restart that app and rerun `scripts/webcam_demo.py`."
        )
    return message


if __name__ == "__main__":
    main()
