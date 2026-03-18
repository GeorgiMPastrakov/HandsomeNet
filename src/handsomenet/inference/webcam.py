"""Webcam inference helpers for HandsomeNet live testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn

from handsomenet.constants import INPUT_HEIGHT, INPUT_WIDTH, SKELETON_EDGES
from handsomenet.data.geometry import (
    build_geometry_metadata,
    invert_geometry_on_points,
    invert_normalized_points,
)
from handsomenet.models.factory import build_model
from handsomenet.types import GeometryMetadata


@dataclass(frozen=True)
class PreparedFrame:
    """Tensorized frame plus geometry metadata for inverse mapping."""

    image_bgr: np.ndarray
    input_tensor: torch.Tensor
    geometry: GeometryMetadata


class FpsTracker:
    """Simple exponential moving average FPS tracker."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0.0, 1.0).")
        self.smoothing = smoothing
        self._previous_time: float | None = None
        self._fps: float = 0.0

    def tick(self) -> float:
        current_time = perf_counter()
        if self._previous_time is None:
            self._previous_time = current_time
            return self._fps

        delta = current_time - self._previous_time
        self._previous_time = current_time
        if delta <= 0.0:
            return self._fps

        instant_fps = 1.0 / delta
        if self._fps == 0.0:
            self._fps = instant_fps
        else:
            self._fps = self.smoothing * self._fps + (1.0 - self.smoothing) * instant_fps
        return self._fps


class LandmarkSmoother:
    """Exponential moving average smoother for predicted landmarks."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0.")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def update(self, points_xy: np.ndarray) -> np.ndarray:
        current = np.asarray(points_xy, dtype=np.float32)
        if self._state is None or self.alpha == 0.0:
            self._state = current.copy()
            return current

        self._state = self.alpha * self._state + (1.0 - self.alpha) * current
        return self._state.copy()


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: Path,
    device: str,
) -> nn.Module:
    """Load a trained model checkpoint for inference."""

    model = build_model(model_name)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def prepare_frame(frame_bgr: np.ndarray, device: str) -> PreparedFrame:
    """Resize-pad a camera frame into the HandsomeNet input contract."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame_bgr.shape}")

    original_height, original_width = frame_bgr.shape[:2]
    geometry = build_geometry_metadata(
        original_width=original_width,
        original_height=original_height,
        final_width=INPUT_WIDTH,
        final_height=INPUT_HEIGHT,
    )
    resized_width = max(1, int(round(original_width * geometry.resize_scale)))
    resized_height = max(1, int(round(original_height * geometry.resize_scale)))

    resized_bgr = cv2.resize(
        frame_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_bgr = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    pad_x = int(round(geometry.pad_x))
    pad_y = int(round(geometry.pad_y))
    padded_bgr[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized_bgr

    rgb_input = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = (
        torch.from_numpy(rgb_input)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
        / 255.0
    )
    return PreparedFrame(image_bgr=frame_bgr, input_tensor=input_tensor, geometry=geometry)


def predict_frame_landmarks(
    model: nn.Module,
    prepared_frame: PreparedFrame,
) -> np.ndarray:
    """Run one inference step and map landmarks back to original frame pixels."""

    with torch.no_grad():
        normalized_points = model(prepared_frame.input_tensor)[0].detach().cpu().numpy()

    input_points = invert_normalized_points(normalized_points, INPUT_WIDTH, INPUT_HEIGHT)
    original_points = invert_geometry_on_points(input_points, prepared_frame.geometry)
    return original_points.astype(np.float32)


def draw_landmarks(
    frame_bgr: np.ndarray,
    points_xy: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw the fixed hand skeleton on a frame."""

    output = frame_bgr.copy()
    points = np.asarray(points_xy, dtype=np.float32)
    for start_index, end_index in SKELETON_EDGES:
        start = tuple(np.round(points[start_index]).astype(int).tolist())
        end = tuple(np.round(points[end_index]).astype(int).tolist())
        cv2.line(output, start, end, color, thickness=2, lineType=cv2.LINE_AA)

    for x_coord, y_coord in np.round(points).astype(int):
        cv2.circle(output, (int(x_coord), int(y_coord)), radius=3, color=color, thickness=-1)

    return output


def annotate_frame(
    frame_bgr: np.ndarray,
    model_name: str,
    device: str,
    fps: float,
) -> np.ndarray:
    """Render lightweight status text for live inference."""

    output = frame_bgr.copy()
    cv2.putText(
        output,
        f"{model_name} | {device} | {fps:.1f} FPS",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "q/esc quit | s save frame",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output
