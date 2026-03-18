"""Webcam inference helpers for HandsomeNet live testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from handsomenet.constants import INPUT_HEIGHT, INPUT_WIDTH, SKELETON_EDGES
from handsomenet.models.factory import build_model


@dataclass(frozen=True)
class RoiBox:
    """Square ROI in original frame coordinates."""

    left: int
    top: int
    size: int

    @property
    def right(self) -> int:
        return self.left + self.size

    @property
    def bottom(self) -> int:
        return self.top + self.size


@dataclass(frozen=True)
class PreparedFrame:
    """Tensorized ROI crop plus original-frame mapping metadata."""

    source_frame_bgr: np.ndarray
    crop_bgr: np.ndarray
    input_tensor: torch.Tensor
    roi_box: RoiBox


@dataclass(frozen=True)
class TrackerResult:
    """ROI tracker output for one frame."""

    roi_box: RoiBox | None
    detected_this_frame: bool
    using_grace_window: bool


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

    def reset(self) -> None:
        self._state = None


class HandRoiTracker:
    """Single-hand ROI tracker using MediaPipe only for hand localization."""

    def __init__(
        self,
        detection_confidence: float = 0.5,
        roi_expansion: float = 1.45,
        grace_frames: int = 8,
    ) -> None:
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be in [0.0, 1.0].")
        if roi_expansion < 1.0:
            raise ValueError("roi_expansion must be at least 1.0.")
        if grace_frames < 0:
            raise ValueError("grace_frames must be non-negative.")

        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MediaPipe is required for webcam ROI detection. "
                "Install project dependencies again so `mediapipe` is available."
            ) from exc

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
        )
        self.roi_expansion = roi_expansion
        self.grace_frames = grace_frames
        self._active_roi: RoiBox | None = None
        self._missed_frames = 0

    def detect(self, frame_bgr: np.ndarray) -> TrackerResult:
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        candidate_roi: RoiBox | None = None
        if results.multi_hand_landmarks:
            candidate_roi = _select_best_roi(
                multi_hand_landmarks=results.multi_hand_landmarks,
                multi_handedness=results.multi_handedness,
                frame_width=frame_width,
                frame_height=frame_height,
                roi_expansion=self.roi_expansion,
            )

        roi_box, missed_frames, detected_this_frame = resolve_tracked_roi(
            candidate_roi=candidate_roi,
            previous_roi=self._active_roi,
            missed_frames=self._missed_frames,
            grace_frames=self.grace_frames,
        )
        self._active_roi = roi_box
        self._missed_frames = missed_frames
        return TrackerResult(
            roi_box=roi_box,
            detected_this_frame=detected_this_frame,
            using_grace_window=roi_box is not None and not detected_this_frame,
        )

    def close(self) -> None:
        self._hands.close()


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


def prepare_frame(
    frame_bgr: np.ndarray,
    device: str,
    roi_box: RoiBox | None = None,
) -> PreparedFrame:
    """Extract an ROI crop and resize it into the HandsomeNet input contract."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame_bgr.shape}")

    if roi_box is None:
        size = max(frame_bgr.shape[0], frame_bgr.shape[1])
        roi_box = RoiBox(left=0, top=0, size=size)

    crop_bgr = extract_square_roi(frame_bgr, roi_box)
    resized_bgr = cv2.resize(crop_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb_input = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = (
        torch.from_numpy(rgb_input)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
        / 255.0
    )
    return PreparedFrame(
        source_frame_bgr=frame_bgr,
        crop_bgr=crop_bgr,
        input_tensor=input_tensor,
        roi_box=roi_box,
    )


def predict_frame_landmarks(
    model: nn.Module,
    prepared_frame: PreparedFrame,
) -> np.ndarray:
    """Run one inference step and map landmarks back to original frame pixels."""

    with torch.no_grad():
        normalized_points = model(prepared_frame.input_tensor)[0].detach().cpu().numpy()

    original_points = invert_roi_predictions(normalized_points, prepared_frame.roi_box)
    return sanitize_points(original_points, prepared_frame.source_frame_bgr.shape)


def draw_landmarks(
    frame_bgr: np.ndarray,
    points_xy: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw the fixed hand skeleton on a frame."""

    output = frame_bgr.copy()
    points = sanitize_points(points_xy, frame_bgr.shape)
    for start_index, end_index in SKELETON_EDGES:
        start = tuple(np.round(points[start_index]).astype(int).tolist())
        end = tuple(np.round(points[end_index]).astype(int).tolist())
        cv2.line(output, start, end, color, thickness=2, lineType=cv2.LINE_AA)

    for x_coord, y_coord in np.round(points).astype(int):
        cv2.circle(output, (int(x_coord), int(y_coord)), radius=3, color=color, thickness=-1)

    return output


def draw_roi_box(
    frame_bgr: np.ndarray,
    roi_box: RoiBox,
    color: tuple[int, int, int] = (255, 191, 0),
) -> np.ndarray:
    """Draw an ROI box, clipped to the visible frame."""

    output = frame_bgr.copy()
    frame_height, frame_width = output.shape[:2]
    left = max(0, min(frame_width - 1, roi_box.left))
    top = max(0, min(frame_height - 1, roi_box.top))
    right = max(0, min(frame_width - 1, roi_box.right))
    bottom = max(0, min(frame_height - 1, roi_box.bottom))
    cv2.rectangle(output, (left, top), (right, bottom), color, thickness=2)
    return output


def annotate_frame(
    frame_bgr: np.ndarray,
    model_name: str,
    device: str,
    fps: float,
    status_text: str,
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
        f"{status_text} | q/esc quit | s save frame",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def resolve_tracked_roi(
    candidate_roi: RoiBox | None,
    previous_roi: RoiBox | None,
    missed_frames: int,
    grace_frames: int,
) -> tuple[RoiBox | None, int, bool]:
    """Keep the previous ROI alive briefly while the detector reacquires the hand."""

    if candidate_roi is not None:
        return candidate_roi, 0, True
    if previous_roi is not None and missed_frames < grace_frames:
        return previous_roi, missed_frames + 1, False
    return None, 0, False


def landmarks_to_roi_box(
    normalized_points_xy: np.ndarray,
    frame_width: int,
    frame_height: int,
    expansion: float,
) -> RoiBox:
    """Convert normalized hand landmarks into an expanded square ROI."""

    if normalized_points_xy.shape != (21, 2):
        raise ValueError(
            "Expected normalized hand landmarks with shape (21, 2), "
            f"got {normalized_points_xy.shape}"
        )

    x_pixels = normalized_points_xy[:, 0] * float(frame_width)
    y_pixels = normalized_points_xy[:, 1] * float(frame_height)
    min_x = float(np.min(x_pixels))
    max_x = float(np.max(x_pixels))
    min_y = float(np.min(y_pixels))
    max_y = float(np.max(y_pixels))
    side = max(max_x - min_x, max_y - min_y) * expansion
    side = max(side, 1.0)
    side_int = max(1, int(round(side)))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    left = int(round(center_x - side_int / 2.0))
    top = int(round(center_y - side_int / 2.0))
    return RoiBox(left=left, top=top, size=side_int)


def extract_square_roi(frame_bgr: np.ndarray, roi_box: RoiBox) -> np.ndarray:
    """Extract a square ROI, padding with black when the box leaves the frame."""

    frame_height, frame_width = frame_bgr.shape[:2]
    crop = np.zeros((roi_box.size, roi_box.size, 3), dtype=frame_bgr.dtype)

    source_left = max(0, roi_box.left)
    source_top = max(0, roi_box.top)
    source_right = min(frame_width, roi_box.right)
    source_bottom = min(frame_height, roi_box.bottom)

    if source_right <= source_left or source_bottom <= source_top:
        return crop

    destination_left = source_left - roi_box.left
    destination_top = source_top - roi_box.top
    destination_right = destination_left + (source_right - source_left)
    destination_bottom = destination_top + (source_bottom - source_top)
    crop[destination_top:destination_bottom, destination_left:destination_right] = frame_bgr[
        source_top:source_bottom,
        source_left:source_right,
    ]
    return crop


def invert_roi_predictions(normalized_points_xy: np.ndarray, roi_box: RoiBox) -> np.ndarray:
    """Map normalized 224x224 prediction points back into original frame coordinates."""

    points = np.asarray(normalized_points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points.shape}")

    original_points = points.copy()
    original_points[:, 0] = roi_box.left + original_points[:, 0] * float(roi_box.size)
    original_points[:, 1] = roi_box.top + original_points[:, 1] * float(roi_box.size)
    return original_points


def sanitize_points(points_xy: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Clamp prediction points into the visible frame and replace non-finite values."""

    frame_height, frame_width = frame_shape[:2]
    points = np.asarray(points_xy, dtype=np.float32).copy()
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    points[:, 0] = np.clip(points[:, 0], 0.0, max(0, frame_width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, max(0, frame_height - 1))
    return points


def _select_best_roi(
    multi_hand_landmarks: Any,
    multi_handedness: Any,
    frame_width: int,
    frame_height: int,
    roi_expansion: float,
) -> RoiBox:
    candidates: list[tuple[float, RoiBox]] = []
    handedness_list = list(multi_handedness) if multi_handedness is not None else []
    for index, hand_landmarks in enumerate(multi_hand_landmarks):
        points = np.array(
            [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark],
            dtype=np.float32,
        )
        roi_box = landmarks_to_roi_box(
            normalized_points_xy=points,
            frame_width=frame_width,
            frame_height=frame_height,
            expansion=roi_expansion,
        )
        score = float(roi_box.size)
        if index < len(handedness_list) and handedness_list[index].classification:
            score = float(handedness_list[index].classification[0].score)
        candidates.append((score, roi_box))
    return max(candidates, key=lambda item: item[0])[1]
