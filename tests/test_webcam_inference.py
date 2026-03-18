import numpy as np
import torch
from torch import nn

from handsomenet.inference.webcam import (
    LandmarkSmoother,
    RoiBox,
    draw_landmarks,
    draw_roi_box,
    invert_roi_predictions,
    landmarks_to_roi_box,
    predict_frame_landmarks,
    prepare_frame,
    resolve_tracked_roi,
)


class DummyModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        del images
        points = torch.full((1, 21, 2), 0.5, dtype=torch.float32)
        return points


def test_prepare_frame_returns_input_tensor_and_identity_geometry() -> None:
    frame = np.full((224, 224, 3), 127, dtype=np.uint8)

    prepared = prepare_frame(frame, device="cpu")

    assert prepared.source_frame_bgr.shape == (224, 224, 3)
    assert prepared.crop_bgr.shape == (224, 224, 3)
    assert prepared.input_tensor.shape == (1, 3, 224, 224)
    assert prepared.input_tensor.dtype == torch.float32
    assert prepared.roi_box.left == 0
    assert prepared.roi_box.top == 0
    assert prepared.roi_box.size == 224


def test_predict_frame_landmarks_maps_back_to_original_pixels() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    prepared = prepare_frame(frame, device="cpu", roi_box=RoiBox(left=50, top=0, size=100))

    predicted = predict_frame_landmarks(DummyModel(), prepared)

    assert predicted.shape == (21, 2)
    np.testing.assert_allclose(predicted[0], np.array([100.0, 50.0], dtype=np.float32), atol=1e-4)


def test_draw_landmarks_changes_frame() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    points = np.full((21, 2), 64.0, dtype=np.float32)

    rendered = draw_landmarks(frame, points)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_landmark_smoother_applies_exponential_average() -> None:
    smoother = LandmarkSmoother(alpha=0.5)
    first = np.zeros((21, 2), dtype=np.float32)
    second = np.ones((21, 2), dtype=np.float32) * 10.0

    updated_first = smoother.update(first)
    updated_second = smoother.update(second)

    np.testing.assert_allclose(updated_first, first)
    np.testing.assert_allclose(updated_second, np.ones((21, 2), dtype=np.float32) * 5.0)


def test_landmarks_to_roi_box_expands_square_near_frame_edge() -> None:
    points = np.array(
        [
            [0.02, 0.04],
            [0.20, 0.05],
            [0.18, 0.22],
        ]
        + [[0.10, 0.10]] * 18,
        dtype=np.float32,
    )

    roi_box = landmarks_to_roi_box(points, frame_width=640, frame_height=480, expansion=1.5)

    assert roi_box.size > 0
    assert roi_box.left < 50
    assert roi_box.top < 50


def test_invert_roi_predictions_maps_crop_points_back_to_frame() -> None:
    normalized_points = np.full((21, 2), 0.5, dtype=np.float32)

    restored = invert_roi_predictions(normalized_points, RoiBox(left=30, top=70, size=120))

    np.testing.assert_allclose(restored[0], np.array([90.0, 130.0], dtype=np.float32))


def test_resolve_tracked_roi_holds_then_drops_previous_box() -> None:
    previous = RoiBox(left=10, top=20, size=100)

    retained, missed_frames, detected = resolve_tracked_roi(
        candidate_roi=None,
        previous_roi=previous,
        missed_frames=1,
        grace_frames=2,
    )
    assert retained == previous
    assert missed_frames == 2
    assert not detected

    dropped, missed_frames, detected = resolve_tracked_roi(
        candidate_roi=None,
        previous_roi=previous,
        missed_frames=2,
        grace_frames=2,
    )
    assert dropped is None
    assert missed_frames == 0
    assert not detected


def test_draw_roi_box_and_landmarks_handle_out_of_bounds_points() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    points = np.full((21, 2), 1000.0, dtype=np.float32)

    rendered = draw_landmarks(frame, points)
    rendered = draw_roi_box(rendered, RoiBox(left=-10, top=-20, size=100))

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)
