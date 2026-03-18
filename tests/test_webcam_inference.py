import numpy as np
import torch
from torch import nn

from handsomenet.inference.webcam import (
    LandmarkSmoother,
    draw_landmarks,
    predict_frame_landmarks,
    prepare_frame,
)


class DummyModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        del images
        points = torch.full((1, 21, 2), 0.5, dtype=torch.float32)
        return points


def test_prepare_frame_returns_input_tensor_and_identity_geometry() -> None:
    frame = np.full((224, 224, 3), 127, dtype=np.uint8)

    prepared = prepare_frame(frame, device="cpu")

    assert prepared.image_bgr.shape == (224, 224, 3)
    assert prepared.input_tensor.shape == (1, 3, 224, 224)
    assert prepared.input_tensor.dtype == torch.float32
    assert prepared.geometry.original_width == 224
    assert prepared.geometry.original_height == 224
    assert prepared.geometry.resize_scale == 1.0
    assert prepared.geometry.pad_x == 0.0
    assert prepared.geometry.pad_y == 0.0


def test_predict_frame_landmarks_maps_back_to_original_pixels() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    prepared = prepare_frame(frame, device="cpu")

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
