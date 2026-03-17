import torch

from handsomenet.models.baseline import BaselineHandPoseModel


def test_baseline_model_forward_shape_and_range() -> None:
    model = BaselineHandPoseModel()
    images = torch.randn(2, 3, 224, 224)

    outputs = model(images)

    assert outputs.shape == (2, 21, 2)
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
