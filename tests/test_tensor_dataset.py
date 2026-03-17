from pathlib import Path

import pytest
import torch

from handsomenet.data.tensor_dataset import FreiHANDTensorDataset


@pytest.mark.skipif(
    not Path("data/raw/freihand/training_xyz.json").exists(),
    reason="FreiHAND dataset not available locally.",
)
def test_freihand_tensor_dataset_shapes_and_dtypes() -> None:
    dataset = FreiHANDTensorDataset(Path("data/raw/freihand"))

    sample = dataset[0]

    assert sample.image.shape == (3, 224, 224)
    assert sample.targets.shape == (21, 2)
    assert sample.image.dtype == torch.float32
    assert sample.targets.dtype == torch.float32
    assert sample.image.min().item() >= 0.0
    assert sample.image.max().item() <= 1.0
    assert sample.targets.min().item() >= 0.0
    assert sample.targets.max().item() <= 1.0
    assert sample.annotation_index == 0
    assert sample.image_variant_index == 0
