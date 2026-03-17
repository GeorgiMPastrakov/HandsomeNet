from pathlib import Path

import pytest

from handsomenet.data.freihand import FreiHANDDataset, verify_sample_inversion


@pytest.mark.skipif(
    not Path("data/raw/freihand/training_xyz.json").exists(),
    reason="FreiHAND dataset not available locally.",
)
def test_freihand_dataset_sample_contract() -> None:
    dataset = FreiHANDDataset(Path("data/raw/freihand"))

    sample = dataset[0]

    assert len(dataset) == 130240
    assert sample.image.shape == (224, 224, 3)
    assert sample.targets.projected_original_pixels.shape == (21, 2)
    assert sample.targets.projected_input_pixels.shape == (21, 2)
    assert sample.targets.normalized_2d.shape == (21, 2)
    assert sample.geometry.original_width == 224
    assert sample.geometry.original_height == 224
    assert sample.geometry.final_width == 224
    assert sample.geometry.final_height == 224
    assert sample.geometry.resize_scale == pytest.approx(1.0)
    assert sample.geometry.pad_x == pytest.approx(0.0)
    assert sample.geometry.pad_y == pytest.approx(0.0)
    assert sample.targets.normalized_2d.min() >= 0.0
    assert sample.targets.normalized_2d.max() <= 1.0


@pytest.mark.skipif(
    not Path("data/raw/freihand/training_xyz.json").exists(),
    reason="FreiHAND dataset not available locally.",
)
def test_freihand_inverse_mapping_is_exact_for_native_training_images() -> None:
    dataset = FreiHANDDataset(Path("data/raw/freihand"))
    sample = dataset[1234]

    restored = verify_sample_inversion(sample)

    assert restored.shape == (21, 2)
    assert restored == pytest.approx(sample.targets.projected_original_pixels, abs=1e-5)
