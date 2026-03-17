from pathlib import Path

from handsomenet.data.contracts import expected_freihand_paths


def test_expected_freihand_paths() -> None:
    root = Path("data/raw/freihand")
    paths = expected_freihand_paths(root)

    assert paths.training_rgb_dir == root / "training" / "rgb"
    assert paths.training_xyz_json == root / "training_xyz.json"
    assert paths.training_k_json == root / "training_K.json"

