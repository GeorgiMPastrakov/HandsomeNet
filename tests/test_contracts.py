from pathlib import Path

import numpy as np

from handsomenet.data.contracts import expected_freihand_paths
from handsomenet.data.geometry import (
    apply_geometry_to_points,
    build_geometry_metadata,
    invert_geometry_on_points,
    invert_normalized_points,
    normalize_points,
    project_xyz_with_intrinsics,
)
from handsomenet.data.splits import (
    build_freihand_split,
    image_index_to_annotation_index,
    image_index_to_variant_index,
)


def test_expected_freihand_paths() -> None:
    root = Path("data/raw/freihand")
    paths = expected_freihand_paths(root)

    assert paths.training_rgb_dir == root / "training" / "rgb"
    assert paths.training_xyz_json == root / "training_xyz.json"
    assert paths.training_k_json == root / "training_K.json"
    assert paths.evaluation_rgb_dir == root / "evaluation" / "rgb"
    assert paths.evaluation_k_json == root / "evaluation_K.json"


def test_project_xyz_with_intrinsics() -> None:
    xyz = np.zeros((21, 3), dtype=np.float32)
    xyz[:, 2] = 2.0
    xyz[0, :3] = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    projected = project_xyz_with_intrinsics(xyz, intrinsics)

    np.testing.assert_allclose(projected[0], np.array([0.5, 1.0], dtype=np.float32))


def test_geometry_round_trip_for_points() -> None:
    geometry = build_geometry_metadata(
        original_width=100,
        original_height=50,
        final_width=224,
        final_height=224,
    )
    points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    transformed = apply_geometry_to_points(points, geometry)
    normalized = normalize_points(transformed, 224, 224)
    restored_input = invert_normalized_points(normalized, 224, 224)
    restored_original = invert_geometry_on_points(restored_input, geometry)

    np.testing.assert_allclose(restored_original, points, atol=1e-5)


def test_freihand_split_keeps_variants_together() -> None:
    split = build_freihand_split(num_unique_samples=10, val_fraction=0.2, seed=7)

    train_annotations = {
        image_index_to_annotation_index(index, 10) for index in split.train_indices
    }
    val_annotations = {image_index_to_annotation_index(index, 10) for index in split.val_indices}

    assert train_annotations.isdisjoint(val_annotations)
    assert len(split.train_indices) == len(split.train_unique_ids) * 4
    assert len(split.val_indices) == len(split.val_unique_ids) * 4


def test_freihand_image_variant_mapping() -> None:
    assert image_index_to_annotation_index(32561, 32560) == 1
    assert image_index_to_variant_index(32561, 32560) == 1
