"""FreiHAND dataset integration entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from handsomenet.constants import INPUT_HEIGHT, INPUT_WIDTH
from handsomenet.data.contracts import expected_freihand_paths, validate_freihand_paths
from handsomenet.data.geometry import (
    apply_geometry_to_points,
    build_geometry_metadata,
    invert_geometry_on_points,
    invert_normalized_points,
    normalize_points,
    project_xyz_with_intrinsics,
)
from handsomenet.types import DatasetSample, SampleTargets


class FreiHANDDataset:
    """FreiHAND dataset for the HandsomeNet v1 geometry contract."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = expected_freihand_paths(root)
        validate_freihand_paths(self.paths)

        self.training_xyz = self._load_json(self.paths.training_xyz_json)
        self.training_k = self._load_json(self.paths.training_k_json)
        self.image_paths = sorted(self.paths.training_rgb_dir.glob("*.jpg"))

        self.num_unique_samples = len(self.training_xyz)
        expected_num_images = self.num_unique_samples * 4

        if len(self.training_k) != self.num_unique_samples:
            raise ValueError(
                "FreiHAND training_K.json length does not match training_xyz.json length."
            )
        if len(self.image_paths) != expected_num_images:
            raise ValueError(
                "Expected "
                f"{expected_num_images} training RGB images, found {len(self.image_paths)}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DatasetSample:
        image_path = self.image_paths[index]
        annotation_index = int(image_path.stem) % self.num_unique_samples

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_array = np.asarray(image, dtype=np.uint8)
            width, height = image.size

        geometry = build_geometry_metadata(
            original_width=width,
            original_height=height,
            final_width=INPUT_WIDTH,
            final_height=INPUT_HEIGHT,
        )

        xyz = np.asarray(self.training_xyz[annotation_index], dtype=np.float32)
        intrinsics = np.asarray(self.training_k[annotation_index], dtype=np.float32)
        projected_original = project_xyz_with_intrinsics(xyz, intrinsics)
        projected_input = apply_geometry_to_points(projected_original, geometry)
        normalized = normalize_points(projected_input, INPUT_WIDTH, INPUT_HEIGHT)

        return DatasetSample(
            image=image_array,
            targets=SampleTargets(
                projected_original_pixels=projected_original,
                projected_input_pixels=projected_input,
                normalized_2d=normalized,
            ),
            geometry=geometry,
            sample_id=image_path.stem,
            image_path=image_path,
        )

    @staticmethod
    def _load_json(path: Path) -> list[Any]:
        with path.open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload in {path}, got {type(payload).__name__}")
        return payload


def verify_sample_inversion(sample: DatasetSample) -> np.ndarray:
    """Invert a sample's normalized coordinates back into original-image space."""

    input_space_pixels = invert_normalized_points(
        sample.targets.normalized_2d,
        width=sample.geometry.final_width,
        height=sample.geometry.final_height,
    )
    return invert_geometry_on_points(input_space_pixels, sample.geometry)
