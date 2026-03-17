"""Shared typed structures for dataset and model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class GeometryMetadata:
    """Geometry metadata required for exact inverse mapping."""

    original_width: int
    original_height: int
    resize_scale: float
    pad_x: float
    pad_y: float
    final_width: int
    final_height: int


@dataclass(frozen=True)
class SampleTargets:
    """Projected and normalized target coordinates for one sample."""

    projected_original_pixels: np.ndarray
    projected_input_pixels: np.ndarray
    normalized_2d: np.ndarray


@dataclass(frozen=True)
class DatasetSample:
    """Single sample contract for HandsomeNet v1."""

    image: np.ndarray
    targets: SampleTargets
    geometry: GeometryMetadata
    sample_id: str
    image_path: Path


@dataclass(frozen=True)
class TensorSample:
    """Training-facing tensorized sample."""

    image: torch.Tensor
    targets: torch.Tensor
    geometry: GeometryMetadata
    sample_id: str
    annotation_index: int
    image_variant_index: int
    image_path: Path
