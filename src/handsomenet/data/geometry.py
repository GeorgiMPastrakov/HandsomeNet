"""Geometry utilities for projection and inverse mapping.

These functions are intentionally left unimplemented until the dataset
integration checkpoint is executed.
"""

from __future__ import annotations

import numpy as np


def project_xyz_with_intrinsics(xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project 3D joints into image space."""

    raise NotImplementedError("Projection logic is deferred to the dataset checkpoint.")


def normalize_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize 2D points into [0, 1] image coordinates."""

    raise NotImplementedError("Normalization logic is deferred to the dataset checkpoint.")


def invert_normalized_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized [0, 1] coordinates back to image-space pixels."""

    raise NotImplementedError("Inverse mapping logic is deferred to the dataset checkpoint.")

