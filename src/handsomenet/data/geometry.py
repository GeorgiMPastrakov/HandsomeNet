"""Geometry utilities for projection and inverse mapping."""

from __future__ import annotations

import numpy as np

from handsomenet.types import GeometryMetadata


def project_xyz_with_intrinsics(xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project 3D joints into image space."""

    xyz_array = np.asarray(xyz, dtype=np.float32)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float32)

    if xyz_array.shape != (21, 3):
        raise ValueError(f"Expected xyz shape (21, 3), got {xyz_array.shape}")
    if intrinsics_array.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3), got {intrinsics_array.shape}")

    z = xyz_array[:, 2:3]
    if np.any(z <= 0):
        raise ValueError("Cannot project joints with non-positive depth values.")

    homogeneous = (intrinsics_array @ xyz_array.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def build_geometry_metadata(
    original_width: int,
    original_height: int,
    final_width: int,
    final_height: int,
) -> GeometryMetadata:
    """Compute resize-plus-pad geometry metadata for one image."""

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive.")
    if final_width <= 0 or final_height <= 0:
        raise ValueError("Final image dimensions must be positive.")

    resize_scale = min(final_width / original_width, final_height / original_height)
    resized_width = original_width * resize_scale
    resized_height = original_height * resize_scale
    pad_x = (final_width - resized_width) / 2.0
    pad_y = (final_height - resized_height) / 2.0

    return GeometryMetadata(
        original_width=original_width,
        original_height=original_height,
        resize_scale=resize_scale,
        pad_x=pad_x,
        pad_y=pad_y,
        final_width=final_width,
        final_height=final_height,
    )


def apply_geometry_to_points(points_xy: np.ndarray, geometry: GeometryMetadata) -> np.ndarray:
    """Apply resize-plus-pad geometry to 2D pixel coordinates."""

    points_array = np.asarray(points_xy, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points_array.shape}")

    transformed = points_array.copy()
    transformed[:, 0] = transformed[:, 0] * geometry.resize_scale + geometry.pad_x
    transformed[:, 1] = transformed[:, 1] * geometry.resize_scale + geometry.pad_y
    return transformed


def normalize_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize 2D points into [0, 1] image coordinates."""

    points_array = np.asarray(points_xy, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points_array.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    normalized = points_array.copy()
    normalized[:, 0] = normalized[:, 0] / float(width)
    normalized[:, 1] = normalized[:, 1] / float(height)
    return normalized


def invert_normalized_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized [0, 1] coordinates back to image-space pixels."""

    points_array = np.asarray(points_xy, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points_array.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    pixels = points_array.copy()
    pixels[:, 0] = pixels[:, 0] * float(width)
    pixels[:, 1] = pixels[:, 1] * float(height)
    return pixels


def invert_geometry_on_points(points_xy: np.ndarray, geometry: GeometryMetadata) -> np.ndarray:
    """Undo resize-plus-pad geometry on 2D pixel coordinates."""

    points_array = np.asarray(points_xy, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2), got {points_array.shape}")
    if geometry.resize_scale <= 0:
        raise ValueError("Resize scale must be positive for inverse mapping.")

    restored = points_array.copy()
    restored[:, 0] = (restored[:, 0] - geometry.pad_x) / geometry.resize_scale
    restored[:, 1] = (restored[:, 1] - geometry.pad_y) / geometry.resize_scale
    return restored
