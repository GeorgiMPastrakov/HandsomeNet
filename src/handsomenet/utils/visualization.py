"""Visualization utilities for HandsomeNet overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handsomenet.constants import SKELETON_EDGES


def save_prediction_overlay(
    image_chw: np.ndarray,
    target_points_xy: np.ndarray,
    prediction_points_xy: np.ndarray,
    output_path: Path,
) -> None:
    """Write an overlay comparing target and predicted 2D joints."""

    image_hwc = np.transpose(image_chw, (1, 2, 0))
    image_uint8 = np.clip(image_hwc * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_uint8)
    draw = ImageDraw.Draw(image)

    _draw_skeleton(draw, target_points_xy, color=(0, 255, 0))
    _draw_skeleton(draw, prediction_points_xy, color=(255, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_skeleton(draw: ImageDraw.ImageDraw, points_xy: np.ndarray, color: tuple[int, int, int]):
    for start, end in SKELETON_EDGES:
        draw.line(
            [tuple(points_xy[start].tolist()), tuple(points_xy[end].tolist())],
            fill=color,
            width=2,
        )
    for x_coord, y_coord in points_xy.tolist():
        radius = 2
        draw.ellipse(
            (x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius),
            fill=color,
        )
