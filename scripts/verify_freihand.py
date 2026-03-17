"""Verify the HandsomeNet v1 FreiHAND geometry contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handsomenet.constants import SKELETON_EDGES
from handsomenet.data.freihand import FreiHANDDataset, verify_sample_inversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/freihand"),
        help="Canonical FreiHAND dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/freihand_verification"),
        help="Directory where verification overlays are written.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[0, 1000, 32000],
        help="Dataset indices to visualize.",
    )
    return parser.parse_args()


def draw_overlay(image: Image.Image, points_xy: np.ndarray) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for start, end in SKELETON_EDGES:
        start_point = tuple(points_xy[start].tolist())
        end_point = tuple(points_xy[end].tolist())
        draw.line([start_point, end_point], fill=(0, 255, 0), width=2)

    for x_coord, y_coord in points_xy.tolist():
        radius = 3
        draw.ellipse(
            (x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius),
            fill=(255, 0, 0),
        )

    return overlay


def main() -> None:
    args = parse_args()
    dataset = FreiHANDDataset(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_inversion_error = 0.0
    for index in args.indices:
        sample = dataset[index]
        restored = verify_sample_inversion(sample)
        inversion_error = np.abs(restored - sample.targets.projected_original_pixels).max()
        max_inversion_error = max(max_inversion_error, float(inversion_error))

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            overlay = draw_overlay(image, sample.targets.projected_original_pixels)
            overlay.save(args.output_dir / f"{sample.sample_id}_overlay.jpg")

    print(f"verified_samples={len(args.indices)}")
    print(f"dataset_length={len(dataset)}")
    print(f"max_inversion_error={max_inversion_error:.8f}")


if __name__ == "__main__":
    main()
