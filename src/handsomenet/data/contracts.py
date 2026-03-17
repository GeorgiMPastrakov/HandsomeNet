"""Dataset contract definitions for FreiHAND integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FreiHANDPaths:
    """Expected file locations for the FreiHAND v1 data contract."""

    root: Path
    training_rgb_dir: Path
    training_xyz_json: Path
    training_k_json: Path
    evaluation_rgb_dir: Path
    evaluation_k_json: Path


def expected_freihand_paths(root: Path) -> FreiHANDPaths:
    """Return the expected FreiHAND file layout under a local root."""

    return FreiHANDPaths(
        root=root,
        training_rgb_dir=root / "training" / "rgb",
        training_xyz_json=root / "training_xyz.json",
        training_k_json=root / "training_K.json",
        evaluation_rgb_dir=root / "evaluation" / "rgb",
        evaluation_k_json=root / "evaluation_K.json",
    )


def validate_freihand_paths(paths: FreiHANDPaths) -> None:
    """Raise a clear error if the expected FreiHAND layout is missing."""

    missing = []
    for path in (
        paths.training_rgb_dir,
        paths.training_xyz_json,
        paths.training_k_json,
    ):
        if not path.exists():
            missing.append(path)

    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing expected FreiHAND paths: {missing_str}")
