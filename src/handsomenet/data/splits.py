"""Split helpers for FreiHAND training and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handsomenet.constants import NUM_IMAGE_VARIANTS_PER_SAMPLE


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation image indices split by unique annotation ID."""

    train_indices: list[int]
    val_indices: list[int]
    train_unique_ids: list[int]
    val_unique_ids: list[int]


def build_freihand_split(
    num_unique_samples: int,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_variants: int = NUM_IMAGE_VARIANTS_PER_SAMPLE,
    limit_train_unique: int | None = None,
    limit_val_unique: int | None = None,
) -> DatasetSplit:
    """Build a deterministic split by unique annotation ID."""

    if num_unique_samples <= 0:
        raise ValueError("num_unique_samples must be positive.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if num_variants <= 0:
        raise ValueError("num_variants must be positive.")

    rng = np.random.default_rng(seed)
    unique_ids = np.arange(num_unique_samples, dtype=np.int32)
    rng.shuffle(unique_ids)

    num_val_unique = max(1, int(round(num_unique_samples * val_fraction)))
    val_unique_ids = np.sort(unique_ids[:num_val_unique]).tolist()
    train_unique_ids = np.sort(unique_ids[num_val_unique:]).tolist()

    if limit_train_unique is not None:
        if limit_train_unique <= 0:
            raise ValueError("limit_train_unique must be positive when provided.")
        train_unique_ids = train_unique_ids[:limit_train_unique]

    if limit_val_unique is not None:
        if limit_val_unique <= 0:
            raise ValueError("limit_val_unique must be positive when provided.")
        val_unique_ids = val_unique_ids[:limit_val_unique]

    return DatasetSplit(
        train_indices=_expand_unique_ids(train_unique_ids, num_unique_samples, num_variants),
        val_indices=_expand_unique_ids(val_unique_ids, num_unique_samples, num_variants),
        train_unique_ids=train_unique_ids,
        val_unique_ids=val_unique_ids,
    )


def image_index_to_annotation_index(
    image_index: int,
    num_unique_samples: int,
) -> int:
    """Map a FreiHAND image index to the corresponding unique annotation index."""

    if image_index < 0:
        raise ValueError("image_index must be non-negative.")
    if num_unique_samples <= 0:
        raise ValueError("num_unique_samples must be positive.")

    return image_index % num_unique_samples


def image_index_to_variant_index(
    image_index: int,
    num_unique_samples: int,
) -> int:
    """Map a FreiHAND image index to the corresponding image variant index."""

    if image_index < 0:
        raise ValueError("image_index must be non-negative.")
    if num_unique_samples <= 0:
        raise ValueError("num_unique_samples must be positive.")

    return image_index // num_unique_samples


def _expand_unique_ids(
    unique_ids: list[int],
    num_unique_samples: int,
    num_variants: int,
) -> list[int]:
    indices: list[int] = []
    for variant_index in range(num_variants):
        variant_offset = variant_index * num_unique_samples
        indices.extend(variant_offset + unique_id for unique_id in unique_ids)
    return indices
