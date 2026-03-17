"""Metrics for HandsomeNet training and validation."""

from __future__ import annotations

import torch


def mean_2d_pixel_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Compute mean Euclidean 2D pixel error in model-input space."""

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape, got {predictions.shape} and "
            f"{targets.shape}."
        )
    if predictions.ndim != 3 or predictions.shape[-1] != 2:
        raise ValueError(
            "Expected predictions and targets to have shape (B, N, 2), got "
            f"{predictions.shape}."
        )

    scale = torch.tensor([width, height], dtype=predictions.dtype, device=predictions.device)
    diff_pixels = (predictions - targets) * scale
    distances = torch.linalg.norm(diff_pixels, dim=-1)
    return distances.mean()
