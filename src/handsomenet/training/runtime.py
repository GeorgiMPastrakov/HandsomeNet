"""Runtime helpers for HandsomeNet experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch


def resolve_device(requested_device: str) -> str:
    """Resolve the requested training device with MPS-first auto priority."""

    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device, but MPS is not available.")
        return "mps"

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return "cuda"

    return "cpu"


def build_run_name(
    model_name: str,
    limit_train_unique: int | None,
    limit_val_unique: int | None,
) -> str:
    """Build a unique run directory name for one experiment."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_suffix = []
    if limit_train_unique is not None:
        limit_suffix.append(f"train{limit_train_unique}")
    if limit_val_unique is not None:
        limit_suffix.append(f"val{limit_val_unique}")
    limits = "_".join(limit_suffix) if limit_suffix else "full"
    return f"{model_name}_{limits}_{timestamp}"


def resolve_run_name(
    model_name: str,
    limit_train_unique: int | None,
    limit_val_unique: int | None,
    resume_checkpoint: Path | None,
) -> str:
    """Resolve the run directory, reusing the original run when resuming."""

    if resume_checkpoint is not None:
        return resume_checkpoint.parent.name
    return build_run_name(model_name, limit_train_unique, limit_val_unique)
