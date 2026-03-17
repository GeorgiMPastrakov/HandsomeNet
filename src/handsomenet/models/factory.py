"""Model factory for HandsomeNet training."""

from __future__ import annotations

from torch import nn

from handsomenet.models.baseline import BaselineHandPoseModel
from handsomenet.models.handsomenet import HandsomeNet


def build_model(model_name: str) -> nn.Module:
    """Build a supported HandsomeNet project model by name."""

    normalized_name = model_name.strip().lower()
    if normalized_name == "baseline":
        return BaselineHandPoseModel()
    if normalized_name == "handsomenet":
        return HandsomeNet()
    raise ValueError(f"Unsupported model name: {model_name}")
