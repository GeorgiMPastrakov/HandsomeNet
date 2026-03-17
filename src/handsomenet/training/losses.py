"""Training losses for HandsomeNet."""

from __future__ import annotations

import torch
from torch import nn


class CoordinateMSELoss(nn.Module):
    """Mean squared error on normalized `(x, y)` coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(predictions, targets)
