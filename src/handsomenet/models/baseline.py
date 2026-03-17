"""Minimal baseline model for HandsomeNet training sanity checks."""

from __future__ import annotations

import torch
from torch import nn

from handsomenet.constants import NUM_JOINTS


class BaselineHandPoseModel(nn.Module):
    """Small CNN baseline that regresses normalized joint coordinates directly."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flattened_dim = hidden_dim * 14 * 14
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, NUM_JOINTS * 2),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        coordinates = self.regressor(features)
        return coordinates.view(-1, NUM_JOINTS, 2)
