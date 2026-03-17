"""Minimal baseline model for HandsomeNet training sanity checks."""

from __future__ import annotations

import torch
from torch import nn

from handsomenet.constants import NUM_JOINTS
from handsomenet.models.coordinate_head import CoordinateHead


class BaselineHandPoseModel(nn.Module):
    """Small CNN baseline that regresses normalized joint coordinates directly."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.coordinate_head = CoordinateHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_joints=NUM_JOINTS,
            bounded_output=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        pooled = self.pool(features).flatten(start_dim=1)
        token_like_features = pooled.unsqueeze(1).expand(-1, NUM_JOINTS, -1)
        return self.coordinate_head(token_like_features)
