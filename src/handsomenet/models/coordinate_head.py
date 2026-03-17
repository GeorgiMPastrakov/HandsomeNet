"""Coordinate heads for HandsomeNet models."""

from __future__ import annotations

import torch
from torch import nn


class CoordinateHead(nn.Module):
    """Shared per-joint coordinate regressor with optional bounded output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_joints: int,
        bounded_output: bool = True,
    ) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.num_joints = num_joints
        self.bounded_output = bounded_output

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError(
                f"Expected token_features to have shape (B, N, C), got {token_features.shape}."
            )
        if token_features.shape[1] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, got {token_features.shape[1]}."
            )

        coordinates = self.regressor(token_features)
        if self.bounded_output:
            coordinates = torch.sigmoid(coordinates)
        return coordinates
