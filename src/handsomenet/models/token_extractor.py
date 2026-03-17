"""Joint token extraction from spatial backbone features."""

from __future__ import annotations

import torch
from torch import nn

from handsomenet.constants import NUM_JOINTS
from handsomenet.models.architecture import BACKBONE_OUTPUT_CHANNELS, NUM_ATTENTION_HEADS, TOKEN_DIM


class JointTokenExtractor(nn.Module):
    """Extract learned joint tokens via query attention over spatial features."""

    def __init__(
        self,
        input_channels: int = BACKBONE_OUTPUT_CHANNELS,
        token_dim: int = TOKEN_DIM,
        grid_size: tuple[int, int] = (14, 14),
        num_joints: int = NUM_JOINTS,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_joints = num_joints
        self.token_projection = nn.Linear(input_channels, token_dim)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, grid_size[0] * grid_size[1], token_dim)
        )
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, token_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=NUM_ATTENTION_HEADS,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        if (height, width) != self.grid_size:
            raise ValueError(
                f"Expected feature map spatial size {self.grid_size}, got {(height, width)}."
            )

        spatial_tokens = feature_map.flatten(start_dim=2).transpose(1, 2)
        spatial_tokens = self.token_projection(spatial_tokens)
        spatial_tokens = spatial_tokens + self.position_embeddings

        queries = self.joint_queries.expand(batch_size, -1, -1)
        attended, _ = self.attention(queries, spatial_tokens, spatial_tokens)
        return self.output_norm(attended)
