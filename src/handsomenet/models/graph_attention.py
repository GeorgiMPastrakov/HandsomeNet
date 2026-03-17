"""Graph-attention refinement over the fixed hand skeleton."""

from __future__ import annotations

import math

import torch
from torch import nn

from handsomenet.constants import NUM_JOINTS, SKELETON_EDGES
from handsomenet.models.architecture import NUM_ATTENTION_HEADS, TOKEN_DIM


def build_adjacency_mask(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Build a boolean adjacency mask including self-connections."""

    adjacency = torch.eye(num_joints, dtype=torch.bool)
    for start, end in SKELETON_EDGES:
        adjacency[start, end] = True
        adjacency[end, start] = True
    return adjacency


class GraphAttentionLayer(nn.Module):
    """Masked self-attention layer restricted to graph neighborhoods."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        num_joints: int = NUM_JOINTS,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")

        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.num_joints = num_joints

        self.query = nn.Linear(token_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.output = nn.Linear(token_dim, token_dim)

        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim),
        )

        adjacency_mask = build_adjacency_mask(num_joints)
        self.register_buffer("adjacency_mask", adjacency_mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_joints, token_dim = tokens.shape
        if num_joints != self.num_joints or token_dim != self.token_dim:
            raise ValueError(
                f"Expected token tensor shape (B, {self.num_joints}, {self.token_dim}), "
                f"got {tokens.shape}."
            )

        q = self.query(tokens).view(batch_size, num_joints, self.num_heads, self.head_dim)
        k = self.key(tokens).view(batch_size, num_joints, self.num_heads, self.head_dim)
        v = self.value(tokens).view(batch_size, num_joints, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.adjacency_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_joints, token_dim)
        attended = self.output(attended)

        tokens = self.norm1(tokens + attended)
        ff = self.feedforward(tokens)
        return self.norm2(tokens + ff)
