"""Top-level HandsomeNet model."""

from __future__ import annotations

import torch
from torch import nn

from handsomenet.constants import NUM_JOINTS
from handsomenet.models.architecture import (
    COORDINATE_HEAD_HIDDEN_DIM,
    NUM_GRAPH_LAYERS,
    TOKEN_DIM,
)
from handsomenet.models.backbone import HandsomeNetBackbone
from handsomenet.models.coordinate_head import CoordinateHead
from handsomenet.models.graph_attention import GraphAttentionLayer
from handsomenet.models.token_extractor import JointTokenExtractor


class HandsomeNet(nn.Module):
    """HandsomeNet v1 architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = HandsomeNetBackbone()
        self.token_extractor = JointTokenExtractor()
        self.graph_layers = nn.ModuleList(
            [GraphAttentionLayer(token_dim=TOKEN_DIM) for _ in range(NUM_GRAPH_LAYERS)]
        )
        self.coordinate_head = CoordinateHead(
            input_dim=TOKEN_DIM,
            hidden_dim=COORDINATE_HEAD_HIDDEN_DIM,
            num_joints=NUM_JOINTS,
            bounded_output=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        joint_tokens = self.token_extractor(feature_map)
        for graph_layer in self.graph_layers:
            joint_tokens = graph_layer(joint_tokens)
        return self.coordinate_head(joint_tokens)
