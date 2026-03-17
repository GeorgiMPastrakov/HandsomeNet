"""Mobile-style CNN backbone for HandsomeNet v1."""

from __future__ import annotations

from torch import nn

from handsomenet.models.architecture import BACKBONE_OUTPUT_CHANNELS, BACKBONE_STEM_CHANNELS


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class HandsomeNetBackbone(nn.Module):
    """Backbone mapping 224x224 RGB images to a 192x14x14 feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, BACKBONE_STEM_CHANNELS, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(BACKBONE_STEM_CHANNELS),
            nn.ReLU6(inplace=True),
        )
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(BACKBONE_STEM_CHANNELS, 48, stride=2),
            InvertedResidualBlock(48, 48, stride=1),
        )
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(48, 96, stride=2),
            InvertedResidualBlock(96, 96, stride=1),
        )
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(96, 160, stride=2),
            InvertedResidualBlock(160, 160, stride=1),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(160, BACKBONE_OUTPUT_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(BACKBONE_OUTPUT_CHANNELS),
            nn.ReLU6(inplace=True),
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.projection(x)
