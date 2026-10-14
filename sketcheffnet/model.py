"""
SketchEffNet - a lightweight CNN for 64x64 sketch recognition.

Three ideas, combined in one block (:class:`EfficientSketchBlock`):

* **Depthwise-separable convolution** (MobileNet) - a 3x3 per-channel conv
  followed by a 1x1 channel mixer; far fewer weights than a dense 3x3 conv.
* **Squeeze-and-Excitation** (SENet) - global-pool the feature map, learn one
  gate per channel, re-weight the channels.
* **Residual shortcut** (ResNet) - identity, or a strided 1x1 projection when
  the shape changes.

Layout for a 64x64 input::

    stem      3 -> 32   stride 2              32x32
    stage 1   32 -> 64   2 blocks             32x32
    stage 2   64 -> 128  3 blocks (1st s=2)   16x16
    stage 3  128 -> 256  4 blocks (1st s=2)    8x8
    stage 4  256 -> 512  2 blocks (1st s=2)    4x4
    head     GAP -> Dropout(0.2) -> Linear(512, num_classes)

1,001,586 parameters for 50 classes. In code the class keeps its original
name, ``SketchNet``; ``SketchEffNet`` is an alias with the name used in the paper.

This module has no side effects on import - it only defines classes.
"""
from __future__ import annotations

import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 convolution, then BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class EfficientSketchBlock(nn.Module):
    """Depthwise-separable conv -> SE attention -> + residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.se(self.conv(x)) + self.skip(x)


class SketchNet(nn.Module):
    """SketchEffNet: efficient CNN for 64x64 sketches (see module docstring)."""

    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            EfficientSketchBlock(32, 64, stride=1),
            EfficientSketchBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            EfficientSketchBlock(64, 128, stride=2),
            EfficientSketchBlock(128, 128, stride=1),
            EfficientSketchBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            EfficientSketchBlock(128, 256, stride=2),
            EfficientSketchBlock(256, 256, stride=1),
            EfficientSketchBlock(256, 256, stride=1),
            EfficientSketchBlock(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            EfficientSketchBlock(256, 512, stride=2),
            EfficientSketchBlock(512, 512, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def features(self, x):
        """Backbone output after global pooling: shape (N, 512)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))


SketchEffNet = SketchNet


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def fp32_size_mb(model: nn.Module) -> float:
    """Size of the parameters in float32, in MiB (the figure the reports use)."""
    return count_parameters(model) * 4 / (1024 ** 2)
