"""
SketchNet: custom efficient CNN architecture for sketch recognition.

Extracted from day5_custom_efficient_model.py into its own module so it can
be imported (e.g. by master_pipeline_v2.py) without re-running that script's
top-level training pipeline as a side effect.
"""
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet-style)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EfficientSketchBlock(nn.Module):
    """Custom block optimized for sketch recognition"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(EfficientSketchBlock, self).__init__()

        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = out + self.skip(x)
        return out


class SketchNet(nn.Module):
    """
    Custom Efficient Model for Sketch Recognition

    Architecture optimized for:
    - Small image size (64x64)
    - Abstract sketch patterns
    - Parameter efficiency
    - High accuracy
    """
    def __init__(self, num_classes=50):
        super(SketchNet, self).__init__()

        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 1: 32x32
        self.stage1 = nn.Sequential(
            EfficientSketchBlock(32, 64, stride=1, use_se=True),
            EfficientSketchBlock(64, 64, stride=1, use_se=True)
        )

        # Stage 2: 16x16
        self.stage2 = nn.Sequential(
            EfficientSketchBlock(64, 128, stride=2, use_se=True),
            EfficientSketchBlock(128, 128, stride=1, use_se=True),
            EfficientSketchBlock(128, 128, stride=1, use_se=True)
        )

        # Stage 3: 8x8
        self.stage3 = nn.Sequential(
            EfficientSketchBlock(128, 256, stride=2, use_se=True),
            EfficientSketchBlock(256, 256, stride=1, use_se=True),
            EfficientSketchBlock(256, 256, stride=1, use_se=True),
            EfficientSketchBlock(256, 256, stride=1, use_se=True)
        )

        # Stage 4: 4x4
        self.stage4 = nn.Sequential(
            EfficientSketchBlock(256, 512, stride=2, use_se=True),
            EfficientSketchBlock(512, 512, stride=1, use_se=True)
        )

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
