"""
Baseline architectures compared against SketchEffNet.

Pretrained models use the same ImageNet weights the original ``pretrained=True``
calls resolved to (``*_Weights.IMAGENET1K_V1``). This avoids the torchvision
deprecation warning without changing the starting weights.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .model import SketchNet

_TORCHVISION = {
    # name: (constructor, weights enum, how to replace the classifier)
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, "classifier[1]"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier[1]"),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "classifier"),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, "classifier[6]"),
}

DISPLAY_NAMES = {
    "simple_cnn": "Simple CNN", "mobilenet_v2": "MobileNetV2", "resnet18": "ResNet18",
    "resnet34": "ResNet34", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
    "vgg16": "VGG16", "sketcheffnet": "SketchEffNet",
}


def simple_cnn(num_classes: int = 50) -> nn.Module:
    """Three conv/pool stages + a 256-unit MLP head; trained from scratch."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


def create_model(name: str, num_classes: int = 50, pretrained: bool = True) -> nn.Module:
    """Build a classifier by name with a ``num_classes``-way head."""
    if name == "simple_cnn":
        return simple_cnn(num_classes)
    if name in ("sketcheffnet", "sketchnet"):
        return SketchNet(num_classes)
    if name not in _TORCHVISION:
        raise ValueError(f"Unknown model: {name}. Options: {sorted(_TORCHVISION) + ['simple_cnn', 'sketcheffnet']}")
    ctor, weights, head = _TORCHVISION[name]
    model = ctor(weights=weights if pretrained else None)
    if head == "fc":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif head == "classifier":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:  # classifier[i]
        i = int(head[head.index("[") + 1:-1])
        model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
    return model
