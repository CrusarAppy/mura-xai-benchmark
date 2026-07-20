"""CNN backbone registry. Phase 1 ships DenseNet121; the other two are registered
and ready for later phases. Each backbone keeps the torchvision model intact (so the
standard forward is used) and only replaces the final Linear layer with a binary head.

`target_layer_for(model)` returns the module whose output feature maps (C x 7 x 7)
Grad-CAM family methods hook — the final convolutional feature block.
"""
from __future__ import annotations

BACKBONES = ("densenet121", "efficientnet_b0", "convnext_tiny")


def build_backbone(name: str, num_classes: int = 2, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    if name == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=w)
        in_f = net.classifier.in_features           # 1024
        net.classifier = nn.Linear(in_f, num_classes)
        net._feature_module = "features"            # hook net.features output
        return net

    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=w)
        in_f = net.classifier[1].in_features        # 1280
        net.classifier[1] = nn.Linear(in_f, num_classes)
        net._feature_module = "features"
        return net

    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=w)
        in_f = net.classifier[2].in_features        # 768
        net.classifier[2] = nn.Linear(in_f, num_classes)
        net._feature_module = "features"
        return net

    raise KeyError(f"Unknown backbone '{name}'. Available: {BACKBONES}")


def target_layer_for(model):
    """Return the final feature-map module used by CAM methods."""
    name = getattr(model, "_feature_module", "features")
    mod = getattr(model, name, None)
    if mod is None:
        raise AttributeError(f"Backbone has no '{name}' module for CAM hooking.")
    return mod
