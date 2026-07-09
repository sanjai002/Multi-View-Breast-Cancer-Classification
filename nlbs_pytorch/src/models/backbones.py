"""Backbone factory (timm) returning a pooled feature extractor + feature dim."""
from __future__ import annotations

import timm
import torch.nn as nn

SUPPORTED = ["resnet50", "densenet121", "tf_efficientnetv2_s", "convnext_tiny"]


def create_backbone(name: str, pretrained: bool = True, in_chans: int = 3):
    """Return (module, feat_dim). Module maps (B,C,H,W) -> (B, feat_dim)."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=0,
                              global_pool="avg", in_chans=in_chans)
    return model, model.num_features


def last_conv_module(name: str, backbone: nn.Module) -> nn.Module:
    """Best target layer for CAM methods, per backbone family."""
    if name.startswith("resnet"):
        return backbone.layer4[-1]
    if name.startswith("densenet"):
        return backbone.features[-1]
    if "efficientnet" in name:
        return backbone.conv_head if hasattr(backbone, "conv_head") else backbone.blocks[-1]
    if name.startswith("convnext"):
        return backbone.stages[-1]
    # Fallback: last Conv2d found.
    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1]
