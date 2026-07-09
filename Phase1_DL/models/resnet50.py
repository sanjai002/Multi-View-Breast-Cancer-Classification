"""ResNet-50 backbone adapted for single-channel mammography.

* Loads ImageNet-pretrained weights.
* Adapts ``conv1`` from 3 to ``in_channels`` inputs. For a single channel the
  new kernel is the *sum* of the RGB kernels, which makes the 1-channel path
  numerically equivalent to replicating the grayscale image across three
  channels while keeping all pretrained information.
* Exposes stage-wise freeze / unfreeze for progressive transfer learning and a
  handle on the last convolutional stage for Grad-CAM.
"""

from __future__ import annotations

from typing import List, Set

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# ResNet-50 stages ordered from the network *head* (deepest) to the *stem*.
# Progressive unfreezing walks this list top-down.
STAGE_ORDER: List[str] = ["layer4", "layer3", "layer2", "layer1", "stem"]


class ResNet50Backbone(nn.Module):
    """Feature extractor returning either a spatial map or a pooled vector."""

    def __init__(self, in_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)

        self._adapt_first_conv(net, in_channels)

        # Keep stages as named attributes so freezing/hooking is explicit.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = 2048

    # ------------------------------------------------------------------ #
    @staticmethod
    def _adapt_first_conv(net: nn.Module, in_channels: int) -> None:
        if in_channels == 3:
            return
        old = net.conv1
        new = nn.Conv2d(
            in_channels, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=old.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
            else:
                # Tile/average the pretrained kernels to the requested channels.
                mean_kernel = old.weight.mean(dim=1, keepdim=True)
                new.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1))
        net.conv1 = new

    # ------------------------------------------------------------------ #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the layer4 feature map ``(B, 2048, h, w)``."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector ``(B, 2048)``."""
        x = self.forward_features(x)
        return self.pool(x).flatten(1)

    # ------------------------------------------------------------------ #
    def get_target_layer(self) -> nn.Module:
        """Module hooked by Grad-CAM family methods (last conv stage)."""
        return self.layer4

    def _stage_module(self, name: str) -> nn.Module:
        return getattr(self, name)

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def set_trainable_stages(self, trainable: Set[str]) -> None:
        """Enable grad only for the named stages (others frozen)."""
        self.freeze_all()
        for name in trainable:
            if name in STAGE_ORDER:
                for p in self._stage_module(name).parameters():
                    p.requires_grad = True

    def stages_up_to(self, deepest: str) -> Set[str]:
        """Return every stage from the head down to (and including) ``deepest``."""
        if deepest not in STAGE_ORDER:
            return set()
        return set(STAGE_ORDER[: STAGE_ORDER.index(deepest) + 1])
