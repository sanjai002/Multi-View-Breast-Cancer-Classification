"""Grad-CAM for the multi-view fusion model.

A single forward/backward through the shared backbone yields one activation
tensor of shape ``(B*V, C, h, w)`` covering all four views, so per-view class
activation maps come for free. Row ``b*V + v`` corresponds to view ``v`` of
patient ``b`` (following ``config.VIEW_ORDER``).

The backbone is often frozen during inference, so the input tensor is marked
``requires_grad`` to guarantee the activation participates in the autograd graph.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def normalize_maps(maps: torch.Tensor) -> torch.Tensor:
    """Per-map min-max normalisation to [0, 1]. ``maps``: (N, H, W)."""
    n = maps.shape[0]
    flat = maps.view(n, -1)
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    flat = (flat - mins) / (maxs - mins + 1e-8)
    return flat.view_as(maps)


def finalize_maps(cam_flat: torch.Tensor, b: int, v: int,
                  out_hw: Tuple[int, int]) -> np.ndarray:
    """Upsample, normalise and reshape ``(B*V, h, w)`` -> ``(B, V, H, W)``."""
    cam = F.interpolate(cam_flat.unsqueeze(1), size=out_hw, mode="bilinear",
                        align_corners=False).squeeze(1)
    cam = normalize_maps(cam)
    return cam.view(b, v, out_hw[0], out_hw[1]).detach().cpu().numpy()


class GradCAM:
    """Class-discriminative localisation via gradient-weighted activations."""

    def __init__(self, model: nn.Module, cfg: Config,
                 target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.target_layer = target_layer or model.backbone.get_target_layer()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [self.target_layer.register_forward_hook(self._forward_hook)]

    # ------------------------------------------------------------------ #
    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ #
    def _run(self, views: torch.Tensor, mask: torch.Tensor,
             target: Optional[torch.Tensor]):
        self.model.eval()
        # Clone so we never flip requires_grad on the caller's tensor (on CPU
        # ``.to(device)`` returns the same object).
        views = views.detach().clone().to(self.device).requires_grad_(True)
        mask = mask.to(self.device)
        with torch.enable_grad():
            out = self.model(views, mask)
            logits = out["logits"].float()
            probs = F.softmax(logits, dim=1)
            if target is None:
                target = logits.argmax(dim=1)
            score = logits.gather(1, target.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            if views.grad is not None:
                views.grad = None
            score.backward()
        return probs.detach(), target.detach()

    def _weights(self) -> torch.Tensor:
        """Grad-CAM channel weights: global-average-pooled gradients."""
        return self.gradients.mean(dim=(2, 3), keepdim=True)

    def attribute(self, views: torch.Tensor, mask: torch.Tensor,
                  target: Optional[torch.Tensor] = None):
        """Return ``(maps (B,V,H,W), targets (B,), probs (B,C))``."""
        b, v = views.shape[0], views.shape[1]
        probs, target = self._run(views, mask, target)
        weights = self._weights()                              # (B*V, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)          # (B*V, h, w)
        cam = F.relu(cam)
        maps = finalize_maps(cam, b, v, (views.shape[-2], views.shape[-1]))
        return maps, target.cpu().numpy(), probs.cpu().numpy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
        return False
