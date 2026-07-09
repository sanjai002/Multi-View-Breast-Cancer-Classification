"""Integrated Gradients (Sundararajan et al., 2017).

Attributes the target-class score to input pixels by integrating gradients along
a straight path from a black baseline (in normalised space) to the input. Works
directly on the four-view input tensor and returns per-view attribution maps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from explainability.gradcam import normalize_maps


class IntegratedGradients:
    def __init__(self, model: nn.Module, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        # Normalised value of a fully black pixel: (0 - mean) / std.
        mean = float(cfg.data.normalize_mean[0])
        std = float(cfg.data.normalize_std[0])
        self.blank_value = (0.0 - mean) / (std + 1e-8)

    def attribute(self, views: torch.Tensor, mask: torch.Tensor,
                  target: Optional[torch.Tensor] = None, steps: Optional[int] = None):
        steps = steps or self.cfg.explain.ig_steps
        b, v, c, h, w = views.shape
        views = views.detach().to(self.device)
        mask = mask.to(self.device)
        baseline = torch.full_like(views, self.blank_value)

        with torch.no_grad():
            logits = self.model(views, mask)["logits"].float()
            probs = F.softmax(logits, dim=1)
            if target is None:
                target = logits.argmax(dim=1)

        total_grad = torch.zeros_like(views)
        for i in range(1, steps + 1):
            alpha = i / steps
            x = (baseline + alpha * (views - baseline)).clone().requires_grad_(True)
            with torch.enable_grad():
                out = self.model(x, mask)
                logit = out["logits"].float()
                score = logit.gather(1, target.view(-1, 1)).sum()
                self.model.zero_grad(set_to_none=True)
                grad = torch.autograd.grad(score, x, retain_graph=False)[0]
            total_grad += grad.detach()

        avg_grad = total_grad / steps
        attributions = (views - baseline) * avg_grad          # (B, V, C, H, W)
        attr_map = attributions.sum(dim=2).abs()               # (B, V, H, W)

        flat = attr_map.view(b * v, h, w)
        maps = normalize_maps(flat).view(b, v, h, w).detach().cpu().numpy()
        return maps, target.cpu().numpy(), probs.cpu().numpy()
