"""Score-CAM (Wang et al., 2020) for the multi-view fusion model.

Gradient-free: each activation channel of the target view is upsampled, used to
mask that view's input, and the resulting change in the target-class score
becomes the channel's weight. To stay tractable on 2048-channel ResNet features
only the top-K highest-energy channels are scored (``cfg.explain.scorecam_channels``)
and forwards are batched (``cfg.explain.scorecam_batch``).

Masking happens on the normalised input tensor, which is the standard practical
approximation when a model consumes standardised images.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from explainability.gradcam import normalize_maps


class ScoreCAM:
    def __init__(self, model: nn.Module, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.target_layer = model.backbone.get_target_layer()
        self.activations: Optional[torch.Tensor] = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def remove_hooks(self) -> None:
        self._handle.remove()

    @torch.no_grad()
    def attribute(self, views: torch.Tensor, mask: torch.Tensor, target=None):
        cfg = self.cfg
        b, v, c, h, w = views.shape
        views = views.to(self.device)
        mask = mask.to(self.device)

        out = self.model(views, mask)
        logits = out["logits"].float()
        probs = F.softmax(logits, dim=1)
        if target is None:
            target = logits.argmax(dim=1)

        acts = self.activations                          # (B*V, Ca, h', w')
        n_channels = acts.shape[1]
        topk = min(cfg.explain.scorecam_channels, n_channels)
        maps = np.zeros((b, v, h, w), dtype=np.float32)

        for bi in range(b):
            tgt = int(target[bi].item())
            for vi in range(v):
                if mask[bi, vi] < 0.5:
                    continue
                a = acts[bi * v + vi]                    # (Ca, h', w')
                a_up = F.interpolate(a.unsqueeze(1), size=(h, w), mode="bilinear",
                                     align_corners=False).squeeze(1)   # (Ca, H, W)
                a_norm = normalize_maps(a_up)
                energy = a_up.flatten(1).sum(dim=1)
                idx = torch.topk(energy, topk).indices
                a_sel = a_norm[idx]                      # (topk, H, W)

                weights = torch.zeros(topk, device=self.device)
                for start in range(0, topk, cfg.explain.scorecam_batch):
                    chunk = a_sel[start:start + cfg.explain.scorecam_batch]
                    k = chunk.shape[0]
                    batch_views = views[bi].unsqueeze(0).repeat(k, 1, 1, 1, 1).clone()
                    batch_views[:, vi] = batch_views[:, vi] * chunk.unsqueeze(1)
                    batch_mask = mask[bi].unsqueeze(0).repeat(k, 1)
                    logit_k = self.model(batch_views, batch_mask)["logits"].float()
                    weights[start:start + k] = F.softmax(logit_k, dim=1)[:, tgt]

                weights = F.relu(weights)
                cam = (weights.view(-1, 1, 1) * a_sel).sum(dim=0)
                cam = F.relu(cam)
                maps[bi, vi] = normalize_maps(cam.unsqueeze(0)).squeeze(0).cpu().numpy()

        return maps, target.cpu().numpy(), probs.cpu().numpy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
        return False
