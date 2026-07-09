"""Grad-CAM++ (Chattopadhyay et al., 2018).

Refines Grad-CAM's channel weighting with higher-order gradient terms, giving
sharper maps and better localisation when several instances of the target
appear. Only the weight computation differs, so it subclasses :class:`GradCAM`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from explainability.gradcam import GradCAM, finalize_maps


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ variant with pixel-wise weighting coefficients."""

    def attribute(self, views: torch.Tensor, mask: torch.Tensor, target=None):
        b, v = views.shape[0], views.shape[1]
        probs, target = self._run(views, mask, target)

        grads = self.gradients                     # (B*V, C, h, w)
        acts = self.activations                    # (B*V, C, h, w)
        grads_relu = F.relu(grads)

        grads_sq = grads ** 2
        grads_cu = grads_sq * grads
        # alpha coefficients (Eq. 19 in the paper).
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grads_sq + sum_acts * grads_cu
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alphas = grads_sq / denom

        weights = (alphas * grads_relu).sum(dim=(2, 3), keepdim=True)  # (B*V,C,1,1)
        cam = (weights * acts).sum(dim=1)          # (B*V, h, w)
        cam = F.relu(cam)
        maps = finalize_maps(cam, b, v, (views.shape[-2], views.shape[-1]))
        return maps, target.cpu().numpy(), probs.cpu().numpy()
