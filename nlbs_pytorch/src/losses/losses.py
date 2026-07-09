"""
Loss functions + factory.

Comparison (see README):
  CrossEntropy        - baseline; biased toward the majority (normal) class.
  Weighted CE         - re-weights classes; helps imbalance, can be noisy.
  Focal Loss          - down-weights easy negatives, focuses on hard/positive
                        cancer cases. ***RECOMMENDED*** for this imbalanced task.
  BCEWithLogits       - equivalent binary form (pos_weight for imbalance).
  Dice                - segmentation loss; not appropriate for whole-image
                        classification (included only for completeness/comparison).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        # alpha weights the positive (cancer) class; (1-alpha) the negative.
        self.alpha = alpha
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        # per-class alpha vector: [1-alpha, alpha]
        alpha = torch.tensor([1 - self.alpha, self.alpha], device=logits.device)
        if self.ls > 0:                       # label smoothing on the hard target
            n = logits.size(1)
            true = torch.full_like(logp, self.ls / n)
            true.scatter_(1, target.unsqueeze(1), 1 - self.ls + self.ls / n)
            ce = -(true * logp).sum(1)
            focal = ((1 - p).gather(1, target.unsqueeze(1)).squeeze(1) ** self.gamma)
            return (alpha[target] * focal * ce).mean()
        ce = F.nll_loss(logp, target, reduction="none")
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        return (alpha[target] * (1 - pt) ** self.gamma * ce).mean()


def build_loss(cfg_loss, class_weights: torch.Tensor | None = None) -> nn.Module:
    name = cfg_loss.name
    ls = cfg_loss.get("label_smoothing", 0.0)
    if name == "focal":
        return FocalLoss(cfg_loss.focal_gamma, cfg_loss.focal_alpha, ls)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if name == "bce":
        pos_weight = None
        if class_weights is not None:
            pos_weight = (class_weights[1] / class_weights[0]).reshape(1)
        return _BCEAdapter(pos_weight)
    raise ValueError(f"unknown loss {name}")


class _BCEAdapter(nn.Module):
    """BCEWithLogits on the positive-class logit (2-logit model -> binary)."""
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        pos_logit = logits[:, 1] - logits[:, 0]
        return self.bce(pos_logit, target.float())
