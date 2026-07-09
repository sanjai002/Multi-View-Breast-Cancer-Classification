"""Attention building blocks: Squeeze-and-Excitation and gated attention pooling.

* :class:`SEBlock` performs channel recalibration on a convolutional feature map
  (Hu et al., 2018).
* :class:`AttentionPooling` fuses a *set* of view/branch embeddings into one
  vector with masked, gated attention (Ilse et al., 2018, attention-based MIL).
  The mask lets the module ignore missing mammography views cleanly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for ``(B, C, H, W)`` maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class AttentionPooling(nn.Module):
    """Masked gated-attention pooling over a set of embeddings.

    Given ``x`` of shape ``(B, N, D)`` and a binary ``mask`` of shape ``(B, N)``
    it returns the attention-weighted sum ``(B, D)`` and the attention weights
    ``(B, N)``. Fully-masked rows return a zero vector.
    """

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.attn_v = nn.Linear(dim, hidden)
        self.attn_u = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gated attention scores: w^T (tanh(Vx) * sigmoid(Ux)).
        gate = torch.tanh(self.attn_v(x)) * torch.sigmoid(self.attn_u(x))
        logits = self.attn_w(gate).squeeze(-1)  # (B, N)

        # Mask out absent elements before the softmax.
        logits = logits.masked_fill(mask < 0.5, -1e4)
        weights = F.softmax(logits, dim=1)  # (B, N)

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)

        # Zero the output for rows that had no valid element at all.
        valid = (mask.sum(dim=1, keepdim=True) > 0).float()
        pooled = pooled * valid
        return pooled, weights
