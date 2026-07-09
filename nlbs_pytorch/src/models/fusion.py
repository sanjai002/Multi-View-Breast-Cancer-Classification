"""
Dual-view fusion models (CC + MLO).

Strategies (config.model.fusion):
  early      : channel-concat the two views (6ch) -> 1x1 proj -> single backbone.
  late       : two backbones + two heads -> average the logits.
  feature /
  dualbranch : two backbones -> concat feature vectors -> MLP head.
  attention  : two backbones -> learn per-view attention weights -> weighted
               feature -> head.  ***RECOMMENDED*** (see README): keeps view-specific
               features (unlike early), and learns which view is informative for a
               given breast (unlike fixed averaging in late fusion).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import create_backbone


class _Head(nn.Sequential):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                         nn.Linear(in_dim, num_classes))


class ViewAttention(nn.Module):
    """Attention over the two view features -> weighted sum."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(),
                                   nn.Linear(dim // 2, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:      # (B, 2, D)
        w = torch.softmax(self.score(feats), dim=1)              # (B, 2, 1)
        return (w * feats).sum(dim=1)                            # (B, D)


class DualViewModel(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.fusion = cfg_model.fusion
        name = cfg_model.backbone
        pre = cfg_model.pretrained
        nc = cfg_model.num_classes
        drop = cfg_model.dropout

        if self.fusion == "early":
            self.proj = nn.Conv2d(6, 3, kernel_size=1)
            self.backbone, d = create_backbone(name, pre, in_chans=3)
            self.head = _Head(d, nc, drop)
        else:
            self.enc_cc, d = create_backbone(name, pre)
            if cfg_model.shared_backbone:
                self.enc_mlo = self.enc_cc
            else:
                self.enc_mlo, _ = create_backbone(name, pre)
            if self.fusion == "late":
                self.head_cc = _Head(d, nc, drop)
                self.head_mlo = _Head(d, nc, drop)
            elif self.fusion in ("feature", "dualbranch"):
                self.head = _Head(2 * d, nc, drop)
            elif self.fusion == "attention":
                self.attn = ViewAttention(d)
                self.head = _Head(d, nc, drop)
            else:
                raise ValueError(f"unknown fusion {self.fusion}")
        self.feat_dim = d

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        if self.fusion == "early":
            x = self.proj(torch.cat([cc, mlo], dim=1))
            return self.head(self.backbone(x))
        f_cc = self.enc_cc(cc)
        f_mlo = self.enc_mlo(mlo)
        if self.fusion == "late":
            return 0.5 * (self.head_cc(f_cc) + self.head_mlo(f_mlo))
        if self.fusion in ("feature", "dualbranch"):
            return self.head(torch.cat([f_cc, f_mlo], dim=1))
        # attention
        feats = torch.stack([f_cc, f_mlo], dim=1)                # (B,2,D)
        return self.head(self.attn(feats))

    # --- helpers for transfer-learning schedule ---
    def backbone_parameters(self):
        for n, p in self.named_parameters():
            if any(k in n for k in ("enc_cc", "enc_mlo", "backbone")):
                yield p

    def head_parameters(self):
        for n, p in self.named_parameters():
            if not any(k in n for k in ("enc_cc", "enc_mlo", "backbone")):
                yield p

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad = flag


def build_model(cfg_model) -> DualViewModel:
    return DualViewModel(cfg_model)
