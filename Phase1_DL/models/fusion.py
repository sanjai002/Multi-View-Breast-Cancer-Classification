"""Dual-branch multi-view feature-fusion classifier.

Architecture
------------
* A **single shared** ResNet-50 backbone encodes each of the four views (memory
  efficient and standard practice for multi-view mammography).
* Each view's ``layer4`` map is recalibrated by an **SE block**, globally pooled
  and projected to a per-view embedding.
* **Dual branch fusion**: a *CC branch* fuses {LCC, RCC} and an *MLO branch*
  fuses {LMLO, RMLO}, each with masked gated **attention pooling**.
* A final **attention fusion** combines the two branch embeddings into a single
  patient embedding, which a small MLP head maps to the three classes.

The forward pass returns the logits, the patient embedding and the per-view
embeddings; the latter two are exported for Phase 2 (RL).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from config import Config
from models.attention import AttentionPooling, SEBlock
from models.resnet50 import ResNet50Backbone

# Indices into VIEW_ORDER = (LCC, LMLO, RCC, RMLO).
CC_INDICES: List[int] = [0, 2]    # LCC, RCC
MLO_INDICES: List[int] = [1, 3]   # LMLO, RMLO


class MultiViewFusionModel(nn.Module):
    """Full patient-level three-class classifier over four mammography views."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        self.backbone = ResNet50Backbone(
            in_channels=cfg.data.in_channels, pretrained=m.pretrained
        )
        self.se = SEBlock(self.backbone.out_channels, reduction=m.se_reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Shared per-view projection: 2048 -> embed_dim.
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.out_channels, m.embed_dim),
            nn.BatchNorm1d(m.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(m.dropout),
        )

        # Branch-level and patient-level attention fusion.
        self.cc_fusion = AttentionPooling(m.embed_dim, m.attention_hidden)
        self.mlo_fusion = AttentionPooling(m.embed_dim, m.attention_hidden)
        self.patient_fusion = AttentionPooling(m.embed_dim, m.attention_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(m.embed_dim, m.embed_dim // 2),
            nn.BatchNorm1d(m.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(m.dropout),
            nn.Linear(m.embed_dim // 2, cfg.data.num_classes),
        )

        if m.freeze_backbone:
            self.backbone.freeze_all()

    # ------------------------------------------------------------------ #
    def encode_views(self, views: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, V, C, H, W)`` into per-view embeddings ``(B, V, D)``."""
        b, v, c, h, w = views.shape
        x = views.reshape(b * v, c, h, w)
        fmap = self.backbone.forward_features(x)      # (B*V, 2048, h', w')
        fmap = self.se(fmap)
        pooled = self.gap(fmap).flatten(1)            # (B*V, 2048)
        emb = self.projection(pooled)                 # (B*V, D)
        return emb.view(b, v, -1)

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        view_emb = self.encode_views(views)           # (B, V, D)

        cc_emb, cc_attn = self.cc_fusion(
            view_emb[:, CC_INDICES], mask[:, CC_INDICES]
        )
        mlo_emb, mlo_attn = self.mlo_fusion(
            view_emb[:, MLO_INDICES], mask[:, MLO_INDICES]
        )

        branch_stack = torch.stack([cc_emb, mlo_emb], dim=1)          # (B, 2, D)
        branch_mask = torch.stack(
            [mask[:, CC_INDICES].amax(dim=1), mask[:, MLO_INDICES].amax(dim=1)],
            dim=1,
        )                                                             # (B, 2)
        patient_emb, branch_attn = self.patient_fusion(branch_stack, branch_mask)

        logits = self.classifier(patient_emb)
        return {
            "logits": logits,
            "patient_embedding": patient_emb,
            "view_embeddings": view_emb,
            "cc_attention": cc_attn,
            "mlo_attention": mlo_attn,
            "branch_attention": branch_attn,
        }

    @torch.no_grad()
    def extract_features(self, views: torch.Tensor,
                         mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference-only feature extraction used for the Phase 2 export."""
        was_training = self.training
        self.eval()
        out = self.forward(views, mask)
        if was_training:
            self.train()
        return {
            "logits": out["logits"],
            "patient_embedding": out["patient_embedding"],
            "view_embeddings": out["view_embeddings"],
        }

    # ------------------------------------------------------------------ #
    def get_param_groups(self, cfg: Config) -> List[Dict]:
        """Differential learning-rate groups: backbone vs. everything else."""
        backbone_params, head_params = [], []
        for name, param in self.named_parameters():
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": backbone_params, "lr": cfg.train.backbone_lr,
             "weight_decay": cfg.train.weight_decay, "name": "backbone"},
            {"params": head_params, "lr": cfg.train.head_lr,
             "weight_decay": cfg.train.weight_decay, "name": "head"},
        ]

    def apply_unfreeze_schedule(self, epoch: int, cfg: Config) -> bool:
        """Unfreeze backbone stages according to ``cfg.train.unfreeze_schedule``.

        Returns ``True`` if the set of trainable stages changed this epoch.
        """
        schedule = cfg.train.unfreeze_schedule
        active_events = {ep: stage for ep, stage in schedule.items() if ep <= epoch}
        if not active_events:
            if cfg.model.freeze_backbone:
                self.backbone.freeze_all()
            return False
        deepest_stage = active_events[max(active_events)]
        trainable = self.backbone.stages_up_to(deepest_stage)
        self.backbone.set_trainable_stages(trainable)
        return epoch in schedule

    def feature_extractor_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict of the encoder (everything except the classifier head)."""
        return {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("classifier.")
        }


def build_model(cfg: Config) -> MultiViewFusionModel:
    return MultiViewFusionModel(cfg)
