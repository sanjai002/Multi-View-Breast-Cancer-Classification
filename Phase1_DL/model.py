"""
model.py

Minimal shared-ResNet50 multi-view classifier for NLBS.

Design:
- One shared ResNet50 backbone for all four views.
- Each view is encoded independently.
- Missing views are handled through a binary mask.
- View embeddings are fused with masked average pooling.
- A small MLP head produces the final binary prediction.
- The backbone exposes a target layer for Grad-CAM.

Expected input from dataset.py:
    views: FloatTensor [B, 4, 1, H, W]
    mask:  FloatTensor [B, 4]

Output:
    dict with keys:
        logits: [B, 2]
        probs: [B, 2]
        patient_embedding: [B, D]
        view_embeddings: [B, 4, D]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------
class ResNet50Encoder(nn.Module):
    """
    Shared ResNet50 encoder adapted for single-channel mammography.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = resnet50(weights=weights)

        # Adapt conv1 from 3-channel RGB to 1-channel grayscale.
        old_conv = self.net.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self.net.conv1 = new_conv
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Remove the classification head; keep everything up to global pooling.
        self.feature_dim = self.net.fc.in_features
        self.net.fc = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return feature vectors of shape [B, 2048].
        """
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        # Save this for Grad-CAM via hooks on layer4.
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

    def get_target_layer(self) -> nn.Module:
        """
        Target layer for Grad-CAM / Score-CAM.
        """
        return self.net.layer4[-1]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

        # Keep BN layers in eval when frozen.
        self.eval()

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------
# Multi-view fusion classifier
# ---------------------------------------------------------------------
class MammographyClassifier(nn.Module):
    """
    Shared-backbone four-view mammography classifier.

    Simple masked fusion:
        - Encode each available view.
        - Multiply by mask.
        - Average only over present views.
        - Classify with a small MLP.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512,
        num_classes: int = 2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ResNet50Encoder(pretrained=pretrained, dropout=0.0)
        self.embed_dim = self.backbone.feature_dim

        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    # -----------------------------------------------------------------
    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    # -----------------------------------------------------------------
    def encode_views(self, views: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all views.

        Args:
            views: [B, V, 1, H, W]

        Returns:
            view_embeddings: [B, V, D]
            feature_maps: [B*V, C, h, w] from the last conv block, useful for CAM.
        """
        if views.ndim != 5:
            raise ValueError(f"Expected views shape [B, V, 1, H, W], got {tuple(views.shape)}")

        b, v, c, h, w = views.shape
        x = views.reshape(b * v, c, h, w)

        # Forward through the backbone while retaining the last conv feature map.
        feature_maps = self._forward_backbone_feature_maps(x)
        pooled = F.adaptive_avg_pool2d(feature_maps, output_size=1).flatten(1)
        pooled = self.backbone.dropout(pooled)
        view_embeddings = pooled.reshape(b, v, -1)

        return view_embeddings, feature_maps

    def _forward_backbone_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to the last convolutional feature map (before global pooling).
        """
        x = self.backbone.net.conv1(x)
        x = self.backbone.net.bn1(x)
        x = self.backbone.net.relu(x)
        x = self.backbone.net.maxpool(x)

        x = self.backbone.net.layer1(x)
        x = self.backbone.net.layer2(x)
        x = self.backbone.net.layer3(x)
        x = self.backbone.net.layer4(x)
        return x

    # -----------------------------------------------------------------
    def fuse(self, view_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked average pooling over the four views.

        Args:
            view_embeddings: [B, V, D]
            mask: [B, V] with 1 for present view, 0 for missing view

        Returns:
            patient_embedding: [B, hidden_dim]
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape [B, V], got {tuple(mask.shape)}")

        mask = mask.to(view_embeddings.dtype).unsqueeze(-1)  # [B, V, 1]
        masked = view_embeddings * mask

        denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        patient_embedding = masked.sum(dim=1) / denom

        patient_embedding = self.proj(patient_embedding)
        return patient_embedding

    # -----------------------------------------------------------------
    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            views: [B, 4, 1, H, W]
            mask:  [B, 4]

        Returns:
            dict with logits, probs, patient_embedding, view_embeddings
        """
        view_embeddings, _ = self.encode_views(views)
        patient_embedding = self.fuse(view_embeddings, mask)
        logits = self.classifier(patient_embedding)
        probs = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
            "patient_embedding": patient_embedding,
            "view_embeddings": view_embeddings,
        }


def build_model(
    pretrained: bool = True,
    dropout: float = 0.5,
    hidden_dim: int = 512,
    num_classes: int = 2,
    freeze_backbone: bool = False,
) -> MammographyClassifier:
    """
    Convenience constructor.
    """
    return MammographyClassifier(
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    # Quick smoke test.
    model = build_model(pretrained=False)
    x = torch.randn(2, 4, 1, 224, 224)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1]], dtype=torch.float32)
    out = model(x, mask)
    print("logits:", out["logits"].shape)
    print("probs:", out["probs"].shape)
    print("patient_embedding:", out["patient_embedding"].shape)
    print("view_embeddings:", out["view_embeddings"].shape)