from typing import Optional

import torch
import torch.nn as nn


class SharedBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, feat_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv(x).flatten(1)
        return self.fc(h)


class SimpleFusionModel(nn.Module):
    """Tiny fusion model with shared backbone and masked averaging."""

    def __init__(self, num_views: int = 4, feat_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.backbone = SharedBackbone(in_channels=1, feat_dim=feat_dim)
        self.num_views = num_views
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, images, mask=None):
        # images: (B, V, C, H, W)
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)
        feats = self.backbone(images)  # (B*V, feat_dim)
        feats = feats.view(B, V, -1)  # (B, V, feat_dim)

        if mask is None:
            mask = torch.ones(B, V, device=feats.device)
        mask = mask.unsqueeze(-1)  # (B, V, 1)

        summed = (feats * mask).sum(dim=1)  # (B, feat_dim)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts

        logits = self.classifier(pooled)
        return logits
