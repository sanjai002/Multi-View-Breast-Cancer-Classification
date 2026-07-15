"""Multi-view mammography classifier with attention-based fusion."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionFusionHead(nn.Module):
    """Attention-based fusion of 4 mammography views.

    Computes attention weights for each view and fuses features.
    Missing views (mask=0) have zero attention weight.
    """

    def __init__(self, feature_dim: int, num_views: int = 4) -> None:
        """Initialize attention fusion.

        Args:
            feature_dim: Feature dimension per view.
            num_views: Number of views (4).
        """
        super().__init__()
        self.num_views = num_views

        # Attention: (num_views, feature_dim) -> (num_views,)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse features with attention weighting.

        Args:
            features: (B, num_views, feature_dim) tensor.
            mask: (B, num_views) binary mask (1=present, 0=missing).

        Returns:
            Tuple of:
            - fused_features: (B, feature_dim) weighted average.
            - attention_weights: (B, num_views) for interpretability.
        """
        batch_size = features.shape[0]

        # Compute attention for each view: (B, num_views, 1)
        att_logits = self.attention(features)  # (B, num_views, 1)
        att_weights = att_logits.squeeze(-1)  # (B, num_views)

        # Mask out missing views
        att_weights = att_weights * mask  # (B, num_views)

        # Normalize to probability distribution
        att_weights = att_weights / (att_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted average: (B, feature_dim)
        fused = (features * att_weights.unsqueeze(-1)).sum(dim=1)

        return fused, att_weights


class MultiViewMammographyClassifier(nn.Module):
    """ConvNeXt-based multi-view mammography classifier.

    Architecture:
    - 4 separate ConvNeXt-Large backbones (weight-shared)
    - Per-view feature extraction
    - Attention-based multi-view fusion
    - Binary classification head
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        fusion_dim: int = 768,
    ) -> None:
        """Initialize classifier.

        Args:
            num_classes: Number of output classes (2 for binary).
            pretrained: Whether to use ImageNet-1K pretrained backbone.
            fusion_dim: Feature dimension for fusion (ConvNeXt-Large outputs 768).
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim

        # Load pretrained ConvNeXt-Large backbone
        # NOTE: Model expects RGB (3-channel) input, but we'll handle grayscale
        self.backbone = models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Modify input layer for grayscale (1 channel -> 3 channels)
        # Repeat grayscale to RGB by duplicating channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize with pretrained weights (copy to 1st channel)
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight.copy_(
                    original_conv.weight[:, :1, :, :]
                )
                if original_conv.bias is not None:
                    self.backbone.features[0][0].bias.copy_(original_conv.bias)

        # Get output feature dimension from backbone
        self.feature_dim = self.backbone.classifier[-1].in_features

        # Remove classification head (we'll add custom fusion + head)
        self.backbone.classifier = nn.Identity()

        # Attention fusion layer
        self.fusion = AttentionFusionHead(self.feature_dim, num_views=4)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        views: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            views: (B, 4, H, W) tensor of 4 mammography views.
            mask: (B, 4) binary mask (1=present, 0=missing).

        Returns:
            Tuple of:
            - logits: (B, num_classes) classification logits.
            - attention_weights: (B, 4) for interpretability.
        """
        batch_size = views.shape[0]

        # Process each view through backbone
        # Reshape to (B*4, 1, H, W) for batch processing
        views_flat = views.reshape(batch_size * 4, 1, views.shape[2], views.shape[3])

        # Forward through backbone: (B*4, feature_dim)
        features_flat = self.backbone(views_flat)

        # Reshape back to (B, 4, feature_dim)
        features = features_flat.reshape(batch_size, 4, self.feature_dim)

        # Attention-based fusion: (B, feature_dim)
        fused_features, attention_weights = self.fusion(features, mask)

        # Classification head: (B, num_classes)
        logits = self.head(fused_features)

        return logits, attention_weights

    def get_attention_map(self, views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get attention weights for each view (for interpretability).

        Args:
            views: (B, 4, H, W) tensor.
            mask: (B, 4) binary mask.

        Returns:
            (B, 4) attention weights.
        """
        with torch.no_grad():
            _, attention_weights = self.forward(views, mask)
        return attention_weights


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = "cuda",
) -> MultiViewMammographyClassifier:
    """Factory function to build and move model to device.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet-1K pretrained weights.
        device: Device to move model to.

    Returns:
        Model on specified device.
    """
    model = MultiViewMammographyClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = model.to(device)
    return model
