"""Loss functions for binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Loss = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t: model probability for the true class
    - α_t: balancing weight for the true class
    - γ: focusing parameter (controls how much focus on hard examples)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Balance weight. If alpha=0.25, the negative class gets 0.75.
            gamma: Focusing parameter. Recommended: 1.0-2.5 for imbalanced data.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.

        Args:
            logits: (B, num_classes) model output logits.
            targets: (B,) ground truth labels.

        Returns:
            Loss value (scalar if reduction='mean').
        """
        # Cross-entropy + focal term
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # Apply alpha balancing
        alpha_t = self.alpha if targets == 1 else (1 - self.alpha)
        focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy for class imbalance.

    Useful when negative class (Normal) is much more frequent than positive (Cancer).
    Weight = 1 / class_frequency
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        """Initialize weighted BCE loss.

        Args:
            pos_weight: Weight for positive class. >1 penalizes false negatives more.
                        Recommended: use inverse of class ratio.
                        E.g., if 90% negative and 10% positive, pos_weight=9.0
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: (B,) or (B, 1) logits for positive class.
            targets: (B,) or (B, 1) binary targets {0, 1}.

        Returns:
            Loss value (scalar).
        """
        # Ensure correct shapes
        logits = logits.squeeze()
        targets = targets.float().squeeze()
        return self.bce(logits, targets)
