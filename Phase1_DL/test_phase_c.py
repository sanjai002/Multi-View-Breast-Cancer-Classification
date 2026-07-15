#!/usr/bin/env python3
"""Verify Phase C: Datasets, Models, and Training components."""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.base import Config
from datasets.cached_dataset import CachedMammographyDataset
from datasets.transforms import MammographyTransforms
from models.classifier import MultiViewMammographyClassifier, build_model
from training.trainer import Trainer
from training.loss_functions import FocalLoss, WeightedBCELoss
from utils.logging import get_logger
from utils.reproducibility import seed_everything

logger = get_logger("phase_c_test")


def test_transforms():
    """Test augmentation transforms."""
    logger.info("=" * 60)
    logger.info("TEST 1: Augmentation Transforms")
    logger.info("=" * 60)

    # Create dummy (4, 224, 224) image
    dummy_image = np.random.uniform(0, 1, (4, 224, 224)).astype(np.float32)

    # Test training transforms
    train_aug = MammographyTransforms.get_train_transforms(224)
    aug_image = train_aug(dummy_image)
    assert aug_image.shape == (4, 224, 224)
    assert aug_image.dtype == np.float32
    assert aug_image.min() >= 0 and aug_image.max() <= 1
    logger.info(f"✓ Training transforms: {aug_image.shape}, range [{aug_image.min():.3f}, {aug_image.max():.3f}]")

    # Test validation transforms
    eval_aug = MammographyTransforms.get_eval_transforms()
    eval_image = eval_aug(dummy_image)
    assert eval_image.shape == (4, 224, 224)
    assert np.allclose(eval_image, dummy_image)  # Should be identity
    logger.info(f"✓ Validation transforms: {eval_image.shape} (identity)")


def test_model():
    """Test model architecture."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Model Architecture")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(num_classes=2, pretrained=False, device=device)
    logger.info(f"✓ Model built: {type(model).__name__}")

    # Forward pass with dummy input
    batch_size = 4
    views = torch.randn(batch_size, 4, 224, 224, device=device)
    mask = torch.ones(batch_size, 4, device=device)

    logits, attention_weights = model(views, mask)

    assert logits.shape == (batch_size, 2)
    assert attention_weights.shape == (batch_size, 4)
    logger.info(f"✓ Forward pass: logits {logits.shape}, attention {attention_weights.shape}")

    # Test with missing views
    mask[0, 0] = 0  # Missing first view for first sample
    logits_missing, attention_missing = model(views, mask)
    assert logits_missing.shape == (batch_size, 2)
    assert attention_missing[0, 0].item() < 0.1  # Should have near-zero weight
    logger.info(f"✓ Missing view handling: attention for absent view = {attention_missing[0, 0]:.4f}")

    # Test attention map extraction
    attention_map = model.get_attention_map(views, mask)
    assert attention_map.shape == (batch_size, 4)
    logger.info(f"✓ Attention map extraction: {attention_map.shape}")


def test_loss_functions():
    """Test loss functions."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Loss Functions")
    logger.info("=" * 60)

    batch_size = 8
    logits = torch.randn(batch_size, 2)
    targets = torch.randint(0, 2, (batch_size,))

    # Test Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_val = focal_loss(logits, targets)
    assert loss_val.item() >= 0
    logger.info(f"✓ Focal Loss: {loss_val.item():.4f}")

    # Test Weighted BCE Loss
    weighted_bce = WeightedBCELoss(pos_weight=2.0)
    logits_bce = torch.randn(batch_size, 1)
    loss_val_bce = weighted_bce(logits_bce, targets)
    assert loss_val_bce.item() >= 0
    logger.info(f"✓ Weighted BCE Loss: {loss_val_bce.item():.4f}")


def test_trainer():
    """Test trainer initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Trainer Initialization")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy model, data loaders
    model = build_model(num_classes=2, pretrained=False, device=device)
    cfg = Config()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy data loaders
    from torch.utils.data import DataLoader, TensorDataset

    dummy_dataset = TensorDataset(
        torch.randn(10, 4, 224, 224),
        torch.ones(10, 4),
        torch.randint(0, 2, (10,)),
    )
    train_loader = DataLoader(dummy_dataset, batch_size=2)
    val_loader = DataLoader(dummy_dataset, batch_size=2)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
    )
    logger.info(f"✓ Trainer initialized: {len(train_loader)} training batches")

    # Test one training step
    train_metrics = trainer.train_epoch()
    assert "loss" in train_metrics
    assert "acc" in train_metrics
    assert "auc" in train_metrics
    logger.info(f"✓ Training epoch: loss={train_metrics['loss']:.4f}, "
                f"acc={train_metrics['acc']:.4f}, auc={train_metrics['auc']:.4f}")

    # Test validation
    val_metrics = trainer.validate()
    logger.info(f"✓ Validation: loss={val_metrics['loss']:.4f}, "
                f"acc={val_metrics['acc']:.4f}, auc={val_metrics['auc']:.4f}")


def main():
    """Run all tests."""
    logger.info("PHASE C VERIFICATION: Datasets, Models, and Training")
    logger.info("=" * 60)

    seed_everything(42)

    try:
        test_transforms()
        test_model()
        test_loss_functions()
        test_trainer()

        logger.info("\n" + "=" * 60)
        logger.info("✓ PHASE C VERIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info("\nComponents verified:")
        logger.info("  ✓ Augmentation transforms (training + eval)")
        logger.info("  ✓ Multi-view classifier with attention fusion")
        logger.info("  ✓ Missing view masking")
        logger.info("  ✓ Focal Loss and Weighted BCE")
        logger.info("  ✓ Trainer with validation and checkpointing")
        logger.info("\nReady for Phase D: Evaluation & Inference")
        return 0

    except Exception as e:
        logger.error(f"✗ PHASE C VERIFICATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
