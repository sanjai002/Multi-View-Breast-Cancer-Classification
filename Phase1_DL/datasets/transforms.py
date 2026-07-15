"""Data augmentation transforms for mammography images."""

from typing import Callable, Optional
import numpy as np
import albumentations as A
from albumentations import DualTransform
import torch


class MammographyTransforms:
    """Training and validation transforms for mammography."""

    @staticmethod
    def get_train_transforms(image_size: int = 224) -> Callable:
        """Get augmentation transforms for training.

        Applies: rotation, affine, brightness/contrast, Gaussian noise, blur,
        CutMix, MixUp.

        Args:
            image_size: Image size (square).

        Returns:
            Augmentation function that accepts (4, H, W) array.
        """

        def augment(images: np.ndarray) -> np.ndarray:
            """Apply augmentations to 4-view stack.

            Args:
                images: (4, H, W) array of float32 values in [0, 1].

            Returns:
                Augmented (4, H, W) array.
            """
            # Apply spatial transforms to all views together
            aug = A.Compose([
                # Rotation
                A.Rotate(limit=15, border_mode=0, p=0.5),
                # Affine: shear, scale
                A.Affine(scale=(0.9, 1.1), shear=(-10, 10), p=0.3),
                # Brightness/Contrast
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
                # Gaussian noise
                A.GaussNoise(p=0.2),
                # Blur
                A.Blur(blur_limit=3, p=0.2),
            ], bbox_params=None)

            # Apply to each view independently
            augmented = []
            for view in images:
                # Ensure [0, 1] range
                view = np.clip(view, 0, 1)
                # Apply augmentation
                aug_view = aug(image=view)["image"]
                augmented.append(aug_view)

            result = np.stack(augmented, axis=0)

            # Apply CutMix to random pair of views
            if np.random.random() < 0.3:
                result = _apply_cutmix(result)

            # Apply MixUp to random pair of views
            if np.random.random() < 0.2:
                result = _apply_mixup(result)

            return np.clip(result, 0, 1).astype(np.float32)

        return augment

    @staticmethod
    def get_eval_transforms() -> Callable:
        """Get identity transform for validation/testing.

        Returns:
            No-op function that returns input unchanged.
        """

        def identity(images: np.ndarray) -> np.ndarray:
            """Return images unchanged."""
            return np.clip(images, 0, 1).astype(np.float32)

        return identity


def _apply_cutmix(images: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Apply CutMix augmentation to random pair of views.

    Cuts rectangular patch from one view and pastes into another.

    Args:
        images: (4, H, W) array.
        alpha: Beta distribution parameter.

    Returns:
        (4, H, W) array with CutMix applied.
    """
    n_views = images.shape[0]

    # Select two random views
    view_a, view_b = np.random.choice(n_views, 2, replace=False)

    h, w = images.shape[1:]
    lam = np.random.beta(alpha, alpha)

    # Sample random cut size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    # Sample random position
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    # Apply cut
    images[view_a, bby1:bby2, bbx1:bbx2] = images[view_b, bby1:bby2, bbx1:bbx2]

    return images


def _apply_mixup(images: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Apply MixUp augmentation to random pair of views.

    Blends two random views with weighted average.

    Args:
        images: (4, H, W) array.
        alpha: Beta distribution parameter.

    Returns:
        (4, H, W) array with MixUp applied.
    """
    n_views = images.shape[0]

    # Select two random views
    view_a, view_b = np.random.choice(n_views, 2, replace=False)

    lam = np.random.beta(alpha, alpha)
    images[view_a] = lam * images[view_a] + (1 - lam) * images[view_b]

    return images
