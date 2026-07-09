"""Data augmentation for training.

Two layers of augmentation are provided:

* **Per-image** transforms built with Albumentations (applied independently to
  each of the four views inside the dataset). Includes rotation, affine,
  brightness, contrast, gamma, random crop, Gaussian noise and random erasing.
  The version-fragile transforms (noise / erasing / resized-crop) are
  implemented as small ``ImageOnlyTransform`` subclasses so the pipeline behaves
  identically across Albumentations 1.4–2.x.

* **Batch-level** MixUp and CutMix, applied in the training loop to the stacked
  multi-view tensor. Both return mixed targets so the loss can be interpolated.

Augmentation is only ever applied to the training split.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from config import Config


# --------------------------------------------------------------------------- #
# Version-robust custom Albumentations transforms
# --------------------------------------------------------------------------- #
class _CompatImageOnly(ImageOnlyTransform):
    """Base that swallows the ``always_apply`` signature change across versions."""

    def __init__(self, p: float = 0.5) -> None:
        try:
            super().__init__(p=p)
        except TypeError:  # pragma: no cover - older Albumentations
            super().__init__(always_apply=False, p=p)


class GaussianNoiseGray(_CompatImageOnly):
    """Additive Gaussian noise for float images in [0, 1]."""

    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.3):
        super().__init__(p=p)
        self.std_range = std_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(0.0, std, size=img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0.0, 1.0)

    def get_transform_init_args_names(self):
        return ("std_range",)


class RandomErasingGray(_CompatImageOnly):
    """Randomly erase a rectangular region (a.k.a. Cutout / Random Erasing)."""

    def __init__(self, area_range: Tuple[float, float] = (0.02, 0.12),
                 aspect_range: Tuple[float, float] = (0.3, 3.3),
                 fill: float = 0.0, p: float = 0.25):
        super().__init__(p=p)
        self.area_range = area_range
        self.aspect_range = aspect_range
        self.fill = fill

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(*self.area_range) * area
            aspect = np.random.uniform(*self.aspect_range)
            eh = int(round(np.sqrt(target_area * aspect)))
            ew = int(round(np.sqrt(target_area / aspect)))
            if eh < h and ew < w:
                y0 = np.random.randint(0, h - eh)
                x0 = np.random.randint(0, w - ew)
                img[y0:y0 + eh, x0:x0 + ew] = self.fill
                break
        return img

    def get_transform_init_args_names(self):
        return ("area_range", "aspect_range", "fill")


class RandomResizedCropGray(_CompatImageOnly):
    """Random scale/aspect crop that is resized back to the canvas size.

    Implemented directly (rather than ``A.RandomResizedCrop``) because that
    transform's constructor signature changed between Albumentations versions.
    """

    def __init__(self, size: int, scale: Tuple[float, float] = (0.8, 1.0),
                 ratio: Tuple[float, float] = (0.85, 1.18), p: float = 0.5):
        super().__init__(p=p)
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = np.exp(np.random.uniform(*log_ratio))
            cw = int(round(np.sqrt(target_area * aspect)))
            ch = int(round(np.sqrt(target_area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                x0 = np.random.randint(0, w - cw + 1)
                y0 = np.random.randint(0, h - ch + 1)
                crop = img[y0:y0 + ch, x0:x0 + cw]
                return self._resize(crop, img.ndim)
        return self._resize(img, img.ndim)

    def _resize(self, arr: np.ndarray, ndim: int) -> np.ndarray:
        out = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if ndim == 3 and out.ndim == 2:  # cv2 drops a singleton channel axis
            out = out[..., None]
        return out

    def get_transform_init_args_names(self):
        return ("size", "scale", "ratio")


class BrightnessContrastGray(_CompatImageOnly):
    """Brightness/contrast jitter for [0, 1] floats, always clipped to range.

    Replaces ``A.RandomBrightnessContrast`` so values never leave [0, 1] and
    poison a subsequent gamma (``power`` of a negative number -> NaN).
    """

    def __init__(self, brightness_limit: float = 0.15,
                 contrast_limit: float = 0.15, p: float = 0.5):
        super().__init__(p=p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
        beta = np.random.uniform(-self.brightness_limit, self.brightness_limit)
        out = img.astype(np.float32) * alpha + beta
        return np.clip(out, 0.0, 1.0)

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit")


class GammaGray(_CompatImageOnly):
    """Gamma correction for [0, 1] floats (input clipped so ``power`` is safe)."""

    def __init__(self, gamma_limit: Tuple[float, float] = (0.8, 1.2), p: float = 0.3):
        super().__init__(p=p)
        self.gamma_limit = gamma_limit

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        gamma = np.random.uniform(*self.gamma_limit)
        out = np.clip(img.astype(np.float32), 0.0, 1.0)
        return np.power(out, gamma)

    def get_transform_init_args_names(self):
        return ("gamma_limit",)


# --------------------------------------------------------------------------- #
# Transform pipelines
# --------------------------------------------------------------------------- #
def build_train_transforms(cfg: Config) -> A.Compose:
    """Full training augmentation pipeline ending in normalise + to-tensor."""
    size = cfg.data.image_size
    mean = list(cfg.data.normalize_mean)
    std = list(cfg.data.normalize_std)
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.06),
                rotate=(-15, 15),
                shear=(-7, 7),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7,
            ),
            RandomResizedCropGray(size=size, scale=(0.85, 1.0), p=0.4),
            BrightnessContrastGray(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            GammaGray(gamma_limit=(0.8, 1.2), p=0.3),
            GaussianNoiseGray(std_range=(0.01, 0.04), p=0.25),
            RandomErasingGray(p=0.25),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(cfg: Config) -> A.Compose:
    """Deterministic pipeline for validation / test: normalise + to-tensor."""
    return A.Compose(
        [
            A.Normalize(
                mean=list(cfg.data.normalize_mean),
                std=list(cfg.data.normalize_std),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------- #
# Batch-level MixUp / CutMix for the multi-view tensor
# --------------------------------------------------------------------------- #
def _rand_bbox(size: Tuple[int, int], lam: float) -> Tuple[int, int, int, int]:
    h, w = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x0, x1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    return y0, y1, x0, x1


def mixup_cutmix(views: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
                 cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       torch.Tensor, float]:
    """Optionally apply MixUp or CutMix to a multi-view batch.

    Parameters
    ----------
    views: (B, V, C, H, W)
    mask:  (B, V)
    labels:(B,)

    Returns ``(views, mask, y_a, y_b, lam)``. When no mixing is applied,
    ``y_a == y_b`` and ``lam == 1.0`` so the loss reduces to the standard case.
    """
    if np.random.rand() > cfg.train.mix_prob:
        return views, mask, labels, labels, 1.0

    b = views.size(0)
    perm = torch.randperm(b, device=views.device)
    use_cutmix = np.random.rand() < 0.5

    if use_cutmix and cfg.train.cutmix_alpha > 0:
        lam = float(np.random.beta(cfg.train.cutmix_alpha, cfg.train.cutmix_alpha))
        h, w = views.shape[-2:]
        y0, y1, x0, x1 = _rand_bbox((h, w), lam)
        views = views.clone()
        views[..., y0:y1, x0:x1] = views[perm][..., y0:y1, x0:x1]
        lam = 1.0 - ((y1 - y0) * (x1 - x0) / (h * w))
    else:
        alpha = max(cfg.train.mixup_alpha, 1e-3)
        lam = float(np.random.beta(alpha, alpha))
        views = lam * views + (1.0 - lam) * views[perm]

    mask = torch.maximum(mask, mask[perm])
    return views, mask, labels, labels[perm], lam


def mix_criterion(loss_fn, logits: torch.Tensor, y_a: torch.Tensor,
                  y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Interpolated loss for a (possibly) mixed batch."""
    return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
