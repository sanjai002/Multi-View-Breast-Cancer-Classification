"""
Medically-appropriate augmentation for mammograms (Albumentations).

WHY these (see README for full rationale):
  USE   : small rotation (+/-12 deg), mild affine (scale/translate), random crop,
          contrast/brightness, gamma, mild Gaussian noise, elastic (mild),
          random erasing (CoarseDropout). MixUp/CutMix are applied in the
          training loop (batch level), not here.
  AVOID : horizontal/vertical flips AFTER laterality standardization (they move
          the chest wall / invert anatomy), heavy color jitter (images are
          grayscale), large elastic (distorts lesion morphology).

Both CC and MLO views of the same breast receive the SAME random transform
parameters (via a shared additional_target) so the pair stays consistent.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _norm_totensor():
    # Images are single-channel [0,1] replicated to 3ch upstream; normalize with
    # ImageNet stats so pretrained backbones see a familiar distribution.
    return [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        max_pixel_value=1.0), ToTensorV2()]


def train_transforms(cfg_aug, img_size: int) -> A.Compose:
    tfl = [
        A.Rotate(limit=cfg_aug.rotation_deg, border_mode=0, p=0.7),
        A.Affine(scale=(1 - cfg_aug.affine_scale, 1 + cfg_aug.affine_scale),
                 translate_percent=(0.0, cfg_aug.affine_translate),
                 rotate=0, mode=0, p=0.5),
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=tuple(cfg_aug.random_crop_scale), ratio=(0.9, 1.1),
                            p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=cfg_aug.contrast, p=0.5),
        A.RandomGamma(gamma_limit=tuple(cfg_aug.gamma), p=0.3),
        A.GaussNoise(var_limit=tuple(cfg_aug.gaussian_noise_var), p=0.2),
    ]
    if cfg_aug.get("elastic", True):
        tfl.append(A.ElasticTransform(alpha=20, sigma=6, alpha_affine=0,
                                      border_mode=0, p=0.15))
    if cfg_aug.horizontal_flip:                    # off by default (see README)
        tfl.append(A.HorizontalFlip(p=0.5))
    if cfg_aug.random_erasing_p > 0:
        tfl.append(A.CoarseDropout(max_holes=6, max_height=img_size // 8,
                                   max_width=img_size // 8, fill_value=0,
                                   p=cfg_aug.random_erasing_p))
    tfl += _norm_totensor()
    # Apply the SAME transform to CC ('image') and MLO ('mlo').
    return A.Compose(tfl, additional_targets={"mlo": "image"})


def eval_transforms(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(img_size, img_size), *_norm_totensor()],
                     additional_targets={"mlo": "image"})
