"""
Mammogram preprocessing for NLBS DICOMs.

Pipeline (all steps toggleable via config.preprocess):
    read DICOM -> VOI-LUT -> MONOCHROME1 fix -> to 0-255 uint8
    -> breast segmentation (Otsu + largest connected component)
    -> remove black border (crop to breast bbox)
    -> optional median denoise
    -> CLAHE (contrast-limited adaptive histogram equalization)
    -> flip left breast to a canonical orientation (chest wall on the left)
    -> resize (square)
Returns a single-channel float32 image in [0, 1]; the model replicates to 3ch.

Also provides ``quality_ok`` for data-quality filtering.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pydicom

try:
    from pydicom.pixels import apply_voi_lut
except Exception:  # pragma: no cover
    from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_dicom(path: str | Path) -> tuple[np.ndarray, str]:
    """Return (uint8 grayscale image 0-255, PhotometricInterpretation)."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return (arr * 255.0).astype(np.uint8), photometric


def segment_breast(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the breast (Otsu threshold + largest connected component)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return th > 0
    # Largest non-background component = the breast.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask.astype(bool)


def crop_to_mask(gray: np.ndarray, mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to the breast bounding box (removes black border)."""
    if not mask.any():
        return gray, mask
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0 = max(int(rows[0]) - margin, 0)
    r1 = min(int(rows[-1]) + margin + 1, gray.shape[0])
    c0 = max(int(cols[0]) - margin, 0)
    c1 = min(int(cols[-1]) + margin + 1, gray.shape[1])
    return gray[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def orient_left(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flip so the chest wall (dense side) is on the LEFT — canonical orientation.

    The breast tissue mass sits against the chest wall; we put it on the left by
    comparing foreground area of the left vs right half.
    """
    w = gray.shape[1]
    left_area = mask[:, : w // 2].sum()
    right_area = mask[:, w // 2:].sum()
    if right_area > left_area:
        gray = np.ascontiguousarray(np.fliplr(gray))
    return gray


def quality_ok(gray: np.ndarray, mask: np.ndarray,
               min_frac: float = 0.03, max_frac: float = 0.98) -> bool:
    """Reject blank/corrupt frames: breast must occupy a sensible image fraction."""
    if gray is None or gray.size == 0:
        return False
    frac = float(mask.mean())
    if not (min_frac < frac < max_frac):
        return False
    return float(gray.std()) > 3.0            # not a flat/constant image


def preprocess_image(path: str | Path, pcfg, return_mask: bool = False):
    """Full preprocessing -> float32 [0,1] image of size (img_size, img_size)."""
    gray, _ = read_dicom(path)

    if pcfg.get("breast_segmentation", True):
        mask = segment_breast(gray)
    else:
        mask = np.ones_like(gray, dtype=bool)

    if pcfg.get("remove_border", True):
        gray, mask = crop_to_mask(gray, mask)

    if pcfg.get("denoise", False):
        gray = cv2.medianBlur(gray, 3)

    if pcfg.get("clahe", True):
        gray = apply_clahe(gray, pcfg.get("clahe_clip", 2.0), pcfg.get("clahe_grid", 8))

    # Zero-out non-breast background AFTER CLAHE to suppress labels/artifacts.
    if pcfg.get("breast_segmentation", True) and mask.shape == gray.shape:
        gray = np.where(mask, gray, 0).astype(np.uint8)

    if pcfg.get("flip_left_to_right", True):
        gray = orient_left(gray, mask if mask.shape == gray.shape
                           else np.ones_like(gray, bool))

    size = int(pcfg.get("img_size", 512))
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    img = gray.astype(np.float32) / 255.0
    if return_mask:
        return img, mask
    return img
