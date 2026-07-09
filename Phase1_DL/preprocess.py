"""Spatial preprocessing pipeline for mammography images.

Consumes the normalised float image produced by :mod:`dicom` and applies, in
order:

1. CLAHE contrast enhancement.
2. Breast segmentation (Otsu/triangle threshold + largest connected component).
3. Artifact removal (keep only the breast component, drop tags/labels/scanner
   markings that survive as separate components).
4. Black-border removal (crop to the breast bounding box).
5. Resize to a square canvas.
6. Left/right orientation normalisation (flip so the chest wall is consistent).
7. Intensity normalisation (mean/std standardisation).

Everything operates on single 2-D ``float32`` arrays so it is cheap enough to
run inside a DataLoader worker on demand.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import Config


class MammoPreprocessor:
    """Deterministic preprocessing shared by every split.

    The object is stateless with respect to individual images (only holds the
    configuration) so it is safe to share across DataLoader workers.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.image_size = cfg.data.image_size
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.data.clahe_clip_limit,
            tileGridSize=(cfg.data.clahe_grid_size, cfg.data.clahe_grid_size),
        )

    # ------------------------------------------------------------------ #
    def __call__(self, image: np.ndarray, laterality: str) -> np.ndarray:
        """Run the full pipeline. ``image`` is float32 in [0, 1]."""
        img = self._ensure_2d_float(image)
        img8 = self._to_uint8(img)
        img8 = self.apply_clahe(img8)

        mask = self.segment_breast(img8)
        img8, mask = self.remove_artifacts(img8, mask)
        img8, mask = self.remove_black_borders(img8, mask)

        img = img8.astype(np.float32) / 255.0
        img = self.resize(img)
        img = self.orient(img, laterality)
        # Return in [0, 1]; mean/std standardisation is applied by the
        # Albumentations ``Normalize`` step inside the dataset transform so the
        # image is never normalised twice.
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_2d_float(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = image[..., 0]
        return image.astype(np.float32)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        img = np.clip(image, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    def apply_clahe(self, image_u8: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalisation."""
        return self._clahe.apply(image_u8)

    # ------------------------------------------------------------------ #
    def segment_breast(self, image_u8: np.ndarray) -> np.ndarray:
        """Return a binary mask (uint8 0/255) of the breast region."""
        # Slight blur removes sensor noise before thresholding.
        blur = cv2.GaussianBlur(image_u8, (5, 5), 0)
        if self.cfg.data.breast_threshold_method == "triangle":
            flag = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        else:
            flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        _, mask = cv2.threshold(blur, 0, 255, flag)

        # Morphological closing then opening to consolidate the breast and drop
        # small speckle.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def remove_artifacts(self, image_u8: np.ndarray,
                         mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only the largest connected component (the breast).

        Labels, patient tags and scanner markings appear as separate bright
        blobs; retaining just the biggest component removes them. Falls back to
        the original image when the mask is empty.
        """
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return image_u8, mask
        # Component 0 is the background; pick the largest of the rest.
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))

        total_px = mask.shape[0] * mask.shape[1]
        if areas.max() < self.cfg.data.min_breast_area_ratio * total_px:
            # Segmentation failed (e.g. very dark image); keep the raw image.
            return image_u8, mask

        breast_mask = (labels == largest).astype(np.uint8) * 255
        cleaned = cv2.bitwise_and(image_u8, image_u8, mask=breast_mask)
        return cleaned, breast_mask

    @staticmethod
    def remove_black_borders(image_u8: np.ndarray,
                             mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crop both image and mask to the bounding box of the breast."""
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return image_u8, mask
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return image_u8[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # ------------------------------------------------------------------ #
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to a square canvas, letter-boxing to preserve aspect ratio."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((self.image_size, self.image_size), dtype=image.dtype)
        scale = self.image_size / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((self.image_size, self.image_size), dtype=image.dtype)
        y_off = (self.image_size - new_h) // 2
        x_off = (self.image_size - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def orient(self, image: np.ndarray, laterality: str) -> np.ndarray:
        """Flip so every breast points the same way.

        We estimate the side the breast occupies from the column-wise mass and
        flip so the dense (chest-wall) side matches ``cfg.data.flip_to``. Using
        image content rather than only the laterality tag is robust to
        inconsistent header conventions.
        """
        col_mass = image.sum(axis=0)
        half = image.shape[1] // 2
        left_mass = col_mass[:half].sum()
        right_mass = col_mass[half:].sum()
        chest_on_left = left_mass >= right_mass

        want_left = self.cfg.data.flip_to.lower() == "left"
        if want_left != chest_on_left:
            image = np.fliplr(image)
        return np.ascontiguousarray(image)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Standardise with the configured single-channel mean/std."""
        mean = float(self.cfg.data.normalize_mean[0])
        std = float(self.cfg.data.normalize_std[0])
        return (image - mean) / (std + 1e-8)


def blank_view(cfg: Config) -> np.ndarray:
    """Return a normalised all-zero image used for missing views."""
    mean = float(cfg.data.normalize_mean[0])
    std = float(cfg.data.normalize_std[0])
    return (np.zeros((cfg.data.image_size, cfg.data.image_size), np.float32) - mean) / (std + 1e-8)
