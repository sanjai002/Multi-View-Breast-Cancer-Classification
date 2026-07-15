"""Breast region segmentation."""

import numpy as np
import cv2
from typing import Tuple


def segment_breast(
    image: np.ndarray,
    method: str = "otsu",
    min_area_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment breast region from mammography image.

    Args:
        image: Input image (H, W) in [0, 1].
        method: Thresholding method: 'otsu' or 'triangle'.
        min_area_ratio: Drop segment if area < this fraction of image.

    Returns:
        Tuple of (segmented_image, mask) where:
        - segmented_image: Original image with non-breast regions zeroed.
        - mask: Binary mask of breast region.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D")

    # Convert to uint8 for OpenCV
    img_uint8 = (image * 255).astype(np.uint8)

    # Apply thresholding
    if method == "otsu":
        _, mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "triangle":
        _, mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find contours and filter by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create new mask with only valid contours
    h, w = mask.shape
    new_mask = np.zeros_like(mask)
    min_area = min_area_ratio * (h * w)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(new_mask, [contour], 0, 255, -1)

    # Apply mask to image
    segmented = image.copy()
    segmented[new_mask == 0] = 0

    return segmented, (new_mask > 0).astype(np.float32)
