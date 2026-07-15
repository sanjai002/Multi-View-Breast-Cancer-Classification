"""Orientation normalization for mammography."""

import numpy as np
import cv2


def normalize_orientation(image: np.ndarray, laterality: str = "L") -> np.ndarray:
    """Normalize breast orientation (flip left).

    In standard mammography, images are oriented so that:
    - Left breast: patient's right side on left of image
    - Right breast: patient's left side on left of image

    This function flips right-side images horizontally so that all images
    have a consistent left-to-right orientation.

    Args:
        image: Input image (H, W).
        laterality: 'L' (left) or 'R' (right).

    Returns:
        Oriented image.
    """
    if laterality.upper() == "R":
        # Flip right breast horizontally
        return cv2.flip(image, 1)  # 1 = horizontal flip
    return image
