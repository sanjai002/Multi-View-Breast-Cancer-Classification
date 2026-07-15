"""Intensity normalization."""

import numpy as np


def normalize_intensity(
    image: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalize image intensity.

    Args:
        image: Input image (H, W) in [0, 1].
        method: Normalization method:
            - 'zscore': (x - mean) / std
            - 'minmax': (x - min) / (max - min)
            - 'none': no normalization

    Returns:
        Normalized image.
    """
    if method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        return image

    elif method == "minmax":
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image

    elif method == "none":
        return image

    else:
        raise ValueError(f"Unknown method: {method}")
