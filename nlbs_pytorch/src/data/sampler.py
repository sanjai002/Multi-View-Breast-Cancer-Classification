"""Class-imbalance samplers.

Recommended: WeightedRandomSampler (balanced mini-batches without discarding
data) combined with Focal Loss. SMOTE is NOT appropriate for full images
(interpolating raw mammograms creates unrealistic tissue), so it is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["label"].to_numpy()
    class_count = np.bincount(labels, minlength=2).astype(np.float64)
    class_w = 1.0 / np.maximum(class_count, 1)
    sample_w = class_w[labels]
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=len(sample_w), replacement=True)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    labels = df["label"].to_numpy()
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    w = counts.sum() / (2.0 * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
