"""
Patient-level splitting + class balancing.

Patient-level (never image-level): every breast/image of a patient stays in one
split, so the model can't cheat by recognizing a patient seen in training. We
use 70/15/15 — with only ~150 cancer patients, 80/10/10 leaves a test set too
small (~15 cancer patients) for stable AUC/CI, so 70/15/15 is preferred here.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd


def balance(idx: pd.DataFrame, neg_per_pos: int, seed: int) -> pd.DataFrame:
    """Downsample normal breasts to at most ``neg_per_pos`` per cancer breast."""
    rng = random.Random(seed)
    pos = idx[idx.label == 1]
    neg = idx[idx.label == 0]
    keep = min(len(neg), len(pos) * neg_per_pos)
    neg_ids = list(neg.index)
    rng.shuffle(neg_ids)
    neg = neg.loc[neg_ids[:keep]]
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def patient_level_split(idx: pd.DataFrame, fracs, seed: int) -> pd.DataFrame:
    """Assign each row a 'split' via a stratified patient-level partition."""
    rng = random.Random(seed)
    # Patient class = 1 if the patient has any cancer breast (stratify on this).
    pcls = idx.groupby("patient_id")["label"].max()
    splits = {"train": [], "val": [], "test": []}
    for cls in (0, 1):
        pids = list(pcls[pcls == cls].index)
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        splits["train"] += pids[:n_tr]
        splits["val"] += pids[n_tr:n_tr + n_va]
        splits["test"] += pids[n_tr + n_va:]
    pid_to_split = {pid: s for s, pids in splits.items() for pid in pids}
    idx = idx.copy()
    idx["split"] = idx["patient_id"].map(pid_to_split)
    return idx


def make_splits(idx: pd.DataFrame, cfg) -> pd.DataFrame:
    if cfg.data.balance == "downsample":
        idx = balance(idx, int(cfg.data.neg_per_pos), cfg.seed)
    idx = patient_level_split(idx, list(cfg.data.split), cfg.seed)
    # Safety: assert no patient leakage across splits.
    per = idx.groupby("patient_id")["split"].nunique()
    assert (per == 1).all(), "patient leakage detected across splits!"
    return idx
