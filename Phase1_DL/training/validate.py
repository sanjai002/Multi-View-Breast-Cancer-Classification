"""Validation / inference loop shared by training and the final test export.

``evaluate`` runs a full pass over a dataloader under ``torch.no_grad`` and mixed
precision, returning per-patient probabilities, labels, ids and (optionally) the
fused patient embedding and per-view embeddings needed for the Phase 2 export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from utils import amp_settings, autocast_context


@dataclass
class EvalResult:
    """Container for the outputs of a single evaluation pass."""

    loss: float = 0.0
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_prob: np.ndarray = field(default_factory=lambda: np.empty(0))
    patient_ids: List[str] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)
    patient_embeddings: Optional[np.ndarray] = None
    view_embeddings: Optional[np.ndarray] = None
    view_masks: Optional[np.ndarray] = None

    @property
    def y_pred(self) -> np.ndarray:
        return self.y_prob.argmax(axis=1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             cfg: Config, loss_fn: Optional[nn.Module] = None,
             collect_features: bool = False) -> EvalResult:
    """Evaluate ``model`` over ``loader``.

    Parameters
    ----------
    collect_features:
        When ``True`` the patient/view embeddings and masks are gathered for the
        feature export. Left ``False`` during in-loop validation to save memory.
    """
    model.eval()
    use_amp, amp_dtype, _ = amp_settings(cfg, device)

    total_loss, n_batches = 0.0, 0
    probs_list, labels_list = [], []
    pids: List[str] = []
    ages: List[float] = []
    patient_emb_list, view_emb_list, mask_list = [], [], []

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device, use_amp, amp_dtype):
            out = model(views, mask)
            logits = out["logits"].float()
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, labels).item())
                n_batches += 1

        probs = F.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(labels.cpu().numpy())
        pids.extend(list(batch["patient_id"]))
        ages.extend([float(a) for a in batch["age"]])

        if collect_features:
            patient_emb_list.append(out["patient_embedding"].float().cpu().numpy())
            view_emb_list.append(out["view_embeddings"].float().cpu().numpy())
            mask_list.append(mask.cpu().numpy())

    result = EvalResult(
        loss=total_loss / max(n_batches, 1),
        y_true=np.concatenate(labels_list) if labels_list else np.empty(0),
        y_prob=np.concatenate(probs_list) if probs_list else np.empty(0),
        patient_ids=pids,
        ages=ages,
    )
    if collect_features:
        result.patient_embeddings = np.concatenate(patient_emb_list, axis=0)
        result.view_embeddings = np.concatenate(view_emb_list, axis=0)
        result.view_masks = np.concatenate(mask_list, axis=0)
    return result
