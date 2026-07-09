"""
Training engine.

Implements the modern medical-imaging training stack:
  * discriminative LRs (low for backbone, high for head) + ImageNet transfer
  * phase-1 frozen-backbone warmup, then unfreeze (layer-unfreezing schedule)
  * AdamW / SGD / SAM optimizers
  * OneCycle / cosine / cosine+warmup schedulers
  * Automatic Mixed Precision (GPU), gradient accumulation, gradient clipping
  * MixUp / CutMix (batch level), label smoothing (in loss)
  * EMA weights, early stopping, best-checkpointing, TensorBoard
  * Test-Time Augmentation at evaluation
  * Progressive resizing (via a train-loader factory)
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.ema import ModelEMA
from src.utils.sam import SAM
from src.utils.metrics import compute_metrics


# --------------------------- MixUp / CutMix --------------------------- #
def _rand_bbox(h, w, lam):
    cut = math.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h)
    x2, y2 = np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def mix_batch(cc, mlo, y, mixup_alpha, cutmix_alpha):
    """Apply MixUp or CutMix consistently to both views. Returns mixed, (ya,yb,lam)."""
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    alpha = cutmix_alpha if use_cut else mixup_alpha
    if alpha <= 0:
        return cc, mlo, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(cc.size(0), device=cc.device)
    yb = y[perm]
    if use_cut:
        x1, y1, x2, y2 = _rand_bbox(cc.size(2), cc.size(3), lam)
        for t in (cc, mlo):
            t[:, :, y1:y2, x1:x2] = t[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (cc.size(2) * cc.size(3)))
    else:
        cc = lam * cc + (1 - lam) * cc[perm]
        mlo = lam * mlo + (1 - lam) * mlo[perm]
    return cc, mlo, (y, yb, lam)


class Trainer:
    def __init__(self, model, loss_fn, cfg, device, out_dir):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.tc = cfg.train
        self.device = device
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "checkpoints").mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.out / "tb"))
        self.use_amp = bool(self.tc.amp) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.ema = ModelEMA(self.model, self.tc.ema_decay) if self.tc.ema else None
        self.best = -np.inf
        self.best_path = self.out / "checkpoints" / "best.pt"

    # ------------------------- optim / sched ------------------------- #
    def build_optimizer(self):
        base_lr = float(self.tc.base_lr)
        groups = [
            {"params": list(self.model.backbone_parameters()), "lr": base_lr},
            {"params": list(self.model.head_parameters()),
             "lr": base_lr * float(self.tc.head_lr_mult)},
        ]
        wd = float(self.tc.weight_decay)
        if self.tc.optimizer == "sam":
            return SAM(groups, torch.optim.AdamW, rho=0.05, weight_decay=wd)
        if self.tc.optimizer == "sgd":
            return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd, nesterov=True)
        return torch.optim.AdamW(groups, weight_decay=wd)

    def build_scheduler(self, optimizer, steps_per_epoch):
        epochs = self.tc.epochs
        opt = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
        if self.tc.scheduler == "onecycle":
            max_lrs = [g["lr"] for g in opt.param_groups]
            pct = min(0.5, max(self.tc.warmup_epochs / max(epochs, 1), 0.01))
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=max_lrs, epochs=epochs, steps_per_epoch=steps_per_epoch,
                pct_start=pct), "step"
        if self.tc.scheduler == "cosine_warmup":
            warm = self.tc.warmup_epochs

            def fn(e):
                if e < warm:
                    return (e + 1) / max(warm, 1)
                t = (e - warm) / max(epochs - warm, 1)
                return 0.5 * (1 + math.cos(math.pi * t))
            return torch.optim.lr_scheduler.LambdaLR(opt, fn), "epoch"
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), "epoch"

    # ----------------------------- steps ----------------------------- #
    def _forward_loss(self, batch):
        cc = batch["cc"].to(self.device, non_blocking=True)
        mlo = batch["mlo"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device, non_blocking=True)
        cc, mlo, (ya, yb, lam) = mix_batch(cc, mlo, y,
                                           self.cfg.augment.mixup_alpha,
                                           self.cfg.augment.cutmix_alpha)
        logits = self.model(cc, mlo)
        loss = lam * self.loss_fn(logits, ya) + (1 - lam) * self.loss_fn(logits, yb)
        return loss

    def train_epoch(self, loader, optimizer, scheduler, sched_when):
        self.model.train()
        total, n = 0.0, 0
        accum = max(int(self.tc.grad_accum_steps), 1)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            if isinstance(optimizer, SAM):                 # SAM: 2 passes, no AMP
                loss = self._forward_loss(batch)
                loss.backward(); optimizer.first_step(zero_grad=True)
                self._forward_loss(batch).backward(); optimizer.second_step(zero_grad=True)
            else:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss = self._forward_loss(batch) / accum
                self.scaler.scale(loss).backward()
                if (i + 1) % accum == 0:
                    if self.tc.grad_clip:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
                    self.scaler.step(optimizer); self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                loss = loss * accum
            if self.ema:
                self.ema.update(self.model)
            if scheduler is not None and sched_when == "step":
                scheduler.step()
            total += float(loss) * batch["label"].size(0); n += batch["label"].size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader, use_ema=True, tta=False):
        model = self.ema.ema if (use_ema and self.ema) else self.model
        model.eval()
        ys, ps, pids = [], [], []
        for batch in loader:
            cc = batch["cc"].to(self.device); mlo = batch["mlo"].to(self.device)
            views = [(cc, mlo)]
            if tta:
                views += [(torch.flip(cc, [3]), torch.flip(mlo, [3])),
                          (torch.flip(cc, [2]), torch.flip(mlo, [2]))]
            probs = []
            for c, m in views:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    probs.append(torch.softmax(model(c, m), dim=1)[:, 1])
            p = torch.stack(probs).mean(0)
            ps.append(p.float().cpu().numpy()); ys.append(batch["label"].numpy())
            pids += list(batch["patient_id"])
        return np.concatenate(ys), np.concatenate(ps), pids

    # ------------------------------ fit ------------------------------ #
    def fit(self, train_loader_fn, val_loader, size_schedule=None):
        size_schedule = dict(size_schedule or {})
        cur_size = size_schedule.get(0, self.cfg.preprocess.img_size)
        train_loader = train_loader_fn(cur_size)
        optimizer = self.build_optimizer()
        scheduler, sched_when = self.build_scheduler(optimizer, len(train_loader))

        # Phase 1: freeze backbone for a few epochs.
        self.model.set_backbone_trainable(False)
        frozen = True
        patience, bad = self.tc.early_stop_patience, 0

        for epoch in range(self.tc.epochs):
            if epoch in size_schedule and size_schedule[epoch] != cur_size:
                cur_size = size_schedule[epoch]
                train_loader = train_loader_fn(cur_size)     # progressive resizing
            if frozen and epoch >= self.tc.freeze_backbone_epochs:
                self.model.set_backbone_trainable(True); frozen = False

            tr_loss = self.train_epoch(train_loader, optimizer, scheduler, sched_when)
            if scheduler is not None and sched_when == "epoch":
                scheduler.step()

            y, p, _ = self.evaluate(val_loader, use_ema=True, tta=False)
            m = compute_metrics(y, p)
            score = m["roc_auc"] if self.tc.monitor == "val_auc" else m["f1"]
            self.writer.add_scalar("train/loss", tr_loss, epoch)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            print(f"[epoch {epoch:02d}] loss={tr_loss:.4f} "
                  f"val_auc={m['roc_auc']:.4f} val_f1={m['f1']:.4f} "
                  f"val_acc={m['accuracy']:.4f} (img={cur_size}, frozen={frozen})")

            if score > self.best:
                self.best = score; bad = 0
                self._save_checkpoint()
            else:
                bad += 1
                if bad >= patience:
                    print(f"[early-stop] no {self.tc.monitor} gain in {patience} epochs")
                    break
        self.writer.close()
        return self.best

    def _save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "ema": self.ema.ema.state_dict() if self.ema else None,
            "cfg": dict(self.cfg), "best": self.best,
        }
        torch.save(state, self.best_path)
        print(f"[checkpoint] saved best ({self.tc.monitor}={self.best:.4f}) -> {self.best_path}")
