"""Training callbacks and optimisation helpers.

Grouped here for convenience:

* :class:`SAM`           - Sharpness-Aware Minimisation wrapper (Foret et al.).
* :class:`EMA`           - Exponential moving average of model weights.
* :class:`EarlyStopping` - Stop when the monitored metric stops improving.
* :class:`ModelCheckpoint` - Track the best metric for checkpoint saving.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterable

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Sharpness-Aware Minimisation
# --------------------------------------------------------------------------- #
class SAM(torch.optim.Optimizer):
    """SAM wraps a base optimiser and performs the ascent/descent two-step.

    Usage::

        optimizer = SAM(param_groups, torch.optim.AdamW, rho=0.05, lr=1e-3)
        # first forward/backward on the perturbed weights, then:
        optimizer.first_step(zero_grad=True)
        # second forward/backward, then:
        optimizer.second_step(zero_grad=True)
    """

    def __init__(self, params: Iterable, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs) -> None:
        if rho < 0:
            raise ValueError(f"Invalid rho={rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                       # climb to the local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])      # back to the original weights
        self.base_optimizer.step()                # actual descent step
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):                 # noqa: D401 - Optimizer API
        raise RuntimeError("SAM requires explicit first_step()/second_step() calls.")

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                w = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((w * p.grad).norm(p=2).to(device))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# --------------------------------------------------------------------------- #
# Exponential Moving Average
# --------------------------------------------------------------------------- #
class EMA:
    """Maintain an exponential moving average of floating-point parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily swap in the EMA weights (restored on exit)."""
        self._backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        msd = model.state_dict()
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])
        try:
            yield
        finally:
            msd = model.state_dict()
            for k, v in self._backup.items():
                msd[k].copy_(v)
            self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stop training when the monitored metric plateaus."""

    def __init__(self, patience: int = 10, mode: str = "max",
                 min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Return ``True`` if this value is a new best."""
        if self._is_improvement(value):
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# --------------------------------------------------------------------------- #
# Best-metric checkpoint tracker
# --------------------------------------------------------------------------- #
class ModelCheckpoint:
    """Track the best monitored metric so the trainer knows when to save."""

    def __init__(self, mode: str = "max") -> None:
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")

    def is_improvement(self, value: float) -> bool:
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best = value
        return better
