"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM seeks flat minima (better generalization on small medical datasets). It
needs two forward/backward passes per step; the trainer handles the closure.
Reference: Foret et al., "Sharpness-Aware Minimization", ICLR 2021.
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        assert rho >= 0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared = self.param_groups[0]["params"][0].device
        norms = [p.grad.norm(p=2).to(shared)
                 for group in self.param_groups for p in group["params"]
                 if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0)
