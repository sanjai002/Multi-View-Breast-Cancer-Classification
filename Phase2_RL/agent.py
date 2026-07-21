"""
Step 2 — networks and offline algorithms (METHODOLOGY.md §9, §10).

One network, feature-flagged, so the ablation ladder in §9.2 is a config sweep
rather than five codebases:

    BC  ->  DQN  ->  DoubleDQN  ->  Dueling  ->  +QR distributional  ->  +CQL

Discounting is plain-MDP by default: one gamma per decision step. Note the
consequence (METHODOLOGY.md §5.1) — a 36-month wait is then discounted exactly
as hard as a 12-month one, so the return carries no time cost for waiting
longer. Set CFG.smdp = True for the semi-Markov variant (gamma**tau, tau in
years) as an ablation.

CQL is implemented here rather than pulled from d3rlpy: it is ~30 lines, and it
avoids a dependency on a machine that is CPU-only with 28 GB of free disk.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, N_ACTIONS


class QNetwork(nn.Module):
    """MLP torso + optional dueling + optional quantile (QR) output head.

    A GRU belief encoder (METHODOLOGY.md §10.2) is deliberately NOT the default:
    max trajectory length is 6 and the state already carries explicit history
    summaries (prior recalls, cumulative screens, density deltas, previous
    action), so a recurrent net adds parameters that 16k transitions cannot
    identify. Kept as an option for the ablation table.
    """

    def __init__(self, d_state: int, hidden=None, n_quantiles=None,
                 dueling=None, recurrent: bool = False):
        super().__init__()
        hidden = hidden or CFG.hidden
        self.nq = n_quantiles or CFG.n_quantiles
        self.dueling = CFG.dueling if dueling is None else dueling
        self.recurrent = recurrent

        self.torso = nn.Sequential(
            nn.Linear(d_state, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        if recurrent:
            self.gru = nn.GRU(hidden, CFG.gru_hidden, batch_first=True)
            hidden = CFG.gru_hidden

        self.adv = nn.Linear(hidden, N_ACTIONS * self.nq)
        self.val = nn.Linear(hidden, self.nq) if self.dueling else None

        # Quantile midpoints tau_hat_i = (2i+1)/2N  (Dabney et al., QR-DQN).
        self.register_buffer(
            "tau_hat", (torch.arange(self.nq, dtype=torch.float32) + 0.5) / self.nq)

    def quantiles(self, s: torch.Tensor) -> torch.Tensor:
        """-> (B, n_actions, n_quantiles)"""
        h = self.torso(s)
        if self.recurrent:
            h, _ = self.gru(h.unsqueeze(1))
            h = h.squeeze(1)
        a = self.adv(h).view(-1, N_ACTIONS, self.nq)
        if self.dueling:
            v = self.val(h).view(-1, 1, self.nq)
            a = v + a - a.mean(dim=1, keepdim=True)
        return a

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Scalar Q(s,a) -> (B, n_actions)"""
        return self.quantiles(s).mean(-1)


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                        tau_hat: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """pred (B,Nq), target (B,Nq) -> per-sample loss (B,)"""
    # u[i,j] = target_j - pred_i
    u = target.unsqueeze(1) - pred.unsqueeze(2)              # (B, Nq_pred, Nq_tgt)
    huber = torch.where(u.abs() <= kappa,
                        0.5 * u.pow(2),
                        kappa * (u.abs() - 0.5 * kappa))
    w = (tau_hat.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    return (w * huber / kappa).sum(dim=1).mean(dim=1)


class OfflineAgent:
    """Discrete offline agent: BC / DQN / DoubleDQN / Dueling / QR / CQL."""

    def __init__(self, d_state: int, algo: str = "cql", cfg=CFG,
                 device="cpu", **net_kw):
        self.cfg, self.algo, self.device = cfg, algo.lower(), device
        self.distributional = self.algo in ("qr", "cql")
        self.cql = self.algo == "cql"
        self.double = cfg.double and self.algo in ("ddqn", "dueling", "qr", "cql")
        nq = cfg.n_quantiles if self.distributional else 1
        dueling = net_kw.pop("dueling", None)
        if self.algo in ("dqn", "ddqn"):
            dueling = False

        self.q = QNetwork(d_state, n_quantiles=nq, dueling=dueling, **net_kw).to(device)
        self.q_t = QNetwork(d_state, n_quantiles=nq, dueling=dueling, **net_kw).to(device)
        self.q_t.load_state_dict(self.q.state_dict())
        for p in self.q_t.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.q.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)

    # ---------------------------------------------------------------- update
    def update(self, batch: dict) -> dict:
        s, a, r = batch["s"], batch["a"], batch["r"]
        s2, done, tau, w = batch["s2"], batch["done"], batch["tau"], batch["w"]

        if self.algo == "bc":
            logits = self.q(s)
            loss = (F.cross_entropy(logits, a, reduction="none") * w).mean()
            self._step(loss)
            return {"loss": float(loss)}

        # MDP: one gamma per decision step (CFG.smdp=True switches to gamma**tau).
        disc = self.cfg.discount(tau)

        with torch.no_grad():
            if self.double:
                a2 = self.q(s2).argmax(1)              # action from ONLINE net
            else:
                a2 = self.q_t(s2).argmax(1)
            if self.distributional:
                qt = self.q_t.quantiles(s2)                      # (B,A,Nq)
                nxt = qt[torch.arange(len(a2)), a2]              # (B,Nq)
                target = r.unsqueeze(1) + disc.unsqueeze(1) * (1 - done).unsqueeze(1) * nxt
            else:
                nxt = self.q_t(s2)[torch.arange(len(a2)), a2]
                target = r + disc * (1 - done) * nxt

        if self.distributional:
            pred = self.q.quantiles(s)[torch.arange(len(a)), a]  # (B,Nq)
            td = quantile_huber_loss(pred, target, self.q.tau_hat)
        else:
            pred = self.q(s)[torch.arange(len(a)), a]
            td = F.smooth_l1_loss(pred, target, reduction="none")

        loss = (td * w).mean()
        logs = {"td": float(loss)}

        if self.cql:
            # Conservatism: push DOWN values of actions the clinicians never
            # took in this state, push UP the observed one. This is what stops
            # the net inventing a large Q for out-of-support intervals (§0.1).
            qall = self.q(s)
            gap = torch.logsumexp(qall, dim=1) - qall[torch.arange(len(a)), a]
            cql_loss = (gap * w).mean()
            loss = loss + self.cfg.cql_alpha * cql_loss
            logs["cql"] = float(cql_loss)

        self._step(loss)
        logs["loss"] = float(loss)
        return logs

    def _step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        with torch.no_grad():                       # Polyak target update
            for p, pt in zip(self.q.parameters(), self.q_t.parameters()):
                pt.mul_(1 - self.cfg.target_tau).add_(self.cfg.target_tau * p)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def q_values(self, S: np.ndarray, bs: int = 4096) -> np.ndarray:
        self.q.eval()
        out = [self.q(torch.as_tensor(S[i:i + bs], device=self.device)).cpu().numpy()
               for i in range(0, len(S), bs)]
        self.q.train()
        return np.concatenate(out)

    @torch.no_grad()
    def act(self, S: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Greedy action; `mask` (n,A) bool marks DISALLOWED actions (§7)."""
        q = self.q_values(S)
        if mask is not None:
            q = np.where(mask, -np.inf, q)
        return q.argmax(1)

    def policy_probs(self, S: np.ndarray, mask=None, eps: float = 0.0) -> np.ndarray:
        """Deterministic-greedy policy as a distribution, for OPE."""
        a = self.act(S, mask)
        P = np.full((len(S), N_ACTIONS), eps / N_ACTIONS, np.float32)
        P[np.arange(len(S)), a] += 1.0 - eps
        return P

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "algo": self.algo}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(sd["q"])
        self.q_t.load_state_dict(sd["q"])
        return self
