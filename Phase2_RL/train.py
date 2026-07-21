"""
Step 3 — offline training loop (METHODOLOGY.md §11).

    python train.py --algo cql
    python train.py --algo bc dqn ddqn dueling qr cql     # the ablation ladder

There is NO exploration and NO environment: the buffer is fixed. Checkpoint
selection therefore cannot use rollouts — it uses validation FQE (§11.6).

The gate that matters (§11.7): a policy that always picks "36mo + no work-up"
scores well on mean return over a cohort that is ~90% healthy while being
clinically useless. Cancer-specific behaviour is logged every eval and a
degenerate policy is flagged loudly rather than discovered at write-up.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from config import CFG, N_ACTIONS, ACTION_NAMES, decode_action, INTERVALS_YEARS
from agent import OfflineAgent


def load_buffer(path=None):
    d = np.load(path or (CFG.out / "buffer.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def split_idx(buf, name):
    return np.where(buf["split"] == name)[0]


def make_batcher(buf, idx, X, device, rng, batch_size):
    """Return a function that draws a random minibatch of transitions.

    Sampling is uniform over transitions, which is correct here because the
    state already carries explicit history features -- there is no recurrent
    hidden state that would require whole trajectories.
    """
    states = X[buf["s"]]
    next_states = X[buf["s2"]]

    def sample():
        picked = rng.choice(idx, size=min(batch_size, len(idx)), replace=True)

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array[picked], dtype=dtype, device=device)

        return dict(
            s=tensor(states),
            s2=tensor(next_states),
            a=tensor(buf["a"], torch.long),
            r=tensor(buf["r"]),
            done=tensor(buf["done"]),
            tau=tensor(buf["tau"]),
            w=tensor(buf["w"]),
        )

    return sample


# ─────────────────────────────────────────────────── policy diagnostics ──
def policy_report(agent, buf, X, idx) -> dict:
    """Action mix overall and on cancer trajectories — the collapse gate."""
    actions = agent.act(X[buf["s"][idx]])
    interval_years = np.array([INTERVALS_YEARS[decode_action(int(a))[0]]
                               for a in actions])
    recalls = np.array([decode_action(int(a))[1] for a in actions])
    is_case = buf["is_case"][idx] == 1

    def mean_or_zero(values, mask):
        return float(values[mask].mean()) if mask.any() else 0.0

    action_freq = np.bincount(actions, minlength=N_ACTIONS) / len(actions)
    entropy = float(-(action_freq * np.log(action_freq + 1e-12)).sum())

    report = {
        "mean_interval_months": float(interval_years.mean() * 12),
        "recall_rate": float(recalls.mean()),
        "action_entropy": entropy,
        "recall_rate_cases": mean_or_zero(recalls, is_case),
        "recall_rate_controls": mean_or_zero(recalls, ~is_case),
        "mean_interval_cases": mean_or_zero(interval_years, is_case) * 12,
    }

    # Two ways this can go wrong, both of which look fine on mean reward:
    #   collapsed        -> the agent picks one action for everybody
    #   no_discrimination-> it varies, but treats cancer and non-cancer alike
    report["collapsed"] = bool(entropy < 0.05)
    report["no_discrimination"] = bool(
        abs(report["recall_rate_cases"] - report["recall_rate_controls"]) < 0.01
        and abs(report["mean_interval_cases"] - report["mean_interval_months"]) < 0.5)
    return report


def train(algo="cql", steps=None, seed=0, buf=None, X=None, verbose=True,
          device="cpu"):
    from evaluate import fqe          # imported here to avoid a circular import

    buf = buf if buf is not None else load_buffer()
    X = X if X is not None else buf["X"]
    steps = steps or CFG.steps
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    tr, va = split_idx(buf, "train"), split_idx(buf, "val")
    agent = OfflineAgent(X.shape[1], algo=algo, device=device)
    batch = make_batcher(buf, tr, X, device, rng, CFG.batch_size)

    best = {"fqe": -np.inf, "step": -1}
    CFG.out.mkdir(exist_ok=True, parents=True)
    ckpt = CFG.out / f"{algo}_seed{seed}.pt"
    history = []

    for step in range(1, steps + 1):
        logs = agent.update(batch())

        if step % CFG.eval_every == 0 or step == steps:
            rep = policy_report(agent, buf, X, va)
            v = fqe(agent, buf, X, va, device=device)
            rep.update(fqe=v, step=step, **logs)
            history.append(rep)

            if v > best["fqe"] and not rep["collapsed"]:
                best = {"fqe": v, "step": step}
                agent.save(ckpt)

            if verbose:
                flag = ""
                if rep["collapsed"]:
                    flag = "  ** COLLAPSED (single action) **"
                elif rep["no_discrimination"]:
                    flag = "  ** no case/control discrimination **"
                print(f"  step {step:6d} | loss {logs['loss']:8.4f} | "
                      f"FQE {v:+.4f} | interval {rep['mean_interval_months']:5.1f}mo | "
                      f"recall {rep['recall_rate']:.3f} "
                      f"(ca {rep['recall_rate_cases']:.2f}/co {rep['recall_rate_controls']:.2f})"
                      f"{flag}")

    if best["step"] < 0:                       # never beat -inf without collapsing
        agent.save(ckpt)
        best = {"fqe": history[-1]["fqe"] if history else float("nan"),
                "step": steps, "note": "no non-collapsed checkpoint"}
    else:
        agent.load(ckpt)

    if verbose:
        print(f"  best: step {best['step']} FQE {best['fqe']:+.4f} -> {ckpt.name}")
    return agent, {"algo": algo, "seed": seed, "best": best, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", nargs="+", default=["cql"],
                    choices=["bc", "dqn", "ddqn", "dueling", "qr", "cql"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="publication runs need >=5 (METHODOLOGY.md §11.5)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cql-alpha", type=float, default=None)
    args = ap.parse_args()

    if args.cql_alpha is not None:
        CFG.cql_alpha = args.cql_alpha

    buf = load_buffer()
    X = buf["X"]
    print(f"[train] state dim {X.shape[1]}  gamma {CFG.gamma:.4f}/yr  "
          f"cql_alpha {CFG.cql_alpha}  steps {args.steps or CFG.steps}")

    out = []
    for algo in args.algo:
        for seed in range(args.seeds):
            print(f"\n=== {algo}  seed {seed} ===")
            _, res = train(algo, steps=args.steps, seed=seed, buf=buf, X=X)
            out.append(res)

    (CFG.out / "train_history.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {CFG.out / 'train_history.json'}")


if __name__ == "__main__":
    main()
