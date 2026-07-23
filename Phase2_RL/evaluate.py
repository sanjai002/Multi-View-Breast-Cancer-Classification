"""
Step 4 — off-policy evaluation, clinical metrics, baselines (METHODOLOGY.md §12-14).

    python evaluate.py                 # scores CQL against all baselines
    python evaluate.py --split val

Two kinds of claim live here and they must not be blurred:

  * The RECALL component is directly evaluable. "Was there cancer at this exam"
    is observed, so sensitivity / specificity / PPV / AUC are ordinary
    supervised metrics against ground truth.

  * The INTERVAL component is NOT directly evaluable. Nobody was screened on a
    counterfactual schedule, so its value comes only from OPE, with all the
    caveats of §0.1. ESS is reported alongside every IS-based number; if ESS is
    small, the honest read is "FQE only".

Everything is prevalence-reweighted (§8.5), else the numbers describe a cohort
where 1 in 10 women has cancer.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from config import CFG, N_ACTIONS, ACTION_NAMES, INTERVALS_YEARS, decode_action, encode_action
from agent import OfflineAgent


# ─────────────────────────────────────────────────────────────────── FQE ──
class _NegSoftplus(nn.Module):
    """Constrains FQE output to Q <= 0, matching the non-positive cost reward."""

    def forward(self, x):
        return -nn.functional.softplus(x)


def fqe(agent, buf, X, idx, steps: int = 1500, device="cpu", seed=0, policy_fn=None):
    """Fitted Q Evaluation of a target policy (METHODOLOGY.md §12.1).

    The most reliable estimator in this regime: unlike importance sampling it
    does not degrade when the behaviour policy is near-deterministic.

    `policy_fn(exam_rows) -> actions` evaluates an arbitrary policy (used for
    the baselines); if omitted, the agent's greedy policy is evaluated. Passing
    a policy_fn lets every row of the results table share one estimator, which
    is the only way the baseline comparison is apples-to-apples.
    """
    torch.manual_seed(seed)
    S, S2 = X[buf["s"][idx]], X[buf["s2"][idx]]
    a = torch.as_tensor(buf["a"][idx], dtype=torch.long, device=device)
    r = torch.as_tensor(buf["r"][idx], device=device)
    d = torch.as_tensor(buf["done"][idx], device=device)
    tau = torch.as_tensor(buf["tau"][idx], device=device)
    w = torch.as_tensor(buf["w"][idx], device=device)
    Ts, Ts2 = (torch.as_tensor(z, device=device) for z in (S, S2))

    # pi(s') under evaluation: fixed actions from the target policy.
    a2_np = policy_fn(buf["s2"][idx]) if policy_fn is not None else agent.act(S2)
    a2 = torch.as_tensor(np.asarray(a2_np), dtype=torch.long, device=device)
    disc = CFG.discount(tau)

    # Every reward is a non-positive cost, so Q^pi <= 0 by construction. The
    # final -softplus enforces that. Without it FQE happily returns positive
    # values for off-support policies (e.g. annual screening), which is a pure
    # extrapolation artefact and silently corrupts checkpoint selection.
    def make_net():
        return nn.Sequential(
            nn.Linear(X.shape[1], 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_ACTIONS), _NegSoftplus()).to(device)

    net, tgt = make_net(), make_net()
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ar = torch.arange(len(a), device=device)

    for i in range(steps):
        with torch.no_grad():
            y = r + disc * (1 - d) * tgt(Ts2)[ar, a2]
        loss = ((net(Ts)[ar, a] - y).pow(2) * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 25 == 0:
            tgt.load_state_dict(net.state_dict())

    # Value at the cohort's initial states (first visit of each patient).
    first = _first_visit_mask(buf, idx)
    a0_np = (policy_fn(buf["s"][idx][first]) if policy_fn is not None
             else agent.act(S[first]))
    with torch.no_grad():
        a0 = torch.as_tensor(np.asarray(a0_np), dtype=torch.long, device=device)
        q0 = net(Ts[first])[torch.arange(len(a0), device=device), a0]
        w0 = w[torch.as_tensor(first, device=device)]
    return float((q0 * w0).sum() / w0.sum())


def _first_visit_mask(buf, idx) -> np.ndarray:
    """Exams are ordered by (patient, year), so min state-row per patient = visit 1."""
    pid, s = buf["pid"][idx], buf["s"][idx]
    order = np.lexsort((s, pid))
    keep = np.zeros(len(idx), bool)
    seen = set()
    for i in order:
        if pid[i] not in seen:
            seen.add(pid[i])
            keep[i] = True
    return keep


# ────────────────────────────────────────────── importance sampling / DR ──
def is_estimators(policy_probs, buf, idx, eps=0.05):
    """WIS, per-decision WIS and ESS (METHODOLOGY.md §12.1).

    The evaluation policy is smoothed to eps-greedy purely so trajectory weights
    are not identically zero the moment it disagrees with the clinician once.
    Report eps; it is an estimator artefact, not part of the policy.
    """
    behaviour = np.clip(buf["pi_b"][idx], 1e-6, 1.0)
    target = policy_probs * (1 - eps) + eps / N_ACTIONS

    actions = buf["a"][idx]
    rewards = buf["r"][idx]
    taus = buf["tau"][idx]
    patients = buf["pid"][idx]
    prevalence_w = buf["w"][idx]

    rows = np.arange(len(actions))
    step_ratio = target[rows, actions] / behaviour[rows, actions]

    traj_weight, traj_return, traj_prevalence = [], [], []
    perdec_num = perdec_den = 0.0

    for patient in np.unique(patients):
        steps = np.where(patients == patient)[0]
        steps = steps[np.argsort(buf["s"][idx][steps])]      # chronological

        cumulative_ratio = np.cumprod(step_ratio[steps])
        discount = CFG.gamma ** CFG.elapsed(taus[steps])
        w = prevalence_w[steps]

        traj_weight.append(cumulative_ratio[-1])
        traj_return.append(float((discount * rewards[steps]).sum()))
        traj_prevalence.append(float(w[0]))

        perdec_num += float((cumulative_ratio * discount * rewards[steps] * w).sum())
        perdec_den += float((cumulative_ratio * w).sum())

    traj_weight = np.asarray(traj_weight) * np.asarray(traj_prevalence)
    traj_return = np.asarray(traj_return)

    return {
        "WIS": float((traj_weight * traj_return).sum() / max(traj_weight.sum(), 1e-12)),
        "PDWIS": perdec_num / max(perdec_den, 1e-12),
        # Effective sample size: how many trajectories the weights really use.
        "ESS": float(traj_weight.sum() ** 2 / max((traj_weight ** 2).sum(), 1e-12)),
        "n_traj": len(traj_return),
    }


# ───────────────────────────────────────────────────── clinical metrics ──
def recall_metrics(pred_recall, buf, idx, ex, score=None):
    """Directly evaluable: was there cancer at THIS exam (rad_timing == 1)?"""
    from sklearn.metrics import roc_auc_score

    exam_timing = ex.timing.to_numpy()[buf["s"][idx]]
    cancer_here = (exam_timing == 1.0).astype(int)   # screen-detectable cancer present
    weight = buf["w"][idx]                           # prevalence reweighting
    recalled = pred_recall.astype(int)

    # Prevalence-weighted confusion matrix.
    def total(predicted, actual):
        return float((weight * (recalled == predicted) * (cancer_here == actual)).sum())

    true_pos = total(1, 1)
    false_pos = total(1, 0)
    false_neg = total(0, 1)
    true_neg = total(0, 0)
    n_exams = true_pos + false_pos + false_neg + true_neg

    safe = lambda num, den: num / max(den, 1e-12)
    sensitivity = safe(true_pos, true_pos + false_neg)
    specificity = safe(true_neg, true_neg + false_pos)
    precision = safe(true_pos, true_pos + false_pos)

    metrics = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_PPV": precision,
        "F1": safe(2 * precision * sensitivity, precision + sensitivity),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "recall_rate": safe(true_pos + false_pos, n_exams),
        "cancer_detection_per_1000": 1000 * safe(true_pos, n_exams),
        "unnecessary_recalls_per_1000": 1000 * safe(false_pos, n_exams),
        "n_cancer_exams": int((cancer_here == 1).sum()),
    }
    if score is not None and len(np.unique(cancer_here)) > 1:
        metrics["ROC_AUC"] = float(
            roc_auc_score(cancer_here, score, sample_weight=weight))
    return metrics


def interval_metrics(actions):
    iv = np.array([INTERVALS_YEARS[decode_action(int(a))[0]] for a in actions])
    return {
        "mean_interval_months": float(iv.mean() * 12),
        "screens_per_woman_per_decade": float((10.0 / iv).mean()),
        "pct_12mo": float((iv == 1).mean()),
        "pct_24mo": float((iv == 2).mean()),
        "pct_36mo": float((iv == 3).mean()),
    }


# ──────────────────────────────────────────────────────────── baselines ──
def baseline_policy(name, ex):
    """METHODOLOGY.md §13 -> a policy_fn(exam_rows) -> actions.

    Interval baselines keep the OBSERVED recall decision. That is deliberate:
    recall is made by a radiologist reading the mammogram, so holding it at
    standard of care isolates the interval choice, which is the actual
    contribution. Comparing an interval rule that never recalls against
    clinicians who do would be a rigged comparison.
    """
    age_all = ex.age_bin.to_numpy()
    dens_all = ex[["percentdensity_left", "percentdensity_right"]].mean(1).to_numpy()
    rec_all = ex.recall.fillna(0.0).to_numpy().astype(int)
    thr = np.quantile(dens_all, 0.75)

    def fn(rows):
        rows = np.asarray(rows)
        rec = rec_all[rows]
        if name == "annual":
            iv = np.zeros(len(rows), int)
        elif name == "biennial":
            iv = np.ones(len(rows), int)
        elif name == "triennial":
            iv = np.full(len(rows), 2)
        elif name == "standard_of_care":
            # Swedish programme: 18mo for 40-54 (-> 12mo arm), 24mo for 55+.
            iv = np.where(age_all[rows] == 1, 0, 1)
        elif name == "density_rule":
            # Hardest non-RL baseline: shorten the interval for dense breasts.
            iv = np.where(dens_all[rows] > thr, 0, 1)
        elif name == "random":
            rng = np.random.default_rng(0)
            return rng.integers(0, N_ACTIONS, len(rows))
        else:
            raise ValueError(name)
        return np.array([encode_action(i, r) for i, r in zip(iv, rec)])

    return fn


# ───────────────────────────────────────────────────────────── evaluate ──
def evaluate_policy(name, actions, buf, idx, ex, X, score=None, agent=None,
                    policy_fn=None):
    rec = np.array([decode_action(int(a))[1] for a in actions])
    out = {"policy": name}
    out.update(interval_metrics(actions))
    out.update(recall_metrics(rec, buf, idx, ex, score=score))

    P = np.zeros((len(actions), N_ACTIONS), np.float32)
    P[np.arange(len(actions)), actions] = 1.0
    out.update(is_estimators(P, buf, idx))
    out.update(support_diagnostics(actions, buf, idx))
    # FQE is a fitted estimator and is noticeably seed-sensitive on 2.4k
    # transitions; a single fit is not a number worth ranking on.
    vals = [fqe(agent, buf, X, idx, policy_fn=policy_fn, seed=s)
            for s in range(CFG.n_fqe_seeds)]
    out["FQE"] = float(np.mean(vals))
    out["FQE_sd"] = float(np.std(vals))
    return out


def support_diagnostics(actions, buf, idx):
    """How much of this policy's behaviour the data can actually speak to.

    FQE is only meaningful where the behaviour policy put mass. With a
    near-deterministic clinician policy (METHODOLOGY.md §0.1) a rule like
    "annual screening for everyone" asks the estimator about state-action pairs
    that barely occur, and FQE answers by extrapolating. Reporting support next
    to FQE is what stops the results table being read as a ranking.
    """
    pb = buf["pi_b"][idx][np.arange(len(actions)), actions]
    return {
        "support_mean_pi_b": float(pb.mean()),
        "support_frac_ok": float((pb >= CFG.propensity_floor).mean()),
    }


def main():
    import pandas as pd
    from data import build_exams
    from train import load_buffer, split_idx

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    buf, ex = load_buffer(), build_exams()
    X = buf["X"]
    idx = split_idx(buf, args.split)
    print(f"[eval] {args.split}: {len(idx):,} transitions, "
          f"{len(np.unique(buf['pid'][idx])):,} patients\n")

    # Observed action per exam row, so the clinicians' own policy can be scored
    # with the same estimator as everything else. Exam rows with no logged
    # transition (censored terminals) fall back to the modal action.
    obs_a = np.full(len(ex), encode_action(1, 0), dtype=np.int64)
    obs_a[buf["s"]] = buf["a"]

    rows = []
    rows.append(evaluate_policy("observed", buf["a"][idx], buf, idx, ex, X,
                                policy_fn=lambda r: obs_a[np.asarray(r)]))
    for b in ("standard_of_care", "annual", "biennial", "triennial",
              "density_rule", "random"):
        fn = baseline_policy(b, ex)
        rows.append(evaluate_policy(b, fn(buf["s"][idx]), buf, idx, ex, X,
                                    policy_fn=fn))

    ckpt = CFG.out / "cql_seed0.pt"
    if ckpt.exists():
        agent = OfflineAgent(X.shape[1]).load(ckpt)
        q = agent.q_values(X[buf["s"][idx]])
        a = q.argmax(1)
        # Score for AUC: preference for recalling over not, marginalised.
        qr = q.reshape(len(q), len(INTERVALS_YEARS), 2)
        score = qr[:, :, 1].max(1) - qr[:, :, 0].max(1)
        rows.append(evaluate_policy("RL-cql", a, buf, idx, ex, X,
                                    score=score, agent=agent))
    else:
        print(f"!! no checkpoint {ckpt.name}; run train.py first. Baselines only.\n")
        return

    df = pd.DataFrame(rows)
    df["FQE_trustworthy"] = df.support_frac_ok >= 0.5
    cols = ["policy", "FQE", "FQE_sd", "FQE_trustworthy", "support_frac_ok", "ESS",
            "mean_interval_months", "screens_per_woman_per_decade",
            "sensitivity", "specificity", "precision_PPV", "recall_rate",
            "cancer_detection_per_1000", "unnecessary_recalls_per_1000"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.4f}")
    print(df[cols].to_string(index=False))

    df.to_csv(CFG.out / f"eval_{args.split}.csv", index=False)
    print(f"\nsaved -> {CFG.out / f'eval_{args.split}.csv'}")

    print("\n" + "-" * 70)
    ess = df.loc[df.policy.str.startswith("RL"), "ESS"]
    if len(ess) and ess.iloc[0] < 100:
        print(f"ESS = {ess.iloc[0]:.1f} << 100: WIS/PDWIS carry almost no "
              "information here.\n  This is the positivity violation of "
              "METHODOLOGY.md §0.1, measured. Report FQE as primary.")
    bad = df.loc[~df.FQE_trustworthy, "policy"].tolist()
    if bad:
        print(f"\nFQE is EXTRAPOLATING for: {', '.join(bad)}")
        print("  These policies choose actions the clinicians rarely took, so their")
        print("  FQE values are model artefacts, not estimates. Do not rank on them.")
    print("\nThe recall column is only learnable from images (none on this machine);")
    print("interval baselines therefore hold recall at the observed standard of care.")


if __name__ == "__main__":
    main()
