"""
Step 5 — calibrated latent progression simulator (METHODOLOGY.md §5.5).

    python simulator.py                 # calibrate, then compare policies
    python simulator.py --quick         # small n, for iteration

Why this exists. The offline arm cannot answer "what if we screened sooner?":
the logged transition kernel does not respond to the action (§12.3), and the
fitted estimators are too noisy to rank policies anyway (§5.1). This module
supplies the missing causal mechanism explicitly, as a natural-history model:

    Healthy --onset--> Preclinical (screen-detectable, growing) --> Clinical

A tumour grows between screens, so a longer interval mechanically means
detection at a larger size and a higher chance of nodal spread. That is the
"wait longer -> worse stage" link the observational data cannot identify, and
it is calibrated to CSAW-CC's own screen-detected/interval contrast rather
than assumed.

Everything is a modelling assumption and is labelled as such. Simulator results
must be reported in their own table, never merged with offline-evaluated
numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from config import CFG

# Calibration targets measured from CSAW-CC by data_audit.py / data.py.
# These are the observed behaviour of the Swedish programme, so calibration
# must be run under the standard-of-care schedule (18mo <55y, 24mo 55+).
TARGETS = {
    "p_screen_detected": 524 / (524 + 217),   # 0.707
    "node_pos_screen": 0.229,
    "node_pos_interval": 0.341,
    "large_screen": 0.408,                    # invasive > 15 mm
    "large_interval": 0.470,
    "recall_rate_control": 0.015,
}
# Stage targets carry weight equal to the screen/interval split: stage at
# detection is what the reward function is built on (U(sigma)), so a model that
# nails the split while getting tumour size wrong is fit to the wrong quantity.
TARGET_WEIGHTS = {
    "p_screen_detected": 2.0, "node_pos_screen": 1.0, "node_pos_interval": 1.0,
    "large_screen": 2.0, "large_interval": 2.0, "recall_rate_control": 2.0,
}


@dataclass
class Params:
    """Natural-history parameters. Units: size in mm, time in years.

    Symptomatic presentation is a size-dependent HAZARD, not a fixed threshold
    size. This matters and was got wrong first time round: with a heterogeneous
    threshold, a tumour with a low threshold surfaces quickly and gets few
    screening opportunities, so interval cancers were selected for *small*
    thresholds and came out smaller than screen-detected ones -- the opposite of
    what CSAW-CC shows. Under a hazard, interval cancers are the fast-growing
    ones that outrun the screening schedule, which is both the real mechanism
    and the one that reproduces the observed stage contrast.
    """

    size_onset: float = 2.0        # size at which the tumour becomes detectable-in-principle
    log_growth_mean: float = -0.3  # log of annual exponential growth rate
    log_growth_sd: float = 0.60    # heterogeneity in growth -- the dominant source
    k_sym: float = 0.35            # symptomatic hazard at 10 mm (per year)
    a_sym: float = 1.8             # hazard exponent: h(size) = k_sym * (size/10)**a_sym
    #   a_sym must stay superlinear (>1.2): a sublinear hazard lets 10 cm
    #   tumours remain silent, which produced a 273 mm 95th percentile.
    se_alpha: float = -2.0         # screening sensitivity: logistic(a + b*log size - c*density)
    se_beta: float = 1.6
    se_density: float = 0.02
    node_a: float = -4.0           # P(node+) = logistic(a + b*log size)
    node_b: float = 1.1
    fp_rate: float = 0.015         # per-screen false-recall probability
    onset_rate: float = 0.010      # annual hazard of entering the preclinical state


def sample_population(n, ex, rng):
    """Draw (age bin, percent density) jointly from the real CSAW-CC cohort."""
    d = ex[["age_bin", "percentdensity_left", "percentdensity_right"]].dropna()
    dens = d[["percentdensity_left", "percentdensity_right"]].mean(1).to_numpy()
    age = d.age_bin.to_numpy()
    i = rng.integers(0, len(d), n)
    return age[i], dens[i]


def soc_intervals(age, density=None, visit=None):
    """Swedish standard of care: 18 months under 55, 24 months at 55+."""
    return np.where(age == 1, 1.5, 2.0)


def _policy_intervals(policy, age, density, visit):
    if callable(policy):
        return policy(age, density, visit)
    return np.full(len(age), float(policy))


def simulate(params: Params, age, density, policy, rng, horizon=10.0, dt=1.0 / 12):
    """Vectorised natural history + screening on a monthly grid.

    Symptomatic presentation runs as a competing risk against screen detection
    at every time step, so it happens whether or not a screening round follows.
    (An earlier round-based version registered symptomatic cancers only at the
    next screen, which let long-interval policies avoid recording their own
    misses.)
    """
    n = len(age)
    p = params

    # --- latent natural history -------------------------------------------
    has_cancer = rng.random(n) < (1 - np.exp(-p.onset_rate * horizon))
    t_onset = rng.uniform(0, horizon, n)
    t_onset[~has_cancer] = np.inf
    growth = np.exp(rng.normal(p.log_growth_mean, p.log_growth_sd, n))

    detected = np.zeros(n, bool)       # screen-detected
    interval = np.zeros(n, bool)       # presented symptomatically
    size_at_dx = np.zeros(n)
    t_dx = np.full(n, np.inf)
    n_screens = np.zeros(n, int)
    n_fp = np.zeros(n, int)

    visit = np.zeros(n, int)
    next_screen = _policy_intervals(policy, age, density, visit).astype(float)
    dz = (density - density.mean()) / (density.std() + 1e-9)

    t = 0.0
    for _ in range(int(round(horizon / dt))):
        t += dt
        alive = ~detected & ~interval
        if not alive.any():
            break

        pre = alive & (t_onset <= t)
        # Cap the exponent: fast growth over a long horizon otherwise overflows,
        # and no breast tumour is metres across. 300 mm is far past any
        # clinically reachable size, so the cap never binds on real trajectories.
        expo = np.clip(growth * np.where(pre, t - t_onset, 0.0), 0.0, 5.0)
        size = np.where(pre, np.minimum(p.size_onset * np.exp(expo), 300.0), 0.0)

        # Competing risk 1: symptomatic presentation, hazard rising with size.
        h = np.where(pre, p.k_sym * (np.maximum(size, 1e-6) / 10.0) ** p.a_sym, 0.0)
        present = pre & (rng.random(n) < 1 - np.exp(-h * dt))
        interval |= present
        size_at_dx = np.where(present, size, size_at_dx)
        t_dx = np.where(present, t, t_dx)

        # Competing risk 2: a scheduled screen falls in this step.
        due = alive & ~present & (t >= next_screen) & (t <= horizon)
        if due.any():
            n_screens += due
            se = 1 / (1 + np.exp(-(p.se_alpha
                                   + p.se_beta * np.log(np.maximum(size, 1e-6))
                                   - p.se_density * dz * 10)))
            found = due & pre & (rng.random(n) < se)
            detected |= found
            size_at_dx = np.where(found, size, size_at_dx)
            t_dx = np.where(found, t, t_dx)

            n_fp += due & ~pre & (rng.random(n) < p.fp_rate)
            visit = visit + due
            gap = _policy_intervals(policy, age, density, visit)
            next_screen = np.where(due, t + gap, next_screen)

    node = rng.random(n) < 1 / (1 + np.exp(
        -(p.node_a + p.node_b * np.log(np.maximum(size_at_dx, 1e-6)))))
    node &= (detected | interval)

    return dict(detected=detected, interval=interval, size=size_at_dx, node=node,
                t_dx=t_dx, n_screens=n_screens, n_fp=n_fp, has_cancer=has_cancer)


def summary(out) -> dict:
    """Reduce a cohort to the quantities we calibrate against."""
    dx = out["detected"] | out["interval"]
    s, i = out["detected"], out["interval"]
    f = lambda m, v: float(v[m].mean()) if m.any() else 0.0
    return {
        "p_screen_detected": float(s.sum() / max(dx.sum(), 1)),
        "node_pos_screen": f(s, out["node"]),
        "node_pos_interval": f(i, out["node"]),
        "large_screen": f(s, out["size"] > 15),
        "large_interval": f(i, out["size"] > 15),
        "recall_rate_control": float(out["n_fp"].sum() / max(out["n_screens"].sum(), 1)),
        "cancer_rate": float(dx.mean()),
        # Not calibration targets -- plausibility guards (see `plausibility`).
        "mean_size": float(out["size"][dx].mean()) if dx.any() else 0.0,
        "p_size_gt50": float((out["size"][dx] > 50).mean()) if dx.any() else 0.0,
    }


# Clinical plausibility bounds. CSAW-CC records only the >15 mm indicator, so a
# fit can satisfy every binary target while producing a grotesque size tail --
# an early calibration matched all six targets with 19% of tumours over 50 mm
# and a 95th percentile of 273 mm. These bounds close that loophole.
MAX_MEAN_SIZE_MM = 25.0      # mean size at detection in a screened population
MAX_FRAC_OVER_50MM = 0.06    # locally advanced disease is uncommon under screening
SOJOURN_RANGE = (2.0, 4.0)   # mean preclinical sojourn, years (literature)


def plausibility(stats, p: Params) -> float:
    """Penalty for parameter sets that fit the targets but are not credible."""
    penalty = 0.0

    sojourn = sojourn_years(p)
    low, high = SOJOURN_RANGE
    if not (low <= sojourn <= high):
        penalty += 0.5 * min(abs(sojourn - low), abs(sojourn - high)) ** 2

    if stats["mean_size"] > MAX_MEAN_SIZE_MM:
        penalty += 0.02 * (stats["mean_size"] - MAX_MEAN_SIZE_MM) ** 2
    if stats["p_size_gt50"] > MAX_FRAC_OVER_50MM:
        penalty += 20.0 * (stats["p_size_gt50"] - MAX_FRAC_OVER_50MM) ** 2
    return penalty


_NEVER = 1e6  # a policy that never screens


def sojourn_years(p: Params, n=4000, seed=0) -> float:
    """Mean preclinical sojourn: onset -> symptomatic, with screening switched off.

    No closed form once presentation is a hazard, so it is measured. This is the
    single most influential quantity in any screening model, so it is measured
    rather than assumed and is then constrained to the literature range.
    """
    rng = np.random.default_rng(seed)
    age = np.ones(n, int)
    dens = np.full(n, 25.0)
    out = simulate(p, age, dens, _NEVER, rng, horizon=30.0)
    m = out["interval"] & np.isfinite(out["t_dx"])
    if not m.any():
        return 99.0
    # t_onset is uniform on [0, horizon); recover sojourn from recorded sizes.
    return float(np.mean(np.log(out["size"][m] / p.size_onset))
                 / np.exp(p.log_growth_mean + p.log_growth_sd ** 2 / 2))


def loss(stats, p: Params | None = None) -> float:
    """Weighted mismatch against the CSAW-CC targets, plus plausibility penalties.

    The targets alone do not pin down a credible natural history: they are all
    proportions, so a model can match every one of them with an implausible
    sojourn time or a wild tumour-size tail. `plausibility` supplies the
    clinical constraints that the data cannot.
    """
    mismatch = sum(TARGET_WEIGHTS[k] * (stats[k] - v) ** 2 for k, v in TARGETS.items())
    return mismatch + (plausibility(stats, p) if p is not None else 0.0)


def calibrate(ex, n_pop=20000, n_iter=400, seed=0, verbose=True):
    """Random search over natural-history parameters (METHODOLOGY.md §5.5).

    Deliberately simple: with 6 summary targets and ~9 free parameters the
    problem is under-determined, so this returns *a* plausible parameterisation,
    not a unique one. Report the calibration table and run policy conclusions
    across the retained top-k parameter sets, not just the argmin.
    """
    rng = np.random.default_rng(seed)
    age, dens = sample_population(n_pop, ex, rng)
    base = Params()
    best, best_l, keep = base, np.inf, []

    for it in range(n_iter):
        p = base if it == 0 else replace(
            base,
            log_growth_mean=rng.uniform(-1.0, 0.6),
            log_growth_sd=rng.uniform(0.3, 1.0),   # growth is the dominant heterogeneity
            k_sym=rng.uniform(0.05, 1.2),
            a_sym=rng.uniform(1.2, 3.2),
            se_alpha=rng.uniform(-5.0, 0.0),
            se_beta=rng.uniform(0.5, 3.0),
            se_density=rng.uniform(0.0, 0.06),
            node_a=rng.uniform(-6.0, -1.5),
            node_b=rng.uniform(0.4, 2.2),
            fp_rate=rng.uniform(0.008, 0.025),
        )
        st = summary(simulate(p, age, dens, soc_intervals,
                              np.random.default_rng(1000 + it)))
        l = loss(st, p)
        keep.append((l, p, st))
        if l < best_l:
            best_l, best = l, p
            if verbose:
                print(f"  rand {it:4d}  loss {l:.5f}  "
                      f"split {st['p_screen_detected']:.3f}  "
                      f"large {st['large_screen']:.3f}/{st['large_interval']:.3f}")

    # --- local refinement -------------------------------------------------
    # Pure random search over 10 parameters leaves a lot on the table; a simple
    # shrinking-scale hill climb from the best draw closes most of the gap.
    FIELDS = ["log_growth_mean", "log_growth_sd", "k_sym", "a_sym", "se_alpha",
              "se_beta", "se_density", "node_a", "node_b", "fp_rate"]
    scales = {"log_growth_mean": .25, "log_growth_sd": .15, "k_sym": .15,
              "a_sym": .3, "se_alpha": .5, "se_beta": .3, "se_density": .008,
              "node_a": .4, "node_b": .2, "fp_rate": .002}
    n_ref = max(n_iter // 2, 50)
    for it in range(n_ref):
        shrink = 1.0 - it / n_ref
        cand = replace(best, **{
            f: float(getattr(best, f) + rng.normal(0, scales[f] * shrink))
            for f in FIELDS})
        cand = replace(cand, log_growth_sd=max(cand.log_growth_sd, 0.05),
                       k_sym=max(cand.k_sym, 0.01), a_sym=max(cand.a_sym, 1.2),
                       fp_rate=float(np.clip(cand.fp_rate, 0.002, 0.05)),
                       se_density=max(cand.se_density, 0.0))
        st = summary(simulate(cand, age, dens, soc_intervals,
                              np.random.default_rng(50_000 + it)))
        l = loss(st, cand)
        keep.append((l, cand, st))
        if l < best_l:
            best_l, best = l, cand
            if verbose and it % 10 == 0:
                print(f"  refine {it:4d}  loss {l:.5f}  "
                      f"split {st['p_screen_detected']:.3f}  "
                      f"large {st['large_screen']:.3f}/{st['large_interval']:.3f}")

    keep.sort(key=lambda z: z[0])
    return best, best_l, [k[1] for k in keep[:10]], keep[0][2]


# ─────────────────────────────────────────────────────────── policy value ──
def policy_value(params, age, dens, policy, rng, horizon=10.0, costs=None):
    """True (simulated) value + clinical metrics. No fitted estimator involved."""
    costs = costs or CFG.costs
    out = simulate(params, age, dens, policy, rng, horizon)

    u = np.zeros(len(age))
    dx = out["detected"] | out["interval"]
    big, node = out["size"] > 15, out["node"]
    u[dx & ~big & ~node] = costs.u_inv_small_n0
    u[dx & ~big & node] = costs.u_inv_small_n1
    u[dx & big & ~node] = costs.u_inv_large_n0
    u[dx & big & node] = costs.u_inv_large_n1

    # Discount by ELAPSED TIME, always — even though the offline arm uses
    # per-round MDP discounting (CFG.smdp=False).
    #
    # Per-round discounting is not valid for cross-policy comparison when the
    # policies differ in how many rounds they generate. A 6-month policy
    # produces ~19 rounds in 10 years and a 36-month policy ~3, so discounting
    # per round would shrink the frequent screener's harm term by gamma**19 vs
    # gamma**3 and make frequent screening look good for a purely bookkeeping
    # reason. Time-discounting is also the health-economics standard and is what
    # makes gamma = 1/(1+0.03) mean what it claims to mean.
    disc = CFG.gamma ** np.where(np.isfinite(out["t_dx"]), out["t_dx"], 0.0)

    cost = (costs.exam * out["n_screens"]
            + costs.false_positive * out["n_fp"]
            + u * disc)
    n_dx = max(dx.sum(), 1)
    return {
        "value": float(-cost.mean()),
        # Components, so the value can be recomputed for any cost parameters
        # without re-simulating (used by the threshold analysis).
        "mean_screens": float(out["n_screens"].mean()),
        "mean_fp": float(out["n_fp"].mean()),
        "mean_harm": float((u * disc).mean()),
        "screens_per_woman": float(out["n_screens"].mean()),
        "cancers": int(dx.sum()),
        "pct_screen_detected": float(out["detected"].sum() / n_dx),
        "interval_cancer_rate_per_1000": float(1000 * out["interval"].sum() / len(age)),
        "mean_size_mm": float(out["size"][dx].mean()) if dx.any() else 0.0,
        "node_pos_rate": float(node[dx].mean()) if dx.any() else 0.0,
        "fp_per_1000_screens": float(1000 * out["n_fp"].sum() / max(out["n_screens"].sum(), 1)),
    }


def main():
    from data import build_exams

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-pop", type=int, default=20000)
    ap.add_argument("--n-iter", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.quick:
        args.n_pop, args.n_iter, args.seeds = 4000, 60, 2

    ex = build_exams()
    print(f"\n[calibrate] {args.n_iter} draws x {args.n_pop:,} women, "
          "under the standard-of-care schedule")
    best, l, topk, st = calibrate(ex, args.n_pop, args.n_iter)

    print("\n[calibration fit]")
    print(f"{'target':24s} {'observed':>10s} {'simulated':>10s}")
    for k, v in TARGETS.items():
        print(f"{k:24s} {v:10.3f} {st[k]:10.3f}")
    print(f"\nweighted loss {l:.5f}")
    print("\n[calibrated parameters]")
    for k, v in asdict(best).items():
        print(f"  {k:18s} {v:.4f}")
    sojourn = sojourn_years(best)
    flag = "" if 2.0 <= sojourn <= 4.0 else "   <-- OUTSIDE the literature range"
    print(f"\n  implied mean sojourn time ~ {sojourn:.2f} years "
          f"(literature: ~2-4y){flag}")

    # ---- counterfactual interval comparison, including sub-12-month ----
    rng = np.random.default_rng(7)
    age, dens = sample_population(args.n_pop, ex, rng)
    policies = {
        "6mo": 0.5, "12mo": 1.0, "18mo": 1.5, "24mo": 2.0, "36mo": 3.0,
        "standard_of_care": soc_intervals,
        "density_adaptive": lambda a, d, v: np.where(d > np.quantile(dens, 0.75), 1.0, 2.0),
    }
    rows = []
    for name, pol in policies.items():
        runs = [policy_value(best, age, dens, pol, np.random.default_rng(100 + s))
                for s in range(args.seeds)]
        r = {"policy": name}
        for k in runs[0]:
            r[k] = float(np.mean([x[k] for x in runs]))
            if k == "value":
                r["value_sd"] = float(np.std([x[k] for x in runs]))
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["policy", "value", "value_sd", "screens_per_woman",
            "interval_cancer_rate_per_1000", "pct_screen_detected",
            "mean_size_mm", "node_pos_rate", "fp_per_1000_screens"]
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.4f}")
    print("\n[simulated policy comparison — SIMULATOR RESULTS, do not merge with OPE]")
    print(df[cols].to_string(index=False))

    # ---- what actually decides the answer: the per-exam disutility ----
    print("\n[threshold analysis — which policy wins as a function of c_e]")
    print("The optimal interval is governed almost entirely by the disutility of a")
    print("single screening exam relative to the harm averted. c_e is a PLACEHOLDER")
    print("in config.py, so this sweep — not the single row above — is the result.")
    grid = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    tab = []
    for ce in grid:
        vals = {r["policy"]: -(ce * r["mean_screens"]
                               + CFG.costs.false_positive * r["mean_fp"]
                               + r["mean_harm"]) for r in rows}
        bestp = max(vals, key=vals.get)
        tab.append({"c_e": ce, "c_e_days": round(ce * 365, 2),
                    "optimal": bestp, **{k: round(v, 4) for k, v in vals.items()}})
    tdf = pd.DataFrame(tab)
    print(tdf.to_string(index=False))
    print(f"\ncurrent config c_e = {CFG.costs.exam} "
          f"({CFG.costs.exam * 365:.2f} days of perfect health per mammogram)")

    CFG.out.mkdir(exist_ok=True, parents=True)
    tdf.to_csv(CFG.out / "simulator_threshold.csv", index=False)
    df.to_csv(CFG.out / "simulator_policies.csv", index=False)
    pd.DataFrame([asdict(p) for p in topk]).to_csv(
        CFG.out / "simulator_topk_params.csv", index=False)
    print(f"\nsaved -> {CFG.out / 'simulator_policies.csv'}")
    print("        (top-10 parameter sets saved too: rerun conclusions across them,")
    print("         the calibration is under-determined by design)")


if __name__ == "__main__":
    main()
