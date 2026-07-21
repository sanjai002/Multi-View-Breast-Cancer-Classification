"""
Step 1 — CSAW-CC csv  ->  MDP transition buffer.

Covers METHODOLOGY.md §1 (organisation/QC), §3 (trajectories/censoring),
§6 (state), §8 (reward), §12.1 (behaviour policy for OPE).

    python data.py            # builds outputs/buffer.npz + prints the audit

Design points that are easy to get wrong and are enforced here:
  * outcome variables (x_case, x_type, x_lymphnode_met, rad_timing,
    x_cancer_laterality) are REWARD-ONLY and are asserted out of the state;
  * censoring is not "stayed healthy" — censored terminals bootstrap;
  * the 217 interval-cancer terminals are kept via an imputed final action,
    because they carry the entire miss signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG, DETECTED, INTERVAL, CENSORED, N_ACTIONS, encode_action

# Known retrospectively only. Used to build rewards, never states.
OUTCOME_COLS = [
    "x_case", "x_type", "x_lymphnode_met", "rad_timing", "x_cancer_laterality",
]


# ─────────────────────────────────────────────────────────────── exam level ──
def build_exams(csv=None) -> pd.DataFrame:
    """98,788 image rows -> 24,694 exam rows, with QC (METHODOLOGY.md §1.2-1.3)."""
    df = pd.read_csv(csv or CFG.csv)

    # QC: 3 exams carry 8 images (duplicate acquisitions). Keep the first of
    # each (laterality, view) pair by filename so every exam has exactly 4.
    before = len(df)
    df = (df.sort_values("anon_filename")
            .groupby(["anon_patientid", "exam_year", "imagelaterality", "viewposition"],
                     as_index=False).first())
    if before != len(df):
        print(f"[qc] dropped {before - len(df)} duplicate view rows")

    # libra_* are image-level -> keep per side (asymmetry is a real risk cue).
    side = (df.pivot_table(index=["anon_patientid", "exam_year"],
                           columns="imagelaterality",
                           values=["libra_percentdensity", "libra_densearea",
                                   "libra_breastarea"],
                           aggfunc="mean"))
    side.columns = [f"{a}_{b}".lower().replace("libra_", "") for a, b in side.columns]

    # rad_* are exam-level and repeat across the 4 rows; x_* are patient-level.
    ex = (df.groupby(["anon_patientid", "exam_year"])
            .agg(n_img=("anon_filename", "count"),
                 age_bin=("x_age", "max"),
                 case=("x_case", "max"),
                 timing=("rad_timing", "max"),
                 recall=("rad_recall", "max"),
                 r1=("rad_r1", "max"),
                 r2=("rad_r2", "max"),
                 x_type=("x_type", "max"),
                 node=("x_lymphnode_met", "max"))
            .join(side).reset_index()
            .sort_values(["anon_patientid", "exam_year"]).reset_index(drop=True))

    assert (ex.n_img == 4).all(), "every exam must have exactly 4 views after QC"

    ex["visit_idx"] = ex.groupby("anon_patientid").cumcount()
    ex["n_visits"] = ex.groupby("anon_patientid").exam_year.transform("size")
    ex["gap_prev"] = ex.groupby("anon_patientid").exam_year.diff()
    ex["gap_next"] = ex.groupby("anon_patientid").exam_year.diff(-1).mul(-1)

    # Missingness is informative (8.1% of rad_recall) -> flag, never impute to 0.
    for c in ("recall", "r1", "r2"):
        ex[f"{c}_missing"] = ex[c].isna().astype(float)
    return ex


def terminal_types(ex: pd.DataFrame) -> pd.Series:
    """Per patient: DETECTED / INTERVAL / CENSORED, from the FINAL exam's timing.

    rad_timing is monotone within a patient (0 violations, see data_audit.py),
    so only the last exam determines the outcome; an intermediate timing==2 just
    means that exam was already close to the eventual diagnosis.
    """
    last = ex.groupby("anon_patientid").timing.last()
    t = pd.Series(CENSORED, index=last.index, dtype=int)
    t[last == 1.0] = DETECTED
    t[last == 2.0] = INTERVAL
    return t


# ──────────────────────────────────────────────────────────────────  state ──
def build_states(ex: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Exam -> state vector (METHODOLOGY.md §6). Outcome columns are excluded."""
    by_patient = ex.groupby("anon_patientid")
    feat = pd.DataFrame(index=ex.index)

    # --- who she is -------------------------------------------------------
    feat["age_40_55"] = (ex.age_bin == 1).astype(float)
    feat["age_55p"] = (ex.age_bin == 2).astype(float)

    # --- what her breasts look like now -----------------------------------
    for measure in ("percentdensity", "densearea", "breastarea"):
        for side in ("left", "right"):
            feat[f"{measure}_{side}"] = ex[f"{measure}_{side}"]
    # Bilateral asymmetry: radiologists read the left-vs-right difference.
    feat["density_asym"] = (ex.percentdensity_left - ex.percentdensity_right).abs()
    # Change since her previous visit: the core longitudinal cue.
    for side in ("left", "right"):
        feat[f"d_density_{side}"] = (by_patient[f"percentdensity_{side}"]
                                     .diff().fillna(0.0))

    # --- where she is in her screening career -----------------------------
    feat["visit_idx"] = ex.visit_idx
    feat["time_since_last"] = ex.gap_prev.fillna(0.0)
    feat["cum_screens"] = ex.visit_idx + 1
    feat["calendar_year"] = ex.exam_year

    # --- what happened at her previous visits -----------------------------
    # Every one of these uses shift(1), which is what keeps them strictly
    # causal: a feature at visit t may only see visits 1..t-1.
    recalled = ex.recall.fillna(0.0)
    feat["prior_recall_count"] = _running_total_before_now(ex, recalled)
    feat["prior_recall_last"] = by_patient.recall.shift(1).fillna(0.0)
    feat["prior_recall_ever"] = (feat.prior_recall_count > 0).astype(float)

    # Radiologists 1 and 2 disagreeing is a genuine uncertainty signal.
    readers_disagreed = (ex.r1.fillna(0) != ex.r2.fillna(0)).astype(float)
    feat["prior_disagree_last"] = (readers_disagreed.groupby(ex.anon_patientid)
                                   .shift(1).fillna(0.0))
    feat["prior_disagree_count"] = _running_total_before_now(ex, readers_disagreed)

    # Missingness is informative, so it gets its own channel (never imputed).
    for column in ("recall", "r1", "r2"):
        feat[f"{column}_missing"] = ex[f"{column}_missing"]

    # --- what was decided last time (one-hot) -----------------------------
    previous_action = np.full(len(ex), -1)
    has_previous = ex.gap_prev.notna().to_numpy()
    previous_action[has_previous] = [
        encode_action(_bucket(gap), int(r))
        for gap, r in zip(ex.gap_prev[has_previous], recalled[has_previous])
    ]
    for action in range(N_ACTIONS):
        feat[f"prev_action_{action}"] = (previous_action == action).astype(float)

    names = list(feat.columns)
    X = feat.to_numpy(np.float32)
    assert not np.isnan(X).any(), "NaNs in state matrix"
    assert not any(c in names for c in OUTCOME_COLS), "LABEL LEAK: outcome in state"

    # Optional image block (no DICOMs on this machine; wired for when they land).
    if CFG.image_features is not None:
        Z = np.load(CFG.image_features).astype(np.float32)
        assert len(Z) == len(ex), "image feature rows must align with exams"
        X = np.concatenate([X, Z], axis=1)
        names += [f"img_{i}" for i in range(Z.shape[1])]
        print(f"[state] +{Z.shape[1]}-d image embedding")
    return X, names


def _running_total_before_now(ex: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Cumulative sum of `values` over a patient's EARLIER visits only.

    shift(1) before cumsum is what makes this causal -- at visit t it counts
    visits 1..t-1 and never t itself.
    """
    return (values.groupby(ex.anon_patientid)
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=0, drop=True)
            .fillna(0.0))


def _bucket(gap: float) -> int:
    """Realised gap in years -> interval action index (12 / 24 / 36+ months)."""
    if gap <= 1:
        return 0
    if gap == 2:
        return 1
    return 2          # 3y and beyond (526 transitions) collapse to the 36mo arm


# ─────────────────────────────────────────────────────────── transitions ──
def build_buffer(ex: pd.DataFrame, X: np.ndarray, costs=None) -> dict:
    """Assemble transitions with rewards (METHODOLOGY.md §3, §8).

    tau (the realised gap in years) is still recorded per transition: it is a
    state feature and the SMDP ablation needs it, even though MDP discounting
    ignores it.
    """
    costs = costs or CFG.costs
    outcome_of = terminal_types(ex)
    exam_row_of = {(patient, year): row for row, (patient, year)
                   in enumerate(zip(ex.anon_patientid, ex.exam_year))}

    transitions = []          # list of dicts; assembled into arrays at the end

    for patient, visits in ex.groupby("anon_patientid", sort=False):
        visits = visits.sort_values("exam_year")
        exam_rows = [exam_row_of[(patient, y)] for y in visits.exam_year]
        recalled = visits.recall.fillna(0.0).to_numpy()
        timing = visits.timing.to_numpy()
        years = visits.exam_year.to_numpy()
        n_visits = len(exam_rows)

        outcome = outcome_of[patient]
        # Stage-dependent QALY loss, known only for patients who were diagnosed.
        harm = costs.utility(visits.x_type.iloc[-1], visits.node.iloc[-1])

        def screening_cost(visit_i):
            """Exam cost, plus a false-positive cost if this recall found nothing."""
            was_false_positive = recalled[visit_i] == 1 and timing[visit_i] != 1.0
            return costs.exam + costs.false_positive * was_false_positive

        # --- one transition per observed visit-to-visit gap ------------------
        for i in range(n_visits - 1):
            gap_years = years[i + 1] - years[i]
            action = encode_action(_bucket(gap_years), int(recalled[i]))

            # The diagnosis is attributed to the last wait of the trajectory.
            ends_in_diagnosis = (i == n_visits - 2) and outcome == DETECTED
            reward = -screening_cost(i) - (harm if ends_in_diagnosis else 0.0)

            transitions.append(dict(
                s=exam_rows[i], a=action, r=reward, s2=exam_rows[i + 1],
                done=float(ends_in_diagnosis), tau=float(gap_years),
                pid=patient, imputed=0.0))

        # --- terminal transitions that would otherwise be lost ---------------
        # INTERVAL: the cancer surfaced AFTER the last screen, so the miss belongs
        #   to the interval chosen there -- but there is no next exam to read it
        #   from.
        # DETECTED with a single visit: 219 patients whose cancer was found at
        #   their only screen, so there is no preceding transition to carry it.
        # Both need an imputed interval action (the programme default, which is
        # also the modal observed action). Flagged so it can be ablated.
        needs_terminal = outcome == INTERVAL or (outcome == DETECTED and n_visits == 1)
        if needs_terminal and CFG.impute_terminal_action:
            transitions.append(dict(
                s=exam_rows[-1], a=encode_action(1, int(recalled[-1])),
                r=-screening_cost(n_visits - 1) - harm,
                s2=exam_rows[-1],          # unused: done = 1
                done=1.0, tau=2.0, pid=patient, imputed=1.0))

        # CENSORED terminals get NO transition. We stopped observing them; we did
        # not observe "stayed healthy". Bootstrapping handles them via the last
        # next_state of the preceding transition.

    def column(name, dtype):
        return np.asarray([t[name] for t in transitions], dtype)

    patient_ids = column("pid", np.int64)
    case_per_patient = ex.groupby("anon_patientid").case.max()
    is_case = case_per_patient.reindex(patient_ids).to_numpy()

    # Case-control reweighting (METHODOLOGY.md §8.5): this file is ~10% cancer,
    # the screening population is ~0.6%. Without this the agent over-screens.
    sample_prevalence = float(case_per_patient.mean())
    if CFG.reweight_prevalence:
        pop = CFG.population_prevalence
        weights = np.where(is_case == 1,
                           pop / sample_prevalence,
                           (1 - pop) / (1 - sample_prevalence))
    else:
        weights = np.ones(len(patient_ids))

    return dict(
        s=column("s", np.int64), a=column("a", np.int64),
        r=column("r", np.float32), s2=column("s2", np.int64),
        done=column("done", np.float32), tau=column("tau", np.float32),
        imputed=column("imputed", np.float32),
        pid=patient_ids, w=weights.astype(np.float32),
        is_case=is_case.astype(np.int64), sample_prevalence=sample_prevalence,
    )


# ──────────────────────────────────────────────────────────────── splits ──
def make_splits(ex: pd.DataFrame, seed=None) -> pd.Series:
    """Patient-level, stratified on (case, age bin, n_visits). §1.4."""
    rng = np.random.default_rng(seed if seed is not None else CFG.seed)
    p = ex.groupby("anon_patientid").agg(
        case=("case", "max"), age=("age_bin", "max"), nv=("exam_year", "size"))
    out = pd.Series("train", index=p.index)
    tr, va, _ = CFG.split
    for _, sub in p.groupby(["case", "age", "nv"]):
        ids = rng.permutation(sub.index.to_numpy())
        n = len(ids)
        i, j = int(round(tr * n)), int(round((tr + va) * n))
        out[ids[i:j]] = "val"
        out[ids[j:]] = "test"
    return out


# ────────────────────────────────────────────────── behaviour policy (OPE) ──
def fit_behaviour_policy(X, buf, split, ex):
    """pi_b(a|s) for importance sampling, propensity floors and BC (§12.1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    tr = split.reindex(buf["pid"]).to_numpy() == "train"
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X[buf["s"]][tr], buf["a"][tr])
    P = np.zeros((len(buf["a"]), N_ACTIONS), np.float32)
    P[:, m.classes_] = m.predict_proba(X[buf["s"]]).astype(np.float32)
    acc = (P.argmax(1) == buf["a"]).mean()
    print(f"[pi_b] train acc {(P.argmax(1)[tr] == buf['a'][tr]).mean():.3f} | all {acc:.3f}")
    return P, m


# ────────────────────────────────────────── delay-cost analysis (§8.3) ──
def delay_cost_table(ex: pd.DataFrame, costs=None) -> pd.DataFrame:
    """The paper's key empirical claim: delay cost ESTIMATED, not assumed."""
    costs = costs or CFG.costs
    det = ex[ex.timing.isin([1.0, 2.0])].drop_duplicates("anon_patientid", keep="last")
    rows = []
    for name, code in (("screen-detected", 1.0), ("interval", 2.0)):
        d = det[det.timing == code]
        u = [costs.utility(a, b) for a, b in zip(d.x_type, d.node)]
        rows.append(dict(group=name, n=len(d),
                         node_pos=float((d.node == 1).mean()),
                         inv_gt15=float((d.x_type == 3).mean()),
                         mean_U=float(np.mean(u))))
    t = pd.DataFrame(rows)
    t.attrs["delta_U"] = t.mean_U.iloc[1] - t.mean_U.iloc[0]
    return t


# ─────────────────────────────────────────────────────────────────  main ──
def load_all(verbose=True):
    ex = build_exams()
    X, names = build_states(ex)
    buf = build_buffer(ex, X)
    split = make_splits(ex)
    P, _ = fit_behaviour_policy(X, buf, split, ex)
    buf["pi_b"] = P
    buf["split"] = split.reindex(buf["pid"]).to_numpy()

    # Standardise states using TRAIN statistics only.
    tr_rows = np.unique(buf["s"][buf["split"] == "train"])
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    if verbose:
        print(f"\n[data] exams {len(ex):,}  patients {ex.anon_patientid.nunique():,}"
              f"  state dim {X.shape[1]}")
        print(f"[data] transitions {len(buf['a']):,} "
              f"(observed {int((buf['imputed'] == 0).sum()):,}, "
              f"imputed-terminal {int(buf['imputed'].sum()):,})")
        tt = terminal_types(ex).value_counts()
        print(f"[data] terminals  DETECTED {tt.get(DETECTED,0)}  "
              f"INTERVAL {tt.get(INTERVAL,0)}  CENSORED {tt.get(CENSORED,0)}")
        for s in ("train", "val", "test"):
            m = buf["split"] == s
            print(f"[split] {s:5s} transitions {m.sum():6,}  "
                  f"patients {len(np.unique(buf['pid'][m])):5,}  "
                  f"case-transitions {int(buf['is_case'][m].sum()):5,}")
        print(f"[reward] mean {buf['r'].mean():.4f}  min {buf['r'].min():.3f}  "
              f"max {buf['r'].max():.3f}")
        print(f"[prevalence] sample {buf['sample_prevalence']:.3f} -> "
              f"population {CFG.population_prevalence}")
        print("\n[delay cost, METHODOLOGY §8.3]")
        t = delay_cost_table(ex)
        print(t.to_string(index=False))
        print(f"  Delta E[U] interval vs screen-detected = {t.attrs['delta_U']:+.3f} QALY")
        print("  -> the cost of delay is measured from this dataset, not assumed.")
    return ex, Xn, buf, names


def main():
    ex, X, buf, names = load_all()
    CFG.out.mkdir(exist_ok=True, parents=True)
    arrays = {k: v for k, v in buf.items() if isinstance(v, np.ndarray)}
    arrays["split"] = buf["split"].astype("U5")
    np.savez_compressed(
        CFG.out / "buffer.npz", X=X, names=np.array(names, dtype=object),
        sample_prevalence=np.float32(buf["sample_prevalence"]), **arrays,
    )
    print(f"\nsaved -> {CFG.out / 'buffer.npz'}")


if __name__ == "__main__":
    main()
