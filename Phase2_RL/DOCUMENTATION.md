# Phase 2 — Full Project Documentation

**Personalized Breast Cancer Screening Using Deep Reinforcement Learning on
Longitudinal Mammography Data**

This is the complete reference for the project: what it does, what the data
supports, how the code is organised, how to run it, what the results currently
are, and what is still wrong with them.

Three documents, three jobs:

| Document | Purpose | Read it when |
|---|---|---|
| **DOCUMENTATION.md** (this file) | Everything: data, code, design, results, how-to | You want to understand or run the project |
| [METHODOLOGY.md](METHODOLOGY.md) | The publication-facing scientific write-up, with equations | You are writing the paper |
| [README.md](README.md) | Quickstart and current status | You just want to run something |

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [The dataset](#2-the-dataset)
3. [The four hard constraints](#3-the-four-hard-constraints)
4. [Pipeline overview](#4-pipeline-overview)
5. [File-by-file reference](#5-file-by-file-reference)
6. [The decision problem (MDP)](#6-the-decision-problem-mdp)
7. [The reward function](#7-the-reward-function)
8. [Algorithms](#8-algorithms)
9. [The simulator](#9-the-simulator)
10. [How to run everything](#10-how-to-run-everything)
11. [Results so far](#11-results-so-far)
12. [Known problems and open work](#12-known-problems-and-open-work)
13. [Traps that have already bitten](#13-traps-that-have-already-bitten)
14. [Glossary](#14-glossary)

---

## 1. What this project does

**The clinical question.** Every woman in a breast screening programme is invited
back on a fixed schedule — in Sweden, every 18 months if she is under 55 and
every 24 months if she is older. That schedule ignores everything known about
*her*: her breast density, how that density is changing, whether she has been
recalled before, whether the two radiologists reading her films disagreed.

**The research question.** Can a reinforcement learning agent, trained on real
longitudinal screening histories, recommend a *personalized* screening interval
that detects more cancers earlier without screening everyone more often?

**The honest answer so far.** Partly. The project has two arms because one arm
alone cannot answer the question:

- The **offline arm** learns from 8,723 real patients' screening histories. It is
  trustworthy where the data has support, but it cannot establish that screening
  sooner *causes* better outcomes — see [§3](#3-the-four-hard-constraints).
- The **simulator arm** supplies that missing causal mechanism explicitly, as a
  calibrated model of tumour growth, and can therefore evaluate schedules nobody
  was ever put on (including 6-monthly).

The contribution is the combination, plus an unusually careful account of what
each arm can and cannot support.

---

## 2. The dataset

**CSAW-CC** (Cohort of Screen-Aged Women, case-control subset), curated by
Fredrik Strand at Karolinska Institutet. One CSV, one codebook.

```
CSAW-CC_breast_cancer_screening_data.csv    98,788 rows (one per mammogram image)
CSAW-CC_Readme_annon_230516.docx            official codebook
```

### Shape

| | Count |
|---|---|
| Image rows | 98,788 |
| Screening exams (4 images each) | 24,694 |
| Patients | 8,723 |
| Cancer patients | 873 |
| Control patients | 7,850 |
| Observed visit-to-visit transitions | 15,971 |

### Columns that matter

| Column | Meaning | Notes |
|---|---|---|
| `anon_patientid` | Patient ID | Splits are done on this |
| `exam_year` | Year of exam | **Year resolution only** — no finer timing exists |
| `x_age` | Age band | **Only 2 values**: 1 = 40–55, 2 = 55+ |
| `x_case` | Cancer during study | Patient-level. **Outcome — never a state feature** |
| `x_type` | 1 = in situ, 2 = invasive ≤15mm, 3 = invasive >15mm | Outcome |
| `x_lymphnode_met` | Nodal metastasis | Outcome |
| `rad_timing` | 1 = screen-detected (<60d), 2 = interval cancer (60–729d), 3 = prior (730d+) | Outcome. Drives everything |
| `rad_recall` | Recalled for work-up | A genuine per-exam decision |
| `rad_r1`, `rad_r2` | Two independent radiologist reads | Their disagreement is a useful signal |
| `libra_percentdensity` | Breast density % | Continuous, per image. **The main personalization signal** |
| `libra_breastarea`, `libra_densearea` | Area measures | Per image |
| `imagelaterality`, `viewposition` | Left/Right, CC/MLO | Defines the 4 standard views |

### Trajectory structure

Visits per patient:

| Visits | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Patients | 2,137 | 1,250 | 1,770 | 3,086 | 477 | 3 |

Gaps between exams: 1yr → 2,662; **2yr → 11,106**; 3yr → 1,677; 4yr+ → 526.

### The outcome signal

`rad_timing` sequences per patient are perfectly monotone (0 violations across
873 cancer patients): `(3, 3, …, 3, 1)` or `(3, 3, …, 3, 2)`. The final exam
determines the outcome:

| Outcome | n | Meaning |
|---|---|---|
| `DETECTED` | 524 | Cancer found *at* that screen |
| `INTERVAL` | 217 | Cancer surfaced *after* it — **a miss** |
| `CENSORED` | 7,982 | Observation stopped; **not** "stayed healthy" |

**Interval cancers are measurably worse**, and this is the empirical foundation
of the whole reward function:

| | Node-positive | Invasive >15mm |
|---|---|---|
| Screen-detected | 22.9% | 40.8% |
| Interval | 34.1% | 47.0% |

The cost of delayed detection is therefore *measured from CSAW-CC*, not assumed.

Reproduce every number above with `python data_audit.py`.

---

## 3. The four hard constraints

These are not caveats to bury in a limitations section — they determined the
entire design.

### 3.1 Positivity is violated

The screening interval is almost entirely determined by age band:

| Gap | 40–55y | 55+y |
|---|---|---|
| 1 year | 2,637 | 25 |
| 2 years | 4,750 | 6,356 |

That is the Swedish protocol showing through. Consequently **nobody was ever
screened at 6 months**, and no estimator can recover the value of an action with
zero support. A Q-network will confidently output a value for "screen in 6
months" purely as extrapolation.

*Response:* restrict the offline action space to observed intervals, use a
conservative algorithm (CQL) that explicitly penalises out-of-support actions,
and get sub-12-month answers from the simulator instead.

### 3.2 Several requested actions do not exist in the data

| Action | Status | Why |
|---|---|---|
| 6 months | Simulator only | Zero support |
| 12 / 24 / 36 months | Supported | 2,662 / 11,106 / 1,677 transitions |
| 18 months | Folded into 12/24 | Year resolution can't separate it |
| Immediate biopsy | **Removed** | No biopsy/BI-RADS/pathology variable exists |
| Additional imaging | **Removed** | Not recorded |
| Recall for work-up | Supported | `rad_recall`, 3,116 positives |

### 3.3 Age carries about one bit

`x_age ∈ {1, 2}`. Any claim of "age-personalized" screening is unsupportable.
Personalization rides on **breast density**, its **change over time**, and
**screening history**.

### 3.4 The dataset is case-enriched

Patient-level cancer prevalence here is **10.0%**; a real screening population is
~0.6%. Cases were included exhaustively, controls sampled randomly. Every
reported rate and expected return is prevalence-reweighted
(`CFG.reweight_prevalence`). Without it the agent believes 1 woman in 10 has
cancer and over-screens wildly.

---

## 4. Pipeline overview

```
CSAW-CC csv
     │
     ├─ data_audit.py ······ verifies every claim in the docs
     │
     ▼
  data.py                    98,788 image rows
     │                          ↓ collapse + QC
     │                       24,694 exams
     │                          ↓ 29 features per exam, causal only
     │                       state matrix X
     │                          ↓ trajectories, rewards, censoring
     ▼                       16,407 transitions
 outputs/buffer.npz
     │
     ├──────────────► train.py ──► agent.py (CQL / DQN / … )
     │                   │              │
     │                   │         outputs/cql_seed0.pt
     │                   ▼
     │              FQE checkpoint selection + collapse gate
     │
     └──────────────► evaluate.py ──► FQE, WIS, ESS, support,
                                       clinical metrics, baselines

  simulator.py  ── calibrate to CSAW-CC ──► counterfactual schedules
                   (independent of the offline arm)
```

---

## 5. File-by-file reference

### `config.py` — every knob in one place

Nothing elsewhere hard-codes a cost, a discount rate, or a path.

```python
CFG.costs.exam            # c_e: disutility of one mammogram (QALY)
CFG.costs.false_positive  # c_r: cost of a recall that finds nothing
CFG.costs.utility(...)    # U(σ): QALY loss by stage at detection
CFG.gamma                 # 1/(1+0.03) = 0.9709, health-economic discount
CFG.smdp                  # False = one γ per round; True = γ^τ ablation
CFG.reweight_prevalence   # case-control correction (§3.4)
CFG.image_features        # path to cached image embeddings, or None
```

Sensitivity analysis is a sweep over this file.

### `data_audit.py` — the evidence base

Prints every empirical claim used in the design: scale, view counts, trajectory
lengths, the gap-vs-age table, timing semantics, recall behaviour, stage
contrasts, missingness. Run it first, and after any change to the CSV.

### `data.py` — CSV → transition buffer

| Function | Does |
|---|---|
| `build_exams` | Collapses 4 image rows per exam, drops 12 duplicate view rows, adds visit index and gaps |
| `terminal_types` | DETECTED / INTERVAL / CENSORED from the final exam's `rad_timing` |
| `build_states` | 29 causal features per exam (see below) |
| `build_buffer` | Trajectories → transitions with rewards, terminals, censoring |
| `make_splits` | Patient-level 70/15/15, stratified on (case, age, n_visits) |
| `fit_behaviour_policy` | π̂_b for importance sampling and propensity floors |
| `delay_cost_table` | The measured delay cost (§7) |

**State features (29).** Age band (2), density/dense-area/breast-area per side
(6), bilateral asymmetry (1), density change since last visit (2), visit index,
time since last screen, cumulative screens, calendar year (4), prior recall
count/last/ever (3), prior reader disagreement last/count (2), missingness flags
(3), previous action one-hot (6).

Every history feature uses `shift(1)` so a feature at visit *t* sees only visits
1…*t−1*.

**Excluded by assertion** — `x_case`, `x_type`, `x_lymphnode_met`, `rad_timing`,
`x_cancer_laterality`. These are known only retrospectively. They build rewards;
they must never enter a state. The assert exists because this is the single
easiest way to publish a fraudulent AUC.

### `agent.py` — networks and algorithms

One `QNetwork` (MLP torso → optional dueling → optional quantile head) and one
`OfflineAgent` whose behaviour is feature-flagged, so the ablation ladder is a
config sweep rather than five codebases:

```
bc → dqn → ddqn → dueling → qr → cql
```

CQL adds a conservatism term that pushes down the value of actions the
clinicians never took in a given state — the direct answer to §3.1:

```
L = α·(logsumexp_a Q(s,a) − Q(s,a_observed)) + TD error
```

A GRU belief encoder exists but is **off by default**: trajectories are at most
6 visits and the state already carries explicit history, so a recurrent net adds
parameters that 16k transitions cannot identify.

### `train.py` — the offline loop

No exploration, no environment; the buffer is fixed. Checkpoints are selected by
**validation FQE**, never by training loss.

The **collapse gate** matters more than the loss curve. A policy that always says
"36 months, no work-up" scores well on mean reward over a cohort that is 90%
healthy while being clinically useless. Every evaluation logs action entropy and
whether the policy distinguishes cancer from non-cancer trajectories, and flags
degenerate checkpoints loudly.

### `evaluate.py` — off-policy evaluation and clinical metrics

Two kinds of claim, deliberately kept apart:

- **Recall decisions are directly evaluable.** "Was there cancer at this exam" is
  observed, so sensitivity/specificity/PPV/AUC are ordinary supervised metrics.
- **Interval decisions are not.** Nobody was screened counterfactually, so their
  value comes only from OPE, with all of §3.1's caveats.

Estimators: FQE (primary, seed-averaged), WIS, per-decision WIS, plus two
mandatory diagnostics — **ESS** and **support fraction**. Policies with low
support are flagged and must not be ranked.

### `simulator.py` — the causal arm

See [§9](#9-the-simulator).

---

## 6. The decision problem (MDP)

**Time index:** the screening round, not the calendar year.

**State** `s_t ∈ ℝ²⁹` — the feature vector above (plus image embeddings when
available).

**Action** `a_t = (interval, work-up)`:

```
interval ∈ {12, 24, 36 months}   ×   work-up ∈ {none, recall}   →   6 actions
```

**Transition** — from the observed trajectories (offline arm) or the calibrated
natural history (simulator arm).

**Discount** `γ = 1/(1+0.03) = 0.9709` per round, from the standard 3% annual
health-economic discount rate.

> **Note on the convention.** Per-round discounting means a 36-month wait is
> discounted exactly as hard as a 12-month one, so elapsed time carries no cost
> in the return. `CFG.smdp = True` gives the semi-Markov alternative `γ^τ`. The
> data cannot currently adjudicate between them (§11), so report both. The
> simulator always discounts by elapsed time, because per-round discounting is
> invalid when policies differ in how many rounds they generate.

**Terminal conditions** — the three-way distinction in §2 is the most important
implementation detail in the project. Censored patients bootstrap; only observed
diagnoses get a terminal reward.

---

## 7. The reward function

A negative cost in QALY-equivalent units, so it is directly interpretable to a
clinical audience:

```
r_t = −[ c_e·(exam)  +  c_r·(recall that found nothing)  +  U(σ)·(diagnosis) ]
```

| Term | Symbol | Default | Status |
|---|---|---|---|
| One screening exam | `c_e` | 0.001 QALY (≈0.36 days) | **Placeholder — needs a citation** |
| False-positive recall | `c_r` | 0.05 QALY | Placeholder |
| QALY loss by stage | `U(σ)` | 0.5 – 5.5 | Ordering is data-supported; magnitudes are placeholders |

`U(σ)` is keyed on `(x_type, x_lymphnode_met)`:

| Stage | U |
|---|---|
| In situ | 0.5 |
| Invasive ≤15mm, node− | 1.0 |
| Invasive ≤15mm, node+ | 2.5 |
| Invasive >15mm, node− | 3.0 |
| Invasive >15mm, node+ | 5.5 |

**The delay cost is measured, not assumed.** From `data.py`:

| | n | node+ | >15mm | mean U |
|---|---|---|---|---|
| Screen-detected | 524 | 22.9% | 40.8% | 2.248 |
| Interval | 217 | 34.1% | 47.0% | 2.615 |

**ΔE[U] = +0.367 QALY** for a cancer that surfaces between screens rather than at
one. This is the empirical anchor that satisfies the "no fabricated medical
assumptions" requirement.

> ⚠️ `c_e` decides the answer. At 0.01 QALY (3.65 days per mammogram — roughly
> 10× too high) the simulator selects 36-month screening on its own. Treat
> `simulator.py`'s threshold sweep as the result, never a single value.

---

## 8. Algorithms

| Algorithm | Usable offline? | Verdict |
|---|---|---|
| PPO, A2C | ❌ | On-policy; need environment interaction. **Simulator arm only** |
| SAC | ❌ | Continuous-action design; needs interaction |
| DQN, Double DQN, Dueling | ⚠️ | Off-policy but extrapolate badly off-support. **Ablations only** |
| Rainbow | ⚠️ | Mostly exploration machinery, irrelevant offline |
| **CQL** | ✅ | **Recommended.** Penalises out-of-support action values |
| BCQ, IQL | ✅ | Reasonable alternatives |

**Primary: CQL** on a dueling + double + quantile (QR) backbone. Sweep
`CFG.cql_alpha ∈ {0.1, 1, 5, 10}` and show the policy's deviation from standard
care shrinking as α grows — that plot is the credibility argument.

---

## 9. The simulator

The offline data cannot answer "what if we screened sooner?" because the logged
transition kernel does not respond to the action. The simulator supplies that
mechanism explicitly:

```
Healthy ──onset──► Preclinical (detectable, growing) ──► Clinical (symptomatic)
```

- Tumours grow exponentially at a **heterogeneous rate** (the dominant source of
  variation — fast growers become interval cancers).
- Symptomatic presentation is a **size-dependent hazard**, `h(size) = k·(size/10)^a`,
  running as a competing risk against screen detection at every monthly step.
- Screening sensitivity rises with tumour size and falls with breast density.
- Nodal involvement probability rises with size.

**Calibration** targets CSAW-CC's own statistics under the standard-of-care
schedule: the screen-detected/interval split, both node-positive rates, both
>15mm fractions, and the control recall rate. Random search followed by a
shrinking-scale local refinement.

**Plausibility constraints** exist because all six targets are *proportions*, and
a model can match every one while being clinically absurd:

| Constraint | Bound | Why |
|---|---|---|
| Mean preclinical sojourn | 2–4 years | Literature; unconstrained fits drift to 5–6 |
| Mean size at detection | ≤ 25mm | An early fit matched all targets at 41mm |
| Fraction >50mm | ≤ 6% | The same fit had 19% over 50mm, 95th pct 273mm |
| Hazard exponent `a_sym` | ≥ 1.2 | Sublinear hazards let 10cm tumours stay silent |

**Because it rolls out policies directly, it has no fitted-estimator noise** —
which is exactly why it can rank schedules when FQE cannot (§11).

---

## 10. How to run everything

Requirements: Python 3.10+, numpy, pandas, scikit-learn, torch. CPU is fine —
the whole pipeline is minutes.

```bash
cd Phase2_RL

# 0. verify the dataset matches every documented claim
python data_audit.py

# 1. build the transition buffer  ->  outputs/buffer.npz
python data.py

# 2. train the conservative offline agent  ->  outputs/cql_seed0.pt
python train.py --algo cql --steps 20000

# 3. evaluate against all baselines  ->  outputs/eval_test.csv
python evaluate.py --algo cql --split test

# 4. the causal arm: calibrate + counterfactual schedules
python simulator.py --quick        # ~1 minute
python simulator.py                # full: 600 draws x 15k women
```

Ablation ladder and multi-seed runs (publication needs ≥5 seeds):

```bash
python train.py --algo bc dqn ddqn dueling qr cql --seeds 5
python train.py --algo cql --cql-alpha 0.1     # conservatism sweep
```

Useful switches:

```python
CFG.smdp = True                 # γ^τ semi-Markov ablation
CFG.reweight_prevalence = False # see how wrong things go without it
CFG.impute_terminal_action = False  # drop the 436 imputed terminals
CFG.n_fqe_seeds = 5             # FQE is seed-sensitive; raise for reporting
CFG.image_features = Path(...)  # slot in image embeddings when available
```

### Outputs

| File | Contents |
|---|---|
| `outputs/buffer.npz` | States, transitions, rewards, splits, π̂_b |
| `outputs/{algo}_seed{n}.pt` | Trained agent |
| `outputs/train_history.json` | Per-eval diagnostics and FQE |
| `outputs/eval_{split}.csv` | Full policy comparison |
| `outputs/simulator_policies.csv` | Counterfactual schedule comparison |
| `outputs/simulator_threshold.csv` | Optimal policy vs `c_e` |
| `outputs/simulator_topk_params.csv` | Top-10 calibrations for robustness checks |

---

## 11. Results so far

### 11.1 Off-policy evaluation cannot rank interval policies

Five FQE refits per policy, both discounting conventions:

| Mode | Policy | FQE | SD | Range across seeds |
|---|---|---|---|---|
| MDP | standard of care | −3.600 | 3.351 | −8.84 … −0.07 |
| MDP | biennial | −3.250 | 2.106 | −6.97 … −1.18 |
| MDP | triennial | −2.081 | 2.319 | −5.24 … −0.002 |
| SMDP | standard of care | −3.168 | 4.139 | −11.42 … −0.75 |
| SMDP | biennial | −1.733 | 0.868 | −3.40 … −1.03 |
| SMDP | triennial | −4.737 | 9.461 | −23.66 … −0.003 |

Typical seed SD (2.59 MDP, 4.82 SMDP) **exceeds the entire spread of means**
(1.52, 3.01). Combined with **ESS ≈ 1.2 out of 1,024 trajectories**, which makes
WIS/PDWIS informationless, the conclusion is blunt: **no off-policy estimator in
this pipeline can currently separate a 17-month policy from a 36-month one.**

Do not quote single-fit FQE rankings. Two contradictory "findings" were produced
that way during development and neither survived averaging.

### 11.2 The recall decision needs images

| | Sensitivity |
|---|---|
| Clinicians' observed recalls | 0.76 |
| Learned policy (tabular state only) | 0.00 |

No CSAW-CC DICOMs are on this machine. That gap *is* the value of the mammogram:
density and history cannot tell you a cancer is present at this exam. Interval
baselines therefore hold recall at the observed standard of care, so the interval
comparison stays apples-to-apples.

### 11.3 The simulator does rank policies

**Calibration fit** (weighted loss 0.0071, sojourn 2.50 y — inside the literature
range):

| Target | Observed | Simulated |
|---|---|---|
| P(screen-detected \| cancer) | 0.707 | 0.693 |
| Node+ \| screen-detected | 0.229 | 0.208 |
| Node+ \| interval | 0.341 | 0.303 |
| >15mm \| screen-detected | 0.408 | 0.411 |
| >15mm \| interval | 0.470 | 0.519 |
| Control recall rate | 0.015 | 0.017 |

**Dose–response** — monotone and clinically coherent in every column:

| Schedule | Value | Screens/woman/decade | Interval cancers /1000 | Mean size at dx |
|---|---|---|---|---|
| 6 months | −0.1389 | 17.3 | 8.0 | 8.2mm |
| 12 months | −0.1525 | 8.7 | 14.2 | 12.6mm |
| 18 months | −0.1656 | 5.8 | 21.2 | 16.8mm |
| 24 months | −0.1689 | 3.9 | 27.7 | 21.0mm |
| 36 months | −0.1977 | 2.9 | 38.3 | 28.5mm |
| standard of care | −0.1668 | 4.9 | 23.8 | 18.8mm |

Mean sizes are now clinically plausible (8–28mm). Gaps clear Monte-Carlo noise
comfortably, unlike FQE.

**The result is the threshold sweep, not any single row:**

| `c_e` | days of perfect health per exam | Optimal |
|---|---|---|
| 0.0002 – 0.002 | 0.07 – 0.73 | 6 months |
| 0.005 – 0.010 | 1.82 – 3.65 | 24 months |

> **Caveat worth reporting.** The fitted `log_growth_sd` is 0.051 — essentially
> *homogeneous* growth, sitting on the search floor. So in the calibrated model
> interval cancers arise from unlucky **onset timing** rather than from fast
> growers outrunning the schedule, which is not the mechanism originally assumed
> in §9. The fit is good and plausible either way, but the mechanism claim should
> be softened or the growth-heterogeneity floor investigated.

---

## 12. Known problems and open work

**Blocking for publication**

1. **Cost parameters are placeholders.** `c_e`, `c_r` and the `U(σ)` magnitudes
   need literature citations. The *ordering* of `U(σ)` is data-supported; the
   levels are not.
2. **OPE is too noisy to support any interval claim** (§11.1). Options: cross-fitted
   FQE over all splits rather than the 15% test set, FQE ensembling, more seeds.
3. **Calibration is under-determined** — 6 targets, 10 free parameters. The top-10
   parameter sets are saved but conclusions have not been re-run across them.
4. **`density_rule` baseline is unstable** (FQE −126 ± 156) despite being the
   intended hardest non-RL comparator. Needs diagnosis.

**Needed for the full design**

5. **Image encoder** (METHODOLOGY §4, §10.1) — ConvNeXt-T at 1024×832 with masked
   multi-view fusion. Blocked on obtaining the DICOMs; note only ~28 GB free disk.
6. **Explainability** (METHODOLOGY §14) — Grad-CAM validated against the curators'
   radiologist annotations. Blocked on images. Policy visualisation and Q-gap
   analysis are tabular-only and could be done now.
7. **PPO in the simulator**, for an on-policy comparison against offline CQL.
8. **Multi-seed everything.** Publication needs ≥5 seeds throughout.

**Data/ethics**

9. CSAW-CC's CC BY licence carries a **binding ICMJE condition to invite the data
   curators (Fredrik Strand, Karolinska Institutet) as co-authors.** Contact them
   well before submission.
10. 30% of the cohort is held back by the curators and is unavailable.

---

## 13. Traps that have already bitten

Recorded because each one silently produced plausible-looking wrong answers.

| Trap | Symptom | Fix |
|---|---|---|
| **Censoring treated as "healthy"** | Agent learns skipping screens is free | Censored terminals bootstrap; no terminal reward |
| **Label leak** | Implausibly high AUC | Assertion in `build_states` |
| **Case enrichment ignored** | Cost-effectiveness ~15× optimistic | `CFG.reweight_prevalence` |
| **Policy collapse** | Good mean reward, useless policy | Collapse gate every eval |
| **Single-fit FQE** | Confident but unreproducible rankings | Seed-average; report SD; check support |
| **Unconstrained FQE** | *Positive* values, though all rewards are costs | `−softplus` output constraint |
| **Per-round discounting in the simulator** | Frequent screening looks good for bookkeeping reasons | Simulator discounts by elapsed time |
| **Symptomatic cancer registered only at the next screen** | Long intervals never record their own misses; never-screen costs nothing | Presentation is a competing risk at every step |
| **Heterogeneous symptomatic *threshold*** | Interval cancers came out *smaller* than screen-detected — backwards | Size-dependent hazard; growth is the heterogeneity |
| **Proportion-only calibration targets** | All six targets matched with 19% of tumours >50mm | Explicit plausibility bounds |
| **`c_e` an order of magnitude too high** | 36-month screening "optimal" | Threshold sweep is the result |
| **Balanced sampler + weighted loss** | Total class collapse (inherited from Phase 1) | Use one or the other, never both |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Interval cancer** | A cancer that surfaces *between* screening rounds — a miss. The key quality metric |
| **Screen-detected** | Found at a screening exam |
| **Sojourn time** | How long a tumour is detectable-but-asymptomatic. The most influential parameter in any screening model |
| **Recall** | Called back for further work-up after an abnormal screen |
| **Behaviour policy (π_b)** | What the clinicians actually did — here, near-deterministic given age |
| **Positivity / overlap** | Every action the learned policy might take must have non-zero probability under π_b. Violated here |
| **Support** | Fraction of a policy's chosen actions that π_b gave non-trivial probability |
| **OPE** | Off-policy evaluation: estimating a policy's value from data generated by another |
| **FQE** | Fitted Q Evaluation. Our primary estimator; robust to deterministic π_b but seed-sensitive |
| **WIS / PDWIS** | (Per-decision) weighted importance sampling. Useless here — see ESS |
| **ESS** | Effective sample size. How many trajectories the importance weights really use |
| **CQL** | Conservative Q-Learning. Penalises out-of-support action values |
| **QALY** | Quality-adjusted life year. The reward's unit |
| **Censoring** | Observation stopped for reasons unrelated to outcome |
| **MDP / SMDP** | Markov / Semi-Markov Decision Process — one γ per round vs γ^(elapsed time) |
| **LIBRA** | The software that produced the density measurements |
