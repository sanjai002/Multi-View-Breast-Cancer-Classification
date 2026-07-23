# Personalized Breast Cancer Screening Using Deep Reinforcement Learning on Longitudinal Mammography Data

**Phase 2 — Methodology, MDP formulation, architecture, training and evaluation protocol**
Dataset: CSAW-CC (Karolinska / Cohort of Screen-Aged Women, case-control subset)

**Status:** the tabular pipeline, offline agent, evaluation harness and calibrated
simulator are implemented and running (`data.py`, `agent.py`, `train.py`,
`evaluate.py`, `simulator.py`). The image encoder of §4/§10.1 is *not* built — no
DICOMs are available. See [DOCUMENTATION.md](DOCUMENTATION.md) for the operational
reference and the current results, and §12.3–12.4 here for what the results do and
do not support.

> Note on two versions of the stage contrast quoted in this document: §0.5 reports
> **exam-level** rates (24.1% / 39.3% node-positive) while §8.3 and the reward code
> use **patient-level** rates taken from each patient's final exam (22.9% / 34.1%).
> The patient-level figures are the ones the reward is built on.

---

## 0. Read this section before anything else

Every empirical number below is reproduced by `python data_audit.py`. Four
properties of CSAW-CC materially change the project you described, and pretending
otherwise would make the paper unpublishable at IEEE TMI / MICCAI. I state them
first because they drive every downstream design decision.

### 0.1 The logged behaviour policy is near-deterministic → positivity is violated

| Inter-exam gap | 40–55 y (`x_age=1`) | 55+ y (`x_age=2`) |
|---|---|---|
| 1 year | 2 637 | 25 |
| 2 years | 4 750 | 6 356 |
| 3 years | 642 | 1 035 |
| 4+ years | 194 | 332 |

The screening interval is almost entirely explained by age bin. This is not an
accident: the Swedish national programme invites women aged 40–54 every 18 months
and 55–74 every 24 months, and at year-level resolution that produces exactly this
table. The `x_age` binning is itself the policy boundary.

**Consequence.** Offline RL requires *positivity*: every action the learned policy
might take must have non-zero probability under the behaviour policy in that state.
Here, a 6-month interval was **never** taken by anyone. No estimator — no amount of
DQN, PPO, or importance weighting — can recover the value of an action with zero
support. A Q-network will happily output a large value for "screen in 6 months"
because nothing in the loss ever contradicts it. That is extrapolation error, not a
finding.

This is the single most common fatal flaw in offline-RL-for-healthcare papers and
reviewers at MICCAI/TMI now look for it explicitly. The design in §5 and §9
addresses it head-on rather than hiding it.

### 0.2 Your requested action space is not identifiable from this data

| Requested action | Verdict | Reason |
|---|---|---|
| 6 months | **Remove** | Zero support; `exam_year` has year resolution only |
| 12 / 24 / 36 months | **Keep** | 2 662 / 11 106 / 1 677 observed transitions |
| 18 months | **Fold into 12/24** | Not distinguishable at year resolution |
| Immediate biopsy | **Remove** | No biopsy, BI-RADS, or pathology-workup variable exists |
| Additional imaging | **Remove** | Not recorded |
| Recall for work-up | **Keep** | `rad_recall` is a genuine recorded exam-level decision |
| "Annual screening" | Same as 12 months | — |

### 0.3 Age is two bins, not a number

`x_age ∈ {1, 2}` only. Your requested state variable "Age" carries ~1 bit. The
personalization signal must therefore come from **breast density** (`libra_percentdensity`,
continuous, mean 24.3%, SD 13.2), **imaging features**, and **screening history** —
not from age. This is fine (density is the strongest modifiable-free risk factor in
the literature after genetics) but it must be stated, and any claim of
"age-personalized" screening must be dropped.

### 0.4 The dataset is case-enriched — raw reward numbers will be ~15× too optimistic

Patient-level cancer prevalence here is **10.0%** (873 / 8 723). Real Swedish
screening detects roughly 0.4–0.6% per round. Controls were *randomly sampled*;
cases were *exhaustively* included. Any cancer-detection rate, cost-effectiveness
figure, or expected return computed on the raw file is a case-control artefact.
§8.5 specifies the prevalence-reweighting correction that makes reported numbers
population-meaningful.

### 0.5 What the data *does* give you — a real, un-fabricated reward signal

This is the good news, and it is what makes the project genuinely publishable.
`rad_timing` marks, for each exam, the time from that screen to diagnosis. Per-patient
sequences are perfectly monotone (0 violations across 873 case patients): they run
`(3, 3, …, 3, 1)` or `(3, 3, …, 3, 2)`.

- **`timing = 1`** — cancer found *at* that screen (screen-detected). 524 patients.
- **`timing = 2`** — cancer surfaced 60–729 days *after* that screen: an **interval
  cancer**, i.e. a miss. 217 patients ended this way, and only 19 of 267 such exams
  were recalled.

And critically, interval cancers are measurably worse at detection:

| | Node-positive | Invasive > 15 mm |
|---|---|---|
| Screen-detected (`timing=1`) | 120 / 497 = **24.1%** | 214 / 521 = **41.1%** |
| Interval (`timing=2`) | 96 / 244 = **39.3%** | 125 / 256 = **48.8%** |

**The cost of delayed detection is therefore *estimated from CSAW-CC itself*, not
assumed.** This directly satisfies your "do not fabricate medical assumptions"
requirement and is the empirical anchor of the reward function in §8.

### 0.6 Resulting design: a hybrid, and why

You cannot answer "what if we screened this woman at 6 months?" from data where
nobody was. There are exactly two scientifically valid responses, and the
established screening-policy literature (CISNET, Erasmus MISCAN, Wisconsin models)
uses the second:

1. **Restrict the policy to the data's support** — learn only over {12, 24, 36 mo}
   with a conservative offline algorithm, and evaluate with off-policy estimators.
2. **Build a calibrated latent disease-progression simulator**, fit its parameters
   to CSAW-CC's observed screen-detected/interval split and stage distribution, then
   train and counterfactually evaluate in the simulator.

**This project should do both**, and the contribution is precisely their
combination: a conservative offline policy that is *trustworthy where data exists*,
plus a calibrated simulator that *extrapolates transparently* where it does not,
with agreement between the two reported as a validation result. That framing is
novel enough for a methods paper and honest enough to survive review.

---

## 1. Dataset organization

### 1.1 Target hierarchy

```
patient (anon_patientid)
└── visit t  (ordered by exam_year)          ← screening round = MDP decision epoch
    ├── L_CC   ─┐
    ├── L_MLO   │ 4 standard views, guaranteed by the curators'
    ├── R_CC    │ inclusion criteria ("examinations that did not
    └── R_MLO  ─┘  include all four standard views" were excluded)
```

Filenames encode this directly:
`[anon_patientid]_20990909_[L|R]_[CC|MLO]_[n].dcm` — the date is a fixed
anonymization placeholder and carries no information; **use `exam_year` for
ordering, never the filename**.

### 1.2 Collapse to exam level

98 788 image rows → **24 694 exams** → 8 723 patients. `rad_*` variables are
exam-level and repeat identically across the 4 rows; `x_*` are patient-level;
`libra_*` are image-level and are averaged over the four views (and kept per-side
for laterality-specific features).

### 1.3 Quality control rules

| Check | Finding | Rule |
|---|---|---|
| Views per exam | 24 691 exams have 4; **3 exams have 8** | Duplicate acquisitions — keep the first of each (laterality, view) pair by filename sort; log the IDs |
| Missing `rad_recall` | 2 001 exams (8.1%) | Do **not** impute to 0. Add an explicit missingness indicator channel (§6) |
| Single-visit patients | 2 137 (24.5%) | Retain as length-1 episodes (they still carry one decision + terminal outcome); exclude from transition-dependent statistics |
| Density outliers | max 95.1% | Winsorize at 1st/99th percentile; do not drop |
| Post-diagnosis images | absent by construction | Treat end-of-sequence for cases as **diagnosis**, for controls as **administrative censoring** (§3.4) |

### 1.4 Patient-level splits

Split on `anon_patientid`, **stratified jointly on** (`x_case`, `x_age`, number of
visits) so that trajectory-length and outcome distributions match across folds:

`train 70% / val 15% / test 15%`, plus a **temporal held-out check**: retrain on
exams up to 2013 and test on 2014–2016 to demonstrate no calendar drift. Report
both. Never split on exams — leakage across a patient's own visits would be severe.

---

## 2. Image preprocessing

Applies to each of the 4 views independently.

1. **DICOM load.** Hologic FFDM. Apply `RescaleSlope`/`Intercept`, then the DICOM
   VOI LUT / window-centre-width if present. Honour
   `PresentationIntentType` — use *For Presentation* images if both exist.
2. **Photometric correction.** If `PhotometricInterpretation == MONOCHROME1`,
   invert so that dense tissue is bright.
3. **Breast segmentation / background removal.** Otsu threshold on the
   down-sampled image → largest connected component → morphological closing →
   convex-hull fill. This removes the air background and, importantly, the
   **pectoral muscle is *not* removed for MLO** (it is anatomical context; removing
   it is a common error that also destroys the Libra density reference frame).
   Crop to the breast bounding box.
4. **Laterality normalization.** Flip all `Right` images horizontally so the chest
   wall is on the left in every image. Record the flip as a feature-invariance
   assumption, not as data.
5. **Intensity normalization.** Per-image min–max to [0, 1] *within the breast
   mask only* (background zeros otherwise dominate the statistics), then
   z-score with dataset-level mean/std.
6. **CLAHE.** `clipLimit = 2.0`, `tileGridSize = 8×8`, applied **inside the mask
   only**. Caveat for the paper: CLAHE improves human reading but its benefit for
   CNNs is equivocal — run it as an **ablation**, do not assume it helps.
7. **Resize.** Mammographic lesions are small; aggressive down-sampling destroys
   them. Use **1024 × 832** (aspect-preserving pad) as the primary resolution.
   224×224 is unusable for this task and will silently cap your AUC.
8. **Augmentation (train only).** Horizontal flip, ±10° rotation, ±10% scale,
   small elastic deformation, brightness/contrast jitter ±10%. **No vertical flip**
   (anatomically invalid). Clip after every photometric op — unbounded photometric
   augmentation producing NaNs is a failure mode already encountered in Phase 1 of
   this project.

**Missing-view handling.** By curator inclusion criteria there are none, but the
encoder must still be robust: implement masked fusion (§10.1) so an absent view
contributes a learned `[MISSING]` embedding rather than zeros.

---

## 3. Longitudinal trajectory construction

### 3.1 From exams to episodes

For patient $i$ with exams at years $y_{i,1} < y_{i,2} < \dots < y_{i,T_i}$:

$$\tau_i = \big(s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, \dots, s_{i,T_i}, a_{i,T_i}, r_{i,T_i}, s_\text{term}\big)$$

The observed action $a_{i,t}$ is *inferred from the realised gap*:
$a_{i,t} = \Delta_{i,t} = y_{i,t+1} - y_{i,t}$. This is the standard "actions are
what was done" construction for offline medical RL.

**Observed transitions available: 15 971** across 6 586 multi-visit patients.

### 3.2 Trajectory-length handling

| Visits | Patients | Treatment |
|---|---|---|
| 1 | 2 137 | Length-1 episode: one action, immediate terminal reward. Keeps the 340 single-visit cancer cases, which are disproportionately screen-detected and thus informative |
| 2 | 1 250 | 1 transition + terminal |
| 3 | 1 770 | 2 transitions + terminal |
| 4 | 3 086 | 3 transitions + terminal |
| 5 | 477 | 4 transitions + terminal |
| 6 | 3 | 5 transitions; verify these are not the 8-view duplicates |

No padding to a fixed length. Use packed variable-length sequences and mask the
loss. Bootstrapping handles the horizon difference correctly; artificially padding
short trajectories would inject fake transitions.

### 3.3 Terminal conditions — three distinct kinds, do not conflate

$$
\text{terminal type} =
\begin{cases}
\textsf{DETECTED} & \text{last exam has } \texttt{rad\_timing}=1 \;(n=524)\\
\textsf{INTERVAL} & \text{last exam has } \texttt{rad\_timing}=2 \;(n=217)\\
\textsf{CENSORED} & \text{control, or case with } \texttt{rad\_timing}=3 \;(n=7850+132)
\end{cases}
$$

**This is the most important modelling detail in the whole pipeline.** A censored
patient did *not* "stay healthy" — we simply stopped observing them (study ended
2016, or images stop at diagnosis). Treating censoring as a confirmed-negative
outcome is the classic bug that makes a screening RL agent learn "screen less,
nothing bad happens." Bootstrap through censored terminals ($V(s_\text{term})$
estimated by the network); only `DETECTED`/`INTERVAL` terminals get a true
terminal reward with no bootstrap.

### 3.4 Censoring correction

Weight each trajectory by inverse probability of censoring, $1/\hat{G}(t \mid x)$,
with $\hat G$ from a Kaplan–Meier / Cox fit on (age bin, density quintile, calendar
year). Report results with and without IPCW as a robustness check.

---

## 4. Feature extractor (Phase 1 → Phase 2 interface)

### 4.1 Backbone recommendation

| Backbone | Verdict for CSAW-CC |
|---|---|
| ResNet-50 | Solid, well-understood baseline; weakest ImageNet-transfer of the four but robust at high resolution |
| DenseNet-121 | Popular in CheXNet-lineage work; memory-hungry at 1024px, no accuracy edge here |
| EfficientNet-B3/V2-S | Best FLOP-efficiency; compound scaling suits high-res inputs; BN statistics can be brittle with small batches at 1024px |
| **ConvNeXt-T / ConvNeXt-S** | **Recommended.** Large effective receptive field with LayerNorm (batch-size-robust — you will be running batch 4–8 at 1024px), modern ImageNet-22k weights, and it degrades gracefully at non-square aspect ratios |

**Recommendation: ConvNeXt-T at 1024×832, ImageNet-22k initialization**, with
EfficientNetV2-S as the ablation arm. Justify in the paper on (a) LayerNorm's
independence from batch size — decisive at this resolution, (b) 7×7 depthwise
kernels giving a receptive field able to span an architectural distortion, (c)
2022+ transfer weights.

Also cite and consider **GMIC** (Shen et al., NYU) as a mammography-specific
alternative: it uses a global network plus patch-level attention to retain
localisation at full resolution, and is a stronger prior than generic ImageNet
transfer for this exact task.

> **Migration note.** Phase 1 of this project trained a 3-class (Normal / Cancer /
> False-Positive) multi-view model on the **NLBS** dataset. CSAW-CC has different
> labels, different scanners (Hologic only), and different class semantics. The
> Phase 1 checkpoint is a **reasonable initialization but not a valid frozen
> extractor** — the encoder must be re-fine-tuned on CSAW-CC training patients only.

### 4.2 What Phase 2 consumes

Per exam, the encoder emits $\mathbf{z}_t \in \mathbb{R}^{512}$ (see §10.1 for the
fusion that produces it), plus the auxiliary risk head's calibrated probability
$\hat{p}_t$. **Freeze the encoder when training the RL head** (§11.1) — joint
training of a CNN through a bootstrapped TD loss on 16k transitions is unstable and
is not a fight worth having.

---

## 5. MDP formulation

### 5.1 Formulation: discrete-time MDP over screening rounds

The process is modelled as a finite-horizon MDP whose time index is the
**screening round**, not the calendar year:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

$$Q^*(s,a) = \mathbb{E}\Big[\, r(s,a) \;+\; \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \,\Big]$$

with one discount factor $\gamma$ applied per decision epoch. This matches how
screening is actually delivered — a woman is seen at rounds, and the policy's
output is "when is the next round" — and it keeps the formulation aligned with the
standard offline-RL machinery (CQL, FQE, WIS) without a semi-Markov correction
term in every estimator.

**Consequence, and an important caveat about measuring it.** The structural worry
is that per-round discounting makes a 36-month wait as cheap as a 12-month one, so
elapsed time carries no cost in the return and only the outcome term $U(\sigma)$
opposes long intervals. Conversely, under the SMDP the exponent $\gamma^{\tau}$
actively *rewards* waiting — a 3-year gap discounts all downstream harm three times
as hard, so deferring a cancer's cost can be worth more than the cancer costs.
Neither convention is neutral with respect to the quantity this project estimates.

**We cannot currently say which effect dominates empirically, and the paper must
not claim otherwise.** FQE on the 2 445-transition test split is severely
seed-limited. Five refits per policy, both conventions:

| mode | policy | FQE mean | SD | min … max across seeds |
|---|---|---|---|---|
| MDP | standard of care (≈17 mo) | −3.600 | 3.351 | −8.84 … −0.07 |
| MDP | biennial (24 mo) | −3.250 | 2.106 | −6.97 … −1.18 |
| MDP | triennial (36 mo) | −2.081 | 2.319 | −5.24 … −0.002 |
| SMDP | standard of care | −3.168 | 4.139 | −11.42 … −0.75 |
| SMDP | biennial | −1.733 | 0.868 | −3.40 … −1.03 |
| SMDP | triennial | −4.737 | 9.461 | −23.66 … −0.003 |

In **both** conventions the typical seed SD (2.59 MDP, 4.82 SMDP) exceeds the
entire spread of means across baselines (1.52 MDP, 3.01 SMDP). FQE as specified
cannot distinguish these interval policies at all on this split. Single-fit
rankings are not evidence: an earlier draft of this document quoted "triennial
best under SMDP (−0.004), worst under MDP (−5.24)" — those are the best and worst
single draws from two heavily overlapping distributions, and the apparent effect
vanished under averaging.

What the point estimates weakly suggest, without reaching significance, is that
under MDP discounting longer intervals score *better* (triennial has the best MDP
mean) — the direction predicted by the structural argument above, since per-round
discounting attaches no time cost to waiting.

Practical rules that follow:
- Always report FQE as mean ± SD over $\ge 5$ refits (`CFG.n_fqe_seeds`), never a
  single fit.
- Treat any baseline ordering whose gaps fall inside 1 SD as *undetermined*.
- Report both discounting conventions (`CFG.smdp`) side by side, since neither is
  neutral and the data does not currently adjudicate between them.

The residual identification problem in §12.3 (the logged transition kernel does not
respond to the action) is untouched by the discounting choice and still requires
the simulator arm.

The elapsed gap $\tau$ is still recorded on every transition. It enters as a
*state feature* (time since last screen), and setting `CFG.smdp = True` recovers
the semi-Markov variant $Q^*(s,a) = \mathbb{E}[r + \gamma^{\tau(a)}\max_{a'}Q^*(s',a')]$
as a one-line ablation. Reporting both is cheap and pre-empts the obvious reviewer
question about whether the interval recommendation is an artefact of the
discounting choice.

### 5.2 Partial observability

The true state — latent tumour presence, onset time, and growth rate — is never
observed. Formally this is a **POMDP**; we approximate the belief state with a
recurrent encoding of the observable history (§10.2):

$$b_t \approx f_\theta(o_1, a_1, \dots, o_{t-1}, a_{t-1}, o_t)$$

State the approximation explicitly in the paper; do not claim Markovianity of the
raw observation.

### 5.3 Components

**State** $s_t$ — see §6 (dimension 553).

**Action** $a_t = (d_t, w_t)$, factored:

$$d_t \in \{\textsf{12mo}, \textsf{24mo}, \textsf{36mo}\}, \qquad w_t \in \{\textsf{NO\_WORKUP}, \textsf{RECALL}\}$$

giving $|\mathcal{A}| = 6$. `RECALL` is included because `rad_recall` is a real
recorded decision with 3 116 positive instances. `DISCHARGE` (exit screening at
age 74) is *not* an action — it is age-determined, not policy-determined.

**Transition** $P(s_{t+1} \mid s_t, a_t)$ — from the empirical trajectories in the
offline setting; from the calibrated latent model in the simulator (§5.5).

**Discount factor.** $\gamma = 1/(1+\rho)$ with $\rho = 0.03$, the standard annual
discount rate for health-economic evaluation (WHO-CHOICE, NICE reference case), so

$$\boxed{\gamma = 0.9709 \text{ per year}}$$

This is a *justifiable, citable* choice, unlike the arbitrary 0.99 usually seen.
Run $\rho \in \{0, 0.015, 0.03, 0.05\}$ as a sensitivity analysis.

**Terminal condition.** §3.3.

**Observation space.** $\mathcal{O} = \mathbb{R}^{553}$, continuous, described next.

### 5.4 Episode dynamics (offline arm)

```
 s_1 ──a_1=(24mo, NO_WORKUP)──► s_2 ──a_2=(24mo, RECALL)──► s_3 ──► ⊗ DETECTED
 │                               │                          │
 r_1 = -c_exam            r_2 = -c_exam - c_fp       r_3 = -U(stage) + B_screen
```

### 5.5 Simulator arm (for counterfactual intervals)

A three-state latent progression model per breast, the standard structure in the
screening literature:

$$\textsf{Healthy} \xrightarrow{\;\lambda(x)\;} \textsf{Preclinical (screen-detectable)} \xrightarrow{\;\mu\;} \textsf{Clinical (symptomatic)}$$

- Onset hazard $\lambda(x)$ depends on age bin and density.
- Mean sojourn time (MST) in the preclinical state, $1/\mu$, is the key parameter;
  published estimates for this age range are ~2–4 years.
- Screening at time $t$ detects a preclinical tumour with sensitivity
  $\mathrm{Se}(\text{density}, \text{size})$.

**Calibration targets, taken from CSAW-CC directly:** the screen-detected /
interval split (524 vs 217 → 0.707), the node-positive rates (22.9% vs 34.1%), the
size distribution by timing (40.8% vs 47.0% > 15 mm), and the 1.5% control recall
rate. Fit by random search on growth rate, symptomatic-presentation size,
sensitivity and nodal-spread parameters, with an explicit prior constraining the
mean preclinical **sojourn time to 2–4 years** — fitting the summary targets alone
returns 5–6 years, which matches the stage contrast but implies a natural history
no reviewer would accept. Implemented in `simulator.py`; the top-10 parameter sets
are retained because 6 targets do not identify 9 parameters, and conclusions
should be checked across all of them rather than at the argmin.

**Implementation notes that matter.**

*Symptomatic presentation is screening-independent.* A tumour reaching symptomatic
size presents clinically whether or not a screening round follows. An early version
registered interval cancers only at the next screening visit, which meant
long-interval policies silently failed to record their own misses — and a policy
that never screened incurred no cancer cost at all. That inverts the entire
comparison.

*Discount by elapsed time, not by round.* The offline arm uses per-round MDP
discounting (§5.1), but that convention is invalid for cross-policy comparison in
the simulator: a 6-month policy generates ~19 rounds per decade against ~3 for a
triennial policy, so per-round discounting would shrink the frequent screener's
harm term by $\gamma^{19}$ vs $\gamma^{3}$ and favour frequent screening for
bookkeeping reasons alone. The simulator therefore always uses $\gamma^{t}$ with
$t$ in years, which is also the health-economics standard.

---

## 6. State representation

$s_t \in \mathbb{R}^{553}$, concatenating:

| Block | Dim | Source | Note |
|---|---|---|---|
| Fused image embedding $\mathbf{z}_t$ | 512 | ConvNeXt + view fusion (§10.1) | frozen during RL |
| Calibrated risk score $\hat{p}_t$ | 1 | auxiliary head | isotonic-calibrated on val |
| Age bin (one-hot) | 2 | `x_age` | only 1 bit available (§0.3) |
| Percent density, L / R | 2 | `libra_percentdensity` | z-scored |
| Dense area, L / R | 2 | `libra_densearea` | z-scored |
| Breast area, L / R | 2 | `libra_breastarea` | z-scored |
| Δ density since last visit | 2 | derived | density change is itself a risk signal |
| Visit index $t$, normalized | 1 | derived | |
| Time since last screen (years) | 1 | `exam_year` diff | 0 for first visit |
| Cumulative screens so far | 1 | derived | |
| Prior recalls: count, last, ever | 3 | `rad_recall` history | |
| Prior reader disagreement $\mathbb{1}[r_1 \neq r_2]$, last + count | 2 | `rad_r1`,`rad_r2` | a genuine uncertainty signal |
| Missingness indicators | 3 | `rad_recall`/`r1`/`r2` null flags | §1.3 |
| Previous action (one-hot) | 6 | derived | |
| Calendar year, normalized | 1 | `exam_year` | drift control |
| Prior-visit embedding delta $\lVert \mathbf{z}_t - \mathbf{z}_{t-1}\rVert$ + cosine | 2 | derived | **temporal change is the strongest longitudinal cue** |
| Recurrent belief carry (§10.2) | 8 | GRU hidden projection | |
| **Total** | **553** | | |

**Deliberately excluded — these would leak the label:** `x_case`,
`x_cancer_laterality`, `x_type`, `x_lymphnode_met`, `rad_timing`. These are
*outcome* variables known only retrospectively. They are used **exclusively to
compute rewards during training** and must never enter $s_t$. Enforce this with an
assertion in the state builder; it is the easiest way to accidentally publish an
AUC of 0.99.

"Previous cancer history" (your requested field) does not exist in CSAW-CC — cases
are *first-time* diagnoses by inclusion criterion, and imaging stops at diagnosis.
Drop it.

---

## 7. Action space

$$\mathcal{A} = \{\textsf{12mo}, \textsf{24mo}, \textsf{36mo}\} \times \{\textsf{NO\_WORKUP}, \textsf{RECALL}\}$$

encoded $a \in \{0,\dots,5\}$. Support per component (§0.1) is 2 662 / 11 106 /
1 677 transitions and 3 116 recalls — all adequately represented.

**Action masking.** Mask actions that are structurally invalid:
- If the patient would exceed age 74 at the next screen → episode terminates
  (programme exit), no interval action.
- In the offline arm, mask any action whose behaviour-policy propensity
  $\hat\pi_b(a \mid s) < \epsilon$ (use $\epsilon = 0.01$). Report how often masking
  binds — this is the honest way to show the reader where the data runs out.

**Simulator arm only:** the action set may be extended to
$\{6, 12, 18, 24, 36\}$ months, clearly labelled as *simulator-only,
extrapolative* results in a separate table. Never mix these numbers with the
offline-evaluated ones.

---

## 8. Reward function

### 8.1 Form

The reward is a **negative cost** in QALY-equivalent units, which makes it directly
interpretable to a clinical audience and lets the discount factor mean what it
normally means in health economics.

$$
r_t = -\Big[\; \underbrace{c_e}_{\text{exam}} \cdot \mathbb{1}[\text{screened}]
\;+\; \underbrace{c_r \cdot \mathbb{1}[w_t = \textsf{RECALL}, \, y_t = 0]}_{\text{false positive}}
\;+\; \underbrace{c_m \cdot \mathbb{1}[w_t = \textsf{NO\_WORKUP}, \, y_t = 1]}_{\text{missed at this screen}}
\;+\; \underbrace{U(\sigma_t) \cdot \mathbb{1}[\text{diagnosis in } (t, t+\tau]]}_{\text{outcome}}
\;\Big]
$$

### 8.2 The outcome term $U(\sigma)$ — estimated, not assumed

$\sigma$ is the stage at detection, constructed from CSAW-CC's own fields:

$$\sigma = g(\texttt{x\_type}, \texttt{x\_lymphnode\_met}) \in \{\textsf{in-situ}, \textsf{inv} \le 15\text{mm N0}, \textsf{inv} \le 15\text{mm N+}, \textsf{inv} > 15\text{mm N0}, \textsf{inv} > 15\text{mm N+}\}$$

$U(\sigma)$ is the expected QALY loss for that stage, taken from published
stage-specific breast-cancer survival (Swedish national registry / SEER), **not
invented**. Enter it as a table with a documented source per row and run a full
sensitivity analysis:

| $\sigma$ | 10-y survival (lit.) | $U(\sigma)$ (QALY loss) |
|---|---|---|
| in situ | ~98% | 0.5 |
| invasive ≤15 mm, N0 | ~95% | 1.0 |
| invasive ≤15 mm, N+ | ~85% | 2.5 |
| invasive >15 mm, N0 | ~82% | 3.0 |
| invasive >15 mm, N+ | ~68% | 5.5 |

*(Populate the final column from your chosen citation before submission; the
values shown are placeholders with the right ordering and rough magnitude. The
**ordering** is what the policy is sensitive to, and the ordering is empirically
supported by §0.5.)*

### 8.3 The delay cost is derived from data, not stipulated

Rather than hand-setting a penalty for late detection, estimate the stage
distribution as a function of delay from CSAW-CC:

$$\Pr(\sigma \mid \text{detected after delay } \Delta) \quad\text{fitted from the } (timing{=}1) \text{ vs } (timing{=}2) \text{ contrast}$$

Then the expected delay cost follows:

$$C_\text{delay}(\Delta) = \mathbb{E}_{\sigma \sim \Pr(\cdot \mid \Delta)}\big[U(\sigma)\big] - \mathbb{E}_{\sigma \sim \Pr(\cdot \mid 0)}\big[U(\sigma)\big]$$

Using the node-positivity contrast (24.1% → 39.3%) and size contrast
(41.1% → 48.8%) over a mean interval-cancer delay of ~1.1 years, this yields a
delay cost on the order of **0.6–0.9 QALY per year of delay** — a number your
paper can defend with a table rather than a citation-free assertion. This is the
methodological core of the reward design and should be a figure in the paper.

### 8.4 Cost parameters

| Symbol | Meaning | Value | Basis |
|---|---|---|---|
| $c_e$ | one screening exam | 0.01 | procedure cost + brief anxiety, converted to QALY |
| $c_r$ | false-positive recall | 0.05 | documented recall-anxiety literature (persists ~6–12 mo) |
| $c_m$ | miss at a screen where cancer was detectable | via $C_\text{delay}$ | §8.3 |
| $\rho$ | discount rate | 0.03/yr | health-economic reference case |

All four are **swept** in the sensitivity analysis. A screening-policy paper whose
conclusions are not shown to be stable across the cost ratio $c_r / c_m$ will not
pass review — that ratio *is* the clinical trade-off.

### 8.5 Prevalence reweighting (essential — see §0.4)

Case-control sampling means empirical expectations are biased. Reweight each
patient by

$$
\omega_i = \begin{cases}
\dfrac{\pi_\text{pop}}{\pi_\text{sample}} & \text{if } \texttt{x\_case}=1 \\[2ex]
\dfrac{1-\pi_\text{pop}}{1-\pi_\text{sample}} & \text{if } \texttt{x\_case}=0
\end{cases}
\qquad \pi_\text{sample} = 0.100, \;\; \pi_\text{pop} \approx 0.006
$$

Apply $\omega_i$ in the TD loss, in all reported returns, and in every OPE
estimator. Without it, the agent believes 1 in 10 women has cancer and will
over-screen catastrophically.

### 8.6 Reward normalization

Do **not** z-score rewards — it destroys the QALY interpretation. Instead scale by
a fixed constant $\kappa = 1/\max_\sigma U(\sigma)$ so returns land in roughly
$[-1, 1]$, and report both raw-QALY and scaled numbers. Use reward clipping only
as an ablation, never in the headline model.

---

## 9. RL algorithm selection

### 9.1 The decisive question is not "which algorithm" but "does it work offline"

| Algorithm | Usable here? | Why |
|---|---|---|
| **PPO** | ❌ | On-policy. Requires environment interaction. Usable **only** in the simulator arm |
| **A2C** | ❌ | Same — on-policy |
| **SAC** | ❌ | Designed for continuous actions; off-policy but still needs interaction; discrete variant is not the natural fit |
| **DQN** | ⚠️ | Off-policy but suffers severe extrapolation error offline; overestimates OOD actions. Baseline / ablation only |
| **Double DQN** | ⚠️ | Fixes maximization bias, *not* extrapolation error. Ablation |
| **Dueling DQN** | ⚠️ | Better value estimates under near-identical action values — which is exactly our regime. Good as an architectural component, insufficient alone |
| **Rainbow** | ⚠️ | Its components (PER, n-step, distributional, noisy nets) mostly target exploration, which is irrelevant offline. Distributional (C51/QR) *is* valuable for risk-sensitivity |
| **CQL** | ✅ | **Chosen.** Explicitly penalizes OOD action values — directly addresses §0.1 |
| **BCQ / discrete BCQ** | ✅ | Strong alternative; constrains the policy to behaviour-supported actions. Not implemented |
| **IQL** | ✅ | Avoids querying OOD actions entirely; simplest to tune. Not implemented |

### 9.2 The committed model

**Conservative Q-Learning (CQL) with a Double, Dueling, distributional (QR-DQN)
backbone, in the MDP formulation of §5.1 — and nothing else.** `agent.py`
implements this one algorithm; the project settled on CQL and the DQN-family
ablation ladder was removed from the codebase.

The CQL objective adds a conservatism term to standard TD learning:

$$
\mathcal{L}_\text{CQL}(\theta) = \alpha \underbrace{\mathbb{E}_{s \sim \mathcal{D}}\Big[\log \sum_{a} \exp Q_\theta(s,a) - \mathbb{E}_{a \sim \hat\pi_b}\big[Q_\theta(s,a)\big]\Big]}_{\text{push down OOD actions, push up observed ones}} + \underbrace{\tfrac{1}{2}\mathbb{E}_{\mathcal{D}}\Big[\big(Q_\theta(s,a) - \mathcal{B}^\pi \bar{Q}(s,a)\big)^2\Big]}_{\text{standard TD error}}
$$

with the Bellman operator

$$\mathcal{B}^\pi \bar Q(s,a) = r + \gamma \, \bar Q\big(s', \arg\max_{a'} Q_\theta(s',a')\big) \cdot \mathbb{1}[\text{not terminal-observed}]$$

Rationale to state in the paper: with a near-deterministic behaviour policy, the
learned $Q$ is unconstrained off-support; CQL's $\alpha$ term yields a *lower
bound* on the true value, so reported improvements are conservative rather than
optimistic. Sweep $\alpha \in \{0.1, 1, 5, 10\}$ and show the policy's deviation
from the clinical standard shrinks as $\alpha$ grows — that plot is the paper's
credibility argument.

**Simulator arm.** The simulator ranks schedules by direct policy rollout, not by
a learned on-policy agent (the earlier plan named PPO here). Training a
simulator-side PPO agent and checking that it agrees with offline-CQL remains a
worthwhile extension, but it is not currently implemented.

**On the ablation ladder.** An earlier version of this design proposed a full
comparison table — Behaviour Cloning → DQN → Double DQN → Dueling Double DQN →
+ QR distributional → + CQL — each scored under the same OPE protocol. The code
for the intermediate rungs was removed once CQL was chosen; it is recoverable
from git history if a reviewer asks for the table. What remains as an *internal*
ablation is the CQL conservatism sweep ($\alpha \in \{0.1, 1, 5, 10\}$), which is
the ablation that actually matters for the OOD-extrapolation argument.

**Implementation note.** CQL is implemented directly in `agent.py` (~30 lines on
top of a QR-DQN backbone), not via d3rlpy. This keeps the project dependency-free
beyond numpy/pandas/scikit-learn/torch, which matters on the CPU-only target
machine. d3rlpy remains the sensible route if BCQ or IQL are ever added.

---

## 10. Neural architecture

```
                    ┌────────────────────────────────────────────────────┐
   VISIT t          │  L_CC   L_MLO   R_CC   R_MLO   (1024 × 832, 1-ch)  │
                    └───┬───────┬───────┬───────┬────────────────────────┘
                        │       │       │       │
                     ┌──▼───────▼───────▼───────▼──┐
                     │  ConvNeXt-T  (SHARED wts)   │   ImageNet-22k init
                     │  → 768-d per view           │   frozen during RL
                     └──┬───────┬───────┬───────┬──┘
                        │       │       │       │
                     ┌──▼───────▼──┐ ┌──▼───────▼──┐
                     │  CC branch  │ │  MLO branch │   view-specific
                     │  self-attn  │ │  self-attn  │
                     └──────┬──────┘ └──────┬──────┘
                            └───────┬───────┘
                     ┌──────────────▼──────────────┐
                     │  Masked cross-view fusion   │  handles missing views
                     │  (bilateral + bi-view MHA)  │  via [MISSING] token
                     └──────────────┬──────────────┘
                                    │  z_t ∈ ℝ^512
        ┌───────────────────────────┼──────────────────────────┐
        │                           │                          │
   ┌────▼─────┐            ┌────────▼─────────┐          ┌─────▼──────┐
   │ aux risk │            │  tabular block   │          │  Grad-CAM  │
   │ head p̂_t │            │  age/density/    │          │  (§14)     │
   │ (BCE)    │            │  history → 41-d  │          └────────────┘
   └────┬─────┘            └────────┬─────────┘
        └──────────────┬────────────┘
                       │  s_t ∈ ℝ^553
        ═══════════════▼════════════════  (across visits t = 1..T)
                ┌──────────────────┐
                │  TEMPORAL ENCODER│   2-layer GRU, hidden 256
                │  (belief state)  │   + learned Δt embedding added at each step
                └────────┬─────────┘   packed variable-length; causal (no lookahead)
                         │  b_t ∈ ℝ^256
                 ┌───────▼────────┐
                 │  RL HEAD       │
                 │  MLP 256→256   │
                 └───┬────────┬───┘
              ┌──────▼──┐  ┌──▼──────────┐
              │ V(s)    │  │ A(s,a)      │   dueling
              │ 256→N_q │  │ 256→6×N_q   │   N_q = 32 quantiles (QR)
              └────┬────┘  └──────┬──────┘
                   └───────┬──────┘
                    Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
                           │
                    ┌──────▼───────┐
                    │ DECISION HEAD│  argmax_a Q  →  (interval, recall)
                    │ + action mask│
                    └──────────────┘
```

### 10.1 Masked multi-view fusion

For view set $\mathcal{V} = \{LCC, LMLO, RCC, RMLO\}$ with availability mask
$m_v \in \{0,1\}$:

$$\mathbf{h}_v = \text{ConvNeXt}(x_v) \cdot m_v + \mathbf{e}_\text{missing} \cdot (1 - m_v)$$

$$\mathbf{z}_t = \text{MHA}\big(\{\mathbf{h}_v\}_{v\in\mathcal{V}}\big) \; \Vert \; \big(\mathbf{h}_{L} - \mathbf{h}_{R}\big)$$

The bilateral **difference** term is deliberate: radiologists read
left-vs-right asymmetry, and it is one of the strongest available cues.

### 10.2 Temporal encoder

GRU over visits, with elapsed time injected rather than assumed uniform:

$$b_t = \text{GRU}\big(b_{t-1}, \; [\,s_t \; \Vert \; \phi(\Delta_{t-1}) \; \Vert \; \text{onehot}(a_{t-1})\,]\big)$$

where $\phi$ is a sinusoidal embedding of the elapsed years. Use a GRU rather than
a Transformer: maximum sequence length is 6, and a Transformer has nothing to
attend over while adding parameters you cannot fit with 16k transitions.

---

## 11. Training pipeline

### 11.1 Three stages

**Stage A — encoder pretraining (supervised).** Train ConvNeXt + fusion on
per-exam cancer-presence labels with focal loss ($\gamma = 2$, $\alpha$ balanced).
Progressive unfreezing, differential LRs (backbone $10^{-5}$, head $10^{-3}$),
cosine schedule, EMA, bf16 AMP. Early-stop on **val AUC**.

> ⚠️ **Carry-forward from Phase 1 of this project:** never combine a balanced
> sampler with a class-weighted loss — that combination caused total class
> collapse. Choose exactly one. Focal loss alone is the recommended setting.

**Stage B — freeze encoder, cache states.** Precompute $\mathbf{z}_t$ for all
24 694 exams → a single `states.npy`. RL training then runs in minutes per epoch
instead of hours, and the TD loss never has to backprop through a CNN.

**Stage C — offline RL.** Train CQL over the cached transition buffer.

### 11.2 Experience replay

There is **no exploration and no environment**: the "replay buffer" is the fixed
dataset of 15 971 transitions, loaded once. Sample uniformly at the *trajectory*
level (not the transition level) so that recurrent hidden states are valid; within
a batch, feed whole sequences.

Prioritized replay (PER) is **not recommended** offline — it re-weights toward
high-TD-error transitions, which offline are disproportionately the OOD ones CQL is
trying to suppress. If you include it, do so as an ablation and expect it to hurt.

### 11.3 Target network

Polyak-averaged: $\bar\theta \leftarrow \eta\theta + (1-\eta)\bar\theta$ with
$\eta = 5\times10^{-3}$ every step (smoother than hard periodic copies and better
behaved in small-data offline settings).

### 11.4 Exploration

**None, by construction.** In the offline arm the dataset is fixed; there is
nothing to explore. Say this explicitly in the paper — reviewers will check
whether you understood it. In the simulator arm, use entropy-regularized PPO with
coefficient $0.01 \to 0.001$ annealed.

### 11.5 Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Optimizer | AdamW, weight decay $10^{-4}$ | |
| LR (RL head) | $3\times10^{-4}$, cosine decay | |
| LR (encoder, Stage A) | $10^{-5}$ backbone / $10^{-3}$ head | |
| Batch size | 64 trajectories (Stage C) / 8 exams (Stage A, 1024px) | |
| Gradient clip | 10.0 (global norm) | |
| CQL $\alpha$ | 5.0 (sweep 0.1–10) | |
| Quantiles $N_q$ | 32 | |
| Target update $\eta$ | $5\times10^{-3}$ | |
| Training steps | 200k gradient steps, early stop on val FQE | |
| $\gamma$ | 0.9709 per **screening round** | §5.1, §5.3 |
| Seeds | **5 minimum**, report mean ± SD | non-negotiable for publication |

### 11.6 Model selection without a simulator

You cannot roll out the policy to pick a checkpoint. Select on **Fitted Q
Evaluation (FQE) on the validation split**, never on training TD loss. Report the
selection criterion — checkpoint-selection leakage is a known failure in offline RL
papers.

### 11.7 The collapse mode to watch for

The Phase 1 lesson generalizes: a policy that always outputs `36mo, NO_WORKUP`
will achieve excellent mean reward on a cohort that is 90% healthy while being
clinically useless. **Monitor cancer-specific metrics — detection rate among
cases, and interval-cancer rate — at every checkpoint, not aggregate return.**
Build this into the training loop as a hard gate, not a post-hoc check.

---

## 12. Evaluation protocol

### 12.1 Off-policy evaluation (the primary evidence)

Report **all four**, since disagreement between them is itself diagnostic:

1. **Weighted importance sampling (WIS)** with the estimated behaviour policy
   $\hat\pi_b$ (a multinomial logistic model on age bin, density, visit index —
   report its own calibration and AUC).
   $$\hat{V}_\text{WIS} = \frac{\sum_i w_i \sum_t \gamma^{t_i} r_{i,t}}{\sum_i w_i}, \qquad w_i = \prod_t \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\hat\pi_b(a_{i,t} \mid s_{i,t})}$$
2. **Per-decision WIS** — lower variance.
3. **Fitted Q Evaluation (FQE)** — the most reliable in practice for this regime.
4. **Doubly robust (DR / WDR)** — consistent if either $\hat\pi_b$ or $\hat{Q}$ is correct.

**Mandatory diagnostic 1 — ESS.** Report the **effective sample size**
$\text{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$. With a near-deterministic behaviour
policy, ESS will be small; if ESS < 100, the IS-based estimates are not
trustworthy and you must say so and lean on FQE. Reviewers *will* ask.

> Measured on the test split: **ESS ≈ 1.3 out of 1 024 trajectories.** The
> positivity violation of §0.1 is not hypothetical — WIS and PDWIS carry
> essentially no information on this dataset. FQE is the primary estimator.

**Mandatory diagnostic 2 — support.** FQE degrades off-support too, just more
quietly. For each evaluated policy report

$$\text{support} = \frac{1}{n}\sum_i \mathbb{1}\big[\hat\pi_b(a^{\pi}_i \mid s_i) \ge \epsilon\big], \qquad \epsilon = 0.01$$

and refuse to rank policies whose support is low. Empirically this matters a
great deal here: unconstrained FQE returned **positive** values for annual
screening (impossible — every reward is a non-positive cost) until the estimator
was constrained to $Q \le 0$, and even then annual vs. triennial estimates span
two orders of magnitude because neither is well represented in the data. A
results table without a support column invites exactly the over-claim this
design is built to avoid.

### 12.2 Clinical metrics (per your list)

Computed at the exam level for the recall component and the patient level for
outcomes, all with prevalence reweighting (§8.5) and 1 000-sample bootstrap CIs:

| Metric | Definition / note |
|---|---|
| Sensitivity | detected cancers / all cancers |
| Specificity | correct no-recalls / all non-cancer exams |
| ROC AUC | of the auxiliary risk head $\hat p_t$ (the *policy* is discrete; AUC applies to the score, not the policy) |
| Precision / PPV | PPV of recall — the number clinicians actually care about |
| Recall rate | fraction of exams recalled; Swedish benchmark ≈ **2.6%** — a policy recalling 15% is not deployable regardless of sensitivity |
| F1 / Balanced accuracy | reported for completeness |
| **Cancer detection rate** | per 1 000 screens, reweighted to population prevalence |
| **Interval cancer rate** | per 1 000 screens — *the* key screening-quality metric, and the one your reward is built around |
| **Mean screening interval** | months; the policy's headline output |
| **Screens per cancer detected** | efficiency |
| **Unnecessary recalls** | false positives per 1 000 screens |
| Stage distribution at detection | the clinical benefit endpoint |

### 12.3 A structural limit of the offline arm (measured, not anticipated)

Running the pipeline surfaces something that must be stated plainly in the paper,
because a reviewer will find it otherwise.

In offline RL from observational data, the transition kernel $P(s' \mid s, a)$ is
whatever was logged. Lengthening the screening interval therefore **cannot** raise
the modelled probability of a cancer advancing, because the logged data contains no
such counterfactual: the action barely varied (§0.1), and outcome costs $U(\sigma)$
attach only to the 741 observed diagnosis terminals. Nothing in the offline
objective encodes the causal mechanism "wait longer → detect later → worse stage",
even though §0.5 shows that mechanism is real in this very dataset.

This is compounded by, but independent of, the discounting convention (§5.1) and by
the estimator noise documented there. Whatever the discounting choice, the offline
arm cannot demonstrate that a shorter interval *causes* better outcomes; at best it
ranks policies by observed cost accrual near the data, and on this test split it
currently cannot even do that with confidence.

Two consequences for the write-up:

1. **The offline arm can rank policies near the data** (standard-of-care ≈17 mo vs
   biennial 24 mo vs learned ≈23 mo) but **cannot establish that shorter intervals
   are beneficial.** Any such claim requires the causal mechanism, which is exactly
   what the calibrated progression simulator of §5.5 supplies. This empirically
   vindicates the hybrid design rather than undermining it.
2. **CQL's support constraint is doing the clinical work.** The learned policy sits
   at 100% support and ≈22 months rather than running off to the degenerate
   "never screen" optimum. Report the learned policy's support fraction next to its
   value; that number, not the return, is what makes it trustworthy.

Do not present the offline arm alone as evidence for an interval recommendation.

### 12.4 Simulator results — where policy conclusions actually come from

`simulator.py` compares policies by direct rollout, with no fitted estimator in
the loop, and unlike FQE it separates them cleanly: every gap to the best policy
is 2.8–13.2 standard errors over 5 replications. At the calibrated parameters and
$c_e = 0.001$:

| policy | value | vs 18mo (SE) | screens/woman/decade | interval cancers /1000 | mean size at dx |
|---|---|---|---|---|---|
| **18 mo** | **−0.1114** | — | 5.9 | 22.7 | 10.7 mm |
| 12 mo | −0.1152 | 2.8 | 9.7 | 16.8 | 9.1 mm |
| 6 mo | −0.1168 | 4.3 | 19.4 | 8.3 | 6.8 mm |
| standard of care | −0.1199 | 5.8 | 5.4 | 25.5 | 11.4 mm |
| 36 mo | −0.1246 | 8.6 | 2.9 | 34.7 | 13.7 mm |
| 24 mo | −0.1284 | 13.2 | 4.9 | 28.1 | 12.0 mm |

The dose–response is monotone and clinically coherent in the *epidemiological*
columns — halving the interval roughly halves the interval-cancer rate (34.7 → 8.3
per 1 000) and shrinks mean tumour size at detection (13.7 → 6.8 mm). That is the
causal mechanism §12.3 showed the offline arm cannot represent.

**Report the threshold sweep, not a single optimum.** The value ordering is
governed by $c_e$, the per-exam disutility, which is a placeholder:

| $c_e$ (QALY) | days of perfect health per exam | optimal policy |
|---|---|---|
| 0.0002 – 0.0005 | 0.07 – 0.18 | 6 months |
| 0.001 – 0.005 | 0.36 – 1.82 | **18 months** |
| 0.010 | 3.65 | 36 months |

An 18-month interval — the Swedish programme's own choice for women under 55 — is
optimal across the plausible middle of this range, which is a meaningful external
validation of the calibrated model. Note that the project's original default
$c_e = 0.01$ implied 3.65 days of perfect health lost per mammogram, roughly an
order of magnitude too high, and by itself selected 36 months.

**Calibration (revised).** Weighted loss 0.0071 with all six targets matched and
a mean preclinical sojourn of 2.50 years, inside the literature range. Simulated
mean tumour size at detection now runs 8-28 mm across schedules, which is
clinically plausible; the dose-response is monotone in value, interval-cancer
rate and size simultaneously.

An earlier calibration matched all six targets while producing a mean size of
41 mm, 19% of tumours over 50 mm and a 95th percentile of 273 mm. Because every
CSAW-CC target is a *proportion*, target-matching alone does not constrain the
size distribution; the explicit plausibility bounds in §5.5 are what close that
loophole. This is worth a sentence in the paper -- it is a general hazard when
calibrating to registry summary statistics.

**Remaining calibration caveats.**
1. Fitted growth heterogeneity collapses to the search floor
   ($\sigma_{\log g} = 0.051$), so in the calibrated model interval cancers arise
   from unlucky onset *timing* rather than from fast growers outrunning the
   schedule. The fit is good either way, but the mechanistic claim should be
   softened accordingly.
2. Six targets do not identify ten parameters.
   `outputs/simulator_topk_params.csv` retains the top-10 fits; conclusions must
   be re-run across them before any robustness claim.

### 12.5 The efficiency frontier — your key figure

Sweep the cost ratio $c_r/c_m$ and plot **cancer detection rate vs. total screening
burden**, with the learned policy's frontier against all baselines as points. A
policy is a contribution if it **dominates** annual/biennial screening: equal
detection at fewer screens, or more detection at equal screens. This single figure
carries the paper.

---

## 13. Baselines

| Baseline | Implementation |
|---|---|
| **Annual** | always `12mo` |
| **Biennial** | always `24mo` |
| **Swedish standard of care** | `18mo` if age bin 1 else `24mo` — the actual behaviour policy; the most important comparator |
| **Behaviour cloning** | supervised fit of $\hat\pi_b$; separates "learned the data" from "learned a better policy" |
| **Random** | uniform over $\mathcal{A}$; sanity floor |
| **Density-stratified risk rule** | `12mo` if percent density > 75th pct else `24mo`; a clinically plausible non-RL heuristic and the *hardest* baseline to beat |
| **Tyrer-Cuzick / Gail-style risk model** | ⚠️ cannot be computed — CSAW-CC lacks family history, BMI, parity, menarche age. **State this as a limitation rather than fabricating inputs.** Use the density rule as the risk-model proxy |

The density-stratified rule is the honest bar. If CQL cannot beat it, that is a
publishable negative result — say so rather than tuning until it wins.

---

## 14. Explainability

1. **Grad-CAM / Grad-CAM++** on the ConvNeXt encoder at the final stage, overlaid
   on the breast mask. Validate against the curators' **radiologist PNG
   annotations** — CSAW-CC ships expert lesion annotations, including dots marking
   where cancer later arose on priors. Report **pointing game accuracy** and IoU.
   This turns explainability from a qualitative figure into a *quantitative
   result*, which is rare and reviewer-pleasing.
2. **View-attention weights** from the fusion block — which of the four views drove
   the decision, and whether the model attends to the cancer-bearing laterality.
3. **Policy visualization** — a 2-D heatmap of the chosen interval over
   (percent density × visit index), faceted by age bin. Immediately legible to a
   clinical reader.
4. **Feature importance** — SHAP over the 41-d tabular block, plus a leave-one-block-out
   ablation on the 553-d state to show which blocks the policy actually uses.
5. **Q-value gap analysis** — plot $Q(s, a^*) - Q(s, a_\text{SoC})$ across the
   cohort. Where the gap is near zero, the agent is saying "the standard of care is
   already right"; the interesting cases are the tails. This directly answers "for
   whom does personalization matter?"
6. **Counterfactual trajectories** for the 217 interval-cancer patients: would the
   learned policy have screened them earlier? This is the most compelling single
   result available in this dataset — a concrete, patient-level narrative of
   averted misses.

---

## 15. Publication packaging

**Target venues.** MICCAI (8-page, methods-forward) or IEEE JBHI / TMI (longer,
room for the health-economic framing). The offline-CQL + calibrated-simulator combination
is the methodological novelty; the CSAW-CC-derived delay cost (§8.3) is the
empirical novelty.

**Required figures.**
1. Pipeline overview (data → encoder → MDP → policy)
2. Architecture diagram (§10)
3. Trajectory/timing structure of CSAW-CC (from `data_audit.py`)
4. Delay-cost estimation from the screen-detected vs interval contrast (§8.3)
5. **Efficiency frontier** (§12.3) ← the headline
6. Policy heatmap over density × visit (§14.3)
7. Grad-CAM vs radiologist annotations (§14.1)
8. Sensitivity analysis over $(\alpha, \rho, c_r/c_m)$

**Required algorithm boxes.** (i) trajectory construction with censoring,
(ii) offline CQL training loop, (iii) OPE protocol.

**Limitations section — write it early and honestly.** Positivity violation and
the resulting reliance on the simulator for sub-12-month intervals; case-control
enrichment; 2-bin age; single-vendor (Hologic) single-region (Stockholm) data; 30%
of the cohort held back by the curators; no biopsy/pathology-workup variables; no
mortality endpoint (stage is a surrogate); and the untestable assumption that
detection at an earlier screen would have yielded the earlier-stage distribution.

**Reproducibility.** Fixed seeds, `data_audit.py` committed, exact splits saved as
patient-ID lists, d3rlpy + config YAML, and every cost parameter in one file.

**Ethics / data use.** CSAW-CC is CC BY, and the ICMJE condition requires **inviting
the data curators (Fredrik Strand, Karolinska Institutet) as co-authors** — this is
a binding condition of use, not a courtesy. Contact them before submission.

---

## Appendix A — Implementation file plan

This is the **actual** layout as built: eight flat Python modules, one per
pipeline stage, rather than the nested package the early plan envisaged. ✅ = in
the repo and running; ⬜ = designed here but not yet built (all image-dependent,
blocked on obtaining the DICOMs).

```
Phase2_RL/
├── config.py            ✅ all costs, γ, CQL α, paths in ONE place (§8, §9.2)
├── data_audit.py        ✅ reproduces every empirical number above (§0)
├── data.py              ✅ csv → exams → 29-d causal state → transition buffer
│                           folds in build_exams/splits/mdp/trajectories/reward/
│                           behaviour_policy — with the LABEL-LEAK ASSERTIONS (§6)
├── agent.py             ✅ the CQL agent only: Double + Dueling + QR (§9.2)
├── train.py             ✅ offline loop, FQE checkpoint selection, collapse gate (§11)
├── evaluate.py          ✅ FQE / WIS / PDWIS + ESS, clinical metrics, baselines (§12–13)
├── simulator.py         ✅ 3-state progression + calibration + rollout (§5.5, §9.2)
│                           ranks schedules by direct rollout, not a learned agent
├── DOCUMENTATION.md     ✅ full operational reference
├── LEARNING_GUIDE.md    ✅ study guide
│
├── encoder / images     ⬜ ConvNeXt + masked multi-view fusion (§10.1) — no DICOMs
└── explain.py           ⬜ Grad-CAM vs radiologist annotations (§14) — needs images
```

Notes on the collapse from the early plan:
- `build_exams` / `splits` / `mdp` / `trajectories` / `reward` / `behaviour_policy`
  are all functions inside `data.py`, not separate files.
- The `agents/` package (BC, DQN, Double, Dueling) is gone — the project uses CQL
  only (§9.2). The intermediate rungs are in git history.
- `simulator/train_ppo.py` was never built; the simulator evaluates schedules by
  rollout directly.

## Appendix B — Notation

| Symbol | Meaning |
|---|---|
| $s_t, a_t, r_t$ | state, action, reward at visit $t$ |
| $\tau(a)$ | realised gap in years; a state feature, and the exponent in the SMDP ablation |
| $\gamma = 0.9709$ | per-round discount, $\rho = 3\%$ |
| $\mathbf{z}_t$ | 512-d fused image embedding |
| $b_t$ | 256-d recurrent belief state |
| $\hat\pi_b$ | estimated behaviour policy |
| $\pi_\theta$ | learned policy |
| $U(\sigma)$ | QALY loss at detection stage $\sigma$ |
| $\omega_i$ | prevalence reweighting factor |
| $\alpha$ | CQL conservatism coefficient |
