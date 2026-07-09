# Phase 2 Implementation Guide — complete, self-contained blueprint

**Purpose of this document:** everything needed to implement Phase 2 without
further clarification — exact MDP design, exact per-file specification, network
architectures, training procedure, evaluation protocol, and how to run it. This
was written *before* implementation (per the original project plan: Phase 2 is
scaffolded only, implemented after Phase 1 is approved). Follow it top to
bottom, or hand it to a future session / another engineer as a complete spec.

---

## 0. Prerequisites — don't start until these are true

1. Phase 1 training finished with **healthy, non-collapsed metrics** (see
   `../Phase1_DL/NEXT_STEPS.md` §E — macro-F1 meaningfully above ~0.13, no
   class pinned at exactly 0.0/1.0 recall, `AUC` clearly above 0.5).
2. You ran the export step (`python -m training.test` or Colab's
   `run_colab_test.py`) and have these files in `Phase1_DL/outputs/`:
   ```
   patient_features.npy          patient_feature_index.csv
   image_features.npy            image_feature_index.csv
   prediction_probabilities.csv  patient_predictions.csv
   ```
3. You've looked at the numbers and are satisfied enough to build on top of
   them ("approved Phase 1").

If any of these aren't true yet, stop here and go finish Phase 1 first —
Phase 2 depends on the *shape and quality* of these files (feature
dimensionality, class balance, prediction confidence distribution), so
building it against placeholder assumptions risks having to redo it.

---

## 1. MDP design

NLBS is **cross-sectional** (one screening episode per patient, not real
multi-year follow-up), so "longitudinal RL" needs a defensible way to turn
single-timepoint data into multi-step episodes. Two designs are specified
below. **Implement Design A first** — it's grounded entirely in real data
Phase 1 already produced (no fabrication), directly matches the "use every
available image" spirit of the project, and is simpler to defend in a thesis.
Design B is provided as a documented stretch goal if your advisor specifically
wants a "years of screening" narrative.

### Design A (recommended, primary): Sequential View Acquisition

Reframe the decision problem as **"how much evidence to gather before
committing to a diagnosis"** — mirroring how a radiologist reads CC, then MLO,
then decides whether to recall for more views. This uses the *actual* 4 views
per patient from `image_features.npy` — no synthetic data needed.

- **Episode** = one patient (from any split; train episodes from `train`,
  evaluate policy on `val`/`test` — same patient-level split as Phase 1, reuse
  `patient_manifest.csv` so there's no leakage).
- **View order per episode**: fixed clinical order `[LCC, RCC, LMLO, RMLO]`
  (CC views first, both sides, then MLO both sides) — matches
  `config.VIEW_ORDER` from Phase 1 for consistency, or randomize order per
  episode as an ablation.
- **Step t** (t = 1..4): the agent has observed views `1..t`. Views not yet
  observed are zero-masked, exactly like Phase 1's missing-view handling.
- **State** `s_t` (concatenate):
  - Mean of the observed views' embeddings from `image_features.npy` (512-d,
    zero for unobserved) — or better, feed through Phase 1's *own* attention
    fusion (`MultiViewFusionModel.cc_fusion`/`mlo_fusion`/`patient_fusion`) restricted
    to the observed-view mask, so the state is exactly what the CNN itself
    would produce with partial evidence. **This requires loading
    `feature_extractor.pth` and running the fusion head at each step** —
    more faithful than averaging raw embeddings, and the encoder is already
    exported for exactly this purpose.
  - Phase 1's class probabilities computed from the observed subset (3-d).
  - One-hot step index `t` (4-d) or a scalar `t/4`.
  - Total state dim: 512 + 3 + 4 = **519** (or 516 with scalar step).
- **Actions** (discrete, 2):
  - `CONTINUE` — reveal the next view, pay a small step cost, episode continues
    (unless `t == 4`, in which case it's forced to stop).
  - `STOP` — end the episode now; final prediction = `argmax` of the current
    (partial-evidence) class probabilities.
- **Reward** (given at `STOP`, or forced stop at t=4):
  ```
  R_correct = {Normal: +1, FalsePositive: +2, Cancer: +10}[true_class]   # if predicted == true
  R_miss_cancer = -20   # true=Cancer, predicted != Cancer  (worst outcome)
  R_false_alarm = -5    # true=Normal, predicted=Cancer      (unnecessary alarm)
  R_other_wrong = -2    # any other misclassification
  R_step_cost   = -0.05 # applied at every CONTINUE (cost of extra imaging)
  ```
  This creates a genuine explore/exploit trade-off: stopping early saves the
  step cost but risks a worse (less-informed) prediction; the asymmetric
  rewards encode that missing cancer is far worse than a false alarm, which
  itself is worse than a routine correct call — a standard way to encode
  clinical cost asymmetry in RL reward shaping.
- **Terminal condition**: `STOP` chosen, or `t == 4` (all views observed).
- **Discount factor** `gamma = 0.95` (short horizon, T≤4, so this barely
  matters — could also use `gamma = 1.0` given the episode is finite and short).

### Design B (optional, stretch): Synthetic Longitudinal Screening

For a literal "screening over years" narrative (matches the original
`trajectory_generator.py` name more directly). **Document this clearly as a
modeling assumption / limitation in the paper** — it fabricates temporal
structure that doesn't exist in the source data.

- **Episode** = a synthetic patient history of `T` rounds (e.g. `T=4`,
  representing biennial screens over ~8 years), built by:
  1. Pick a real patient `p` with true label `y` and final feature vector `f_p`
     (from `patient_features.npy`).
  2. Compute the class-conditional centroid `c_y` = mean feature vector of all
     train patients with label `y` (a fixed, precomputed 512-d vector per class).
  3. Also compute the *Normal* centroid `c_Normal` (the common "healthy
     baseline" starting point for everyone, regardless of eventual label).
  4. For round `r = 1..T`, interpolate: `f_r = (1 - alpha_r) * c_Normal + alpha_r * f_p`,
     where `alpha_r` ramps from ~0.1 (round 1, looks mostly normal) to 1.0
     (round T, the real observed presentation) — e.g. `alpha_r = (r/T)^2` for a
     slow-then-fast progression (biologically motivated: early-stage disease
     produces subtler imaging changes). Add small Gaussian noise
     (`sigma≈0.05 * std(f_p)`) per round for realism.
  5. Only the **final round's** label is the true `y`; earlier rounds are
     unlabeled/"Normal-presenting" from the agent's perspective (it must infer
     progression risk from the *trend* across rounds, not a per-round oracle
     label).
- **State**: current round's synthetic feature vector + Phase 1's predicted
  probabilities on that synthetic vector (run it through the classifier head)
  + round index + previous action taken.
- **Actions** (discrete, 3): `ROUTINE` (continue biennial schedule),
  `RECALL` (short-interval follow-up), `REFER` (immediate diagnostic workup /
  biopsy).
- **Reward**: only realized meaningfully at the final round based on whether
  the cumulative action history caught a `Cancer` case early (e.g. `REFER`
  chosen at or before the round where `alpha_r` crosses some detectability
  threshold) vs. missed it — this requires a designed reward function; a
  reasonable starting point mirrors Design A's asymmetric costs but distributed
  across rounds with a bonus for *early* correct escalation and a penalty that
  grows the longer a true Cancer case goes un-referred.

**Recommendation:** implement Design A fully first (it's tractable, honest
about the data, and sufficient for a working RL system + thesis chapter).
Only build Design B if specifically requested — it's substantially more work
for a less defensible trajectory model.

---

## 2. Per-file implementation specification

All paths relative to `Phase2_RL/`. Follow the existing Phase 1 code style
(type hints, dataclass configs, docstrings only where non-obvious).

### `config.py`
Dataclass-based config (mirror `Phase1_DL/config.py`'s pattern):
```python
@dataclass
class MDPConfig:
    view_order: tuple = ("LCC", "RCC", "LMLO", "RMLO")
    state_dim: int = 519          # 512 feature + 3 probs + 4 step one-hot
    num_actions: int = 2          # CONTINUE, STOP  (Design A)
    step_cost: float = 0.05
    reward_correct: dict = field(default_factory=lambda: {"Normal": 1.0, "FalsePositive": 2.0, "Cancer": 10.0})
    reward_miss_cancer: float = -20.0
    reward_false_alarm: float = -5.0
    reward_other_wrong: float = -2.0
    gamma: float = 0.95

@dataclass
class TrainConfig:
    agent: str = "dueling_ddqn"   # "qlearning" | "dqn" | "double_dqn" | "dueling_dqn"
    episodes: int = 20000
    batch_size: int = 64
    buffer_size: int = 100_000
    lr: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 8000
    target_update_every: int = 500     # steps, for DQN/DDQN
    min_buffer_before_train: int = 1000
    hidden_dims: tuple = (256, 128)

@dataclass
class PathConfig:
    phase1_outputs: str = "../Phase1_DL/outputs"
    phase1_checkpoint: str = "../Phase1_DL/checkpoints/feature_extractor.pth"
    manifest_csv: str = "../Phase1_DL/outputs/patient_manifest.csv"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
```
Root `Config` composes these three, same pattern as Phase 1.

### `utils.py`
- `get_logger()`, `set_seed()` — copy directly from `Phase1_DL/utils.py`
  (identical need, no reason to reimplement).
- `load_phase1_features(cfg) -> Dict[str, np.ndarray]`: loads
  `image_features.npy` + `image_feature_index.csv`, `patient_features.npy` +
  `patient_feature_index.csv`, `prediction_probabilities.csv`,
  `patient_manifest.csv`. Returns a dict keyed by patient_id with per-view
  feature arrays `(4, 512)` (ordered by `VIEW_ORDER`, zero-filled for missing
  views, matching Phase 1's `mask` convention) and the true label.
- `load_feature_extractor(cfg) -> nn.Module`: loads `feature_extractor.pth`,
  reconstructs `MultiViewFusionModel`'s encoder (import from
  `../Phase1_DL/models/fusion.py` — add `Phase1_DL` to `sys.path`), sets
  `eval()`, freezes all params. Used by `mdp.py` to compute partial-evidence
  states online (Design A, if you choose the "run the real fusion head" state
  option over raw-average).

### `mdp.py`
- `class ScreeningState`: dataclass holding the current state vector, step
  index, observed-view mask, patient_id (for logging).
- `class ScreeningMDP`: given a loaded feature extractor + patient data,
  implements:
  - `reset(patient_id) -> ScreeningState` — start of episode, no views observed.
  - `step(state, action) -> (next_state, reward, done, info)` — Design A logic
    from §1 above. `info` should include the true label and predicted label at
    termination, for evaluation/reward-shaping debugging.
  - Pure function of (state, action, patient data) — no hidden mutable state
    beyond what's passed in, so it's trivially testable.

### `environment.py`
- `class ScreeningEnv` — thin Gym-style wrapper (`reset()`, `step(action)`,
  `observation_space`, `action_space`) around `ScreeningMDP`, iterating over a
  given split's patient list (shuffled each epoch for training, fixed order
  for eval). If you have `gymnasium` installed, subclass `gym.Env` properly;
  otherwise a duck-typed class with the same method names is fine (agents
  should only depend on `reset`/`step`, not on `gym` internals).

### `reward.py`
- `compute_reward(true_label: int, predicted_label: Optional[int], action: int, cfg: MDPConfig) -> float`
  — pure function implementing the reward table from §1. Keep this separate
  from `mdp.py` so reward-shaping experiments don't require touching the
  transition logic.

### `replay_buffer.py`
- `class ReplayBuffer` — uniform sampling, fixed-size circular buffer of
  `(state, action, reward, next_state, done)` tuples, `push()`/`sample(batch_size)`.
- `class PrioritizedReplayBuffer` — optional, sum-tree based, TD-error priority
  (implement after the uniform version works end-to-end; not required for a
  working baseline).

### `qlearning.py`
- Tabular Q-learning baseline **requires discretizing the continuous 519-d
  state** — not practical directly. Two options: (a) discretize only the
  3 prediction-probability dims + step index (5-ish bins each → few hundred
  discrete states) and ignore the raw 512-d feature vector for this baseline
  only, or (b) skip tabular Q-learning and use a **linear** Q-function
  (`Q(s,a) = w_a . s`) as the "classical baseline" instead — simpler to
  implement correctly and still a fair comparison point against the deep
  agents. Recommend (b): `class LinearQAgent` with per-action weight vectors,
  updated via the standard Q-learning TD update with a linear function
  approximator (this is "linear function approximation SARSA/Q-learning", a
  standard, well-defined method — cite Sutton & Barto if writing this up).

### `dqn.py`
- `class QNetwork(nn.Module)`: MLP, `state_dim -> hidden_dims -> num_actions`,
  ReLU activations, no output activation (raw Q-values). Use `cfg.hidden_dims`.
- `class DQNAgent`: holds online + target `QNetwork`, `ReplayBuffer`,
  epsilon-greedy action selection, `update()` performing one gradient step on
  a sampled batch using the standard DQN target:
  `y = r + gamma * (1-done) * max_a' Q_target(s', a')`.
  Target network synced every `cfg.target_update_every` steps (hard update;
  soft/Polyak update is a reasonable alternative, note it as an option).

### `double_dqn.py`
- `class DoubleDQNAgent(DQNAgent)`: overrides only the target computation —
  action selection from the **online** network, value from the **target**
  network: `y = r + gamma * (1-done) * Q_target(s', argmax_a' Q_online(s', a'))`.
  Everything else (buffer, epsilon schedule, network class) is inherited.

### `dueling_dqn.py`
- `class DuelingQNetwork(nn.Module)`: shared trunk `state_dim -> hidden`,
  then two heads: value stream `hidden -> 1` and advantage stream
  `hidden -> num_actions`, combined as
  `Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))`.
- `class DuelingDQNAgent`: same as `DoubleDQNAgent` but constructed with
  `DuelingQNetwork` instead of the plain `QNetwork` — i.e. **combine dueling
  architecture with the double-DQN target** (this is standard practice; note
  it explicitly in your methodology section as "Dueling Double DQN").

### `policy.py`
- `epsilon_greedy(q_values, epsilon, num_actions) -> int` — pure function.
- `greedy(q_values) -> int` — `argmax`, for evaluation (epsilon=0).
- `linear_epsilon_schedule(episode, cfg) -> float` — linear decay from
  `epsilon_start` to `epsilon_end` over `epsilon_decay_episodes`.
- `extract_policy(agent) -> Callable[[state], int]` — wraps a trained agent's
  greedy action selection for use in `inference.py`.

### `trajectory_generator.py`
- Only needed for **Design B**. Implements the interpolation scheme from §1
  (`generate_synthetic_trajectory(patient_id, T, cfg) -> List[np.ndarray]`).
  Skip / stub this out if only implementing Design A (Design A reads directly
  from `image_features.npy` via `mdp.py`, no separate generator needed).

### `train_rl.py`
Entry point, argparse `--agent {qlearning,dqn,double_dqn,dueling_dqn}`:
1. Load config, set seed, set up logger + TensorBoard (reuse
   `torch.utils.tensorboard.SummaryWriter`, same pattern as Phase 1).
2. Load Phase 1 features/feature-extractor, build `ScreeningEnv` for the
   **train** split (reuse the split from `patient_manifest.csv` — never train
   the RL agent on val/test patients).
3. Instantiate the chosen agent.
4. Standard RL training loop: for each episode, reset env, roll out until
   `done`, push transitions to replay buffer, call `agent.update()` once
   enough transitions are collected (`min_buffer_before_train`), log episode
   return + epsilon + loss to TensorBoard every N episodes.
5. Periodically (e.g. every 500 episodes) run a **greedy** evaluation pass on
   the **val** split (no exploration, no learning) and track: mean episode
   return, but also **clinically meaningful metrics** — Cancer detection rate
   (recall) at episode termination, average number of views used per episode
   (evidence-gathering efficiency), false-alarm rate. Save the checkpoint with
   the best val Cancer-recall (not just best return — same lesson as Phase 1:
   a policy that always says STOP immediately and predicts the majority class
   can "optimize" naive return while being clinically useless; watch per-class
   behavior explicitly, exactly like `metrics_history.csv` did for Phase 1).
6. Save checkpoints to `cfg.paths.checkpoint_dir` every N episodes + best-so-far,
   mirroring Phase 1's `save_every_epoch` / `best_model.pth` pattern for
   consistency.

### `evaluate_rl.py`
- Loads a trained agent checkpoint, runs a **greedy** (epsilon=0) pass over
  the **test** split (held out, never touched during training/tuning).
- Reports: mean return, Cancer/Normal/FalsePositive-conditional accuracy at
  termination, average views used, action distribution, and — critically —
  a **comparison table across all four agents** (linear Q, DQN, Double DQN,
  Dueling DQN) trained under identical config, so the paper can report which
  architecture actually helps.
- Also compare against a **trivial baseline**: "always observe all 4 views,
  predict via Phase 1's full-evidence CNN" (i.e., Phase 1's own test accuracy)
  — the RL agent's value proposition is using *fewer* views for equivalent (or
  better-triaged) outcomes, so this baseline is the natural yardstick.

### `inference.py`
- `run_policy(agent, patient_features) -> Dict` — roll out the trained
  greedy policy on a single new patient's per-view features, returning the
  sequence of actions taken, views used, and final prediction. This is the
  "deployable" entry point — what you'd call to get a recommendation for one
  real patient.

---

## 3. Evaluation protocol (what "success" looks like)

Report, per agent, on the held-out **test** split:
1. **Cancer recall at termination** (the single most clinically important
   number — mirrors Phase 1's own lesson about not trusting aggregate
   accuracy).
2. **Average views used per episode** — lower is better *if* recall is
   maintained; this is the core trade-off the RL system is meant to learn
   (adaptive evidence-gathering vs. Phase 1's fixed "always use all 4 views").
3. **False-alarm rate** (predicted Cancer, true Normal).
4. **Mean episode return** (sanity-check aggregate, not the headline number).
5. Compare all of the above against the **fixed-policy baseline** (always
   observe all 4 views, use Phase 1's full CNN prediction — i.e., Phase 1's
   own test metrics from `test_metrics.json`). The thesis narrative is: *"Does
   sequential, learned evidence-gathering match Phase 1's accuracy while using
   fewer views on average, or safely triage low-risk patients faster?"*

---

## 4. How to actually run it (once implemented)

```bash
cd Phase2_RL
python train_rl.py --agent dueling_dqn      # repeat for qlearning, dqn, double_dqn
tensorboard --logdir outputs/tensorboard
python evaluate_rl.py --agent dueling_dqn --checkpoint checkpoints/dueling_dqn_best.pth
```
Produce a final comparison by running `evaluate_rl.py` for all four agents and
collecting their output tables into one markdown/CSV for the paper — mirrors
how Phase 1's `test_metrics.json` was the single source of truth for its
results tables.

---

## 5. Lessons from Phase 1 that directly apply here

- **Don't trust aggregate reward/accuracy alone** — Phase 1's class-collapse
  bug (see `../Phase1_DL/NEXT_STEPS.md`, "Known issue") showed a model can
  find a degenerate shortcut that scores fine on one metric while being
  useless per-class. Watch **Cancer-specific recall** at every checkpoint, not
  just mean return.
  - Straightforward RL translation: a policy that always chooses `STOP` at
    step 1 and always predicts the majority class (`Normal`) will get a
    *positive* mean return most of the time (since Normal is ~72% of patients
    and gives `+1` with zero step cost) while having near-zero Cancer recall.
    **Reward shaping must be checked against this specific failure mode** —
    if you see mean return climbing while Cancer recall stays at 0, this is
    the RL analogue of the Phase 1 collapse. If it happens, increase
    `reward_miss_cancer`'s magnitude relative to `reward_correct["Normal"]`
    further, or add an explicit Cancer-recall term to early stopping /
    checkpoint selection (as specified in §2's `train_rl.py` step 5).
- **Patient-level splitting, no leakage** — reuse `patient_manifest.csv`
  exactly; never let a test patient's data influence training, same as Phase 1.
- **Save every checkpoint + metrics history + resumability** — mirror
  `Phase1_DL/training/train.py`'s pattern (`save_every_epoch`,
  `metrics_history.csv`, `resume`) so a long RL training run surviving a
  disconnect/crash is a solved problem here too, not re-invented.
