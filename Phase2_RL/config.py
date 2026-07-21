"""
Phase 2 — single source of truth for every knob.

Nothing in this project should hard-code a cost, a discount rate, or a path.
Sensitivity analysis (METHODOLOGY.md §8.4) works by sweeping values here.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).parent
CSV = ROOT / "CSAW-CC_breast_cancer_screening_data.csv"
OUT = ROOT / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Action space (METHODOLOGY.md §7)
#   6-month intervals and biopsy/extra-imaging actions are ABSENT by design:
#   zero support in CSAW-CC. Do not add them back without the simulator arm.
# ─────────────────────────────────────────────────────────────────────────────
INTERVALS_YEARS = (1, 2, 3)            # 12 / 24 / 36 months
WORKUP = ("NO_WORKUP", "RECALL")
N_ACTIONS = len(INTERVALS_YEARS) * len(WORKUP)   # 6

ACTION_NAMES = [
    f"{i * 12}mo+{w}" for i in INTERVALS_YEARS for w in WORKUP
]


def encode_action(interval_idx: int, recall: int) -> int:
    return interval_idx * len(WORKUP) + recall


def decode_action(a: int) -> tuple[int, int]:
    return divmod(a, len(WORKUP))


# ─────────────────────────────────────────────────────────────────────────────
# Terminal types (METHODOLOGY.md §3.3) — never conflate these
# ─────────────────────────────────────────────────────────────────────────────
DETECTED, INTERVAL, CENSORED = 0, 1, 2


@dataclass
class Costs:
    """QALY-equivalent disutilities. All swept in sensitivity analysis."""

    # c_e: disutility of one screening round (attendance + discomfort).
    # 0.001 QALY ~ 0.36 days of perfect health. The previous default of 0.01
    # implied 3.65 days lost per mammogram, which is roughly an order of
    # magnitude too high and by itself flipped the simulator's optimal interval
    # from 18 to 36 months. STILL A PLACEHOLDER - attach a citation, and treat
    # simulator.py's threshold sweep as the actual result rather than any single
    # value of this parameter.
    exam: float = 0.001
    false_positive: float = 0.05       # c_r: recall anxiety, persists 6-12mo

    # U(sigma): QALY loss by stage at detection (METHODOLOGY.md §8.2).
    # Ordering is what the policy is sensitive to and is empirically supported
    # by the screen-detected vs interval contrast in this dataset.
    # PLACEHOLDER MAGNITUDES — attach your survival citation before submission.
    u_in_situ: float = 0.5
    u_inv_small_n0: float = 1.0        # invasive <=15mm, node-negative
    u_inv_small_n1: float = 2.5        # invasive <=15mm, node-positive
    u_inv_large_n0: float = 3.0        # invasive >15mm,  node-negative
    u_inv_large_n1: float = 5.5        # invasive >15mm,  node-positive

    def utility(self, x_type: float, node: float) -> float:
        """Map (x_type, x_lymphnode_met) -> QALY loss."""
        import math

        if x_type is None or (isinstance(x_type, float) and math.isnan(x_type)):
            return self.u_inv_small_n0          # unknown stage -> mid-low
        n = 0 if (node is None or (isinstance(node, float) and math.isnan(node))) else int(node)
        t = int(x_type)
        if t == 1:
            return self.u_in_situ
        if t == 2:
            return self.u_inv_small_n1 if n else self.u_inv_small_n0
        return self.u_inv_large_n1 if n else self.u_inv_large_n0

    @property
    def u_max(self) -> float:
        return self.u_inv_large_n1


@dataclass
class Config:
    # ---- data ----
    csv: Path = CSV
    out: Path = OUT
    seed: int = 0
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # Case-control correction (METHODOLOGY.md §8.5). Sample prevalence is
    # measured from the data; population prevalence is the screening-programme
    # figure. Without this the agent thinks 1 in 10 women has cancer.
    population_prevalence: float = 0.006
    reweight_prevalence: bool = True

    # Impute the final action for interval-cancer terminals (§3.3 / data.py).
    # These 217 trajectories carry the entire "miss" signal; dropping them
    # removes the reason the project exists. Ablate with False.
    impute_terminal_action: bool = True

    # ---- discounting ----
    # gamma = 1/(1+rho), rho = 3% annual health-economic discount rate.
    discount_rate: float = 0.03

    # MDP (default): one gamma per decision step, regardless of how many years
    # the chosen interval spans. SMDP (smdp=True): gamma**tau with tau in years.
    #
    # The difference is not cosmetic. Under the MDP a 36-month wait is
    # discounted exactly as hard as a 12-month one, so waiting longer carries no
    # time cost in the return and the "longer intervals look free" effect
    # documented in METHODOLOGY.md §12.3 gets stronger. Flip to True to
    # reproduce the semi-Markov variant as an ablation.
    smdp: bool = False

    @property
    def gamma(self) -> float:
        return 1.0 / (1.0 + self.discount_rate)

    def discount(self, tau):
        """Per-transition discount, shaped like `tau` (numpy array or tensor)."""
        if self.smdp:
            return self.gamma ** tau
        return self.gamma * (tau * 0 + 1)     # keeps tau's shape/dtype/device

    def elapsed(self, tau):
        """Within-trajectory discount exponent: step index (MDP) or years (SMDP)."""
        import numpy as _np

        return _np.cumsum(tau) - tau if self.smdp else _np.arange(len(tau))

    costs: Costs = field(default_factory=Costs)

    # ---- optional image block ----
    # No CSAW-CC DICOMs are present on this machine; the pipeline runs on the
    # tabular longitudinal state alone. Point this at a cached (n_exams, d)
    # array once images are available and it is concatenated into the state.
    image_features: Path | None = None

    # ---- agent (METHODOLOGY.md §11.5) ----
    hidden: int = 256
    gru_hidden: int = 128
    n_quantiles: int = 32
    cql_alpha: float = 5.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64               # trajectories
    grad_clip: float = 10.0
    target_tau: float = 5e-3
    steps: int = 20_000
    eval_every: int = 500
    dueling: bool = True
    double: bool = True

    # ---- OPE ----
    propensity_floor: float = 0.01     # action masking threshold (§7)
    n_bootstrap: int = 1000
    n_fqe_seeds: int = 3               # FQE is seed-sensitive; average it

    def describe(self) -> str:
        d = asdict(self)
        d["gamma"] = round(self.gamma, 6)
        return "\n".join(f"  {k:24s} {v}" for k, v in d.items())


CFG = Config()
