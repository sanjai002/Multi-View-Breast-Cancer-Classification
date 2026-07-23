# Phase 2 — Personalized Screening Intervals via Offline Deep RL

Learning a personalized breast-cancer screening schedule from **CSAW-CC**, a
longitudinal Swedish screening cohort (8,723 patients, 24,694 exams).

📖 **[DOCUMENTATION.md](DOCUMENTATION.md)** — full reference: data, code, design,
results, how-to, known problems. Start here.
🔬 **[METHODOLOGY.md](METHODOLOGY.md)** — the scientific write-up with equations,
for the paper.

## Quickstart

```bash
python data_audit.py                        # verify the dataset
python data.py                              # -> outputs/buffer.npz
python train.py --steps 20000               # -> outputs/cql_seed0.pt (CQL)
python evaluate.py --split test             # -> outputs/eval_test.csv
python simulator.py --quick                 # counterfactual schedules
```

CPU is fine; the whole pipeline is minutes.

## The files

| File | Step |
|---|---|
| `config.py` | Every cost, γ, and path. Sensitivity analysis = sweep this |
| `data_audit.py` | Reproduces every empirical claim in the docs |
| `data.py` | CSV → exams → states → transition buffer |
| `agent.py` | The network + the **CQL** agent (Double + Dueling + QR) |
| `train.py` | Offline training, FQE selection, collapse gate |
| `evaluate.py` | FQE / WIS / ESS / support, clinical metrics, baselines |
| `simulator.py` | Calibrated natural-history model — the causal arm |

## Status

**Two arms.** The *offline* arm learns from real screening histories but cannot
show that screening sooner *causes* better outcomes — the logged interval barely
varied, so the transition kernel does not respond to the action. The *simulator*
arm supplies that mechanism explicitly and can evaluate schedules nobody was put
on, including 6-monthly.

**What works.** The data pipeline, the conservative offline agent, the full
evaluation harness, and a simulator calibrated to CSAW-CC's own screen-detected
vs interval-cancer stage contrast.

**What doesn't, yet.**

- **OPE cannot rank interval policies.** FQE seed noise exceeds the entire spread
  between baselines, and ESS ≈ 1.2 of 1,024 trajectories makes WIS useless.
  Never quote a single-fit FQE ranking.
- **Recall decisions need images.** No DICOMs on this machine; the learned recall
  sensitivity is 0.00 against clinicians' 0.76. That gap is the value of the
  mammogram.
- **Cost parameters are placeholders.** `c_e` alone decides the optimal interval,
  so read `simulator.py`'s threshold sweep, not any single row.

See [DOCUMENTATION.md §12](DOCUMENTATION.md#12-known-problems-and-open-work) for
the full list, and [§13](DOCUMENTATION.md#13-traps-that-have-already-bitten) for
the failure modes that have already produced plausible-looking wrong answers.

## Licence obligation

CSAW-CC is CC BY with a **binding ICMJE condition to invite the data curators
(Fredrik Strand, Karolinska Institutet) as co-authors.** Contact them before
submission.
