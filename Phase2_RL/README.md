# Phase 2 — Reinforcement Learning (Scaffold Only)

> **Status: NOT IMPLEMENTED.** This directory currently contains only the
> project structure and this README. The reinforcement-learning implementation
> will be completed in the **second phase, after approval of Phase 1**.

Phase 2 builds a reinforcement-learning agent for longitudinal breast-screening
decision-making. It does **not** re-read DICOM images. Instead it consumes the
artefacts produced by Phase 1 (`../Phase1_DL/outputs/`) and treats the learned
CNN representations as the state of a Markov Decision Process (MDP).

## Inputs (produced by Phase 1)

| File | Role in Phase 2 |
|------|-----------------|
| `image_features.npy` | Per-view CNN embeddings — fine-grained state features |
| `patient_features.npy` | Per-patient fused CNN embedding — core state vector |
| `prediction_probabilities.csv` | Class probabilities — confidence signal in the state / reward |
| `patient_predictions.csv` | Predicted vs. true class — supervision for reward shaping and evaluation |
| **Synthetic longitudinal trajectories** | Generated from the above to form multi-step screening episodes |

These files are combined to build the **MDP state representation**: each state
concatenates a patient's CNN feature vector with the model's prediction
probabilities and trajectory context; actions are discrete screening / follow-up
decisions; rewards encode the clinical cost of missed cancers versus unnecessary
recalls.

## Planned structure

```
Phase2_RL/
├── config.py                # RL hyper-parameters
├── trajectory_generator.py  # synthetic longitudinal trajectories from Phase 1 exports
├── mdp.py                   # state / action / transition definitions
├── environment.py           # Gym-style environment over trajectories
├── reward.py                # clinical reward function
├── qlearning.py             # tabular / linear Q-learning baseline
├── dqn.py                   # Deep Q-Network
├── double_dqn.py            # Double DQN
├── dueling_dqn.py           # Dueling DQN
├── replay_buffer.py         # uniform + prioritised experience replay
├── policy.py                # epsilon-greedy / greedy / softmax policies
├── inference.py             # roll out a trained policy on new patients
├── train_rl.py              # training entry point
├── evaluate_rl.py           # evaluation entry point
├── utils.py                 # logging, seeding, Phase 1 export loaders
└── README.md
```

## Data flow

```
Phase 1 (DL)                         Phase 2 (RL)
────────────                         ────────────
DICOM → ResNet-50 multi-view  ──►  patient_features.npy ┐
   fusion classifier          ──►  image_features.npy   ├─► MDP state ─► RL agent
                              ──►  prediction_*.csv      ┘        (DQN / Double / Dueling)
                                        │
                                        └─► synthetic longitudinal trajectories ─► episodes
```

## How to proceed

1. Complete and approve **Phase 1**; run `python -m training.test` to generate
   the exports listed above.
2. Implement the modules in this directory (deferred until approval).
3. Train and evaluate the RL agents on the trajectory environment.
