# Learning Guide — how to study this project with ChatGPT

Three parts:

1. **[The briefing](#part-1--the-briefing)** — paste this into ChatGPT first. It is
   self-contained, so ChatGPT will understand your project without seeing the code.
2. **[Questions to ask](#part-2--questions-to-ask)** — a study path from foundations
   to research-level, in the order that builds understanding.
3. **[Where ChatGPT will mislead you](#part-3--where-chatgpt-will-mislead-you)** —
   generic RL advice that is actively wrong for this project.

---

## Part 1 — The briefing

> Copy everything in this box into a fresh ChatGPT conversation, then start asking
> the questions from Part 2.

---

I am learning a research project. Act as my tutor. Explain concepts from first
principles, check my understanding with questions, and tell me when I am wrong.
Here is the project.

**Goal.** Recommend a *personalized* breast cancer screening interval using
reinforcement learning on real longitudinal screening data, instead of the fixed
national schedule everyone currently gets.

**Clinical background.** Sweden's programme invites women aged 40–54 back every 18
months and 55–74 every 24 months. The schedule ignores individual risk. Screening
too rarely means cancers surface *between* screens ("interval cancers"), which are
found later and at a worse stage. Screening too often means cost, anxiety, and
false-positive recalls that lead to unnecessary work-up.

**The data — CSAW-CC** (Karolinska Institutet, Sweden):
- 98,788 mammogram images → 24,694 screening exams → 8,723 women
- 873 women developed cancer, 7,850 did not
- Each exam has 4 standard views (left/right × CC/MLO)
- Most women have 2–5 visits across 2008–2016, so it is genuinely longitudinal
- Breast density per image, measured by software called LIBRA
- Two independent radiologist reads per exam, plus a recall decision
- Age is given only as a **band** (40–55 or 55+), not a number
- Exam dates are given only as a **year**

**Outcome labels.** A variable `rad_timing` records, for each exam, the time from
that screen to diagnosis: 1 = cancer found at that screen ("screen-detected"),
2 = cancer surfaced 60–729 days later ("interval cancer", i.e. a miss), 3 = a
prior exam more than 2 years before diagnosis. Per patient these sequences are
perfectly monotone. Final counts: 524 screen-detected, 217 interval cancers,
7,982 women whose observation simply ended (censored).

**Key empirical fact.** Interval cancers are measurably worse at detection:
node-positive 34.1% vs 22.9%, and tumour >15mm in 47.0% vs 40.8%. So the cost of
detecting late is *estimated from this dataset*, not assumed from literature. It
works out to about +0.367 QALY of harm per cancer that surfaces between screens.

**The RL formulation.**
- *State* (29 features): breast density each side and how it changed since her last
  visit, left-right asymmetry, age band, visit number, time since last screen,
  number of prior recalls, whether the two radiologists previously disagreed,
  missing-data flags, and the previous action. All strictly causal — a feature at
  visit *t* may only use visits 1…*t−1*.
- *Action* (6 options): come back in {12, 24, 36} months × {no work-up, recall now}.
- *Reward*: negative cost in QALYs = −(exam cost + false-positive cost + harm at
  diagnosis, where harm depends on tumour stage).
- *Discount*: γ = 1/(1.03) = 0.9709 per screening round, from the standard 3%
  annual health-economic discount rate.
- *Terminal states*: three distinct kinds — cancer detected, interval cancer, or
  censored. Censored patients must NOT be treated as "stayed healthy"; we simply
  stopped observing them.

**The central difficulty — positivity violation.** In the data, screening interval
is almost entirely determined by age band (1-year gaps: 2,637 for under-55s vs
25 for over-55s). Everyone got essentially the same schedule. Nobody was ever
screened at 6 months. Offline RL requires *positivity*: every action the learned
policy might choose must have non-zero probability under the behaviour policy. It
is violated here, so a Q-network asked about a 6-month interval will confidently
invent a value that nothing in the loss contradicts.

A second consequence: in offline RL the transition model is frozen — it is just the
recorded data. So lengthening the interval *cannot* increase modelled cancer
progression, because the action never varied enough to reveal that. The causal
mechanism is real but is absent from the objective.

**The solution — two arms.**
1. *Offline arm*: Conservative Q-Learning (CQL), which explicitly penalises the
   value of actions clinicians rarely took, plus off-policy evaluation (FQE,
   weighted importance sampling) that reports its own reliability via effective
   sample size and action support.
2. *Simulator arm*: an explicit natural-history model — healthy → preclinical
   (tumour growing, detectable by screening) → clinical (symptomatic). Tumours grow
   exponentially at heterogeneous rates; becoming symptomatic is a size-dependent
   hazard competing with screen detection; screening sensitivity rises with tumour
   size and falls with breast density. Calibrated so it reproduces the real
   screen-detected/interval split and stage distributions. This supplies the causal
   mechanism the data cannot identify, and can evaluate 6-month schedules.

**Additional complication.** The dataset is case-enriched: 10% of these women have
cancer versus ~0.6% in a real screening population, because cases were included
exhaustively while controls were sampled. Every reported rate is reweighted to true
population prevalence, or the agent would think 1 woman in 10 has cancer.

**Main findings so far.**
- The simulator produces a clean dose–response: halving the screening interval
  roughly halves the interval-cancer rate (38 → 8 per 1,000) and shrinks mean
  tumour size at detection (28mm → 8mm).
- The optimal interval depends almost entirely on one parameter: the disutility of
  a single mammogram. So the result is reported as a threshold sweep, not a single
  recommendation.
- **A negative result:** off-policy evaluation cannot rank screening intervals on
  this data. Effective sample size is 1.2 out of 1,024 trajectories, and FQE's
  seed-to-seed standard deviation exceeds the entire spread between policies. Any
  single-run FQE ranking here is noise.

---

## Part 2 — Questions to ask

Work through these in order. Each block assumes the previous one.

### Block A — RL foundations (start here if RL is new)

1. What is a Markov Decision Process? Define state, action, transition, reward and
   discount factor, using a simple everyday example.
2. What does a *policy* mean, and what makes one policy better than another?
3. What is a value function, and what is a Q-function? How do they differ?
4. Explain the Bellman equation intuitively, then formally.
5. What is Q-learning? Why is it called "off-policy"?
6. What is a discount factor for, and what changes if I set γ = 0.99 vs 0.97?

*Checkpoint:* explain to ChatGPT, in your own words, why a screening schedule is a
sequential decision problem rather than a prediction problem. Ask it to grade you.

### Block B — Offline RL, the heart of this project

7. What is offline (batch) reinforcement learning, and how does it differ from
   normal RL?
8. What is *extrapolation error* in offline RL? Why do standard DQN-style methods
   fail when they meet actions absent from the data?
9. Explain the positivity (overlap) assumption. Why can no algorithm estimate the
   value of an action that was never taken?
10. What is Conservative Q-Learning (CQL)? Walk me through its loss function term
    by term and explain what the conservatism penalty does geometrically.
11. Compare CQL, BCQ and IQL. When would I choose each?
12. Why are PPO, A2C and SAC unusable on a fixed dataset with no simulator?

*Checkpoint:* given my project's gap-vs-age table, explain in your own words why a
6-month action is unlearnable from the data.

### Block C — Off-policy evaluation

13. If I cannot deploy my policy on patients, how can I estimate how good it is?
14. Explain importance sampling for policy evaluation. Derive the weights.
15. What is effective sample size (ESS), and why does ESS = 1.2 out of 1,024
    trajectories mean my importance-sampling estimates are worthless?
16. What is Fitted Q Evaluation (FQE) and why is it more robust than importance
    sampling when the behaviour policy is nearly deterministic?
17. What is a doubly robust estimator and what does "doubly" protect against?
18. Why might FQE still be unreliable, and how would I detect that? (Hint: refit it
    with different random seeds.)

*Checkpoint:* my FQE has a seed-to-seed SD larger than the gap between policies.
What are my options, ranked by cost?

### Block D — Causal inference

19. What is a counterfactual, and why does "what if we had screened sooner?" require
    one?
20. Explain confounding using this example: older women got longer intervals AND
    have different cancer risk.
21. What is censoring in survival analysis? Why is treating a censored patient as
    "healthy" a serious bug rather than a small approximation?
22. What is inverse probability weighting, and how does case-control sampling break
    naive estimates?

### Block E — Screening and medical modelling

23. What is an interval cancer and why is it the key quality metric for a screening
    programme?
24. What is *lead time bias* and *length bias* in cancer screening? Could either
    affect my results?
25. What is overdiagnosis, and does my reward function account for it? (Think hard
    about this one.)
26. What is a QALY? How do health economists put a number on discomfort or anxiety?
27. Explain the natural-history model: healthy → preclinical → clinical. What is
    "sojourn time" and why is it the most influential parameter in any screening
    model?
28. What are the CISNET models and how do they differ from my simulator?
29. Why is breast density both a risk factor for cancer AND a reason screening
    misses it? (This is called *masking*.)

### Block F — Model calibration

30. What does it mean to calibrate a simulation model to summary statistics?
31. My six calibration targets are all proportions, and an early fit matched every
    one of them while producing tumours up to 273mm. What went wrong conceptually,
    and what is the general lesson?
32. My model has 10 free parameters and 6 targets. What is *identifiability* and
    what should I do about being under-determined?
33. What is approximate Bayesian computation, and would it be better than my random
    search plus local refinement?

### Block G — Research-level

34. Critique my two-arm design as a reviewer for a medical imaging journal. What is
    the weakest point?
35. Is a negative result ("off-policy evaluation cannot rank these policies")
    publishable? How should I frame it?
36. What are the ethical issues in an AI system that tells some women to be screened
    less often?
37. What would I need to prove before this could ever be used clinically?

---

## Part 3 — Where ChatGPT will mislead you

ChatGPT gives good *generic* RL answers. It does not know your constraints, and
several standard recommendations are wrong here. Watch for these:

| It will likely say | Why it is wrong for you |
|---|---|
| "Use PPO / A2C / SAC" | Those need a live environment. Your data is fixed. They are only valid inside the simulator |
| "Add more actions like 6-month screening" | Zero support in the data. It is not a modelling choice, it is unlearnable |
| "Use age as a continuous risk feature" | Your age is 2 bands. It carries about one bit |
| "Just use DQN, it's off-policy" | Off-policy ≠ offline. Vanilla DQN extrapolates wildly on unseen actions |
| "Your AUC/accuracy is the key metric" | The clinically meaningful metrics are interval-cancer rate, recall rate and stage at detection |
| "Use prioritized experience replay" | It upweights high-TD-error transitions, which offline are exactly the out-of-support ones CQL is suppressing |
| "Balance your classes with a sampler and weighted loss" | Combining both caused total class collapse earlier in this project. Use one |
| "Just report the best FQE value" | Single-fit FQE here is noise. It must be seed-averaged and paired with a support diagnostic |
| "More training steps will fix the noise" | The noise is in the *estimator*, not the policy. More steps will not help |

**A good habit:** after ChatGPT gives an answer, ask *"what assumption does that
rely on, and does it hold given that the behaviour policy in my data is nearly
deterministic?"* That single follow-up catches most of the errors above.

---

## Part 4 — Concepts worth truly mastering

If you only deeply learn five things, make it these. They are the load-bearing
ideas, and you should be able to explain each to a clinician without notation.

1. **Positivity / overlap** — why you cannot learn about actions nobody took.
2. **Extrapolation error and conservatism** — why a Q-network invents values off
   the data, and how CQL stops it.
3. **Censoring ≠ negative outcome** — why "we stopped watching" is not "she was fine".
4. **Estimator uncertainty** — why a number without an error bar and a support
   diagnostic is not evidence.
5. **The difference between correlation in logged data and a causal mechanism** —
   why the simulator exists at all.

Once those five are solid, the rest of the project is implementation detail.

---

## Suggested study order

| Session | Do this |
|---|---|
| 1 | Paste the briefing. Work Block A. Read `DOCUMENTATION.md §1–3` |
| 2 | Block B. Then read `agent.py` and find the CQL penalty term |
| 3 | Block C. Then run `python evaluate.py --split test` and interpret the ESS and support columns |
| 4 | Blocks D–E. Then read `METHODOLOGY.md §8` on the reward |
| 5 | Block F. Then run `python simulator.py --quick` and read the calibration table |
| 6 | Block G. Then read `DOCUMENTATION.md §12–13` — open problems and traps |

Reading the code after the concepts, rather than before, is deliberate. The code is
short; the ideas are what take time.
