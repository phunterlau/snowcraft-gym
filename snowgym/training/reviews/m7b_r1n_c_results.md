# R1n-c: imitation reaches 65/74 against a 93/91 ceiling; contact is solved, the critic precondition fails, and PPO has headroom

Collected 2026-09-13 from `556db16`.
- The [declaration](m7b_r1n_c_declaration.md) was committed at `3e89316`.
- Its implementation and amendments A1–A6 were committed at `8ade9c6`.
- `556db16` changed documentation only.

Run facts:
- One attempt, with no retries and no aborted runs.
- 535,358 decisions, against a 900,000-decision cap (bound by item: 823,200).
- 2,481 s wall time, 2.04 GB peak memory footprint (1.74 GB maximum RSS).
- Zero rejected actions in every collection, and zero illegal teacher labels.
- All 95 manifest SHA-256 entries were re-verified after the run. The pinned
  E3 digests and the implementation digest still match.
- Archive: `runs/m7b_engage_r1n_c_v0/`.

No PPO update ran, no checkpoint was promoted, and nothing below authorizes
R1n-d. `autonomousQualificationEligible` stays false: training used
coded-teacher labels, and at runtime the policy acted alone.

## Headline

1. **Imitation reaches 65/100 on split A and 74/100 on split B** against
   teacher ceilings of 93 and 91 (deterministic, mean of three optimizer
   seeds).
   - Seed-averaged paired differences from the ceiling: −28 points on A
     (95% CI −35 to −21) and −17 points on B (−24 to −10).
   - A lands in the *PPO has headroom* row and B in the *near ceiling* row.
     When splits disagree the more conservative row applies, so the outcome
     is **PPO has headroom**.
2. **Contact is solved.**
   - Contact rose from 0 in 2,304 R1n-b random-init episodes to 85–98% here.
   - At least 97% of failures are blue deaths, and at most one episode per
     evaluation timed out.
   - Most failures happen on worlds the teacher wins, after dealing part of
     the 80 damage needed to kill.
3. **The critic precondition fails for all three seeds.**
   - Predictive R² was 0.06, 0.11, and 0.06, against the 0.25 gate.
   - R1n-b's clock-only degeneracy is gone: held-out target variance is
     0.76–0.82, and time alone explains at most 5% (time-only R² −0.00 to
     0.05).
   - But the critic fits its *training* returns barely better (R² 0.11–0.12),
     and predicts nothing in the first 100 decisions. This archive cannot say
     whether the critic fits too little or the returns are mostly
     unpredictable from the state (§6).
4. **Execution-mode gap flag: set.**
   - Sampling from the calibrated σ drops success on A to 43, 48, and 54
     (gaps of 11, 31, and 8 points).
   - R1n-d must declare its exploration σ.
5. **Label fit is good except for HOLD, with no throw collapse.**
   - Type accuracy 94–95%.
   - Recall: MOVE 0.97–0.98, THROW 0.68–0.73, HOLD 0.51–0.62.
6. **R1's +20-over-initializer gate is infeasible on split B:** 74 + 20 = 94
   exceeds the 91 ceiling. It is feasible on A only when seed-averaged
   (85 ≤ 93). See §7.

| Declared prediction | Result |
| --- | --- |
| Deterministic success ≥ 60% on both splits | Met seed-averaged (65, 74). Seed 97101 got 54 on A. |
| Contact ≥ 80% | Met: 93% on A, 92% on B. |
| THROW recall below MOVE recall | Met: 0.68–0.73 vs 0.97–0.98. |
| Critic gate passes, clock skill clearly above 0 | **Not met.** 0/3 pass; clock skill 0.04–0.06. |

## 1. Controls

| Split | Source | Success | Contact | Blue deaths | Timeouts | Completion p50/p95 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A (600000–600099) | plan teacher | 93 | 95% | 7% | 0% | 117 / 144 |
| B (601000–601099) | plan teacher | 91 | 97% | 9% | 0% | 120 / 139 |
| A, B | uniform floor | 0, 0 | 0% | 0% | 100% | — |

The ceiling is consistent with R1n-b's 89/100 on 620000–620099. Every
teacher failure is a blue death (7 on A, 9 on B).

## 2. DAgger data and fits

- **Round 0** (teacher, 650000–650127): 15,729 rows.
  - MOVE 13,039, NOOP 1,476, HOLD 673, THROW 541.
  - Shared by all seeds, with digests in `round-0-teacher/dataset.json`.
- **After four on-policy rounds:** 80,433, 78,378, and 79,899 aggregated rows.
  Learner episodes were about as long as the teacher's (15,359–16,755 rows per
  128 episodes).
- **Round 1 had no HOLD labels in any seed.** The fit-0 policies never
  reached a state where the teacher holds. HOLD returned in later rounds
  (245–815 per round).
- **THROW labels in round 1 were 1.7–2.4× those in round 0.**
- **In the last 20 logged steps of fit 4, the near-endpoint term dominates
  the total loss** (0.41–0.44 of 0.55–0.61), although each minibatch holds
  only 3–13 near units. Type contributes about 0.10, move heading about
  0.05, throw aim and power under 0.01.
- **The declared gradient clip of 0.5 was active on ≥ 99% of logged steps**
  (median pre-clip norm about 100). Adam is nearly invariant to a constant
  gradient scale, but this clip rescales each minibatch's gradient to norm
  0.5, so every minibatch carries equal weight regardless of its loss. This
  is an untested candidate contributor to the 25-point seed spread.

## 3. Closed-loop evaluation (final policy after fit 4)

| Seed | A det | B det | A stochastic | Contact A/B | Deaths A/B | A stochastic deaths |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 97101 | 54 | 61 | 43 | 86% / 85% | 45% / 39% | 56% |
| 97102 | 79 | 83 | 48 | 98% / 97% | 21% / 17% | 52% |
| 97103 | 62 | 78 | 54 | 95% / 93% | 37% / 22% | 45% |
| Mean | 65.0 | 74.0 | 48.3 | 93% / 92% | 34% / 26% | 51% |
| Ceiling | 93 | 91 | — | 95% / 97% | 7% / 9% | — |

- **Timeouts:** at most 1%, so the 200-decision horizon does not bind.
- **Timing:** first-hit p50 is 96.5–99 decisions (teacher: 96 on A, 101 on
  B) and completion p50 is 114–120 (teacher: 117 and 120).
- **Paired differences from the ceiling, per seed:**
  - A: −39 (−50 to −28), −14 (−23 to −5), −31 (−41 to −21);
  - B: −30 (−40 to −20), −8 (−16 to 0), −13 (−22 to −4).
- **Seed spread:** 25 points on A and 22 on B, under the 30-point instability
  flag.

## 4. How imitation fails (deterministic evaluations)

| Split / seed | Failures | Died | No contact | Teacher won that world | Imitation won where teacher lost | Mean damage dealt in failures (80 kills) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / 97101 | 46 | 45 | 14 | 42 | 3 | 28 |
| A / 97102 | 21 | 21 | 2 | 18 | 4 | 40 |
| A / 97103 | 38 | 37 | 5 | 33 | 2 | 44 |
| B / 97101 | 39 | 39 | 15 | 32 | 2 | 24 |
| B / 97102 | 17 | 17 | 3 | 13 | 5 | 36 |
| B / 97103 | 22 | 22 | 7 | 18 | 5 | 33 |

- **All three seeds won the same world** 36 times on A and 50 times on B.
  None of them won 10 worlds on A and 5 on B.
- **The deficit is surviving engagements the teacher survives.** It is not
  finding or reaching the enemy. The weakest seed, 97101, also fails to make
  contact in 14–15 episodes per split.

## 5. Held-out label error (split B, on each policy's own deterministic states)

| Seed | Units | Type accuracy | MOVE recall / precision | THROW recall / precision | HOLD recall / precision | Move heading error | Near endpoint error | Throw aim error | Power MAE |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 97101 | 12,156 | 95.1% | 0.98 / 0.96 | 0.68 / 0.84 | 0.51 / 0.63 | 7.1° | 0.51 | 2.6° | 0.038 |
| 97102 | 12,116 | 94.3% | 0.97 / 0.96 | 0.73 / 0.83 | 0.62 / 0.65 | 9.4° | 0.53 | 3.0° | 0.038 |
| 97103 | 12,239 | 94.8% | 0.98 / 0.96 | 0.68 / 0.87 | 0.52 / 0.64 | 7.0° | 0.53 | 2.7° | 0.029 |

- NOOP recall and precision are 1.0.
- The throw collapse flag is not set: every THROW recall is ≥ 0.5.
- When THROW is labeled, the aim and power are accurate. The type decision is
  where the error lies: 27–32% of THROW labels and 38–49% of HOLD labels are
  predicted as something else.

## 6. Critic precondition (R1n-b's Monte Carlo warm start on stochastic episodes)

| Seed | Predictive R² (95% CI) | Explained variance | Time-only R² | Clock skill | Held-out target variance | Train R² after fitting | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 97101 | 0.062 (0.012 to 0.092) | 0.062 | 0.024 | 0.039 | 0.81 | 0.116 | fail (absolute) |
| 97102 | 0.108 (0.059 to 0.134) | 0.108 | 0.052 | 0.059 | 0.82 | 0.118 | fail (absolute) |
| 97103 | 0.059 (−0.043 to 0.113) | 0.089 | −0.003 | 0.062 | 0.76 | 0.112 | fail (absolute) |

- **Episodes:** 384 per seed, using a train fold of 660000+1000i and a
  held-out fold of 670000+1000i.
  - Success: 195, 205, and 210.
  - Contact: 91–94%.
  - Deaths: 45–50%.
- **Gate conditions:** the clock-relative condition passes everywhere. The
  absolute condition (≥ 0.25) fails everywhere.
- **Held-out R² by decision window:**
  - decisions 0–49: −0.02 to 0.00;
  - decisions 50–99: 0.01–0.02;
  - decisions 100 and later: 0.15, 0.26, and 0.13.
- **Diagnosis: not determined.** The target now carries outcome information:
  returns span −1.63 to +1.04. Train and held-out R² are similar, so the
  critic is not overfitting. A low train R² is consistent with two causes:
  - a critic that fits too little (capacity, epochs, or features);
  - returns that are mostly unpredictable from the state, even though a
    better critic would still be limited.

  This archive cannot separate them. Every world seed in the critic folds was
  played once per policy, so there are no repeated rollouts from the same
  state from which to estimate `Var(E[G | s]) / Var(G)`, the largest R² any
  critic could reach.

## 7. Decision rules and what they mean for R1n-d

The predeclared rules give **PPO has headroom** (A: headroom; B: near
ceiling; the conservative row applies). Flags:

| Flag | Value |
| --- | --- |
| Critic healthy | **false** (0/3) |
| Execution-mode gap | **true** (seed 97102: 31 points) |
| Seed instability | false (25 and 22 points) |
| Throw collapse | false |

The table's recommendation is KL-anchored frozen-reward PPO with R1's +20
gate, powered at the measured success. Read with the flags and with the
feasibility arithmetic, R1n-d's declaration has to settle three things first.
These are recommendations; R1n-d decides.

1. **The primary test.**
   - The gate is +20 over *each seed's* initializer. On B that means 81, 103,
     and 98 for seeds 97101, 97102, and 97103, so two of the three exceed the
     91 ceiling and one exceeds 100. On A it means 74, 99, and 82, where 99 is
     above the 93 ceiling. Even seed-averaged, B would need 94 against 91; A
     would need 85.
   - Candidates:
     - an A-only, seed-averaged +20 gate;
     - a smaller gain sized by a power analysis at the measured 65–74;
     - a continuous primary such as blue death rate (26–34% deterministic,
       against the teacher's 7–9%), which is where the measured deficit
       lies (§4).
2. **The critic.**
   - PPO advantages from a critic with R² about 0.1 carry little signal
     before contact.
   - First measure the attainable R² ceiling: repeated stochastic rollouts
     from fixed saved states give `Var(E[G | s]) / Var(G)`. That separates
     "repair the critic" from "the 0.25 threshold is unreachable".
   - Then choose one:
     - repair the critic (capacity, epochs, or features) and check it again;
     - use an estimator that depends less on the critic (for example GAE λ
       close to 1), with a justification;
     - re-declare the gate threshold against the measured ceiling.
3. **Exploration σ.** Stochastic success is 17 points below deterministic on
   average (up to 31). R1n-d must declare σ: lower, calibrated to a
   stochastic-success target, or annealed. It must also say which execution
   mode the gate is measured in.

The two options R1n-b left open stay deferred:
- a second contrasting mission before any plan-following claim;
- leave-one-out ablations.

## 8. What this does and does not decide

**Does show:**
- a full-authority, global-decoder policy imitates the 1v1 plan teacher to
  65–74% success with 92–93% contact and no rejected actions;
- DAgger over four rounds on the declared seeds produces seed-dependent
  policies (54–79 on A);
- the remaining gap is deaths in engagements the teacher wins;
- the Monte Carlo critic on these policies' stochastic episodes fails the
  absolute gate, while no longer being clock-only.

**Does not show:**
- that PPO improves on the initializer (not run);
- whether the critic's failure is capacity or irreducible noise;
- anything about the plan input (uninformative at 1v1), a second mission,
  2v2, or 5v5.

## Verification

- **Gate at `8ade9c6`, before collection:**
  - training 369 passed;
  - Python client 51 passed;
  - build ok;
  - `npm test` 366/367 with only the accepted R1n-b preflight failure.
  - Amendment A2's RNG isolation was checked in-process.
- **After collection:**
  - all 95 manifest digests and the inventory were re-verified;
  - `declaration.json` records `gitCommit 556db16`, matching E3 pins, and an
    implementation digest equal to the committed file;
  - the budget guard was not triggered;
  - `auditSeedDocuments` found no collision in any new JSON;
  - `npm test` still reports only the accepted failure, naming only
    `runs/m7b_engage_r1n_b_v0/declaration.json`.
- **Recomputed from archived artifacts:**
  - the tables in §§3–6 from `report.json`, `episodes.jsonl`,
    `label-error.json`, `critic-warm-start-arrays.npz`, and
    `fit-history.json`;
  - the failure analysis (§4) from per-world episode rows paired with the
    ceiling.
