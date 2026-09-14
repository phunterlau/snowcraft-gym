# R1n-d: halve σ; the 0.25 critic gate is not a meaningful target, and critic capacity is not the bottleneck

Collected 2026-09-14 from `9b89462`.
- The [declaration](m7b_r1n_d_declaration.md) was committed at `e4051f9`.
- Its implementation and amendments A1–A9 were committed at `9b89462`.

Run facts:
- One attempt, with no retries and no aborted runs.
- 807,532 decisions, against a 1,350,000-decision cap (bound by item:
  1,296,000).
- 872 s wall time, 3.43 GB peak memory footprint.
- All 96 manifest SHA-256 entries were re-verified after the run.
  `declaration.json` records `gitCommit 9b89462`, matching E3 pins, matching
  implementation digests, and a verified R1n-c source archive (95 artifacts).
- All 3,184 branched rollouts passed the replay-identity digest check.
- Archive: `runs/m7b_engage_r1n_d_v0/`.

No actor or PPO update ran, and nothing below authorizes R1n-e.
`autonomousQualificationEligible` stays false.

## Headline

1. **Exploration: halve σ.** The recommended scale is **0.5**.
   - The stochastic-execution gap comes almost entirely from continuous
     noise in move, throw, and power: 16.3 points at σ×1.
   - Sampling the action type costs 0.3 points.
   - At σ×0.5 the gap is 8.7 points; at σ×0.25 it is 2.0.
   - `typeSamplingDominates` is false.
2. **The critic outcome is *bracketed*.**
   - Policy 97101 is *gate unreachable*; 97102 and 97103 are *bracketed*.
   - Even the upper bound on attainable R² (a critic that also knows red's
     future random draws) is only **0.16, 0.22, and 0.24**, with 95% CI
     upper bounds of 0.23, 0.34, and 0.33.
   - About 76–84% of return variance lies *within* a state, from blue's own
     action sampling alone.
3. **Capacity, training length, representation, and clipping do not
   matter.** All four variants reach the same held-out R² within 0.011 for
   each policy:
   - 97101: 0.037–0.048;
   - 97102: 0.106–0.109;
   - 97103: 0.050–0.055.

   C0, which replicates R1n-c's 10-epoch warm start, is as good as the
   early-stopped variants. The egocentric critic gives no gain.
4. **The critics recover about 20–35% of the upper-bound state-value
   variance** (point `capture` 0.21–0.35 for C0; the intervals are wide).
   The upper bound includes red's future draws, which no observation-based
   critic can see, so the share of *observable* value recovered is higher by
   an unknown amount.
5. **Predictability rises through the episode.** The per-k ceiling runs from
   0.05–0.18 at k = 0 to 0.23–0.33 at k = 100 and 0.32–0.45 at k = 125.
   Held-out critic R² follows the same shape: −0.07 to 0.00 before
   decision 50, and 0.14–0.24 from decision 100.

| Declared prediction | Result |
| --- | --- |
| `gap(type) > gap(cont1)` | **Not met.** 0.3 vs 16.3 points. |
| `recommendedSigmaScale` = 0.5 | Met. |
| Pooled ceiling `U` between 0.25 and 0.6 | **Not met.** 0.16–0.24, below the range. |
| k ≤ 50 ceiling below k ≥ 100 ceiling | Met, for every policy. |
| C0 held-out R² ≤ 0.15 | Met: 0.05, 0.11, 0.05. |
| An egocentric variant selected and beating C1 | **Mixed.** C2 or C3 was selected for 2 of 3 policies, but beat C1 by only 0.002–0.003 R²; C1 was selected for 97101. |
| Overall outcome *bracketed* | Met. |

## 1. D1 — exploration decomposition (worlds 680000–680099)

**Success out of 100:**

| Policy | `det` | `type` | `cont1` | `full1` | `full05` | `full025` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 97101 | 55 | 60 | 48 | 41 | 53 | 59 |
| 97102 | 72 | 71 | 54 | 58 | 68 | 68 |
| 97103 | 71 | 66 | 47 | 49 | 51 | 65 |
| **Gap from `det`, seed-averaged (points)** | 0 | 0.3 | 16.3 | 16.7 | 8.7 | 2.0 |

**Blue death fraction:**

| Policy | `det` | `type` | `cont1` | `full1` | `full05` | `full025` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 97101 | 46% | 41% | 52% | 59% | 47% | 43% |
| 97102 | 28% | 31% | 44% | 41% | 33% | 32% |
| 97103 | 29% | 35% | 52% | 52% | 49% | 35% |

- **Timeouts** are at most 2% in every mode.
- **Contact** stays 83–97%.
- **Consistency with R1n-c:** deterministic success on this new band (55, 72,
  71) matches R1n-c's splits (A: 54, 79, 62; B: 61, 83, 78).
- **Where the cost is:** noise on move destinations, throw aim, and power
  gets the fighter killed. Sampling among action types does not.
- **Seed 97103 at σ×0.5** stays 20 points below deterministic, so the
  seed-averaged 8.7-point gap hides one policy that needs σ×0.25.

## 2. D3 — critic variants (new folds: train 684000+, held-out 687000+)

| Policy | Variant | Held-out R² (95% CI) | Time-only R² | Epochs (best) | Validation MSE | R² at decisions 0–49 / 50–99 / ≥ 100 | Branch-window R² |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| 97101 | C0 | 0.048 (−0.015 to 0.089) | 0.018 | 10 | — | −0.04 / 0.01 / 0.16 | 0.049 |
| 97101 | **C1 (selected)** | 0.038 (−0.031 to 0.084) | 0.018 | 40 (29) | 0.748 | −0.05 / 0.00 / 0.15 | 0.037 |
| 97101 | C2 | 0.038 (−0.028 to 0.082) | 0.018 | 22 (11) | 0.754 | −0.05 / 0.00 / 0.15 | 0.041 |
| 97101 | C3 | 0.037 (−0.029 to 0.082) | 0.018 | 21 (10) | 0.757 | −0.05 / 0.00 / 0.14 | 0.041 |
| 97102 | C0 | 0.109 (0.047 to 0.149) | 0.046 | 10 | — | 0.00 / 0.05 / 0.24 | 0.095 |
| 97102 | C1 | 0.106 (0.045 to 0.148) | 0.046 | 18 (7) | 0.751 | 0.00 / 0.05 / 0.23 | 0.092 |
| 97102 | C2 | 0.107 (0.049 to 0.144) | 0.046 | 15 (4) | 0.752 | 0.00 / 0.04 / 0.23 | 0.092 |
| 97102 | **C3 (selected)** | 0.109 (0.048 to 0.148) | 0.046 | 16 (5) | 0.748 | 0.00 / 0.04 / 0.24 | 0.094 |
| 97103 | C0 | 0.054 (−0.068 to 0.130) | 0.005 | 10 | — | −0.07 / −0.02 / 0.18 | 0.028 |
| 97103 | C1 | 0.053 (−0.078 to 0.135) | 0.005 | 30 (19) | 0.711 | −0.06 / −0.01 / 0.15 | 0.025 |
| 97103 | **C2 (selected)** | 0.055 (−0.071 to 0.134) | 0.005 | 21 (10) | 0.710 | −0.05 / −0.01 / 0.14 | 0.030 |
| 97103 | C3 | 0.050 (−0.080 to 0.128) | 0.005 | 17 (6) | 0.714 | −0.06 / −0.02 / 0.15 | 0.024 |

- **All variants early-stopped well before 200 epochs** (best epoch 4–29).
- **Validation MSE differs by at most 0.009 within a policy.**
- **The gate fails everywhere** on its absolute condition.
- **The A9 check:** held-out branch-window R² (0.02–0.10) is close to
  `rolloutR2` on the D2 branch states (0.00–0.08). The 690000-band branch
  states are therefore not out of distribution for these critics.

## 3. D2 — upper bound on attainable R² (24 source episodes × k-grid × 8 rollouts)

| Policy | States | Within-state W | Between B | V = B − W/8 | T | **Ceiling R²\*** (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 97101 | 135 | 0.649 | 0.204 | 0.123 | 0.772 | **0.159** (0.088 to 0.225) |
| 97102 | 131 | 0.550 | 0.223 | 0.155 | 0.704 | **0.219** (0.083 to 0.342) |
| 97103 | 132 | 0.589 | 0.260 | 0.186 | 0.775 | **0.240** (0.142 to 0.335) |

**Ceiling by branch decision:**

| Policy | k = 0 | 25 | 50 | 75 | 100 | 125 (n) |
| --- | --- | --- | --- | --- | --- | --- |
| 97101 | 0.05 | 0.08 | 0.09 | 0.15 | 0.23 | 0.32 (15) |
| 97102 | 0.18 | 0.15 | 0.13 | 0.17 | 0.33 | 0.45 (11) |
| 97103 | 0.11 | 0.19 | 0.19 | 0.22 | 0.30 | 0.42 (12) |

- Every k below 125 has 24 states.
- Per-k intervals are wide: at k = 100 they are 0.08–0.39, 0.13–0.50, and
  0.14–0.46.

**Critics on the branch states:**

| Policy | Variant | `rolloutR2` | `capture` (95% CI) |
| --- | --- | ---: | --- |
| 97101 | C0 | 0.032 | 0.21 (−0.30 to 0.50) |
| 97101 | C1 (selected) | 0.035 | 0.23 (−0.35 to 0.57) |
| 97102 | C0 | 0.076 | 0.35 (−0.17 to 0.49) |
| 97102 | C3 (selected) | 0.062 | 0.30 (−0.43 to 0.45) |
| 97103 | C0 | 0.054 | 0.24 (−0.16 to 0.45) |
| 97103 | C2 (selected) | 0.049 | 0.21 (−0.18 to 0.43) |

`capture`'s intervals are wide because V is a small difference of noisy
quantities in resamples with few distinct episodes. They exclude neither 0
nor 0.5.

**Boundary (declared).** Red's future draws are fixed within a branch, so
R²\* is an upper bound for any observation-based critic. The within-state
variance here comes from blue's σ×1 sampling alone. Red's randomness would
add to it, and a smaller σ would reduce it. D1's recommended σ×0.5 was not
measured in D2.

## 4. Decision rules and what they mean for R1n-e

**Critic outcome:** *bracketed*. Recommendation: use the selected variant,
state the critic gate as a capture of the measured ceiling, and state the
red-draw boundary.

| Policy | Row | Lower bound `L` (variant) | Ceiling R²\* | Upper CI bound |
| --- | --- | --- | ---: | ---: |
| 97101 | gate unreachable | 0.038 (C1) | 0.159 | 0.225 |
| 97102 | bracketed | 0.109 (C3) | 0.219 | 0.342 |
| 97103 | bracketed | 0.055 (C2) | 0.240 | 0.335 |

**Exploration:** `recommendedSigmaScale` 0.5; `typeSamplingDominates` false.

Read together, the results put these constraints on R1n-e. These are
recommendations; R1n-e decides.

1. **Critic choice.** The variants are interchangeable (§2). The
   declaration rule names the selected variant (C1, C3, and C2 by policy), but
   the evidence equally supports keeping `OptionCentralCritic` with R1n-b's
   recipe (C0) for continuity. R1n-e should justify whichever it picks.
2. **Critic gate.** An absolute 0.25 is not a meaningful precondition:
   - it exceeds the upper bound for one policy;
   - it is within the noise for the others.

   A capture-style gate needs a D2-like ceiling measurement at R1n-e's own σ.
3. **Advantage variance.** At most about a quarter of return variance is
   state-predictable even in the upper bound. So any baseline removes
   little variance, and per-episode advantage estimates stay noisy.
   - R1n-e's sample-size and power analysis should assume
     near-Monte-Carlo variance.
   - Using λ close to 1 costs little, because the critic contributes little
     anyway.
4. **σ.** Start from σ×0.5 and state the execution mode the gate is measured
   in. The 97103 policy is still 20 points below deterministic at σ×0.5.
   Lower σ may also raise the ceiling by reducing within-state variance; this
   is untested.
5. **Primary test.** Still to be set by R1n-e, using R1n-c's per-seed
   feasibility arithmetic (+20 exceeds the ceiling on B for 2 of 3 seeds) and
   the death-rate deficit.

## 5. What this does and does not decide

**Does show:**
- the execution gap is continuous-action noise, and it closes by σ×0.25;
- critic R² on these policies is bounded well below 0.25 even for an
  omniscient-to-red-draws critic at σ×1;
- critic architecture, features, training length, and clipping do not
  change the achievable fit;
- predictability is concentrated late in engagements.

**Does not show:**
- the observable (red-draws-hidden) ceiling;
- the ceiling at σ×0.5;
- whether PPO improves the policy;
- anything beyond 1v1 Engage.

## Verification

- **Gate at `9b89462`, before collection:**
  - training 380 passed;
  - Python client 51 passed;
  - build ok;
  - `npm test` 366/367 with only the accepted R1n-b preflight failure;
  - A2's `act` reproduction and the replay-identity mechanism were tested
    live.
- **After collection:**
  - all 96 manifest digests and the inventory were re-verified;
  - the implementation and critic digests equal the committed files;
  - all 3,184 rollout rows have `replayIdentity` true;
  - `auditSeedDocuments` found no collision in any new JSON file;
  - `npm test` still reports only the accepted failure.
- **Recomputed from the archive:** all tables were taken from `report.json`,
  whose values are computed from the archived episode rows, rollout rows,
  `branch-states.npz`, and `critic-arrays.npz`.
