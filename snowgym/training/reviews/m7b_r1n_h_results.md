# R1n-h: substituting half of imitation training with scripted-easy exposure transfers to a fully held-out opponent

Collected 2026-09-15 from `953d851`.
- The [declaration](m7b_r1n_h_declaration.md) was committed at `70e8f0a`,
  amendments A1–A2 at `953d851`.
- The implementation is at `4c84b74`.

Run facts:
- **Decisions:** 1,017,713 total (controls + condition M: 491,166;
  condition C: 526,547), against a 2,200,000 cap (bound 1,686,400) —
  actual usage came in well under the bound, as in every prior stage of
  this track, because most episodes terminate before the full 200-decision
  horizon.
- **Pre-collection gate (amendment A2) passed before either condition
  trained:** the plan teacher — the `Labeler(model=None)` path every
  ceiling and DAgger label in this run uses — scored 100% on both
  `eval-easy` and `eval-normal` (91% on `eval-random`, consistent with
  R1n-b/R1n-c's own measurement of the same controller). DAgger's labels
  were valid supervision for this task before any training decision was
  spent.
- **Two conditions, three optimizer seeds each, on fresh 400000–449999
  seed bands:** mixture (M, every round 64 random + 64 scripted-easy) and
  a single-opponent control (C, 128 random). `full_authority_imitation.py`
  and `full_authority_train_v1.py` confirmed byte-identical to the digests
  already recorded in R1n-g's and R1n-f's `declaration.json` at declare
  time — no edits.
- **Manifest:** the top-level manifest (`sha256:6ab9fde9…`) verified after
  aggregation.
- **Archive:** `runs/m7b_engage_r1n_h_v0/`.

R1n-h makes no R1 qualification claim and runs no PPO.
`autonomousQualificationEligible` stays false throughout.

## Headline

1. **The primary result is positive and well-powered: mixture training
   transfers to the fully held-out opponent.** M minus C on `eval-normal`
   (scripted normal, an opponent *neither* condition ever trains on) is
   **+38.0 points [+35.7, +40.3]**, a 100-world bootstrap interval that
   excludes zero by a wide margin. Outcome: **`transfers`**.
2. **The precondition passed comfortably.** M's gap to the teacher on
   `eval-easy` (the opponent it actually trains on) is −1.3 points
   [−2.7, −0.3] — essentially at ceiling, nowhere near the −50-point
   failure threshold. M's own `eval-easy` success is 0.96/1.00/1.00 across
   the three seeds.
3. **The secondary mechanism check confirms the primary isn't a fluke of
   the win/loss metric alone — the specific R1n-g finding is fixed.** Mean
   `throwAimHeadingErrorDegrees` on `eval-normal` is **8.6° for M** versus
   **127.6° for C**. C's number reproduces R1n-g's archived arm-N finding
   almost exactly (74.9°–148.4° per cell there; 116.4°–143.6° per seed
   here) — the *worse-than-random* aim failure is real and repeatable.
   M's number is close to the healthy ~3° baseline this whole track has
   used since R1n-c, on an opponent M never trained on. Mean throw recall
   moves from 0.79 (M) versus 0.02 (C) — C's collapse is even more
   complete here than R1n-g's archived 0.15 mean on arm N, and M's
   recovery is close to ceiling.
4. **Condition C is an exact, independent reproduction of R1n-f/R1n-g's
   floor**, on fresh seeds and a fresh checkpoint: 0/0/0 success on both
   scripted arms across all three seeds, throw recall 0.016–0.026, aim
   116°–144°. This was not R1n-h's question, but it is a second
   confirmation (after R1n-g's own archive cross-check) that the
   generalization failure is real, not an artifact of one specific set of
   checkpoints.
5. **Success and label error tell different stories at the per-seed
   level — worth flagging directly.** M's label-error metrics are good
   for all three seeds (recall 0.865–1.000, aim 0.7°–12.6°), but
   `eval-normal` success is 0.0 / 1.00 / 0.14 — wildly uneven. Fixing the
   aim/recall mechanism did not translate uniformly into fixing win/loss
   outcomes; seed 97102 wins essentially every held-out episode, while
   97101 and 97103 mostly still lose despite comparably healthy label
   error. This is the run's biggest open question (§7).
6. **An unplanned side-finding: M's critic gate passes where C's
   fails.** M: `predictiveR2` 0.26/0.47/0.30, gate passed on all three
   seeds. C: `predictiveR2` 0.07/0.09/0.10, gate failed on all three —
   reproducing R1n-c's own original critic-infeasibility finding (ceiling
   0.16–0.24, per R1n-d) on a freshly-trained checkpoint. Mixing in
   scripted-easy episodes, which end in contact far more often than
   random-opponent episodes, appears to give the critic a substantially
   more learnable Monte Carlo return signal. Not a declared measure —
   reported because it bears directly on whether a future PPO stage
   (`R1n-e`-style) would even clear its own precondition starting from
   condition C's checkpoints, versus M's.

## 1. Precondition: gap(M, eval-easy)

| Metric | Value |
| --- | --- |
| M's `eval-easy` success, by seed | 0.96, 1.00, 1.00 |
| Teacher ceiling, `eval-easy` (fresh worlds, this run) | 1.00 (100/100, controls stage) |
| gap(M, eval-easy), world-paired bootstrap | −0.013 [−0.027, −0.003] |
| `failThreshold` | −0.50 |

Passes by a wide margin. The held-out-opponent comparison below is
interpretable.

## 2. Primary: M minus C on eval-normal

| Metric | Value |
| --- | --- |
| M's `eval-normal` success, by seed | 0.00, 1.00, 0.14 |
| C's `eval-normal` success, by seed | 0.00, 0.00, 0.00 |
| M − C, world-paired bootstrap (100 worlds) | **+0.380 [+0.357, +0.403]** |
| Decision thresholds | interval excludes 0 → `transfers` (or `regresses` if excludes 0 below; `no-detected-transfer` if includes 0) |

Because C is uniformly 0 on every one of the 100 `eval-normal` worlds, the
bootstrap is well-powered for any nonzero M performance regardless of how
unevenly that performance is spread across M's three seeds — the interval
is tight despite the seed-level 0.00/1.00/0.14 spread, because the world
axis (not the seed axis) is what's resampled (`world_paired_difference`,
reused unchanged from `opponent_transfer.py`, the same axis fix R1n-f's
amendment A5 established).

## 3. Secondary: label error on eval-normal, M vs C

| Comparator | Recall (throw) | Support | Aim heading error | Type accuracy |
| --- | ---: | ---: | ---: | ---: |
| M, 97101 | 0.878 | 606 | 12.6° | 0.972 |
| M, 97102 | 1.000 | 400 | 0.7° | 0.983 |
| M, 97103 | 0.865 | 780 | 12.5° | 0.938 |
| **M mean** | **0.914** | | **8.6°** | |
| C, 97101 | 0.023 | 1,347 | 116.4° | 0.868 |
| C, 97102 | 0.026 | 1,310 | 122.7° | 0.901 |
| C, 97103 | 0.016 | 1,298 | 143.6° | 0.909 |
| **C mean** | **0.022** | | **127.6°** | |

C's aim numbers reproduce R1n-g's archived arm-N finding (74.9°–148.4° per
cell, mean 111.9°) in the same range — a fresh, independent confirmation
on different checkpoints and different worlds, not a repeat measurement of
the same run.

## 4. Flags

| Flag | M | C |
| --- | --- | --- |
| `criticHealthy` | **True** (R² 0.26/0.47/0.30, all gates pass) | False (R² 0.07/0.09/0.10, all gates fail — matches R1n-c's known infeasibility) |
| `throwCollapse` (`eval-normal` recall < 0.5) | False | **True** |
| `executionModeGap` (det − stochastic success on `eval-random` > 0.20) | **True** (97101: −0.04, 97102: **+0.29**, 97103: +0.03 — driven entirely by seed 97102) | False |
| `seedInstability` (success spread > 0.30, any split) | **True** (`eval-normal` spread 1.00; `eval-random` spread 0.19, `eval-easy` spread 0.04) | False |

C's flags are clean and uniform — consistent with a policy that simply
never solves either scripted opponent. M's instability and
execution-mode-gap flags both trace to the same underlying seed spread
described in headline point 5 and §7 below, not to independent problems.

## 5. eval-random (in-distribution for both conditions)

| Comparator | M | C |
| --- | --- | --- |
| 97101 | 0.45 (sto. 0.49) | 0.59 (sto. 0.45) |
| 97102 | 0.64 (sto. 0.35) | 0.67 (sto. 0.61) |
| 97103 | 0.51 (sto. 0.48) | 0.74 (sto. 0.60) |

M's `eval-random` success (0.45–0.64) is markedly lower than C's
(0.59–0.74) and lower than prior single-opponent-trained policies in this
track (R1n-c's initializers: 90%+). This is the substitution cost named in
declaration amendment A1: M's round data is 64 random + 64 scripted-easy,
not 128 random, so it gets roughly half the random-opponent practice C or
R1n-c's policies got at the same total training volume. The primary result
does not depend on this number — it is reported because it is the direct,
expected cost of the design, not a defect.

## 6. Predictions vs. results (declaration §6)

| Prediction | Result |
| --- | --- |
| M's aim error on `eval-normal` drops below 90° | **Confirmed** — 8.6° mean, all three seeds under 13°. |
| C's aim error on `eval-normal` reproduces R1n-g's archived arm-N range | **Confirmed** — 116°–144° vs archived 75°–148°. |
| Throw recall improves for M over C, by a smaller relative margin than aim | **Not confirmed as stated** — recall's improvement (0.02→0.91, a ~40× multiple) is at least as large in relative terms as aim's (128°→9°, a ~15× reduction). Both mechanisms moved together and dramatically; the prediction's ordering does not hold, but its premise (aim is not the sole story) is not contradicted either — see §7. |

## 7. What this does and does not decide

**Decides:** substituting half of imitation training's random-opponent
data for scripted-easy data — at fixed total volume (amendment A1) —
produces a policy that transfers to a fully held-out, harder scripted
opponent, both in win/loss terms (+38 points) and in the specific
mechanism R1n-g identified as most severe (aim error resolves from worse
than two random directions to near-healthy). This is not proof that the
two training opponents' data *adds* rather than *substitutes* usefully —
R1n-h has no arm that holds scripted-easy exposure fixed while varying
total volume, so it cannot separate "the mixture works" from "the mixture
works about this well *at this volume*"; a future declaration adding a
volume-controlled arm would settle that.

**Does not decide:** why label error improved uniformly across M's three
seeds while win/loss success on `eval-normal` did not (headline point 5).
Candidate explanations neither confirmed nor ruled out here: small
per-decision errors compounding differently across episode lengths;
critical single-decision failure modes (e.g., one mistimed throw near a
health threshold) invisible to aggregate recall/aim statistics; or a
genuine critic-quality interaction (M's critic gate passed, C's failed —
§6 headline point 6) affecting downstream behavior in a way label error
doesn't capture, since label error is measured on the imitation policy
directly and never touches the critic. This is the natural next diagnostic
if the roadmap continues investigating the mixture curriculum rather than
moving to M8.

**Does not decide:** anything about R1's qualification gates, and trains
no PPO — R1n-h's own next-step question (continue refining the imitation
mixture vs. move to M8's MARL scaffolding vs. run PPO from these
checkpoints) is for the next declaration, not this one.

## Verification

- Training pytest (19 tests for `mixture_imitation.py`), python client
  pytest, `npm run build`: all clean before collection.
- `npm test`: 366 passed, 1 failed — the single documented, accepted R1n-b
  `trainSeedBase: 630000` collision. No new collision from this run's
  artifacts.
- The mandatory pre-collection teacher-ceiling check (amendment A2) passed
  before either condition trained (Run facts, above).
- `validate_critic_fold_sizes` passed at declare time for the real
  configuration (256/128 total fold sizes divide evenly per arm by
  `blockWorlds=64`).
- Top-level manifest re-verified with `death_rate_ppo.verify_sealed` after
  aggregation.
- The `warm_start_critic_mc_mixture` regression guard (bit-for-bit
  reproduction of `v1.warm_start_critic_mc` at an all-random ratio) passed
  live before collection, giving confidence that condition C's critic
  numbers above are not an artifact of the mixture reimplementation.
