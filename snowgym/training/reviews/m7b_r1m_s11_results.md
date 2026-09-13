# R1m-S11: null control does not separate from S9-full real replications

Completed 2026-09-13. The [declaration](m7b_r1m_s11_declaration.md) (committed
`653d420`, budget corrected in `d24e227` before collection) and tested
implementation were committed before fitting. Six points were produced: two
new real training runs (99302, 99303), the archived S9-full/99301 checkpoint
reused read-only, and three new null (permuted-advantage) training runs
(99301, 99302, 99303). Used 468,762 of 550,000 budgeted simulator decisions.
All results below are read directly from `report.json`; no thresholds,
seeds, or measurements were changed after collection.

## Headline

**Real and null are not distinguishable on any of the six predeclared
measurements.** Where one group's extreme exceeds the other's, it goes in
both directions across measurements, and no single run shows a coherent
real-only signal across more than one measurement. This is the predicted
outcome of the review's E1, at the "null training-success gains inside the
real range" resolution rather than the tighter "within ±3 dev outcomes"
resolution originally guessed -- the outcome noise floor itself, visible in
the null runs alone, is wider than that guess.

## 1. Deterministic development success and return versus the shared initializer

Paired bootstrap, 10,000 resamples, RNG 993001 (`horizon_train.paired`).
Historical is 38 seeds, replication is 36.

| Point | Split | Success Δ | 95% CI | Return Δ | 95% CI |
| --- | --- | ---: | --- | ---: | --- |
| 99301-real (archived) | historical | +0.0% | [-10.5, +10.5] | +0.0049 | [-0.168, +0.180] |
| 99301-real (archived) | replication | +2.8% | [0.0, +8.3] | +0.0436 | [-0.005, +0.134] |
| 99302-real | historical | +5.3% | [-5.3, +15.8] | +0.0906 | [-0.078, +0.265] |
| 99302-real | replication | **+11.1%** | **[+2.8, +22.2]** | **+0.1774** | **[+0.039, +0.358]** |
| 99303-real | historical | +2.6% | [-10.5, +15.8] | +0.0405 | [-0.176, +0.258] |
| 99303-real | replication | +8.3% | [-5.6, +22.2] | +0.1380 | [-0.093, +0.370] |
| 99301-null | historical | +7.9% | [-2.6, +18.4] | +0.1338 | [-0.040, +0.313] |
| 99301-null | replication | +13.9% | [-2.8, +30.6] | +0.2299 | [-0.042, +0.505] |
| 99302-null | historical | -2.6% | [-13.2, +7.9] | -0.0476 | [-0.234, +0.137] |
| 99302-null | replication | -2.8% | [-16.7, +11.1] | -0.0518 | [-0.285, +0.181] |
| 99303-null | historical | **-18.4%** | **[-34.2, -5.3]** | **-0.3143** | **[-0.572, -0.092]** |
| 99303-null | replication | **-22.2%** | **[-38.9, -5.6]** | **-0.3815** | **[-0.658, -0.114]** |

Only two of the twelve rows have a paired interval excluding zero, one real
(99302-real, replication, positive) and one null (99303-null, both splits,
negative). The real range (historical [0.0, +5.3]%, replication [+2.8,
+11.1]%) sits entirely inside the null range (historical [-18.4, +7.9]%,
replication [-22.2, +13.9]%); a null run with permuted, uninformative
advantages produced both the single largest positive point estimate
(99301-null) and the only run with a significant interval in either
direction (99303-null, significantly *negative*). Raw outcome-count
deviations from the initializer range from -8 (99303-null, replication) to
+5 (99301-null, replication); no run stays within the originally guessed
+/-3, but the widest deviations are null, not real.

## 2. Update-1 behavior-versus-final destination change: constant-vector fit

Fit of `finalGapReduction ~ worldDirection` on opportunities with
`recommendationWorldGap > 4`, matching
`refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12_reanalysis.py`
section A. `99301-real (archived)` reuses `m7b_engage_r1m_s10_v0`'s existing
`opportunities-full-001.jsonl.gz`; the other five are captured live during
this run's own update 1.

| Point | n | Fitted vector | Magnitude | R2 | Mean shift | SD shift |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 99301-real (archived) | 1712 | (+0.073, +0.711) | 0.715 | 0.972 | 0.729 | 0.120 |
| 99302-real | 2054 | (+0.300, +0.485) | 0.570 | 0.947 | 0.583 | 0.071 |
| 99303-real | 2313 | (+0.484, -0.471) | 0.675 | 0.835 | 0.699 | 0.191 |
| 99301-null | 1712 | (+0.309, +0.028) | 0.310 | 0.548 | 0.407 | 0.152 |
| 99302-null | 2054 | (-0.556, -0.348) | 0.656 | 0.879 | 0.692 | 0.140 |
| 99303-null | 2313 | (-1.454, -0.407) | **1.510** | 0.932 | 1.479 | 0.201 |

Every point's realized policy change is still overwhelmingly a single
constant world-space translation, real or null (R2 0.55-0.97). The three
real vectors point in three different directions with no shared sign
((+0.07,+0.71), (+0.30,+0.49), (+0.48,-0.47)) -- real runs do not agree with
each other on a "correct" direction any more than they disagree with null.
The single largest magnitude of all six (1.51, more than double the largest
real magnitude 0.72) is a null run. The lowest R2 (0.55, the only point
below the original 0.7 falsifier line) is also a null run, in the direction
opposite the falsifier's concern (a null run being *less* translation-like,
not a real run being *more* state-dependent than nulls).

## 3. Per-parameter RMS displacement versus lr*sqrt(steps)

Actor (non-critic) parameter count: 18,890. `lr = 3e-4`.

| Point | L2 | Steps | RMS/param | lr*sqrt(steps) | RMS / (lr*sqrt(steps)) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 99301-real (archived) | 0.4547 | 125 | 0.00331 | 0.00335 | 0.99 |
| 99302-real | 0.4843 | 147 | 0.00352 | 0.00364 | 0.97 |
| 99303-real | 0.4622 | 122 | 0.00336 | 0.00331 | 1.01 |
| 99301-null | 0.5590 | 170 | 0.00407 | 0.00391 | 1.04 |
| 99302-null | 0.6090 | 168 | 0.00443 | 0.00389 | 1.14 |
| 99303-null | 0.4195 | 122 | 0.00305 | 0.00331 | 0.92 |

All six ratios sit within [0.92, 1.14] of the random-walk prediction
`lr*sqrt(steps)`; none approaches `lr*steps` (0.033-0.051, 8-15x every
observed RMS). Parameter displacement scale is indistinguishable between
real and null.

## 4. Stochastic training-success gain (updates 21-30 minus 1-10), adjusted for visited-frame difficulty

Adjustment subtracts each window's deterministic (zero-residual) success
rate on the training frames actually drawn that window, using
`m7b_engage_r1m_s3_v0/snapshots.json`'s baseline.

| Point | Early stochastic/det (of 80) | Late stochastic/det (of 80) | Raw gain | Adjusted gain |
| --- | --- | --- | ---: | ---: |
| 99301-real (archived) | 26/35 | 42/33 | +16 | **+18** |
| 99302-real | 32/31 | 28/26 | -4 | +1 |
| 99303-real | 33/34 | 27/32 | -6 | -4 |
| 99301-null | 25/35 | 39/33 | +14 | +16 |
| 99302-null | 31/31 | 26/26 | -5 | 0 |
| 99303-null | 25/34 | 24/32 | -1 | +1 |

Real adjusted gains: {-4, +1, +18}. Null adjusted gains: {0, +1, +16}. The
one clearly elevated value (+18) is real, but the null range reaches +16,
one point behind it -- not the "exceeds all null runs" the falsifier
requires, and the other two real runs (+1, -4) sit at or below the null
median. No falsifier triggers here.

## 5-6. Deterministic training-frame success (57 frames) and stochastic historical success (38 frames x 2 draws)

| Point | Training det. success | Stochastic historical success |
| --- | ---: | ---: |
| Initializer | 24/57 | 23/76 |
| 99301-real (archived) | 22/57 | 24/76 |
| 99302-real | 24/57 | **32/76** |
| 99303-real | 25/57 | 25/76 |
| 99301-null | **27/57** | 27/76 |
| 99302-null | 24/57 | 17/76 |
| 99303-null | **14/57** | **14/76** |

Real training-frame success (range 22-25) brackets the initializer (24)
tightly; null spans wider on both sides (14-27), including both the single
highest and the single lowest value of all seven points. On stochastic
historical success, 99302-real's 32/76 is the one point that exceeds every
null (max null 27) and the initializer (23) -- the closest this experiment
comes to a real-only signal. It is not corroborated by the same run's other
five measurements: 99302-real's training-frame success (24/57) sits exactly
at the initializer, its adjusted training-success gain (+1) is unremarkable,
and its update-1 R2 (0.947) and parameter RMS ratio (0.97) are unremarkable
too. Per the review's own gate-power finding (S9/S10; a true +20-point gain
passes this initializer-style gate only ~20% of the time), one favorable
draw on one of six measurements, unsupported by the other five for the same
run, is within the behavior expected from noise alone.

## Falsifier check

- **"A real run exceeds all null runs on (4) and on dev return on both
  splits."** Not met: 99302-real's return gains (+0.091 historical, +0.177
  replication) are both *smaller* than 99301-null's (+0.134, +0.230) on the
  same splits, and 99302-real's measurement-4 gain (+1) is far below
  99301-null's (+16).
- **"State-dependent change (constant-vector R2 < 0.7) that nulls lack."**
  Not met in the stated direction: the one point below 0.7 (0.548) is a
  null run, not a real one.
- Neither branch of the conditional falsifier on (4)/(5)/(6) applies,
  because (4) does not separate real from null in the first place.

## Interpretation and decision

All six predeclared measurements point the same way: this bounded PPO
configuration on S9-full's frames produces outcomes statistically
indistinguishable from PPO driven by advantages carrying no information
about the state or action taken. The trust-region envelope and Adam-noise-
scale parameter drift already identified in the review's reanalysis (section
2) are not artifacts of the particular archived seed 99301; three fresh
real seeds and three fresh null seeds land in the same regime. Per the
declaration's stated decision value: **S3/S9 carry no information about
learnability under this configuration; do not run a further automatic PPO
sweep on this movement-residual line.** This does not indict PPO, the
architecture, or the reward in general -- section 4 of the review's
mathematical assessment already identifies *why* (trust-region envelope
below the required change, fixed small sigma, an undertrained critic), and
this experiment confirms the diagnosis empirically rather than replacing it.
Proceed to E2 (representation) and E3 (a smaller, well-posed complete
skill), as recommended; this result does not change either experiment's
design, since neither depends on the movement-residual PPO configuration
tested here.

## Reproducibility and archive

The [manifest](../runs/m7b_engage_r1m_s11_v0/manifest.json) binds every
artifact in the run, including per-update event files, six training/
evaluation directories, `initialization.json`, and `report.json`. The first
collection attempt failed with `simulatorBudget` exceeded at 450,000 (all
five training runs and their evaluations had already completed cleanly,
using 444,484 decisions, before the archived point's fresh measurements 5-6
pushed it over); that partial output was discarded rather than archived. The
corrected budget (550,000) and its justification are recorded in the
declaration; the actual run used 468,762. No provider calls, browser input,
or protocol changes. `autonomousQualificationEligible: false` throughout;
this is an assisted-Engage diagnostic, not an autonomous-qualification
result.

The implementation gate passed 367 TypeScript, 51 Python client, and 350
Python training tests (including 8 targeted `horizon_null_train` tests)
plus build, both before the first (failed) attempt and again after the
budget fix.
