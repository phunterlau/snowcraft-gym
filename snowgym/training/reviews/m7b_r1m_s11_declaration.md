# R1m-S11: null-control and replication of S9-full

Commit this protocol and tested implementation before collection. This is
Experiment E1 of the reviewer handoff
(`refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12.md`, section 6):
it asks whether S9-full's realized learning is distinguishable from PPO driven
by uninformative (permuted) advantages. It does not authorize training, change
a protocol, or promote a checkpoint. This is still an assisted Engage
diagnostic: corrected shots remain in force, the frozen classifier is
unchanged, and `autonomousQualificationEligible: false`.

## Scope

Full arm only. Short is out of scope: E1 targets S9-full specifically. Reuse
the unchanged S3 first-hit frames (57 training, 38 historical-development, 36
replication-development), the unchanged R1h/R1m source, `RecoveryPolicy`
architecture, corrected shots, actor KL stop, fixed latent standard deviation,
rewards, and GAE constants. Verify S3/S8/S9 ancestry (dataset digest, S8
manifest, S9 manifest and its `implementationDigest`) before collection.
`horizon_train.py` (the S9/S10 implementation) is not modified; this
experiment is a new module, `horizon_null_train.py`, that imports and reuses
`horizon_train`'s `inputs`, `episode`, `select_rows`, `subset`, `diagnostics`,
and `recovery_train`'s `ppo_update`, `combine`, `validate_frames`,
`recovery_checkpoint`. Editing `horizon_train.py` would silently invalidate
S9/S10's archived `implementationDigest` checks; a separate module avoids
that.

## Real and null arms

- **Real:** unchanged S9-full code and configuration. Training RNGs 99302 and
  99303 are new collection (the S9 replications that were never run because
  S9's first-seed gate failed). Training RNG 99301 is **not** retrained: S9's
  archived `m7b_engage_r1m_s9_v0/99301/full` checkpoint and history supply the
  real 99301 point, reused read-only. Its `initialStateDigest` is checked
  against a freshly rebuilt initializer before use.
- **Null:** identical code, frames, and per-update frame/row-selection
  schedule. After `select_rows` produces the 240 selected optimizer rows for
  an update, permute the advantage vector across those 240 rows with an
  independent NumPy generator seeded `[trainingRng, updateIndex, 0x45314E31]`
  (a fixed tag distinguishing this stream from the row-selection stream
  `trainingRng*100000+updateIndex` and the per-episode sampling stream
  `trainingRng*100000+updateIndex*batchSize+slot`). Critic targets
  (`returns`) are never touched, only `advantage`. Minibatch advantage
  renormalization inside `movement_loss` is unaffected by which 240 rows were
  permuted beforehand; permuting before the per-epoch minibatch split (not
  within it) is the intended null. Null runs use training RNGs 99301, 99302,
  99303 (three fresh runs; 99301-null has no archived counterpart to reuse).

Five new training runs total: real 99302, real 99303, null 99301, null 99302,
null 99303. This matches the reviewer handoff's stopping rule ("five new
runs").

## Instrumentation added for this experiment

`horizon_train.py`'s `train` has no live capture of per-unit teacher-direction
opportunities; that only exists as a post-hoc audit reconstruction in
`horizon_audit.py` (S10), built for auditing already-archived runs. Because
these five runs are newly collected, `horizon_null_train.train` instruments
its own update-1 collection directly: a `Capture` wrapper stands in for the
model only during update 1's eight episodes, delegating every `.act()` call
through unchanged (so training is not altered), and records each living,
recommendation-available unit's behavior mean, teacher target, and
recommendation gap via the unchanged `recovery_audit.geometry` helper. After
training finishes, one forward pass of the *final* model on the cached
update-1 observations fills in `finalWorldShift` and `finalGapReduction`,
giving the same fields S10 computed by full-episode replay, without
replaying the environment. `horizon_null_train.train` does not support
resume or mid-run pause; a single continuous session runs all five training
loops without interruption, and this is a deliberate scope reduction from
`horizon_train.train`'s resumable design, stated here rather than silently
dropped. Intermediate per-10-update checkpoints are likewise not saved (only
`final`); `events-*.jsonl.gz` retains every update's full trajectories and
history entries regardless.

## Predeclared measurements

For each of the five new checkpoints and the reused archived 99301-real
checkpoint (six points total: 2 real new + 1 real archived + 3 null new):

1. Deterministic historical/replication-development success and discounted
   return versus the shared initializer, using the same paired-bootstrap
   interval helper `horizon_train.paired` (10,000 resamples, RNG 993001).
2. Final-versus-behavior destination change on update-1 states: fit the
   constant-vector regression of `finalGapReduction` on `worldDirection`
   (opportunities with `recommendationWorldGap > 4`, matching
   `refs/..._reanalysis.py` section A), report its R² and fitted vector
   magnitude. For archived 99301-real, reuse
   `m7b_engage_r1m_s10_v0/opportunities-full-001.jsonl.gz` directly rather
   than re-collecting it.
3. Per-parameter RMS displacement (`actorParameterL2Change / sqrt(paramCount)`)
   versus `learningRate * sqrt(totalOptimizerSteps)`.
4. Stochastic training-success gain: updates 21–30 minus updates 1–10 stochastic
   successes, adjusted by the deterministic outcome (`m7b_engage_r1m_s3_v0/snapshots.json`
   baseline) of the training frames actually visited in each window.
5. Deterministic success of each final checkpoint on the 57 training frames
   (S9 never evaluated this split; it is collected fresh for every point,
   including archived 99301-real).
6. σ = 0.02 stochastic success, two independent draws per frame
   (seed `993101_000_003 + frame_seed*10 + draw`), on the 38 historical
   frames, for the shared initializer and every final checkpoint including
   archived 99301-real.

## Prediction and falsifiers (reviewer's, restated for the record)

Real and null are predicted indistinguishable on 1–4: every run within ±3 dev
outcomes per split, constant-vector R² ≥ 0.9 for both, parameter RMS within 2×
of `lr·√steps`, and real training-success gains inside the null range.
Falsifiers: a real run beats every null on (4) *and* on dev return on both
splits, or shows state-dependent change (R² < 0.7) nulls lack — either
indicates a weak real signal bounded by envelope/budget, not by uninformative
advantages. If (4) separates but (1) does not: (5) up with (6) flat means
training-frame fitting without generalization; (6) up with (5) flat means
noise-tolerant execution without genuine placement improvement.

## Budget and gates

Preflight zero-residual parity on all 131 frames, once. Five training runs at
S9-full's per-run cost, five paired dev evaluations, one shared initializer
dev/training baseline, five-plus-one training-frame deterministic evaluations
for measurement 5, six stochastic historical evaluations for measurement 6
(across initializer + 5 new + 1 archived-reused). The first attempt at this
budget (450,000, extrapolated from S9's own per-run figures) undercounted:
"full" episodes here run closer to the option's full remaining horizon than
that estimate assumed, and the five training runs plus their evaluations
alone used 444,484 decisions before measurements 5-6 for the archived point
had even run. Revised declared cap: 550,000 simulator decisions, enforced by
an `account()` guard that raises before it is exceeded; this is a corrected
resource estimate caught by running the implementation, not a change to the
protocol, measurements, arms, or seeds above. No qualification seeds,
provider calls, browser input, or other protocol changes. Stopping rule: one
pass, six new runs' worth of collection (five training runs plus the shared
initializer baseline); no extra seeds,
configurations, or additional measurements beyond 1–6.

Before implementation and results commits: targeted `horizon_null_train`
tests, `npm test`, `npm run build`, Python client tests, Python training
tests.
