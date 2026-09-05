# R1m-S1: isolate the critic update schedule

Declared before collection, 2026-09-05. This is an assisted mechanism diagnostic.
No checkpoint from this experiment qualifies autonomous Engage.

## Evidence and hypothesis

The archived R1m runs stopped all optimization on actor KL. Their independent
critics received 998, 1265 and 1036 of 1600 scheduled minibatch updates. All three
first updates stopped after one minibatch. This establishes a budget coupling;
it does not establish that critic underfitting caused the failed replication.
Value losses from different visited trajectories cannot settle that question.

## Matched experiment

Use the exact R1h source and R1m assistance, architecture, reward, latent noise,
learning rates, clipping, batch size, horizon, epochs and minibatches. Compare:

- Control: existing actor-KL stop ends both optimizers.
- Candidate: the same stop freezes actor parameters and Adam state; the critic
  completes the remaining scheduled minibatches, including the stopping batch.

Use training RNGs 94001, 94002, 94003, each with both arms and exactly 20 updates
(8 worlds × 200 decisions/update; 192,000 maximum collected world decisions
across six runs). Final-update-only evaluation uses the existing historical and
replication-development blocks of 40 seeds each. Both blocks are already exposed;
neither is an untouched qualification set. Training stays in 100000–107999 with
the existing reset schedule. No new seed allocation or data supervision is added.

The candidate uses extra critic computation, explicitly reported. Additional
minibatch permutations after the stop must not consume the subsequent collection
RNG stream: restore the RNG state captured at the actor stop. Before the stop,
control and candidate have identical computations. Later rollouts may diverge
because the critic changes advantages. Equal seeds do not imply equal actions.

## Mechanism checks and decision rule

Before long collection, run both updates on an identical first rollout for each
RNG. Require exact actor parameters, actor Adam state and post-update RNG parity;
report critic parameter change and full common-rollout fixed-return MSE before
and after. This is in-sample fitting evidence, not held-out critic generalization.

For the 20-update runs, report paired success/progress differences and bootstrap
95% intervals separately for each optimizer RNG and each development block
(10,000 resamples, bootstrap RNG 950001). Do not pool runs as independent worlds.
Call the schedule promising only if all three fresh-development paired success
differences are nonnegative, their mean is at least +10 percentage points, at
least one has a positive interval lower bound, and every candidate has rejected
actions below 0.1%. Otherwise archive and stop this schedule branch. This threshold
selects a possible later, separately declared stability experiment; it does not
replace the original R1m or autonomous R1 gates. No automatic budget extension,
aiming experiment, provider call or commander change follows either outcome.

Archive source/config/data digests, first-rollout checks, events, final checkpoints,
evaluations and a concise positive/negative report. Existing artifacts remain
immutable. Targeted tests and the full repository gate precede implementation
commit and collection.
