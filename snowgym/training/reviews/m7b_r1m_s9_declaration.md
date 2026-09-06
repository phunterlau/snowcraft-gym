# R1m-S9: matched learnable control horizon

Commit this protocol and tested implementation before fitting. Compare `short`
(30 learned decisions after first hit, then frozen source) with `full` (learned
movement until the original option terminates). Use the unchanged R1h/R1m source,
RecoveryPolicy architecture, fixed classifier and corrected shots. No runtime
teacher MOVE destinations enter either learned segment. These are assisted
Engage experiments, not autonomous qualification or full-battle win tests.

## Frozen data and control contract

Reuse digest-verified S3 frames: 57 training first-hit states from 100000–100063,
38 historical-development states from 200000–200039, and 36 replication-development
states from 210000–210039. Preserve every archived exclusion. These development
sets have already been used; neither is an untouched qualification set. Verify
all partitions, source checkpoint, S3 source files and S8 ancestor lineage before
and after fitting. S8 teacher-corrected states remain historical diagnostic evidence;
they are not added to the training distribution or used for checkpoint selection.

Replay exact first-hit prefixes before every episode. No horizon extension:
the option still ends by original decision 200. Short uses recoveryWindow=30;
full uses recoveryWindow=200 minus the archived first-hit decision. Both use the
existing normalized control-time input and separate option-budget input, with
the same actor/critic architecture. Early terminal states shorten actual control.
Zero residuals must reproduce archived source actions/hashes on every frame in
both arms. The short frozen tail is folded into its final learned reward using
the existing collector; full has no frozen tail. Only sampled MOVE latents
contribute actor likelihood. Actions, controller rules and reward definitions
remain unchanged at 10 Hz.

## Matched bounded fitting

First paired training RNG: 99301. If its predeclared gate passes on both
development sets, run paired replications 99302 and 99303; otherwise stop.
For every arm/RNG: 30 updates, eight uniformly sampled training frames per update,
with identical frame-index streams between arms. Each episode gets its own Torch
RNG stream seed `training_rng * 100000 + update_index * 8 + slot`, so extra full
episode draws do not change the next episode's stream. Sampling is isolated from
the optimizer RNG. Model and critic initialization match across arms.

Each update selects exactly 240 decision rows from its complete on-policy
collection, using a separate NumPy stream seeded `training_rng * 100000 +
update_index`. Sample without replacement when at least 240 rows are available;
otherwise include every row and draw the deficit uniformly with replacement,
then shuffle. Record selected indices, duplicate counts, unique rows and living
MOVE opportunities. Compute GAE on complete trajectories before selecting rows.
This matches presented optimizer rows, not unique collection support. There is
no additional rollout collection to fill the short arm's sample deficit.

Four PPO epochs, minibatches 120: at most eight optimizer steps/update, 240/run.
Keep Adam LR 3e-4, fixed latent std 0.02, clip ratio 0.2, separate gradient clips
0.5, mean MOVE KL stopping at 0.01, and the coupled critic-stop schedule. Actual
steps can differ after KL stops; report them rather than bypassing the stop.
Keep gamma=0.9976921765, lambda=0.9885140204, no BC or entropy bonus. These are a
matched new pair, not an exact reproduction of S3's original sampling procedure.

Save complete episode actions, hashes, option/reward records, latent samples,
behavior densities, selected optimization rows, advantages, returns and diagnostics.
Checkpoint at updates 10/20/30, with final-update-only selection. Persist optimizer,
Torch/frame-sampler RNG and history; test exact interrupted/resumed training at
update boundaries. Mid-episode continuation is not a new claim of this runner.
Archive partial results on interruption; never overwrite an output directory.

## Verification, evaluation and gates

Before training, both zero-residual arms reproduce all 131 frames exactly. Verify
stored/recomputed densities, zero unused-head gradients, finite boundaries,
source preservation, control-time paths and update-boundary resume. Every final
checkpoint is reloaded before deterministic evaluation on both development sets.
Retain complete evaluation trajectories, not just scalar scores.

For each split, compare full minus short and full minus matched initialization
on success, discounted return, progress, damage and survival. Use seed-paired
bootstrap intervals with 10,000 samples, RNG 993001. Report optimizer-seed
variation separately; do not pool repeated environment seeds as independent.

The first pair triggers replication only if BOTH development splits satisfy:

- Full success >=50%, and >=20 percentage points above initialization with a
  strictly positive paired 95% success-gain lower bound.
- Full exceeds short by >=10 success points, with strictly positive paired 95%
  lower bounds for both success and discounted-return differences.
- Both arms reject <0.1% of evaluation actions; full's actor parameters change
  measurably (L2 >1e-8) and its source remains unchanged.

The same gate is reported independently for each replication. Passing this
diagnostic supports further assisted movement research; it does not close R1.
Failing it does not authorize a noise/reward/decoder sweep or threshold relaxation.

Report per-arm collection/learned/frozen-tail/prefix counts, sample budgets,
actual optimizer steps, KL stops, actor/critic losses and gradient norms, critic
explained variance, advantage spread and action-output movement. Count all
simulator decisions including prefix restoration. Preflight <=52,400 decisions;
each paired fit plus final evaluation <=125,600; maximum three pairs plus preflight
<=429,200. Enforce a 430,000-decision overall cap. Unit tests have separate tiny
budgets and never count as experiment observations.

Before implementation and results commits run targeted tests, npm test, npm run
build, Python client tests and Python training tests. Bind immutable outputs to
declaration, dataset, implementation, ancestor and checkpoint digests. No provider
calls, browser, unrelated-file edits, commander work or automatic next experiment.
