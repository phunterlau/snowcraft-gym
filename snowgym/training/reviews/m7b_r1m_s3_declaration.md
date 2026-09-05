# R1m-S3: learned short movement recovery

Declared before collection, 2026-09-05. S2's 30-decision movement intervention
raised conditional success from 13/38 to 23/38. This experiment asks whether
reward-only PPO can learn such a brief intervention. It does not remove shots
assistance or qualify autonomous Engage.

## Data and boundaries

Freeze the original R1h source checkpoint and its digest used in S2. Collect
baseline first-hit prefixes on training seeds 100000–100063 only. These seeds
are within the already used training allocation; no source-unseen claim is made.
Exclude absent or terminal triggers without replacement. Require at least eight
eligible training snapshots, or stop. Snapshots retain reset seed, action prefix,
physical/plan/option/tensor identity and baseline trajectory digest.

Historical development uses S2's 38 eligible snapshots from 200000–200039.
Replication-development uses baseline first-hit snapshots from 210000–210039;
these seeds are also already exposed. The latter baselines must reproduce the
archived R1m initialization. Neither development set enters collection or updates.
Audit source and archived manifests, snapshot digests and disjoint seed sets.
No new qualification seeds, maps, opponents, providers or commander changes.

## Control and learning contract

Restore each baseline prefix exactly. For at most 30 decisions, the policy may
sample movement latents; categorical choices and corrected shots remain fixed.
Then return to the unchanged source actor until the original 200-decision Engage
budget ends. A no-op retains controller state. Handoff does not cancel movement,
reset scoring, reset simulator state or extend time.

Keep R1m's geometry modules, per-living-unit PPO normalization, fixed latent
standard deviation 0.02, learning rate 3e-4, clip ratio 0.2, separate gradient
clips 0.5 and coupled actor/critic KL stop at 0.01. No BC or entropy bonus.
Use a new checkpoint/observation-input version with a one-feature recovery-time
tensor. A zero-initialized actor path and a fresh independent role-aware critic
receive it. Legacy Gym/option tensors and checkpoint loaders remain unchanged.
Zero-residual deterministic recovery must exactly reproduce Keep.

Use the existing executor reward and discount values: gamma 0.9976921765,
lambda 0.9885140204. After the learned window, collect the actual frozen-policy
continuation to termination. Fold its discounted rewards into the last learned
transition before computing GAE. Thus no learner likelihood or gradient is
assigned to frozen continuation actions, and rewards beyond the window are
retained. Include true terminal potential zeroing and timeout semantics. Store
reward components, tail rewards, sampled latents, behavior log probabilities,
actions and state hashes. The critic sees remaining recovery time as well as
remaining option budget.

## Fixed budget and selection

Run three training RNGs 94101, 94102, 94103, each for 30 updates. Each update
samples eight eligible training snapshots uniformly with replacement, then runs
one recovery plus frozen continuation for each. Four PPO epochs, minibatch size
120, final-update-only evaluation. Maximum 21,600 learner-controlled decisions
and 144,000 simulator decisions including reset-prefix replay across training.
Snapshot construction and deterministic evaluation are separate and reported.
No automatic extra updates, learning-rate sweeps, alternate initializers or
post-hoc checkpoint selection.

Evaluate all three final checkpoints on both development sets against their
matched Keep baselines; report conditional coverage and success/progress paired
bootstrap intervals (10,000 resamples, RNG 970001) separately by optimizer RNG.
A run passes the assisted learning gate only with at least +20 percentage points
success, a positive paired 95% lower bound, changed actor parameters and rejected
actions below 0.1%, on both development sets. Call the mechanism replicated only
if all three runs pass. Failure stops this run at its budget. Existing autonomous
R1/M7b/M7c gates remain unchanged.

## Verification and delivery

Before collection: test recovery-time validation, zero-initialization parity,
stored/reevaluated latent likelihoods, unused-head gradients, correct discounted
tail folding, unchanged handoff, exact prefix/tamper checks, seed separation,
immutable outputs and checkpoint identity. Support exact resume at completed
update boundaries with optimizer, Torch RNG, snapshot sampler RNG, dataset digest
and configuration restored; partial updates are replayed from the last complete
checkpoint, never treated as completed. Test interrupted/resumed equivalence.
Save immutable checkpoints at updates 10 and 20 and the final update 30. Their
intermediate weights are for recovery only, not evaluation or selection.

Run the full TypeScript/build/client/training test gate before implementation
commit and again before results commit. Archive all final models and traces,
whether the result is positive or negative. No automatic aiming experiment follows.
