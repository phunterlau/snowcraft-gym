# R1m scoped movement contract

New option PPO uses `EngageOptionBatchV1` and
`snowgym.engage-option-state.v1`. Its three fractions encode remaining option
budget, activated-target health divided by activation health, and living
assigned fighters divided by activation membership. The server exposes detached
`activationObjectives`; tactical replacement cannot change scoring membership.
Eliminated target IDs score zero health even when another cluster survives.
Timeout terminates the option and sets next shaping potential to zero.

Legacy fixed-option wrappers and Gym v0–v3 tensors remain unchanged for historical
reproduction. This version implements Engage only. Temporal counters and
hold/support geometry still require repair before those missions resume.

`AssistedMovementPolicy` freezes the R1h source's classifier and inherited
geometry. Its movement residual and option-state residual start at zero; the
separate critic pools global entities, plans, role state, and option state.
Corrected R1h shot direction/power is applied outside the actor. Runtime
assistance is recorded as `snowgym.r1h-corrected-shots.v0`; every checkpoint and
collection declares `autonomousQualificationEligible: false`.

Movement samples are independent latent Normals with fixed standard deviation
0.02. Arena coordinates use the existing tanh mapping. Rollouts store latent
samples and their behavior log densities; density is zero for all other action
types. The PPO surrogate sums movement terms and divides by the number of living
fighters in each decision. Shots, categorical choices, and unused coordinates
receive no actor likelihood or gradient. There is no BC or entropy bonus.
Same-batch likelihood reevaluation is exact; changing GEMM batch size has a
tested absolute log-density tolerance of 1e-4 at this small exploration scale.

`MovementCollector` stores exact executed-action prefixes, seeds, physical hashes,
plan identities, option tracker identities, rollout tensors, and Torch RNG state.
Restoration resets and activates each plan, replays prefixes through selected-world
steps, verifies all identities and tensors, then restores sampling RNG. Selective
reset reinstalls activation state without altering other worlds. Artificial
rollout cuts bootstrap values; option timeouts do not.

`snowgym.assisted-movement-checkpoint.v0` binds model, optimizer, RNG, optional
partial collection, source/configuration, counters, and seed cursor with semantic
and file digests. Checkpoint directories are immutable. The current tests cover
exact interrupted collection, unequal prefix lengths, zero-noise parity, finite
boundaries, per-roster normalization, actor/critic isolation, and tamper rejection.
The bounded training/evaluation runner is a separate milestone.
