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
The stochastic action is the stored latent coordinate; tanh and arena scaling
are its deterministic execution map. Ratios use latent densities directly, so
no inverse-tanh reconstruction is needed at floating-point saturation boundaries.
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
The bounded runner below uses this contract; its research outcome is reviewed
separately in [R1m results](../../../reviews/m7b_r1m_results.md).

## Bounded experiment entry point

From the repository root, choose a new output directory:

```bash
snowgym/training/.venv/bin/python -m snowgym_training.options.movement_train \
  --output snowgym/training/runs/my-r1m-movement-run
```

The frozen configuration pins the R1f checkpoint used by R1h; all 40 historical
corrected-shot trajectories must reproduce exactly before training. A separate
baseline uses repaired v1 option scoring. Pre-training noise calibration reports
world-space displacement and tanh saturation while following deterministic
baseline trajectories; it does not tune the noise level.

Training uses 8 worlds, 200 decisions per rollout, 100 PPO updates, four epochs,
minibatches of 400, learning rate 3e-4, clip ratio 0.2, and separate actor/critic
gradient clips of 0.5. Mean movement KL above 0.01 stops the current update.
Gamma 0.9976921765 and lambda 0.9885140204 retain the 30-second return and 5-second
trace half-lives at 10 Hz. Mission, combat, shaping, canonical, and executor
rewards remain separate in compressed per-update event archives. Seeds are drawn
from 100000–107999, excluding the fresh teacher/learner holdout blocks.

Final-update deterministic evaluation compares against matched assisted
initialization on historical and fresh development seeds. Replication runs
94002/94003 require the first run's historical-set success gain of at least
20 percentage points, a positive paired interval, parameter change, and rejected
actions below 0.1%. Environment-seed intervals and optimizer-seed variation are
reported separately. Intermediate checkpoints cannot be selected for reporting.

For a diagnostic interruption, add `--pause-after 100`. Resume into another new
output directory using `--resume <prior-output>/94001/partial`. The configuration,
source, training RNG, seed cursor, executed prefixes, optimizer, and sampling RNG
must match. A resumed report identifies prior completed updates and its parent
checkpoint; its trace covers only updates executed by that invocation.
