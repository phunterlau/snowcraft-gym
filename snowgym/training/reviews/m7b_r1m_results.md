# R1m assisted movement PPO: initial effect, inconsistent replication

Completed 2026-09-05. This is a headless, provider-independent mechanism test.
Every checkpoint uses runtime teacher shot direction and power,
`snowgym.r1h-corrected-shots.v0`, and is marked
`autonomousQualificationEligible: false`.

## Implementation and baseline

The [versioned contract and command](../src/snowgym_training/executor/SCOPED_MOVEMENT.md)
provide activation-frozen target membership and initial health, explicit option
budget/state, a separate option-aware centralized critic, latent movement-only
likelihoods, guarded selective-world steps, and exact interrupted collection
and optimizer resume. Historical option scoring is retained for reproduction;
new training uses `snowgym.engage-option-state.v1`.

The pinned R1f checkpoint used by R1h reproduced **all 40 historical records
exactly**, including action digests, every physical state hash, and 10/40 Engage
successes. The R1i checkpoint was not substituted for it.

Repaired scoring gives **13/40 historical and 16/40 fresh-development** successes
at zero residual. All 40 historical trajectories have identical shared physical
prefixes. Seeds 200006, 200014, and 200017 now finish successfully at decisions
151, 159, and 192. The existing success predicate includes health fraction 0.2;
the old float32 role summary could represent it slightly above 0.2. Scoring now
uses activated members' raw health and their frozen activation denominator.
This baseline change earns no learning credit.

Pre-training calibration followed the deterministic baseline on seed 100000.
Across 950 movement opportunities, independent latent standard deviation 0.02
gave mean target displacement 0.9794 world units and p90 1.7372. No sampled target
coordinate had absolute normalized value at least 0.99. This describes one
calibration trajectory, not a guarantee against saturation during training.

## Frozen experiment

Each RNG ran 8 worlds × 200 decisions × 100 PPO updates: 160,000 world-decision
transitions. Four epochs, minibatches of 400, Adam 3e-4, clip ratio 0.2, actor and
critic gradient clips 0.5, and mean movement KL stopping at 0.01 were unchanged.
Latent noise stayed fixed at 0.02. There was no BC anchor or entropy bonus.
Gamma and lambda retained the declared time-based values. Categorical choice,
inherited source weights, and corrected shots stayed frozen.

Training schedules consumed seeds 100000–101599 in each run, including selective
resets and unfinished episodes at rollout cuts. All remain in the training
partition; fresh label holdouts 108000–108039 and 108100–108139 were not fitted.
Only final-update deterministic policies were evaluated. Historical development
is 200000–200039; fresh development is 210000–210039. Qualification seeds remain
sealed. Repeated evaluations of the same 40 seeds are not 120 independent worlds.

## Final results

Intervals below are paired environment-seed bootstrap intervals for success
gain versus the matched assisted initialization, in percentage points.

| Training RNG | Historical success | Gain, 95% interval | Fresh success | Gain, 95% interval |
|---|---:|---:|---:|---:|
| Assisted initialization | 13/40 | — | 16/40 | — |
| 94001 | 29/40 | +40.0 [25.0, 55.0] | 24/40 | +20.0 [0.0, 37.5] |
| 94002 | 18/40 | +12.5 [0.0, 25.0] | 18/40 | +5.0 [-7.5, 17.5] |
| 94003 | 16/40 | +7.5 [-7.5, 22.5] | 18/40 | +5.0 [-10.0, 20.0] |

Only RNG 94001 passed the historical gate. Its fresh interval touches zero, so
that gate failed. The two subsequent RNGs failed the required success gain and
positive-interval tests. All final evaluations had zero rejected actions.
The report's `replicated: true` means the two declared replication runs were
completed; it does not mean their result gates passed.

Actor parameter L2 changes were 1.2030, 1.2550, and 1.2341. The first two runs
performed 998 and 1,265 optimizer minibatch steps; KL stopped 54 and 32 updates,
respectively. Full traces preserve each run's losses, gradient norms, KL stops,
option completions, and decomposed rewards. A KL stop shortens the update's
optimization, not its already collected 200-decision rollout.

Engage success is the declared target-health option criterion. Evaluation stops
at the option boundary; its `physicalWin` field is not a full-battle win-rate
measurement. No M7c physical-control or composed-mission claim follows here.

## Decision and remaining issues

Reward-driven movement produced a substantial initial assisted effect, but the
fixed protocol did not replicate that effect consistently. The between-RNG
variation is material and must not be hidden by averaging runs or selecting
94001 as though it were a qualified executor. These runs vary new-module
initialization, movement sampling, and optimizer minibatch order together;
the experiment does not isolate which source dominates the variation.

Stop at the predeclared three-run budget. R1n remains open: runtime shot
assistance has not been removed and autonomous Engage has not passed. A next
review should use these traces to choose a narrowly specified stability or
aiming/composition experiment, with a fixed source checkpoint and budget before
new training. Temporal hold/support repairs and later mission qualification
remain required. No additional decoder, PPO sweep, commander, or provider work
was started.

## Artifacts and verification

The immutable [run directory](../runs/m7b_engage_r1m_movement_v0) contains the
historical reproduction, noise calibration, matched baselines, three final
checkpoints, paired episode trajectories, 300 compressed per-update event files,
training traces, configuration, and source/data/checkpoint digests. Its
[report](../runs/m7b_engage_r1m_movement_v0/report.json) preserves positive and
negative gates. File and semantic checkpoint digests verified after completion.

The final gate passed 325 TypeScript tests, production build, 51 Python client
tests, and 257 Python training tests. Live Gymnasium checks passed for v0–v3
using an isolated headless loopback server, which was then stopped. Tests cover
activated multi-target elimination, budgets and terminal potential, selective
reset, unequal exact prefixes, zero-noise parity, stored/recomputed latent
likelihoods, unused-head gradients, roster normalization, finite boundaries,
immutable checkpoints, tamper rejection, and exact resumed PPO state.
