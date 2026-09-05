# R1m-S2: post-hit movement recovery

Completed 2026-09-05. Brief and sustained movement correction passed the declared
criterion for further study. Action-choice correction and combined correction
did not. No learned policy was trained or promoted; autonomous R1n remains open.

## Design and coverage

The [declaration](m7b_r1m_s2_declaration.md) and tested implementation were
committed as `15d08fc` before collection. Every branch used the original frozen
R1h source, corrected shots, open 5v5 and the repaired 200-decision Engage option.
All 40 historical-development baselines reproduced their archived action digests,
complete state-hash trajectories and final scoring exactly.

Thirty-eight seeds reached a nonterminal first-hit state. Seeds 200004 and 200033
never hit and were excluded without replacement. At the common trigger, each
branch restored the exact action prefix and verified physical, plan, option and
observation identities. Five continuations per eligible seed were independently
executed twice: all 190 pairs matched exactly. Keep also matched its original
baseline suffix. The option budget was never reset or extended.

These are already exposed development seeds, and the comparisons are conditional
on a baseline first hit. Duplicates verify determinism; they do not double the
statistical sample. All branches retain teacher-corrected shots. The additional
choice/movement interventions are explicitly marked diagnostic assistance.

## Mission success at the original horizon

Differences are against Keep on 38 paired seeds. Intervals use 10,000 paired
bootstrap resamples, RNG 960001, without multiple-comparison adjustment.

| Post-hit intervention | Success | Difference, percentage points [95% CI] | Further-study criterion |
|---|---:|---:|---|
| Keep | 13/38 (34.2%) | — | Control |
| Teacher choice for 30 decisions | 16/38 (42.1%) | +7.9 [−7.9, 23.7] | Failed |
| Movement correction for 30 decisions | 23/38 (60.5%) | +26.3 [10.5, 42.1] | Passed |
| Both corrections for 30 decisions | 17/38 (44.7%) | +10.5 [−7.9, 28.9] | Failed |
| Movement correction for remaining budget | 35/38 (92.1%) | +57.9 [42.1, 73.7] | Passed |

The declaration required at least +10 points with a positive interval lower
bound. It is a diagnostic evidence rule, not an autonomous qualification gate.
Engage success means the frozen target-health threshold was reached; it is not
a full-team elimination win.

The 30-decision choice×movement interaction for final success was −23.7 points
[−39.5, −7.9]. Its progress interaction was approximately zero [−0.066, 0.069].
Combining the two teacher channels did not produce an additive success gain.
This applies to this trigger, horizon and source; it does not establish that
the teacher's action-choice policy is generally harmful.

## Local consequences and engagement geometry

Within 30 decisions of first hit (or earlier termination), movement correction
increased damage dealt by 55.3 health units [36.3, 76.3] and damage received by
16.8 [3.2, 30.5]. The progress gain was 0.111 [0.073, 0.153]. Action-choice
correction alone reduced local damage dealt by 22.1 [2.1, 40.5] health units.
The two movement arms have identical local outcomes because their actions are
identical during the first 30 decisions.

At the original horizon, movement-30 increased progress by 0.148 [0.084, 0.215];
movement-rest increased it by 0.231 [0.160, 0.303]. Failed completions retain
censoring indicators and actual exposure, rather than fabricated completion times.

| Arm | Mean absolute range error from 6.5 | Occupancy within 9 units | Throws beyond 9 units / all throws |
|---|---:|---:|---:|
| Keep | 8.742 | 12.8% | 1481/1691 |
| Choice-30 | 8.387 | 15.0% | 971/1313 |
| Move-30 | 7.698 | 26.3% | 1082/1486 |
| Both-30 | 7.887 | 26.5% | 939/1554 |
| Move-rest | 3.877 | 43.0% | 610/1151 |

Geometry summaries pool living-unit opportunities over each continuation's
actual exposure. They are descriptive: different termination times change
exposure, and nine units is the teacher firing threshold, not a physical
guarantee of a hit. The movement helper bundles range keeping, formation,
cohesion and dodging; this experiment does not identify the causal contribution
of each component. Readiness, recommendation availability, targets, previous
movement targets and accepted action results are retained in the traces.

## Design feedback and next proposal

The frozen policy can reach contact and make a first hit while failing to
maintain effective subsequent movement. Thirty corrected decisions can produce
a lasting benefit after returning to the frozen policy. Continued correction
has a larger observed effect, while action-choice correction has weak evidence.
Together with R1m-S1, these findings prioritize control consequences over another
critic-budget sweep.

The next proposed learning milestone is a separately declared **short movement-
recovery option**: train from reproducible post-hit snapshots in the training
partition, allow a fixed 30-decision learned movement intervention, then resume
the frozen continuation. Use the corrected-shot initializer as the matched
assisted control, retain the classifier, and evaluate full remaining-horizon
consequences as well as local damage/range. Audit action likelihoods, snapshot
lineage and zero-residual parity before a bounded PPO run. Seed allocation,
training budget and learning gates must be declared before collection. The 38
development snapshots here must not become training data.

This proposal was not executed. Shot assistance would still make that stage
autonomous-ineligible. R1m replication, removal of runtime assistance, autonomous
Engage, remaining M7b missions and M7c composition retain their existing gates.

## Artifacts and verification

The [archive](../runs/m7b_engage_r1m_s2_v0) contains 40 compressed baselines,
190 compressed branches, a [paired report](../runs/m7b_engage_r1m_s2_v0/report.json),
source/configuration digests and a sealed manifest. All 232 listed artifact
digests and the manifest digest verified with exact inventory coverage; stored
duplicate digests also verified. Source weights and historical artifacts remained
unchanged. No provider calls, browser input or optimizer updates occurred.

Baseline and continuation executions, including duplicates and excluding prefix
reconstruction, used 34,008 world decisions and had zero rejected actions among
170,040 recorded action results. Each such decision made one read-only production
teacher query and one call to each movement/shot helper; prefix restoration
replayed recorded actions without those queries. Including 38,750 reconstructed
prefix decisions, the run used 72,758 of its maximum 88,000 world decisions.
The declaration's prohibition
on new labels means no new training-label corpus: diagnostic recommendations
were explicitly required and are archived as telemetry.

Targeted tests passed for window boundaries, channel isolation, source
preservation, baseline reproduction, tampered prefix identities, deterministic
branches and censoring. Full gates passed before implementation commit and
again before result commit: 367 TypeScript tests, 51 client tests, 264 training
tests and the production build. Existing bundle-size and legacy-Gym warnings
remain. No observation contract changed.
