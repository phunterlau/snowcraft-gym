# R1m-S6: teacher correction duration and scope

Commit this declaration and tested implementation before experimental collection.
This is a headless diagnostic with no PPO, provider calls, decoder change or
autonomous qualification. Preserve all earlier checkpoints and artifacts.

## Fixed design

Reuse exactly the 24 S5 training snapshots and selected fighter IDs. Verify the
S5 inventory, its implementation digest, the S3 source-file digests, source
checkpoint and original frames. Do not select new seeds or replace casualties.

Arms: Keep; single-1; squad-1; single-30; squad-30. The first factor is one fixed
fighter versus all living source-selected movers. The second is one decision
versus the first 30 decisions (or earlier option termination). During an active
window, recompute the production-compatible conditional teacher destination on
the actual current state. Override only MOVE targets; keep the source classifier
and corrected-shot policy. A dead single fighter is not replaced. Record living
MOVE opportunities, actual overridden IDs, changed targets and window exposure.

After the intervention window, use the original deterministic source with
corrected shots to the unchanged 200-decision option horizon. Existing persistent
move orders retain normal semantics. This tests closed-loop corrections, not
holding one destination for 30 steps. Action choice may differ across branches
because states differ; categorical weights remain fixed.

Independently reconstruct and repeat all 120 branches twice (240 executions).
Keep and single-1 must match S5 Keep and teacher traces respectively, including
every suffix physical hash, action digest and final outcome. Check physical,
plan, option and observation identity before every branch. Maximum simulation:
24*5*2*200 = 48,000 prefix-plus-continuation decisions. No additional baseline
collection or seed generation is required. Repeats verify determinism and do not
increase statistical sample size.

## Predeclared measurements and interpretation

Primary transfer contrast: squad-30 minus Keep on final success and full discounted
executor return. Further-study evidence requires success gain >=10 percentage
points, positive lower bounds of paired 95% intervals for both success and return,
and rejected-action rate <0.1%. This is a diagnostic gate, not learned-policy
qualification. No other arm can replace the primary contrast after measurement.

Report all arms, all final reward components, success, progress, damage dealt and
received, and living fraction. Report local outcomes at min(30, termination),
actual exposure and early-termination count; do not describe these as common
fixed-time samples. Compute nearest-enemy range occupancy within 9 units and
absolute range error from 6.5, first within each branch, then across seeds.
These ranges are descriptive teacher references, not hit guarantees.

For success, return, progress and damage, report paired contrasts:

- Duration with single fighter: single-30 minus single-1.
- Duration with squad: squad-30 minus squad-1.
- Scope at one decision: squad-1 minus single-1.
- Scope at 30 decisions: squad-30 minus single-30.
- Interaction: (squad-30 minus squad-1) minus (single-30 minus single-1).

Use 10,000 seed-paired bootstrap resamples with RNG 990001. Intervals are
unadjusted; secondary contrasts are descriptive. All runs share 24 exposed
training environments. More fighters and longer windows also change intervention
dose; a scope effect or interaction does not isolate a coordination mechanism.
Equal seed/prefix does not fix future Red actions after trajectories diverge.

Stop and archive either outcome. If the primary contrast passes, consider a
separately declared temporally structured movement-learning experiment after
reviewing duration/scope and exposure. Otherwise inspect transfer of the teacher
effect and scenario coverage before another PPO intervention. Do not widen noise,
change reward or promote an executor within this milestone.

## Verification gates

Test duration boundaries, alive/MOVE masks, stable casualty identity, label
availability and read-only behavior, source-action preservation, exact S5 parity,
duplicate trajectories, seed/identity tamper rejection, reward accounting,
interaction arithmetic, immutable outputs and manifest verification.
Before each implementation/results commit run `npm test`, `npm run build`,
client Python tests and training Python tests. Environment tensors are unchanged.
