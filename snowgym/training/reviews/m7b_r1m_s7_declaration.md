# R1m-S7: frozen handoff and completion inspection

This is an inspection of existing S6 trajectories, not a new policy experiment.
Commit the tested inspection protocol before generating its report. Replay exactly
the saved Keep and squad-30 actions for all 24 S6 training seeds: 48 trajectories,
at most 9,600 prefix-plus-continuation decisions. Do not sample actions, extend
episodes, replace orders, add seeds or train. All original assistance labels and
qualification gates remain intact.

Verify S6's manifest, implementation digest and S3/S5 lineage. Restore each saved
prefix and its physical/plan/option/observation identity. Execute only the recorded
action arrays; compare every resulting state hash, option record, action result,
reward and termination boundary. Check final outcomes and action digest. Verify
all source artifacts unchanged after inspection. Read raw server state to recover
per-unit details omitted from S6; never use a renderer or browser.

Before each recorded action, inspect living allies: current position/velocity,
controller state, existing target, move/throw masks, nearest living enemy ID,
health/distance, incoming threat, recorded action type/target and corrected-shot
recommendation availability. Count throw orders, not inferred projectile launches.
The 9-unit range and 6.5 reference are teacher conventions, not hit guarantees.
Record outward-directed MOVE orders while beyond 9 and relative range-opening
velocity, without treating either as automatically incorrect (dodging may require
retreat). Save enemy identity/health summaries at reset-prefix end, after decision
30 if reached, and terminal. Read-only recommendation helpers must agree with
production and must not mutate the world.

Fixed windows, using zero-based offsets: initial correction [0,30), late correction
[20,30), immediate post-handoff [30,40), first post-handoff block [30,60), and all
post-handoff [30,end). Report actual decision/unit exposure and incomplete windows;
use only complete windows for late-correction versus immediate-post comparisons,
with the same eligible paired seeds for Keep and squad-30. Partial all-tail windows
remain descriptive and do not pretend equal exposure.

Summaries include target-health progress gain and gain/decision, damage dealt and
received, range occupancy/error, throw-order rate, fraction of throw orders beyond
9, readiness/range opportunities not choosing THROW, and MOVE-away/out-of-range
counts. Aggregate within seed/window first. Report per-trajectory data, complete
paired-cohort means and descriptive paired bootstrap intervals (10,000 samples,
RNG 991001). Do not treat before/after changes as the causal effect of handoff:
time, survivors, target replacement, trajectory evolution and intervention history
also differ. Keep is a temporal reference, not an identical post-handoff state.

Stratify squad-30 descriptively by final success/failure and original remaining
budget: <=60, 61–100, >100 decisions. Report unavailable/empty strata. Record last
progress-gain offset and final progress-free decisions; a zero-gain trace uses its
full exposure and is explicitly marked. Report remaining objective health and
headroom to the 0.2 success threshold, without replacing the qualification score.

No outcome gate or optimizer/default changes are included. Use evidence to refine
the next proposal; retain negative findings and uncertainty. Before implementation
and results commits run targeted window/mask/identity/replay/tamper/immutability
tests plus TypeScript, build, client and training test suites.
