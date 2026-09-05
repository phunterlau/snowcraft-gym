# R1m-S2: post-hit continuation diagnostic

Declared before collection, 2026-09-05. No learning or promotion is included.

R1h found a large full-episode movement-replacement effect and a small action-
choice effect under corrected shots. R1l improved geometric regression without
autonomous success. R1m-S1 improved critic fitting without consistent control
improvement. S2 locates the remaining movement problem in post-contact execution.

## Frozen protocol

Use the original R1h source checkpoint, digest
`sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697`,
with deterministic inference and corrected shots. Reuse the open 5v5, single-main
Engage scenario, repaired frozen-target scoring and 200-decision option horizon.
Use the already exposed historical development seeds 200000–200039. First
reproduce every R1m assisted-initialization action digest and state-hash trajectory.
These are diagnostic seeds, not a new holdout or qualification set.

For each baseline, select the state immediately after its first decrease in
activated-target health. If this never occurs or the option is already terminal,
report the exclusion and do not resample. Restore the exact reset/action prefix
before every branch; verify physical hash, grounded plan, option tracker and
all observation tensors. The branch does not reset the option budget.

Five branches share that start:

1. Keep the shot-corrected baseline.
2. Replace action choice with the production teacher for 30 decisions.
3. Replace selected movement destinations with the R1h helper for 30 decisions.
4. Replace both choice and movement for 30 decisions.
5. Replace movement destinations for the remaining option budget.

Select conditional learned heads after changing action type. Keep corrected
shots in every arm, including after assistance windows end. Check teacher/helper
agreement on visited states. Record readiness, teacher firing threshold (9 world
units), recommendations and actual accepted actions separately. These signals
do not guarantee that a shot will hit. No new movement formula is introduced.

Run each branch twice independently and require exact trace reproduction. At
most 40 baseline episodes and 400 branch executions are allowed: at most 88,000
world decisions including branch-prefix reconstruction. No enlarged horizons,
checkpoint search, training RNG search, new labels or provider calls.

## Measurements and interpretation rules

Record complete action/hash traces and prefix identities. At 30 decisions or
earlier termination, and at the original option horizon, report damage dealt,
damage received, living fraction, frozen-target progress and mission success.
Report mean absolute nearest-enemy range error relative to 6.5 units, occupancy
within 9 units, readiness/choice cross-tabs and shots beyond that threshold.
Completion time is measured from first hit; failures remain censored and are
not assigned successful completion times. Record actual exposure for early exits.

Compare each intervention with Keep on eligible paired seeds, reporting separate
success/progress intervals and the choice×movement interaction for the 30-step
factorial. Use 10,000 episode-paired bootstrap resamples, RNG 960001. Intervals
are exploratory, unadjusted for multiple comparisons, conditional on the common
baseline trigger. Do not pool duplicate executions as independent observations.

Classify a channel as supported for further study when its success difference
is at least 10 percentage points and the paired 95% lower bound exceeds zero.
If movement-rest meets this rule and movement-30 does not, prioritize a future
sustained-range diagnostic; this does not prove a statistically significant
difference between those arms. If movement-30 passes, brief recovery merits a
separate learning proposal. Apply the same rule to choice-30 as a timing control.
If none passes, stop and review trigger/window coverage. No result authorizes
automatic training, a production default, or autonomous qualification.

Every branch is explicitly teacher-assisted and autonomous-ineligible. Preserve
source/checkpoint/configuration digests and positive and negative evidence.
Run targeted isolation/reconstruction/tamper tests and the full repository gate
before implementation commit, then repeat the gate before committing results.
