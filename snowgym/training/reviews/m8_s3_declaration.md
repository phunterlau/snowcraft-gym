# M8-S3 — 3v3 plan-conditioned training path on the batch transport, no training

Declared 2026-09-20. Infrastructure and audit, same lightened shape as
[M8-S1](m8_s1_declaration.md) and [M8-S2](m8_s2_declaration.md): declare →
implement. No policy is trained, no seed band is reserved, no run archive is
sealed. Any collection (teacher achievability, floors) is S4, declared
separately.

## 0. The decision this step makes

The external review's first work package ("contract repair and bridge") assumed
the learner would consume `SnowGymUnitParallelEnv`. Orientation found that is
the wrong path for training:

- `SnowGymUnitParallelEnv` wraps `SnowGymParallelEnv`, the HTTP, single-world,
  observation-v2 transport. It has no plan activation, no plan/role tensors and
  no option state.
- The existing training stack runs on the **batch transport**:
  `SnowGymBatchEnv` provides `step`/`step_joint`, `activate_plans`,
  `plan_observations`, `plan_teacher_actions`; `EngageOptionBatchV1` adds the
  three-field option state; `collect_plan_rollout` and the R1n `Collector` merge
  physical and plan tensors into exactly the dict `FullAuthorityPolicyV1`
  consumes. R1n-b..i exercised all of it at roster 1.

**Choice: the batch path is the trainer's path. The unit PettingZoo env stays
the API-conformance artifact** that M8's first checklist item asked for. This
step does not extend it with plan views.

What building the other way would cost: new plan-observation plumbing on the
HTTP path that duplicates the batch path (the server's `/plan-observation` and
`/activate-plan` exist but `SnowGymParallelEnv` does not call them), plus a
second observation version to keep in sync with the trainer.

**Does this satisfy M8's "parameter-shared unit actors, execution uses
actor-local inputs only"?** Partly, by design of the *first* M8 condition.
`FullAuthorityPolicyV1` is one shared network applied to every living unit slot:
`features()` builds one row per unit from the team tensor, subtracting that
unit's own position from every entity (`pair_features`), so each row is egocentric
and parameter-shared. The inputs are still **global** (every entity, the
host-grounded plan, role state), which is the roadmap's stated first condition
("begin with global actor observations"). Restricting inputs to unit-local
sensing is the next M8 checklist item and is not claimed here. The unit env's
role in M8 is conformance, not training.

## 1. Scope

In: a small new module and tests establishing that the *existing* actor, critic,
option tracker and collector run correctly at roster 3, and that Red's action
source is explicit. Out: any training, any collection with a research budget,
local visibility, latency, role/mission expansion, new observation versions,
edits to any existing module.

**Immutability.** Every existing `.py`/`.ts` is treated as frozen (archive
declarations pin source digests; `full_authority_train.py` hardcodes
`ROSTER = 1`, `engage_v1.py` and `full_authority_train_v1.py` are digest-checked
by later declarations). New behavior goes in new modules; the roster-3 scenario
is built from `full_authority_train_v1.scenario()` and `mixture_imitation.EVAL_ARMS` and
passed to the existing wrapper directly (no module-level patching).
`roster_engage.py` therefore depends on R1n-h's `EVAL_ARMS`; a later module needing a
different Red arm should define its own rather than extend that dict.

## 2. Red's action source (declared, not left implicit)

Two transports and two opponent conditions exist; S3 fixes which is used for
what:

- **Training and evaluation against scripted or random Red:** the batch `step`
  path with the scenario's `redController` (`scripted`/`random`) and
  `redDifficulty`. The built-in Red controller runs at its native cadence behind
  the team action. This is the R1n-f/h/i condition, so 3v3 results stay
  comparable to the archived 1v1 ones.
- **Explicit two-team actions:** `step_joint` / `SnowGymUnitParallelEnv`. The
  built-in Red controller is disabled for that interval, so the caller must
  supply Red. This is for conformance tests and future self-play, never silently
  substituted for the scripted benchmark.

A 3v3 scripted-Red result obtained through a joint step with NOOP Red would be a
different, easier opponent and must not be reported as the scripted benchmark.

## 3. Checks (each is a test; live where it needs the simulator)

Roster 3v3, scripted-normal Red on the batch path, arena and horizon as in R1n:

1. **Path runs.** `EngageOptionBatchV1` resets and steps at 3v3; the merged
   tensor dict has the shapes `FullAuthorityPolicyV1` requires; `act()` returns
   actions for exactly the living units; the tracker's activated targets contain
   all three red units and `assigned_ids` all three blue units.
2. **Focal rows are egocentric and shared.** Permuting the ally slots (allies,
   `ally_mask`, `unit_action_mask`, and `plan_unit_roles` rows together) permutes
   the actor's per-unit outputs the same way and leaves each unit's value and
   likelihood unchanged. This is the row-selection convention S3 pins.
3. **No aliasing.** `features()` does not mutate the input observation
   (bitwise-equal before and after); N per-unit reads of one shared observation
   see identical inputs.
4. **Likelihood consistency at roster 3.** The `logp` returned by `act()` equals
   `evaluate_latents` on the stored latents, with only selected heads counted per
   unit, and dead-slot rows contribute zero.
5. **Death handling.** In a real rollout where a blue unit dies while the option
   continues, its `living` mask turns off, its action is a no-op in the joint
   action, its rows contribute zero likelihood, and the option's team value is
   not zeroed by the death. Option termination follows the frozen Engage
   criterion, unchanged.
6. **Batch/unit-env conformance.** Same seed and same explicit two-team action
   script: `SnowGymBatchEnv.step_joint` and `SnowGymUnitParallelEnv` produce
   identical state hashes step by step. This ties the conformance artifact to
   the training transport's physics.
7. **Red routing.** With scripted Red, a native-Red batch `step` diverges from a
   `step_joint` with NOOP Red on the same seed and blue actions (state hashes
   differ), proving the two paths are not interchangeable and the declared
   routing matters.

## 4. What this does not establish

That 3v3 Engage is learnable, that the teacher is achievable at 3v3 against
scripted Red, or that any policy transfers from 1v1. Those are S4 (teacher
achievability and random-initialized floors, with a declared seed band, budget
and archive) and the baseline after it. Passing these checks means the existing
machinery is correct at roster 3, nothing more.

## 5. Verification

Complete gates before commit: `npm test` (366/367, the one documented R1n-b
exception), `npm run build`, Python client `pytest`, Python training `pytest`.
No seeds are reserved; live tests use fixed simulator seeds, not research seeds.
The new module and tests declare no JSON seed band. Wrong if: a check fails
(reported as a finding, not patched around), or an existing module needs
editing (then the path changes to a subclass and this declaration is amended).

## 6. Implementation note (2026-09-20, before commit)

Two checks were established more narrowly than §3 worded them:

- **Check 1** "actions for exactly the living units" is shown as zero likelihood
  on unused and dead slots, not as a filtered action list; the policy still
  samples an action type for a dead slot, and the batch step accepts and ignores
  it (shown by identical state hashes for THROW vs NOOP on a dead slot).
- **Check 5** "the team value is not zeroed by the death" is shown only as the
  option continuing (`tracker.finished` is false). No assertion is made on the
  critic's value or bootstrap arithmetic; that belongs to the trainer step.

Also found while implementing: with three units, a random-init blue policy loses
a unit mid-option in every world tested while the option continues, so R1n's
1v1 "death rate" has no direct 3v3 analogue. S4 must define its 3v3 success and
death metrics before collecting.
