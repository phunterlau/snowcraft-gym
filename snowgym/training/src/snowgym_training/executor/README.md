# SnowGym neural executor

This directory owns the fast, repository-native neural policy used to control
one SnowGym team. It is deliberately separate from the slow LLM commander:
GPT-5.6 Luna may propose a bounded symbolic `CommandPlan`, while this local
PyTorch model converts authoritative observations and host-resolved plan
tensors into physical actions at the normal decision rate.

The canonical implementation is [`model.py`](./model.py). The former
`snowgym_training.model` module is a compatibility export so existing scripts,
tests, and checkpoint tooling continue to load unchanged.
The equations used by this model and its trainers are derived in the
[`training/math` reference](../../../math/README.md).

## Ownership and boundaries

- The architecture and training implementation are part of this repository;
  they are not downloaded foundation-model code.
- Checkpoint weights are trained locally from SnowGym trajectories and rewards.
- The simulator remains independent of Torch. Python receives detached,
  fixed-shape observations from the authoritative TypeScript environment.
- The LLM never emits unit IDs, coordinates, or physical actions. It produces
  only the versioned symbolic plan consumed through host-owned adapters.
- Checkpoint metadata binds architecture, dataset, source revision, training
  configuration, and semantic state digests. Loading uses restricted Torch
  state loading and validates those records.

Repository licensing terms are not defined by this package; they must be
established at the repository level before external distribution.

## Data flow

```text
SnowEnvironment / SnowGymBatchEnv
  -> detached entity tensors and legal-action masks
  -> host-resolved plan tensors and per-unit assignments (when enabled)
  -> EntityPolicy shared entity encoders
  -> per-unit action type, target, and throw-power heads
  -> Gym action validation
  -> authoritative TypeScript step
```

The policy never reads pixels or browser state. Rendering is used only to
replay an already-recorded trajectory.

## Inputs

The physical observation contains masked fixed-capacity tensors for allies,
enemies, projectiles, and obstacles, plus team counts, simulation tick, and a
per-unit legal-action mask. Entity rows are encoded independently and reduced
with masked mean and maximum aggregation, which keeps the shared actor usable
across supported roster sizes.

Optional relational features add per-ally enemy attention, nearest-living-enemy
geometry, or deterministic target priors. A plan-conditioned model additionally
requires:

| Field | Shape | Meaning |
| --- | --- | --- |
| `plan_groups` | `[batch, 3, 38]` | Host-resolved main, maneuver, and reserve directives |
| `plan_group_mask` | `[batch, 3]` | Which group rows are present |
| `plan_unit_roles` | `[batch, units, 3]` | Host-owned assignment from each living ally slot to a group |
| `plan_role_state` | `[batch, 3, 20]` | Physical centroid, velocity, cohesion, health, readiness, objective, support, flank, and phase summaries |
| `mission_progress` | `[batch, 3]` | Host-computed instantaneous mission progress in stable role order |

Per-unit directive features are derived inside the model by selecting the
assigned group row. Raw entity IDs remain outside the learnable observation.

## Outputs

For every present ally slot, `EntityPolicy` emits:

- masked categorical logits for `noop`, `move`, `throw`, and `hold`;
- a bounded normalized two-dimensional target;
- bounded throw power.

Action-conditioned heads can learn distinct move and throw targets. Legal
action masks are applied before selection, and absent slots can only emit the
compatible no-op behavior.

## Architecture variants

`ModelConfig` defines one compatible family rather than unrelated models:

| Option | Purpose |
| --- | --- |
| `pairwise_enemy_attention` | Adds an ally-relative masked attention summary over living enemies |
| `action_conditioned_targets` | Separates move and throw target predictions |
| `nearest_enemy_features` | Adds local nearest-enemy geometry to each ally actor |
| `plan_conditioned` | Encodes the three symbolic group rows into global plan context |
| `plan_target_only` + `separate_target_actor` | Keeps plan target learning from perturbing the physical action classifier |
| `plan_action_adapter` | Adds a zero-initialized plan residual to action logits |
| `plan_role_conditioned` | Conditions residuals on main, maneuver, or reserve assignment |
| `plan_unit_directive_conditioned` | Supplies the full resolved directive for each unit's group |
| `plan_directive_experts` | Routes residuals through separate engage, advance, hold, withdraw, and support experts |
| `physical_role_state_conditioned` | Adds the owning and supported role rows to each fighter residual and selects an independent centralized role-aware critic |
| `plan_ppo_residuals` | Adds one zero-output shared residual for action logits, learned move/throw targets, and power; also selects the independent role-aware critic |

The zero-initialized adapters preserve inherited checkpoint behavior before a
new training step. Invalid option combinations fail during configuration
loading rather than silently changing checkpoint semantics.

The role-aware centralized critic owns separate ally, enemy, projectile, and
obstacle encoders. It pools global entities, symbolic plan rows, all physical
role rows, and mission progress; it does not reuse the actor target pathway.

M7 plan PPO starts from the accepted target-only checkpoint. Its legacy entity
columns are copied into v3 encoders, appended columns are zero initialized, and
a split first-layer calculation preserves inherited policy outputs exactly when
the new fields are zero. Stage 1 trains the shared residual, v3 entity adapters,
and critic while inherited actor parameters remain frozen;
stage 2 opens action, target, and power heads at one tenth of the new-module
learning rate; stage 3 opens final entity-encoder layers only after both
physical and plan gates pass.

## Current checkpoints

Representative committed artifacts are:

| Artifact | Parameters | Status |
| --- | ---: | --- |
| `runs/plan_bc_ablation_qual_v1/plan-conditioned` | 47,649 | Passed the frozen offline plan-target qualification |
| `runs/plan_directive_experts_v3_dev` | 145,269 | Historical mission-expert checkpoint; retained negative closed-loop evidence |
| `checkpoints/bc_10v10_terrain_relational_v0` | 23,495 | Historical v1 random-opponent evidence; requires v2 requalification |

The current R1 recovery checkpoint is
`runs/m7b_engage_r1f_supervised_probe_v0/epoch-020`. It has zero Engage successes
on 40 development seeds under `snowgym.sim.v2`. The frozen R1g intervention
matrix identifies throw direction/implicit enemy selection as a major execution
bottleneck; it does not qualify this checkpoint. See
[R1g results and design feedback](DESIGN_FEEDBACK_R1G.md).

These are small CPU-oriented policies. GPT-5.6 Luna is not embedded in any of
these checkpoints.

## Execute an existing M7 checkpoint

From `snowgym/training`:

```bash
uv sync --extra dev --extra learn

.venv/bin/snowgym-evaluate-plan-closed-loop \
  --ablation runs/plan_bc_ablation_qual_v1 \
  --conditioned-checkpoint runs/plan_directive_experts_v3_dev \
  --suite src/snowgym_training/configs/plan_closed_loop_behaviors_v1.json \
  --output /tmp/snowgym-m7-closed-loop.json \
  --json
```

The command starts the persistent headless batch host itself. It needs neither
the HTTP server nor a browser. Output paths are immutable: choose a new path for
each run.

## Train the present supervised executor

The matched no-plan/plan-input trainer is useful for architecture checks:

```bash
.venv/bin/snowgym-run-plan-ablation \
  --dataset <audited-plan-trajectory-dataset> \
  --config src/snowgym_training/configs/plan_bc_ablation_qual_v1.json \
  --output /tmp/snowgym-plan-ablation \
  --json
```

Replace the placeholder with an existing audited dataset path, and never
overwrite a retained run. The qualification-v1 checkpoint is committed, but
its original generated training corpus is not presented here as a reusable
path.
DAgger collection and correction commands are documented in the package-level
[`README.md`](../../../README.md).

## M7 research status and next execution seam

The plan-conditioned model has passed a frozen offline target-following gate,
but supervised variants have not jointly solved direct, flank, hold, withdraw,
and support behavior in closed loop. The retained failures are evidence against
continuing unguided behavior-cloning variants.

The plan-aware collector and fixed-plan option trainer are implemented in
[`options/train.py`](../options/train.py). They activate plans after resets,
refresh plan and role observations, retain those inputs during PPO, restore
selective resets, and track mission reward separately from canonical returns.
Use the [option commands](../options/README.md) for this path; do not infer
plan-aware collection from the generic `snowgym-train-ppo` command.

R1e PPO and R1f supervised recovery both failed to complete Engage. R1g then
tested frozen throw-channel replacements without training: learned execution
passed 0/40, direction replacement 7/40, direction plus power 10/40, and the
full teacher 40/40. R1h completed the conditional-action diagnostic: with shots
corrected, teacher movement plus learned choice passed 40/40; teacher choice
plus learned movement passed 11/40. This expands the next representation probe
to relative movement as well as shots, while initially retaining the inherited
action classifier. See [R1g design feedback](DESIGN_FEEDBACK_R1G.md) and the
[R1h result and updated design priorities](../../../reviews/m7b_r1h_results.md).
Gradient reachability, improvement over matched priors, and plan selectivity
remain required before further PPO continuation or executor promotion.

The [R1i matched geometry-feature probe](GEOMETRY_PROBE.md) is complete.
Absolute/relative residuals achieved 0/40 and 1/40 Engage successes, improving
progress and HOLD separation without changing the inherited classifier. Relative
features showed no clear advantage over the matched control. See the
[result and next decoder-probe decision](../../../reviews/m7b_r1i_results.md).
Its custom checkpoints are deterministic-only and remain separate from PPO.

The [R1j decoder probe](DECODER_PROBE.md) is complete: all four fitted arms
scored 0/40 Engage successes, and no decoder improved mean progress over the
absolute control. Keep the R1i absolute model as a development reference and
audit learner-state per-head errors and label coverage before a corrective-data
fit. See [R1j results](../../../reviews/m7b_r1j_results.md). No production
controller changed; custom checkpoints remain deterministic-only.

## Design invariants

- Preserve legacy checkpoint shapes unless an architecture flag explicitly
  introduces a new path.
- Keep fixed-size tensors and masks stable across roster sizes.
- Never put raw unit IDs, enemy IDs, or unrestricted coordinates into a
  commander plan.
- Keep reflexes, action validation, target replacement, and lifecycle fallback
  host-owned.
- Report offline imitation, closed-loop execution, and online-LLM orchestration
  as separate evidence classes.
- Freeze evaluation seeds and thresholds before a qualifying run; retain failed
  artifacts instead of selecting only successful checkpoints.
