# Fixed-plan mission options

This package owns the M7b executor objective. Each episode activates one
validated `CommandPlan`; commander scheduling, lifecycle replacement, and
provider calls remain disabled. The TypeScript server continues to own physics,
assignments, target replacement, plan projection, and action validation.

`FixedPlanOptionBatchEnv` combines the v3 physical observation with fresh
`[3,38]` plan tensors, `[3,20]` role state, and mission progress. Its reward
record keeps five values separate:

```text
mission
combat
potential shaping
canonical battle reward
executor = mission + 0.1 * combat + shaping
```

Combat is clipped to `[-1,1]` after damage dealt and received are normalized by
the corresponding team’s initial maximum health. Mission success produces
`+1`. Assigned-group elimination, battle failure before success, or the option
horizon produces `-1`.

## Frozen definitions

| Option | Success condition | Horizon |
| --- | --- | ---: |
| Engage | Selected objective health is at most 20% | 200 |
| Advance | At least 80% of living members are within 10% of arena diagonal from the frozen region anchor for 10 decisions | 150 |
| Hold | Every living member is within 8% of arena diagonal from the activation anchor for at least 90% of 150 decisions, with half the assigned force alive | 150 |
| Withdraw | At least 80% of living members remain within 10% of arena diagonal from the backfield anchor for 20 decisions, with half the assigned force alive | 200 |
| Flank | The group reaches the commanded signed 20% activation-frame lateral extent and damages an enemy within the following 50 decisions | 200 |
| Focus | After damage equal to 10% of one mean enemy maximum-health unit, target-damage HHI is at least 0.65 | 200 |
| Distributed | At least two enemies each receive meaningful aggregate damage and damage entropy normalized over the initial enemy roster is at least 0.65 | 200 |
| Support | Both groups retain half their assignments while their living centroids remain 8–18% of arena diagonal apart for 30 decisions | 300 |

“Settled” in Withdraw is operationalized as maintaining the arrival condition
for 20 consecutive decisions. No velocity threshold is added. Damage
concentration uses health loss per enemy ID as host-side evaluation metadata;
IDs do not enter the learned tensors.

The immutable protocol is
[`m7_option_protocol_v0.json`](../configs/m7_option_protocol_v0.json). It fixes
thresholds, 10 Hz timing, 40 paired development seeds and 100 paired
qualification seeds per mission, plus disjoint training, plan-generation, and
sealed-map ranges.

## Teacher achievability gate

`evaluate_teacher_option` runs the production `PlanAwareTeamController` through
the same persistent server and option tracker used by learning. The test suite
executes all eight options on predeclared seeds. The support construction uses
a 4+1 grouping on a 30×20 arena because the teacher’s bounded lateral support
offset cannot reach the unchanged 8% minimum on the original 100×80 proof
arena. This scenario adjustment was made before PPO thresholds were frozen.

Passing the teacher gate establishes achievability. It does not establish a
learned-policy result or M7b qualification.

## PPO infrastructure smoke

From `snowgym/training`, install the updated entry points and run one immutable
stage-1 artifact:

```bash
uv sync --extra dev --extra learn

.venv/bin/snowgym-train-option-ppo \
  --option engage \
  --worlds 2 \
  --rollout-steps 4 \
  --target-updates 1 \
  --anchor-total-updates 100 \
  --infrastructure-smoke \
  --output /tmp/snowgym-m7b-engage-stage1
```

The runner loads `runs/plan_bc_ablation_qual_v1/plan-conditioned`, migrates its
policy to v3, freezes inherited parameters, and records the root initializer,
plan/seed cursors, reward components, optimizer metrics, source revision, and
semantic checkpoint digests. Exact resume uses `--resume`. A stage-2 run uses
`--ppo-warm-start` with a stage-1 checkpoint and opens inherited action,
move/throw target, and power heads at one tenth of the new-module learning
rate. Stage 3 refuses to start unless both `--physical-gate-passed` and
`--plan-gate-passed` are explicit.

Development mode defaults `--rollout-steps` to the selected option horizon and
rejects shorter values. This guarantees each update can observe mission
success or horizon failure before the restartable checkpoint boundary. Only an
explicit `--infrastructure-smoke` may use a shorter rollout, and its manifest
cannot be confused with development evidence.

Every update records learner and production-teacher action histograms,
separate action/target/power behavior-cloning losses, and target/power
exploration scales. These fields distinguish command classification from
physical target learning when mission reward remains sparse.

The plan residual is part of both the executed move/throw means and their BC
supervision view. The environment-only last-enemy move override remains outside
the supervised move view, preserving the existing hybrid-action boundary.

The BC anchor decays from 0.1 to zero over the first half of
`--anchor-total-updates`; initializer KL decays from 0.01 to zero over its first
three quarters. This total is frozen across exact resume and staged transfer.

Generate the deterministic same-state causal fork:

```bash
.venv/bin/snowgym-option-causal-fork \
  --seed 42001 \
  --decisions 30 \
  --output /tmp/snowgym-hold-withdraw-advance-fork.json
```

The artifact stores all semantic teacher actions and v2 state-hash sequences
for Hold, Withdraw, and Advance from one identical initial physical state.

Once a 100-seed-per-mission evaluation artifact exists, apply the strict
all-mission gate with:

```bash
.venv/bin/snowgym-qualify-m7b \
  --input path/to/m7b-evaluation.json \
  --output path/to/m7b-qualification.json
```

The qualifier verifies the evaluation digest, paired seed alignment, learning
rates, parameter change, bootstrap bound, mission progress, physical retention,
and rejected-action rate. One failed mission fails the checkpoint.

Generate the paired input artifact directly from a staged checkpoint:

```bash
.venv/bin/snowgym-evaluate-m7b \
  --checkpoint path/to/checkpoint \
  --split development \
  --option engage \
  --output /tmp/snowgym-m7b-development.json
```

For every mission and seed, the evaluator completes the battle and records
mission success/progress, physical win/loss, and rejected actions. The correct
condition uses the active grounded plan; the shuffled condition previews and
grounds a deterministic valid alternative at every state without replacing
the tracker’s intended objective; the initializer condition runs the migrated
accepted policy. Qualification mode consumes exactly 100 untouched seeds per
mission and emits `snowgym.m7b-evaluation.v0` for the strict qualifier.
Development may repeat `--option` to evaluate only the missions trained so
far. Qualification rejects every subset and always evaluates all eight frozen
missions.

`--ppo-warm-start` may transfer a checkpoint into the next option in the frozen
training order. A new option starts its own preallocated 10,000-seed range and
option schedule while preserving model state, global update count, anchor
decay, environment-step count, and source checkpoint lineage.
