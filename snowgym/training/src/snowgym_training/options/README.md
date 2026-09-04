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

All of those option boundaries are absorbing terminals for GAE. `truncated`
is reserved for an artificial collector cutoff that would otherwise continue
the same option world. Potential shaping uses a zero next potential at every
terminal boundary.

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
condition uses the active grounded plan; the shuffled condition binds a
deterministic valid alternative to one preview ID at reset, preserves its
stable assignment, and refreshes only its late-bound objective projection.
The initializer condition runs the migrated accepted policy. Qualification
mode consumes exactly 100 untouched seeds per
mission and emits `snowgym.m7b-evaluation.v0` for the strict qualifier.
Development may repeat `--option` to evaluate only the missions trained so
far. Qualification rejects every subset and always evaluates all eight frozen
missions.

`--ppo-warm-start` may transfer a checkpoint into the next option in the frozen
training order. A new option starts its own preallocated 10,000-seed range and
option schedule while preserving model state, global update count, anchor
decay, environment-step count, and source checkpoint lineage.

## M7b-R0 Engage recovery

The failed Engage Stage-1/Stage-2 checkpoints, optimizer states, 40-seed
evaluation, and recovery evidence are immutable under
`runs/m7b_engage_failed_v0`. They are negative diagnostic evidence and cannot
be promoted.

Reproduce the no-training evidence from `snowgym/training`:

```bash
.venv/bin/snowgym-engage-interventions \
  --checkpoint runs/m7b_engage_failed_v0/stage2/checkpoint \
  --output /tmp/snowgym-engage-intervention-matrix-v0.json

.venv/bin/snowgym-export-engage-diagnostics \
  --checkpoint runs/m7b_engage_failed_v0/stage2/checkpoint \
  --output /tmp/snowgym-engage-diagnostics

.venv/bin/snowgym-engage-gradient-diagnostics \
  --checkpoint runs/m7b_engage_failed_v0/stage2/checkpoint \
  --dataset /tmp/snowgym-engage-diagnostics/stochastic_learner_states.npz \
  --output /tmp/snowgym-engage-gradients

.venv/bin/snowgym-summarize-engage-recovery \
  --matrix /tmp/snowgym-engage-intervention-matrix-v0.json \
  --diagnostics /tmp/snowgym-engage-diagnostics \
  --gradients /tmp/snowgym-engage-gradients \
  --output /tmp/snowgym-engage-recovery-report.json
```

These commands use only the 40 development seeds. They never update the model,
touch qualification seeds, invoke a provider, or require a browser. The frozen
R0 report attributes the failure to an action/movement/throw interaction plus
missing successful-state support and selects one R1 intervention: a successful
production-teacher reservoir used only by the auxiliary BC loss. PPO rollouts
remain on-policy.

The R0 audit also repaired a simulator defect: semantic Random Red actions were
previously sampled but not applied. Evidence created after the repair reports
`snowgym.sim.v2`. Earlier random-opponent qualification artifacts remain
available as v1 history, but require v2 requalification before reuse as fighter
performance evidence. Scripted-Red results are unaffected.

## M7b-R1 teacher reservoir

Export the selected BC-only reservoir from successful production-teacher
episodes on training seeds, then run the frozen two-stage experiment:

```bash
cd snowgym/training
.venv/bin/snowgym-export-teacher-reservoir \
  --output runs/m7b_engage_teacher_reservoir_v0 \
  --seed-count 40

.venv/bin/snowgym-run-engage-r1 \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_teacher_reservoir_r1_v0
```

The runner consumes the frozen
`configs/m7b_engage_r1_v0.json`, performs 50 Stage-1 and 50 Stage-2 updates,
evaluates the 40 paired development seeds, and writes a digest-bound bootstrap
report. Reservoir transitions participate only in BC. PPO ratios, advantages,
and returns use learner-executed rollouts.

The first candidate is retained as negative evidence. Stage 1 reached contact
on 28/40 seeds and hit on 10/40 but had no mission successes. The Stage-2
checkpoint regressed to 0/40 contact, hits, and successes. R1 therefore remains
open, and no later option or qualification run is authorized by this result.

### R1b frozen Stage-1 hold

R1b changes only the unfreezing schedule. It resumes the digest-bound update-50
R1 Stage-1 checkpoint with its optimizer, RNG, seed schedule, plan schedule,
reservoir, and anchor schedule intact. Inherited action, target, and power heads
remain frozen through update 100. Updates 50, 75, and 100 are retained and
evaluated on all 40 development seeds; only update 100 is eligible for the
bootstrap decision.

```bash
cd snowgym/training
.venv/bin/snowgym-run-engage-r1b \
  --source-checkpoint runs/m7b_engage_teacher_reservoir_r1_v0/stage1/checkpoint \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_teacher_reservoir_r1b_stage1_hold_v0
```

The immutable configuration is
`configs/m7b_engage_r1b_stage1_hold_v0.json`. Intermediate results describe
the learning trajectory and cannot be selected after evaluation.

The frozen run failed. Update 50 reached 28/40 contacts and 10/40 hits; update
75 retained 27/40 contacts and 8/40 hits; neither checkpoint completed Engage.
Update 100 then reached zero contacts, hits, progress, successes, or physical
wins. All evaluated conditions retained a zero rejected-action rate. This
excludes Stage-2 head unfreezing as the sole cause of collapse and leaves R1
open. The evidence is archived under
`runs/m7b_engage_teacher_reservoir_r1b_stage1_hold_v0/`.

### R1c frozen BC-anchor floor

R1c retrains Stage 1 from update 0 and changes only the BC anchor schedule. The
anchor follows the original decay until it reaches `0.05` at update 50, then
holds that value through update 100. The initializer-KL schedule, reservoir
mixture, PPO data, optimizer, seeds, reward, exploration, and losses remain
unchanged. The update-50 state must exactly match the original R1 Stage-1 state
before later training is accepted.

```bash
cd snowgym/training
.venv/bin/snowgym-run-engage-r1c \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_teacher_reservoir_r1c_bc_floor_v0
```

Updates 50, 75, and 100 are retained, but only update 100 is eligible for the
bootstrap gate. The protocol is frozen in
`configs/m7b_engage_r1c_bc_floor_v0.json`.

R1c passed its update-50 causal parity assertion but failed the final gate.
Update 75 reached 26/40 contacts and 12/40 hits; update 100 retained 26/40
contacts but only 1/40 hit, with no mission successes or physical wins. The BC
floor prevents total contact collapse but does not preserve successful
move/throw coordination. The immutable result is under
`runs/m7b_engage_teacher_reservoir_r1c_bc_floor_v0/`.

### R1d frozen reservoir mixture

R1d retains the R1c `0.05` BC floor and changes only the BC loss weighting from
50% to 90% successful-teacher reservoir loss. Both sources retain equal sample
counts per minibatch. Learner transitions remain the
exclusive source of PPO ratios, advantages, returns, and value targets.

```bash
cd snowgym/training
.venv/bin/snowgym-run-engage-r1d \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_teacher_reservoir_r1d_reservoir90_v0
```

The runner retains updates 50, 75, and 100 and gates only update 100 under the
frozen `configs/m7b_engage_r1d_reservoir90_v0.json` protocol.

R1d failed mission success but recovered the intended trajectory monotonically.
Contact rose from 45% to 67.5% to 85%, hit rate rose from 12.5% to 22.5% to
52.5%, and update-100 mean progress was 8.6%. No episode completed Engage or
won the full battle. The audited run is retained at
`runs/m7b_engage_teacher_reservoir_r1d_reservoir90_v0/`.

### R1e exact continuation to update 200

R1e changes only training duration. It resumes the exact R1d update-100
checkpoint, including optimizer, RNG, seed schedule, and option schedule, then
continues the same Stage-1, 90% reservoir, `0.05` BC-floor protocol.

```bash
cd snowgym/training
.venv/bin/snowgym-run-engage-r1e \
  --source-checkpoint runs/m7b_engage_teacher_reservoir_r1d_reservoir90_v0/update-000100/checkpoint \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_teacher_reservoir_r1e_continue200_v0
```

Updates 100, 150, and 200 are evaluated; only update 200 is eligible for the
bootstrap gate.

R1e failed the final mission gate. Update 200 reached 38/40 contacts, 28/40
hits, and 14% mean progress, but no Engage completion or physical win. The
exact continuation demonstrates that duration improves contact and damage but
does not approach the frozen 80% target-health reduction. The audited result is
under `runs/m7b_engage_teacher_reservoir_r1e_continue200_v0/`.

### Measurement contract after the R1e review

New option checkpoints preserve the complete expanded initializer and its
source/state digests. Evaluation uses those weights; legacy checkpoints use a
seeded reconstruction explicitly labeled in `initializerIdentity`. Legacy
staged transfers without a recorded root identity are marked unverified.
`parameterL2ChangeByGroup` separates inherited heads, new actor modules,
remaining inherited actor parameters, and critic. Learning rates come from
saved named optimizer groups.

The active plan-PPO updater reports `ppoLossComponents`, post-update
`finalPpoDiagnostics`, and optional `target_kl` early stopping.
`bcLossWeights` and actual `bcSampleCounts` replace the ambiguous
`bcSampleMixture` label. Existing artifact bytes remain unchanged; compare
historical episode records separately from their old parameter-distance metric.

Measurement-repair verification: 325 TypeScript tests, build, 50 Gym-client
tests, and 171 training tests passed on 2026-09-04. Added regressions cover
initializer/RNG identity, repeated evaluation digests, stored initializer
restoration, optimizer learning rates, exact resume, and plan-PPO KL stopping.

### R1f supervised-only teacher-trajectory probe

Run from `snowgym/training`:

```bash
.venv/bin/python -m snowgym_training.options.supervised_probe \
  --source-checkpoint runs/m7b_engage_teacher_reservoir_r1e_continue200_v0/update-000200/checkpoint \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_r1f_supervised_probe_v0
```

The frozen configuration is `configs/m7b_engage_r1f_supervised_probe_v0.json`.
The probe warm-starts the final R1e actor and fits only Stage-1 actor modules
using fresh Adam, 20 epochs, minibatches of 256, and the existing BC component
weights. It performs no PPO or critic updates. Epochs 0, 10, and 20 retain
checkpoints and teacher-state diagnostics; only epoch 20 is the final result.
Training-seed closed-loop results compare epochs 0 and 20. Final development
evaluation uses all 40 existing paired seeds and correct/HOLD/initializer
conditions. Qualification seeds are excluded.

Phase diagnostics partition living unit labels into fire (teacher requests a
throw), contact (other actions within 9 world units of a living enemy), and
approach (remaining living labels). Confusion matrices use teacher rows and
predicted columns in noop/move/throw/hold order. Conditional target and ray
errors use the teacher's action branch regardless of classification. Undefined
zero-length rays are counted separately. Distances use world coordinates.

Outputs include `report.json`, per-epoch teacher agreement, per-seed training
and development evaluations, diagnostic bootstrap results, and a recursively
hashed manifest. The shared PPO checkpoint container stores the BC-trained
model and optimizer with gate ID `m7b-engage-supervised-probe` and zero
collected training-environment steps; it is not an option-PPO continuation.
`qualificationEligible` stays false even if diagnostic bootstrap thresholds
pass. Output directories are immutable and cannot be overwritten.
