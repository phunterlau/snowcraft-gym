# SnowGym training

This isolated package contains reproducible data collection and the first
behavior-cloned SnowGym policy. The core `snowgym-client` and simulator remain
free of Torch.

Set up only the data tools:

```bash
uv sync --extra dev
```

Add the isolated Torch learner for training or checkpoint evaluation:

```bash
uv sync --extra dev --extra learn
```

## Quick learned-policy demo

Run the committed behavior-cloned blue policy against the native seeded-random
red controller and write a replay consumable by the existing SnowCraft UI:

```bash
# Terminal 1, from the repository root
npm run snowgym:server

# Terminal 2
cd snowgym/training
uv run snowgym-demo-learned \
  --seed 42 \
  --record ../../public/replays/demo-learned-blue-seed-42.json
```

The command is headless and refuses to overwrite an existing recording. Start
`npm run dev -- --host 127.0.0.1` from the repository root, then open:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/demo-learned-blue-seed-42.json
```

The committed seed-42 demonstration ends with a blue win after 54 decisions
and zero rejected actions. It exercises the learned BC checkpoint on the new
RL-ready server/client path; the PPO module below is infrastructure only and
does not yet provide a committed qualifying PPO checkpoint.

With `npm run snowgym:server` running in another terminal, export one committed
split specification without rendering:

```bash
uv run snowgym-export-trajectories \
  --split train \
  --output artifacts/scripted-blue-1v1-train

uv run snowgym-audit-dataset artifacts/scripted-blue-1v1-train --json
```

The exporter refuses to overwrite an existing directory. It records the
pre-step fixed tensors, exact semantic action returned by `/step-scripted`, its
round-tripped Gym tensors, action results, outcome, versions, and pre/post state
hashes. `snowgym.trajectory.v0` shards use non-pickle compressed NumPy arrays;
the manifest hashes canonical array name/dtype/shape/bytes rather than archive
metadata.

The committed `teacher_1v1_v0` specification keeps train, validation, and
evaluation seeds disjoint. Measure the teacher ceiling and masked-random blue
baseline on the held-out evaluation split with:

```bash
uv run snowgym-teacher-baseline --split evaluation \
  --output artifacts/teacher-1v1-baseline.json
```

These data tools establish the audited input contract used by the neural
executor below.

## Train and evaluate the neural executor

With the headless server still running, train the versioned CPU configuration
from an audited training export:

```bash
uv run snowgym-train-bc \
  --dataset artifacts/scripted-blue-1v1-train \
  --output artifacts/bc-1v1-checkpoint \
  --json

uv run snowgym-evaluate-checkpoint \
  --checkpoint artifacts/bc-1v1-checkpoint \
  --record-dir artifacts/bc-1v1-replays \
  --output artifacts/bc-1v1-evaluation.json \
  --json
```

Outputs refuse to overwrite existing checkpoint, replay, or evaluation paths.

### Closed-loop teacher relabeling

When open-loop BC suffers compounding error, collect teacher labels on states
visited by the learned policy itself. Start the headless server, then run:

```bash
uv run snowgym-export-dagger \
  --spec src/snowgym_training/configs/teacher_3v3_scripted_v0.json \
  --split train \
  --checkpoint checkpoints/bc_3v3_random_v0 \
  --output artifacts/dagger-3v3-scripted \
  --json
```

The collector reads `GET /teacher-action` without advancing the simulator,
then advances with the learned action guarded by the same state hash. The
sharded manifest binds the rollout checkpoint/state digests, source spec,
episode outcomes, and every tensor digest. It refuses rejected learner actions
and re-audits the finished dataset. This first collector emits a pure
learner-state dataset; aggregation/mixing policy remains an explicit later
training decision.

Aggregate audited datasets in a deterministic order before retraining. Repeating
an input is an explicit integer weight; the manifest records every ordered
source digest:

```bash
uv run snowgym-merge-trajectories \
  --input artifacts/teacher-3v3-scripted \
  --input artifacts/teacher-3v3-scripted \
  --input artifacts/dagger-3v3-scripted \
  --output artifacts/aggregate-3v3-scripted \
  --json
```

The merge rejects differences in split, split seeds, source specification,
capacity, or simulator versions and re-audits the complete output.

Gate-4 development uses a 2:1 ordered expert/recovery aggregate: expert dataset
digest `sha256:3fccaa7f0a10f9111b66514874edb099f3bd6c67c594f17650a8340ea8b2cd05`
is repeated twice, followed by DAgger digest
`sha256:116618922841b27d17d21d6371dd0ec05860a6c248071d3ef9f2118c02aa9846`.
The portable 1,745-transition aggregate digest is
`sha256:2fc2770ae2385c16adb14cffde01104a5e4165a4f8154d92021d40ae1fa3e7e4`.
The frozen `bc_3v3_scripted_v0.json` config uses 10,000 steps, target weight 10,
and power weight 1. Its development checkpoint won both BC held-out episodes;
post-commit checkpoint generation remains required for accepted provenance.

The committed `bc_3v3_scripted_v0` checkpoint binds source commit `c17751d`,
aggregate digest `sha256:2fc2770ae2385c16adb14cffde01104a5e4165a4f8154d92021d40ae1fa3e7e4`,
and checkpoint digest
`sha256:7e757f64cd46df87921c744c3580b794199af45df0b4fcddb3347c0942f20f47`.
It won both disjoint BC evaluation episodes in 158 and 73 decisions with one
and two blue survivors, versus 0/2 masked-random and 2/2 teacher. Gate-4 PPO
qualification remains separate and open.

The gate-4 PPO candidate is frozen in `ppo_3v3_scripted_bc_v0.json`. Its
development series passed at every retained update 1/5/10 with 5/8 wins versus
0/8 masked-random and mean 119.875 decisions. Three seeds remain red wins. The
candidate uses the same log-std `-3`, learning rate `1e-8` stability contract as
gate 3, so it demonstrates retention through PPO rather than improvement over
the DAgger initializer. Post-commit qualification remains required.

The committed qualifying artifact is `runs/ppo_3v3_scripted_bc_v0`:

```bash
uv run snowgym-audit-ppo-series runs/ppo_3v3_scripted_bc_v0 --json
```

It is bound to source commit `eef57e4`, config digest
`sha256:de8e80f22a1ebe0d96c96583e3f01de586215a14c932dcbb2faf7965eeebc87b`,
and series digest
`sha256:e82ec0f690c9864ab3d6fa9eb307875c0c182e55fd2657bf5dbdcd8a3cf65f3e`.
Updates 1/5/10 each won 5/8 with mean 119.875 decisions, versus 0/8
masked-random and 8/8 teacher, with zero rejected actions. This advances
`3v3-scripted`; the three losing seeds remain visible in every evaluation.

Record the final checkpoint's seed-6108 3–0 blue win:

```bash
uv run snowgym-demo-ppo \
  --checkpoint runs/ppo_3v3_scripted_bc_v0/checkpoints/update-000010/checkpoint \
  --gate 3v3-scripted \
  --seed 6108 \
  --record ../../public/replays/ppo_3v3_scripted_bc_v0-seed-6108.json \
  --json
```

```text
http://127.0.0.1:5173/replay.html?recording=/replays/ppo_3v3_scripted_bc_v0-seed-6108.json
```

The `snowgym.checkpoint.v0` metadata binds the model and optimizer state digest
to the source commit, audited dataset digest, SnowGym versions, architecture,
optimizer, loss weights, seed, step, and evaluation suite. Loading uses
Torch's restricted `weights_only` mode and verifies the semantic state digest.

The committed `bc_1v1_v0` checkpoint was trained for 5,000 deterministic CPU
steps on 212 teacher transitions. On held-out seeds 201 and 202 it won both
episodes in 54 decisions with zero rejected actions. The teacher won both in
53 decisions; masked random won neither. See the
[checkpoint metadata](./checkpoints/bc_1v1_v0/metadata.json) and
[joined evaluation](./evaluations/bc_1v1_v0.json).

`bc_1v1_easy_v0` is the separate gate-2 initializer trained on 184 audited
native-teacher transitions against easy scripted red. Its checkpoint binds
source commit `fe0d19d`, dataset digest
`sha256:0448fdfa4661645b695bcbc4759f2e40cb2e2bca2aa30fb350e4e215d02722ea`,
and its full training configuration. It won both disjoint BC evaluation seeds,
versus 0/2 masked-random and 2/2 teacher, with zero rejected actions. This is an
initializer result, not the eight-seed PPO gate acceptance.

The gate-2 PPO candidate is frozen in `ppo_1v1_easy_bc_v0.json`. Run it with:

```bash
uv run snowgym-run-ppo-config \
  --config src/snowgym_training/configs/ppo_1v1_easy_bc_v0.json \
  --output artifacts/ppo-1v1-easy-bc-v0-development \
  --json
```

Its predeclared checkpoints stop at update 10 because the earlier development
sweep passed at updates 1/5/10 but regressed at update 25. This is a frozen
tuning decision made before the qualifying run, not post-hoc checkpoint
selection inside a qualifying artifact.

The committed gate-2 qualifying artifact is `runs/ppo_1v1_easy_bc_v0`:

```bash
uv run snowgym-audit-ppo-series runs/ppo_1v1_easy_bc_v0 --json
```

It is bound to source commit `2a3099c` and config digest
`sha256:f1cb3bd658d003deb57396d9668a6aa99c880665fddff2bd91b574afe599a0da`.
Updates 1/5/10 won 5/8, 4/8, and 5/8 respectively against easy scripted red;
masked-random won 0/8 and the teacher 8/8. The final checkpoint averaged 66.75
decisions with zero rejected actions. This advances `1v1-easy-scripted`; the
3v3 and later gates remain closed.

Record the fastest winning seed from the final qualifying checkpoint:

```bash
uv run snowgym-demo-ppo \
  --checkpoint runs/ppo_1v1_easy_bc_v0/checkpoints/update-000010/checkpoint \
  --gate 1v1-easy-scripted \
  --seed 4103 \
  --record ../../public/replays/ppo_1v1_easy_bc_v0-seed-4103.json \
  --json
```

The committed replay is a 43-decision blue win with zero rejected actions.
After starting the root Vite server, view it through the existing UI engine:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/ppo_1v1_easy_bc_v0-seed-4103.json
```

For the next `3v3-random` gate, a direct PPO transfer from the accepted gate-2
checkpoint remained at 0/8 wins at retained updates 1/5/10, matching
masked-random, while the native teacher won 8/8. The gate-specific
`bc_3v3_random_v0` initializer is trained from 424 audited teacher transitions
bound to dataset digest
`sha256:616494b021b437d6b8b641bae03255ffe525607985169b24e218407a4daf5dff`.
On its disjoint BC evaluation seeds it won 2/2 in 105 decisions with all three
blue units alive and zero rejected actions, versus 0/2 masked-random and 2/2
teacher. This establishes a PPO initializer only; `3v3-random` remains closed
until its frozen eight-seed PPO series passes.

The gate-3 PPO candidate is frozen in `ppo_3v3_random_bc_v0.json`. Its
development series retained 8/8 wins at updates 1/5/10 versus 0/8
masked-random, with a constant 105-decision deterministic result and zero
rejections. It uses target/power log-std `-3` and learning rate `1e-8`: this is
a stability configuration that preserves the BC solution while exercising the
PPO collection/update/checkpoint path, not evidence that PPO materially
improves on the BC initializer. Run it before qualification with:

```bash
uv run snowgym-run-ppo-config \
  --config src/snowgym_training/configs/ppo_3v3_random_bc_v0.json \
  --output artifacts/ppo-3v3-random-bc-v0-development \
  --json
```

The committed qualifying artifact is `runs/ppo_3v3_random_bc_v0`. Audit it
with:

```bash
uv run snowgym-audit-ppo-series runs/ppo_3v3_random_bc_v0 --json
```

It is bound to source commit `1829e6a`, config digest
`sha256:f793476d5297c6ebc0c542dfd9a9bd662c81de43e05b92a1aac531ba9c57e341`,
and series digest
`sha256:f54853b4758796fb2cff643d98eeb41787b94c171fffab2de97766367a796adf`.
Updates 1/5/10 each won 8/8 in 105 decisions against 0/8 masked-random and
8/8 teacher, with zero rejected actions. This advances `3v3-random` and meets
the frozen M6.2 exit comparison, subject to the BC-retention caveat above.

Record and replay the final checkpoint's held-out seed-5101 blue win:

```bash
uv run snowgym-demo-ppo \
  --checkpoint runs/ppo_3v3_random_bc_v0/checkpoints/update-000010/checkpoint \
  --gate 3v3-random \
  --seed 5101 \
  --record ../../public/replays/ppo_3v3_random_bc_v0-seed-5101.json \
  --json
```

```text
http://127.0.0.1:5173/replay.html?recording=/replays/ppo_3v3_random_bc_v0-seed-5101.json
```

Replay either learned episode through the existing UI after starting
`npm run dev` from the repository root:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/bc_1v1_v0/learned-seed-201.json
http://127.0.0.1:5173/replay.html?recording=/replays/bc_1v1_v0/learned-seed-202.json
```

This is a narrow 1v1 imitation-learning proof. M6.2 adds reward-driven PPO.

## Persistent batch simulation

The training package can drive the authoritative TypeScript simulation in one
persistent subprocess at 1, 8, 32, or 64 worlds. Run the versioned benchmark:

```bash
uv run snowgym-benchmark-batch \
  --worlds 1 8 32 64 \
  --decisions 20 \
  --output artifacts/batch-throughput.json \
  --json
```

The report separates decisions/s, simulation ticks/s, aggregate real-time
factor, parent-plus-child CPU use, protocol payload bytes, Python JSON
serialization, and transport-plus-simulation time. The committed short-run
[M6.1 result](./benchmarks/batch_throughput_v0.json) reached all 64 worlds; it
is local acceptance evidence rather than a portable performance guarantee.

For a plan-conditioned closed loop, call `SnowGymBatchEnv.activate_plans()`
after reset and `plan_observations()` before every policy decision. The
authoritative TypeScript world re-resolves symbolic objectives and updates
geometry, living assignment fractions, and plan age at the current tick; Python
only validates and stacks the returned `[B,3,38]` tensors.
The adjacent `plan_teacher_actions()` call labels the same learner-visited
states through the production plan-aware executor without advancing them. A
collector must verify the returned state hashes and submit only the learner
action; the teacher label never mutates the episode.

Collect plan-conditioned DAgger data without an HTTP server:

```bash
cd snowgym/training
.venv/bin/python -m snowgym_training.export_plan_dagger \
  --spec src/snowgym_training/configs/plan_dagger_v0.json \
  --split train \
  --checkpoint runs/plan_bc_ablation_qual_v1/plan-conditioned \
  --output artifacts/plan-dagger-v0 \
  --json
```

The `snowgym.plan-dagger-export.v0` spec contains a validated plan catalog and
disjoint train/validation/evaluation episodes that reference catalog names. The
collector activates that plan, executes only the learned policy, and labels
each visited state with the read-only production plan teacher. Output shards
carry aligned plan tensors and bind the rollout checkpoint, source spec, split
seeds, simulator versions, and per-episode plan.

Combine independently seeded expert-state and DAgger corpora with:

```bash
.venv/bin/snowgym-merge-trajectories \
  --input path/to/expert-plan-dataset \
  --input artifacts/plan-dagger-v0-train \
  --output artifacts/plan-dagger-v0-aggregate \
  --independent-sources \
  --json
```

Independent-source mode requires matching split, capacity, simulator versions,
and plan-conditioning status, plus globally disjoint declared seeds. It retains
the intersection of audited tensor fields, records dropped source-specific
auxiliaries, and preserves the plan-conditioned marker. Default merging remains
strict and requires identical source specifications.

For a correction run on a new aggregate dataset, initialize from the qualified
checkpoint while freezing its physical action path:

```bash
.venv/bin/snowgym-train-bc \
  --dataset artifacts/plan-dagger-v0-aggregate-train \
  --config path/to/frozen-correction-config.json \
  --initialize runs/plan_bc_ablation_qual_v1/plan-conditioned \
  --output runs/plan-dagger-correction-v0 \
  --json
```

Set `trainable` to `plan-target-path` only with the plan-conditioned,
target-only, separate-target-actor architecture. The trainer loads model weights
but resets optimizer and step state for the new dataset; only the plan encoder,
target actor, target heads, and power head receive gradients. Initialization
checkpoint/state digests are retained in checkpoint metadata.

For a later action-timing correction, enable `architecture.plan_action_adapter`
with `trainable: "plan-action-target-path"`. This adds a residual plan/action-
context adapter to action logits. Its final layer is initialized to exactly zero,
so loading a target-only checkpoint preserves every initial action logit. The
inherited entity encoders, actor, and action head remain frozen; only the new
adapter and existing plan/target path train.

Measure any conditioned checkpoint on an audited held-out plan corpus with
`snowgym-evaluate-plan-checkpoint`. It reports the same action accuracy, target
error, and counterfactual plan-swap metrics as the matched ablation evaluator,
and binds checkpoint/state plus dataset digests in an audited result. Use this
to enforce action-regression limits before interpreting closed-loop outcomes.

The frozen residual-adapter v0 result is retained in
`evaluations/plan_action_adapter_qualification_v0.json` and does not pass. It
improves offline action accuracy and target error, but fails direct/flank
survival, support combat effect, and first-decision action sensitivity. Because
all five held-out mission cases begin with the same all-unit `move` action, the
next data revision uses the server's guarded `previewPlan` batch operation to
label alternate plans on identical later learner-visited physical states. Plan
preview returns a temporary plan tensor plus the production teacher action
without changing the active plan or advancing the simulator.

Use a `snowgym.plan-dagger-export.v1` spec to retain that intervention as
training data. Every split episode adds `counterfactualPlan`, naming a distinct
entry in the same plan catalog. The exported transition still records only the
learner action actually executed, while auxiliary counterfactual plan/action
tensors are bound to the identical pre-step state. Set a frozen training
config's `counterfactualLossWeight` in `[0,10]` to apply the same hybrid BC loss
to the alternate pair; omitting it keeps the legacy objective exactly.

Evaluate those paired labels across all transitions with:

```bash
.venv/bin/snowgym-evaluate-plan-counterfactual \
  --checkpoint runs/plan-action-adapter \
  --dataset artifacts/plan-counterfactual-evaluation \
  --output evaluations/plan-counterfactual.json --json
```

The result distinguishes teacher action diversity from model sensitivity and
scores both plan conditions, including strict pair accuracy only on unit-state
pairs where the production teacher changes action type.

The frozen v1 result is retained under `runs/plan_action_adapter_v1` and
`evaluations/plan_action_adapter_v1_*.json`. It fails the conjunctive gate, but
establishes partial closed-loop progress: the support case is a 6–0 blue win,
flank preserves five blue, hold lasts 338 decisions, and all physical actions
are accepted. The remaining paired-action failure is localized to low recall
and strict accuracy on teacher-changed positions; overall accuracy or raw
change rate must not be used to claim the gate passed.

For the next imbalance correction, `counterfactualChangedActionWeight` adds a
symmetric action-classification loss only where primary and alternate teacher
action types differ. It requires audited same-state labels and is bounded to
`[0,100]`. Keep the full primary/counterfactual objectives enabled; this term
does not replace their accuracy and target-regression constraints.

The weight-5 v2 development run is retained under
`runs/plan_action_adapter_v2_dev`. It strongly improves changed-teacher recall
and pair accuracy and makes withdraw survive the full 600-decision horizon, but
over-changes action types and regresses the v1 support win and hold duration.
It is diagnostic evidence only; do not promote it over v1 or call it qualified.

The committed weight sweep under `runs/plan_action_adapter_weight*_dev` applies
the frozen selection rule in `evaluations/plan_action_adapter_weight_sweep_v0.json`.
All three candidates pass 7/11 checks; weight 3 wins the declared tie-break but
is not promoted because every candidate misses flank, hold, and withdraw.

For multi-group plans, `snowgym.plan-dagger-export.v2` additionally records
host-resolved `plan_unit_roles` and `counterfactual_plan_unit_roles` as
`[units,3]` main/maneuver/reserve one-hot tensors. Enable
`architecture.plan_role_conditioned` only with the residual plan adapter. Role
features enter zero-initialized action and target residuals, so loading the
qualified target-only checkpoint still preserves its initial outputs exactly.

The frozen role-aware-v0 development artifacts are retained under
`artifacts/plan-role-dagger-v2-{train,validation}` and
`runs/plan_role_adapter_v0_dev`. They validate the complete v2 data and runtime
path, including role-conditioned closed-loop inference, but do not pass M7:
direct and flank each retain four blue without completing, and the support case
loses 0–6. The validation metrics also show over-changing (`0.227865` predicted
versus `0.130642` teacher). Do not generate the sealed v2 evaluation split or
promote this checkpoint. A categorical role alone is insufficient because it
does not carry the unit's resolved group objective, approach, engagement, or
support directive.

Enable `architecture.plan_unit_directive_conditioned` to select the complete
38-feature group directive for every unit from `plan_groups` using that unit's
one-hot role. This option requires `plan_role_conditioned`; it augments the
zero-initialized action and target residuals without changing the v2 dataset
format. It is intended for the next development experiment, not a claim that
the retained role-aware-v0 checkpoint passed.

## PPO foundation

Run a short end-to-end infrastructure smoke against the first frozen curriculum
gate without starting the HTTP server:

```bash
uv run snowgym-train-ppo \
  --output artifacts/ppo-smoke-1v1-random \
  --worlds 8 \
  --rollout-steps 32 \
  --target-updates 1 \
  --json
```

The command drives the persistent batch subprocess, performs a real PPO update,
and atomically writes `manifest.json` plus a restricted-load exact-resume
checkpoint. It refuses to overwrite an existing run. Resume into a new output
directory by passing the prior `checkpoint` directory and a larger target:

```bash
uv run snowgym-train-ppo \
  --output artifacts/ppo-smoke-1v1-random-update-2 \
  --resume artifacts/ppo-smoke-1v1-random/checkpoint \
  --worlds 8 \
  --rollout-steps 32 \
  --target-updates 2 \
  --json
```

These commands validate training plumbing only. Their manifest is labeled
`infrastructure-smoke`; it is not evidence that a curriculum gate was solved.
Canonical rewards are the default. For an explicitly shaped training run, add
`--reward-mode health-potential`; the manifest and resume contract bind that
choice and record canonical and training reward sums separately. Held-out
evaluation always uses canonical `-1/0/+1` returns.

Evaluate any PPO checkpoint on the gate's disjoint held-out seeds, headlessly,
against deterministic masked-random blue and the native scripted blue policy:

```bash
uv run snowgym-evaluate-ppo \
  --checkpoint artifacts/ppo-smoke-1v1-random/checkpoint \
  --gate 1v1-random \
  --output artifacts/ppo-smoke-1v1-random-evaluation.json \
  --json
```

The result retains per-episode winner and canonical `-1/0/+1` return, rejected
actions, policy summaries, frozen gate thresholds, and a result digest. A
threshold failure is reported as evaluation data rather than a process error,
so checkpoint series can be audited without selecting only successful runs.

Preserve and evaluate a complete development checkpoint series with one atomic
command:

```bash
uv run snowgym-run-ppo-series \
  --output artifacts/ppo-1v1-random-series \
  --gate 1v1-random \
  --checkpoints 10 25 50 \
  --worlds 8 \
  --rollout-steps 32 \
  --reward-mode health-potential \
  --json
```

Each listed update is retained under `checkpoints/` and evaluated under
`evaluations/`; the top-level manifest records every digest and the full update
curve. Add `--qualifying` only for a predeclared qualifying run. That label does
not imply success: `finalThresholdPassed` still comes solely from the frozen
held-out evaluation, and earlier failing checkpoints remain in the artifact.

PPO can be initialized from an audited behavior-cloning checkpoint with
`--warm-start checkpoints/bc_1v1_v0`. The policy weights are loaded only before
the first update. Every PPO checkpoint and the series manifest retain the BC
checkpoint, tensor-state, and source-dataset digests; exact resume carries that
provenance forward. Warm-started results must be described as BC-initialized
PPO, never as cold-start reward-only learning.

To advance to a new curriculum gate, use `--ppo-warm-start` with an accepted
checkpoint from the preceding gate. This transfers model/value weights but
starts a fresh optimizer, update counter, and target-gate seed schedule. The
new checkpoint records the source checkpoint/state/curriculum digest, source
gate, and source update. It is deliberately distinct from `--resume`, which
requires the same gate and rollout contract.

The first stable 1v1 candidate is frozen in `ppo_1v1_bc_v0.json`. Run it
without restating or drifting hyperparameters:

```bash
uv run snowgym-run-ppo-config \
  --output artifacts/ppo-1v1-bc-v0-development \
  --json
```

The configuration binds the BC digest, 8 worlds, full 200-decision rollouts,
health-potential training reward, learning rate `3e-5`, one PPO epoch, and
checkpoints 1/5/10/25. Its development reproduction passed the frozen 1v1
random threshold at every checkpoint. Use `--qualifying` only for the immutable
post-commit run intended as acceptance evidence.

The committed qualifying artifact is `runs/ppo_1v1_bc_v0`. Audit all nested
digests and restricted-load every retained checkpoint with:

```bash
uv run snowgym-audit-ppo-series runs/ppo_1v1_bc_v0 --json
```

The qualifying series is provenance-bound to source commit `60459b5` and the
frozen config digest `sha256:df270e5afd04d0a6b8c65892cc955717041aa553ab92cf9329a4ed0df98f3d21`.
Updates 1, 5, 10, and 25 each won 8/8 held-out episodes versus 0/8 for
masked-random; the final checkpoint averaged 60 decisions with zero rejected
actions. This advances only `1v1-random`, and it is explicitly BC-initialized
PPO—not a cold-start PPO result.

Record the final qualifying checkpoint headlessly through the persistent batch
host:

```bash
uv run snowgym-demo-ppo \
  --checkpoint runs/ppo_1v1_bc_v0/checkpoints/update-000025/checkpoint \
  --seed 3101 \
  --record ../../public/replays/ppo_1v1_bc_v0-seed-3101.json \
  --json
```

The committed replay is a 60-decision blue win with zero rejected actions.
After starting the root Vite server, view it in the existing UI engine at:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/ppo_1v1_bc_v0-seed-3101.json
```

`ppo.py` defines the centralized hybrid actor-critic used by the next training
stage. Action masks apply before categorical sampling; target terms contribute
to joint log probability only for move/throw and power only for throw. GAE
bootstraps time-limit truncation but not true terminal states. The optional
health potential is a training signal only and never replaces terminal-only
benchmark reward.

`PPOConfig.initial_target_log_std` and `initial_power_log_std` make the initial
continuous exploration scale explicit (default `-1`). A BC-initialized gate
with coordinated movement may choose a narrower value in its frozen config;
the selected values are retained in checkpoint and run provenance. Exact
resume restores checkpoint state, while PPO-to-PPO transfer restores the
source distribution parameters.

`RolloutBuffer` owns a fixed number of decisions from a fixed number of
persistent worlds. Every transition is validated and detached on insertion;
the completed `PPORollout` retains `[time, world, ...]` tensors for audit and
flattens only the first two axes for optimization. Terminal transitions do not
bootstrap, time-limit truncations bootstrap once but stop recurrence, and an
incomplete rollout cannot be consumed.

`collect_rollout` drives `SnowGymBatchEnv` directly. It assigns every episode a
monotonic seed from a bounded schedule, selectively resets worlds that finish,
and refuses to reuse exhausted seeds. Each collection call is a restartable
rollout boundary: unfinished worlds are recorded as artificial time-limit
truncations with value bootstrap, and the next call resets every world. This
keeps resume exact without claiming that hidden live simulator state is stored.

`ppo_update` normalizes advantages once over the complete rollout, derives
epoch permutations from the training seed and update index, and applies the
hybrid clipped objective in deterministic minibatches. It rejects non-finite
losses or gradients, clips the aggregate gradient norm, and returns
sample-weighted policy/value/entropy/KL/clip diagnostics.

`snowgym.ppo-checkpoint.v0` persists model and optimizer tensors, Torch RNG
state, architecture and PPO configuration, training seed, curriculum digest,
episode-seed schedule and cursor, update index, environment-step count, and
semantic state/metadata digests. Resume compatibility also binds the curriculum
gate, persistent-world count, and rollout horizon.
Loading uses restricted Torch deserialization and rejects incompatible
training provenance before restoring state. The deterministic acceptance test
matches uninterrupted training exactly across a save/restore boundary. This is
rollout-boundary optimizer and sampling resume; simulator worlds intentionally
restart at each collection boundary.

The frozen `ppo_curriculum_v0.json` keeps training ranges disjoint from eight
evaluation seeds per gate and sets thresholds before qualifying runs. Its order
is 1v1 random, 1v1 easy scripted, 3v3 random, 3v3 easy scripted, 3v3 terrain on
`arena4.json`, then 5v5 and 10v10 random terrain on the ten-spawn
`arena6.json`. Each gate has a separate 10,000-seed training range. The live
contract test resets all seven scenarios against the authoritative batch
server. Defining a gate does not advance it: a qualifying checkpoint series
must still pass its held-out threshold, and the current foundation does not yet
claim a PPO result.

The gate-7 relational initializer is frozen in
`bc_10v10_terrain_relational_v0.json`. It retains neural action selection and
an auxiliary learned move target, while the opt-in execution priors aim throws
at the nearest living enemy and use nearest-living-enemy movement only to clean
up the final opponent. Relational masks combine roster presence with the
encoded alive bit; defeated slots must never receive attention or targets.
The committed checkpoint `checkpoints/bc_10v10_terrain_relational_v0` binds
source commit `9f28de6`, dataset digest
`sha256:6f2e7a7ed096d64e980343818d446e721c2bd70468fe25644b47052c0b89df78`,
and checkpoint digest
`sha256:a7f1362cf163fbf23ebc3c8290bb0f772e57fb3a763b6dcf97b21135aa47bc08`.
It won both disjoint BC evaluation episodes in 145 decisions with all ten blue
units alive and zero rejected actions, versus 0/2 masked-random and 2/2 teacher
at 146 decisions. This is a behavior-cloned, code/policy hybrid initializer;
gate-7 PPO qualification remains separate.

The frozen `ppo_10v10_terrain_relational_bc_v0.json` retention configuration
binds that initializer and preserves checkpoints 1/5/10. Its committed
qualifying series is `runs/ppo_10v10_terrain_relational_bc_v0`, bound to source
commit `448f8ba`, config digest
`sha256:39bacab64fca11007617d0698b782a3baf3023e9215a562f851437876525ff47`,
and series digest
`sha256:e5c1b9b540a54c787208a7a4270698846d136132b2207ca9dafc9f3d422e3034`.
Every retained checkpoint won 8/8 held-out episodes in 145 decisions, versus
0/8 masked-random and 8/8 teacher, with zero rejected actions. This advances
the final M6.2 curriculum gate as BC-initialized PPO retention; it does not
claim cold-start or material reward-driven improvement.

Record the final checkpoint as a normal SnowCraft replay:

```bash
uv run snowgym-demo-ppo \
  --checkpoint runs/ppo_10v10_terrain_relational_bc_v0/checkpoints/update-000010/checkpoint \
  --gate 10v10-random-terrain \
  --seed 9101 \
  --max-decisions 600 \
  --record ../../public/replays/ppo_10v10_terrain_relational_bc_v0-seed-9101.json \
  --json
```

The committed replay is a 145-decision 10–0 blue win with zero rejected
actions. View it through the existing UI engine at:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/ppo_10v10_terrain_relational_bc_v0-seed-9101.json
```

## Synthetic command-plan curriculum

M7 begins with the pure TypeScript generator in
`plan/SyntheticPlanCurriculum.ts`. Given one detached server observation, a
safe integer base seed, and a sample count, it emits
`snowgym.synthetic-plan-curriculum.v0` records. Every plan is parsed by the
production command-plan validator and grounded by the production plan grounder
before the record exposes stable role-to-unit assignments. The compact source
section retains the tick, arena, ally/enemy roster geometry and health, plus an
optional public state hash. Generation is deterministic and makes no OpenAI
call. A JSON export and the Python plan-tensor/data join are the next M7 seam.

`plan/PlanTensorEncoder.ts` consumes the production grounded `PlanSnapshot`
directly and emits a row-major `[3, 38]` `Float32Array` plus a three-slot role
mask. Rows are fixed to `main`, `maneuver`, and `reserve`; features include the
bounded directive vocabularies, relative objective/group geometry in the
tactical frame, allocation and live-assignment fractions, support target, and
plan age over a documented 30-second horizon. Unit IDs select group members but
are intentionally absent from the learnable tensor. The command-plan JSON and
grounded plan remain canonical; this array is only the RL adapter.

Export an audited tensor dataset from an authoritative headless reset:

```bash
npx tsx snowgym/training/plan/export-plan-tensors.ts \
  --map arena6.json \
  --blue-units 10 \
  --red-units 10 \
  --environment-seed 42 \
  --plan-seed 1000 \
  --samples 60 \
  --output snowgym/training/artifacts/plan-tensors-10v10.json \
  --json
```

The exporter refuses an existing output unless `--force` is explicit. The
`snowgym.plan-tensor-dataset.v0` artifact binds simulator and state-hash
versions, scenario and environment seed, the complete validated curriculum,
aligned fixed-size tensors, and a canonical semantic SHA-256 digest. The
auditor rejects plan, seed, shape, bounds, mask, or digest corruption. This is
the portable TypeScript-to-Python boundary; model integration and the paired
ablation remain the next M7 step.

Python consumes that artifact with `PlanTensorDataset` from
`snowgym_training.plan_data`. Loading re-verifies the semantic digest and
materializes immutable `plan_groups` (`float32 [samples,3,38]`),
`plan_group_mask` (`int8 [samples,3]`), and `source_seeds` (`int64`). Call
`batch_for_transitions(plan_indices)` to align plan rows with arbitrary
trajectory transitions; the returned arrays are detached copies, so training
augmentation cannot mutate the audited source. Indices must be a one-dimensional
integer array and are range checked before selection.

Set `architecture.plan_conditioned` only for checkpoints trained with aligned
plan data. The entity policy masks absent plan rows, flattens the fixed three
role slots plus presence mask through a small adapter, and appends that global
embedding to the otherwise unchanged physical context. Legacy configurations
omit the flag and retain their exact parameter shapes. `TorchPolicy` requires
`plan_groups [3,38]` and `plan_group_mask [3]` only when loading a conditioned
checkpoint. This adapter makes counterfactual influence possible; it is not by
itself evidence that a model follows plans. Acceptance still requires actions
collected from the production plan-aware executor and a matched no-plan
ablation.

Export plan-caused rollouts from identical authoritative initial states with:

```bash
npx tsx snowgym/training/plan/export-plan-rollouts.ts \
  --map arena6.json \
  --blue-units 10 \
  --red-units 10 \
  --environment-seed 42 \
  --plan-seed 120 \
  --samples 6 \
  --max-decisions 300 \
  --red-difficulty easy \
  --output snowgym/training/artifacts/plan-rollouts-10v10.json \
  --json
```

Each episode resets to the same state hash, grounds one counterfactual plan,
and records the production plan-aware controller's detached observation,
semantic action, dynamic plan tensor, reward, pre/post hashes, and action
acceptance at every decision. The exporter refuses to replace an artifact
unless `--force` is explicit. Episodes that do not reach a simulator terminal
state within the requested horizon are labeled `decisionLimited`; this is
distinct from the environment's own truncation signal.

Convert the exported JSON to the existing sharded training format without a
server:

```bash
cd snowgym/training
.venv/bin/snowgym-convert-plan-rollouts \
  artifacts/plan-rollouts-10v10.json \
  artifacts/plan-training-10v10 \
  --max-team-units 10 \
  --shard-size 1024 \
  --json
```

The converter independently verifies the TypeScript artifact's canonical
digest, public observation hashes, episode continuity, tensor shapes and
bounds, and accepted action results. It then uses the shared Gym encoders for
observations/actions and emits ordinary `snowgym.trajectory.v0` shards with
aligned `observation__plan_groups [3,38]`,
`observation__plan_group_mask [3]`, and `plan_source_seed` arrays. The same
shards can train the no-plan ablation (extra arrays are ignored by that model)
or a `plan_conditioned` architecture; conditioned training fails closed if
those aligned arrays are absent.

Run a matched behavior-cloning experiment with:

```bash
cd snowgym/training
.venv/bin/snowgym-run-plan-ablation \
  --dataset artifacts/plan-training-10v10 \
  --config src/snowgym_training/configs/plan_bc_ablation_v0.json \
  --output artifacts/plan-bc-ablation-v0 \
  --json
```

The ablation config defines one architecture and optimization budget and is not
allowed to set `plan_conditioned`. The runner deterministically derives two
training configs that differ only by that flag, uses one audited dataset for
both, and records the child checkpoint/state digests and loss metrics in a
semantically hashed result. Its auditor reloads both checkpoints and rejects
any difference in dataset, optimizer, loss, seed, or step. A training result is
still only a plumbing check until the paired policies are evaluated on frozen
counterfactual behavior metrics.

Evaluate both checkpoints on a separate aligned dataset with:

```bash
cd snowgym/training
.venv/bin/snowgym-evaluate-plan-ablation \
  --ablation artifacts/plan-bc-ablation-v0 \
  --dataset artifacts/plan-evaluation-10v10 \
  --output artifacts/plan-bc-ablation-v0/evaluation.json \
  --json
```

The evaluator reports overall and first-decision action accuracy and, at the
first decision of each episode, cyclically swaps only the plan tensors while
holding the physical observation fixed. It measures correct versus shuffled
action negative log-likelihood, predicted action changes, and target movement.
For move/throw labels it also reports correct-versus-shuffled target MSE, which
is essential when plans alter destinations without changing the action type.
The no-plan model should have exactly zero counterfactual sensitivity by
construction. Nonzero sensitivity from the conditioned model shows influence,
but acceptance additionally requires better correct-plan fit and later
closed-loop objective-completion metrics; arbitrary sensitivity is not plan
following.

Run the two qualified checkpoints as actual blue-team controllers from matched
initial states with the frozen direct-versus-left-flank suite:

```bash
cd snowgym/training
.venv/bin/snowgym-evaluate-plan-closed-loop \
  --ablation runs/plan_bc_ablation_qual_v1 \
  --suite src/snowgym_training/configs/plan_closed_loop_v0.json \
  --output evaluations/plan_closed_loop_v0.json \
  --json
```

Each policy receives the same physical state and symbolic plan. The conditioned
branch additionally receives the current server-resolved tensor before every
decision. The result records canonical outcomes, action rejection, normalized
group-to-objective progress, first-action target divergence, and final group
trajectory separation. This evaluator measures real covariate-shifted execution;
it does not infer plan following from offline target loss alone.
Pass `--conditioned-checkpoint PATH` to compare a correction checkpoint against
the original ablation's no-plan control. The result binds both checkpoint and
state digests; the auditor reloads the override rather than treating it as part
of the original matched training run.

The frozen 6v6 development run is retained under
`runs/plan_bc_ablation_dev_v0`, with held-out metrics in
`evaluations/plan_bc_ablation_dev_v0.json`. The conditioned model reduced
correct-plan first-decision target MSE from `0.27275` to `0.04821`; cyclic plan
swapping raised its target MSE by `0.30610`, while both no-plan swap deltas were
exactly zero. Its overall action-type accuracy was lower (`0.95503` versus
`0.97309`), and first-decision action types were identical across the sampled
plans. This is positive target-following development evidence, not a passed M7
gate or closed-loop objective-completion result.

The first qualification is predeclared in
`configs/plan_qualification_v0.json`, bound to
`configs/plan_bc_ablation_qual_v0.json`. After generating exactly the rollout
seeds/counts named there, converting, training, and evaluating, apply the gate:

```bash
cd snowgym/training
.venv/bin/python -m snowgym_training.plan_qualification \
  --evaluation evaluations/plan_bc_ablation_qual_v0.json \
  --spec src/snowgym_training/configs/plan_qualification_v0.json \
  --config src/snowgym_training/configs/plan_bc_ablation_qual_v0.json \
  --output evaluations/plan_qualification_v0.json \
  --json
```

The spec enforces disjoint environment and plan-seed ranges and binds the exact
ablation-config digest. The gate is conjunctive: target accuracy, degradation
under plan swapping, target sensitivity, bounded action-accuracy regression,
and exact no-plan invariance must all pass. A failed check remains visible; the
runner never selects or silently substitutes a checkpoint.

Qualification v0 is retained as a failed gate in
`evaluations/plan_qualification_v0.json`. It passed all target-following and
no-plan-invariance checks, but conditioned action accuracy trailed the no-plan
control by `0.04844`, exceeding the frozen `0.03` allowance. The paired
checkpoints and detailed metrics remain under
`runs/plan_bc_ablation_qual_v0` and
`evaluations/plan_bc_ablation_qual_v0.json`. Do not reinterpret this as an M7
pass or change v0 thresholds; the next revision must use a documented model
change and new disjoint seeds.

For that revision, `architecture.plan_target_only` may be enabled alongside
`plan_conditioned`, `action_conditioned_targets`, and `separate_target_actor`.
The ordinary action actor
then receives only physical state; a separate target actor receives the same
physical context plus the plan embedding and drives move/throw targets and
power. Counterfactual plans therefore cannot change action logits by
construction, and target loss is stopped at the shared physical-feature
boundary, while plans can still change destinations. The paired runner
removes this plan-only routing flag from the no-plan control, preserving one
shared separated-target architecture and optimization budget. Existing
checkpoints and the default fully plan-conditioned pathway are unchanged.

The corrected target-only development checkpoints are retained under
`runs/plan_bc_ablation_target_only_dev_v1`, with metrics in
`evaluations/plan_bc_ablation_target_only_dev_v1.json`. The paired branches
have exactly identical action accuracy and action NLL, while conditioned target
MSE is `0.05811` versus `0.24261` without plans and rises by `0.35230` when
plans are swapped. This is development evidence on reused v0 data; it does not
repair the failed v0 qualification or substitute for a new-seed v1 gate.

Qualification v1 is retained under `runs/plan_bc_ablation_qual_v1`, with its
detailed evaluation and gate result in
`evaluations/plan_bc_ablation_qual_v1.json` and
`evaluations/plan_qualification_v1.json`. It passed every frozen threshold on
new seeds: action behavior is exactly matched at `0.96302` accuracy, while
conditioned target MSE is `0.04448` versus `0.26499` without plans and rises by
`0.35338` when plans are swapped. This qualifies the offline matched-training
milestone. It does not yet establish closed-loop mission completion, unseen
directive combinations, or roster transfer.
