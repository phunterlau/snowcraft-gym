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
