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
does not yet provide a trained PPO checkpoint.

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

`ppo_update` normalizes advantages once over the complete rollout, derives
epoch permutations from the training seed and update index, and applies the
hybrid clipped objective in deterministic minibatches. It rejects non-finite
losses or gradients, clips the aggregate gradient norm, and returns
sample-weighted policy/value/entropy/KL/clip diagnostics.

`snowgym.ppo-checkpoint.v0` persists model and optimizer tensors, Torch RNG
state, architecture and PPO configuration, training seed, curriculum digest,
update index, environment-step count, and semantic state/metadata digests.
Loading uses restricted Torch deserialization and rejects incompatible
training provenance before restoring state. The deterministic acceptance test
matches uninterrupted training exactly across a save/restore boundary. This is
rollout-boundary optimizer resume; persistent-world collection and episode-seed
scheduling are the next slice.

The frozen `ppo_curriculum_v0.json` keeps training ranges disjoint from eight
evaluation seeds per gate and sets thresholds before qualifying runs. The
current foundation does not yet claim a PPO result; live batch collection and
the qualifying checkpoint series remain next.
