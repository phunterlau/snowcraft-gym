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

This is a narrow 1v1 imitation-learning proof. M6.1 adds exact-parity batched
simulation before scaling data collection; M6.2 adds reward-driven PPO.
