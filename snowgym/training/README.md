# SnowGym training

This isolated package contains reproducible data collection and, in later
milestones, learning code. The core `snowgym-client` remains free of Torch.

Set up the M6.0a data tools:

```bash
uv sync --extra dev
```

The future neural-policy extra is already declared but is not needed to export
or audit trajectories:

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

These tools are correctness scaffolding, not yet an RL result. M6.0b adds the
first behavior-cloning model and checkpoint evaluator.
