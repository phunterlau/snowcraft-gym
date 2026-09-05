# R1l matched corrective-data experiment

Run from the repository root after R1k passes:

```bash
snowgym/training/.venv/bin/python -m snowgym_training.options.corrective_data \
  --checkpoint snowgym/training/runs/m7b_engage_r1i_geometry_probe_v0/absolute-epoch-020 \
  --reservoir snowgym/training/runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --audit snowgym/training/runs/m7b_engage_r1k_opportunities_v0 \
  --output snowgym/training/runs/m7b_engage_r1l_corrective_v0
```

The immutable configuration fixes four arms: A teacher states/teacher masks;
B mixed states/teacher masks; C teacher states/conditional labels; D mixed
states/conditional labels. The inherited R1i source remains frozen. Only its
geometry modules train, with unchanged R1i loss coefficients, 420 Adam steps,
batch 256, learning rate 3e-4, clip 0.5. Each head normalizes its valid labels.
Sampling chooses episodes uniformly, then states within each episode. B/D use
128 states from each source; A/C use 256 teacher states. A/C and B/D share
sampling streams. No losses or labels change categorical action choice.

Run all four arms with RNG 93001. Replicate only predeclared A/D with RNGs
93002/93003 if D increases historical-set autonomous success without a rejected
action-rate regression. Never choose another arm from observed rankings. Only
the final 420-step checkpoint is evaluated for each fit.

## Lineage and evaluation

The runner verifies all R1k artifacts, its gate and reference identity, then
audits checkpoint ancestors, their consumed seed schedules, the original BC
dataset manifest and tensor shards, and the teacher reservoir. A descendant
dataset's source summary does not replace the original artifact. Missing
ancestry or reserved-seed collisions stop the run before collection.

The original 1,920-transition BC training dataset was recovered byte-for-byte
from a previous temporary run into `training/artifacts/plan-bc-qual-v1-original-train`.
Its digest is `sha256:f64e30e6458ca3d9c9a6e110aae7ae3248e6e2715a938ce323732b1e658d61e6`.
Both the failed local-only preflight and the successful recovered preflight are
retained. This restores provenance; it changes no checkpoint or dataset bytes.

Fresh teacher-regression seeds 108000–108039 and learner-validation seeds
108100–108139 are collected only after ancestry passes, and are never fitted.
Reference and initializer baselines plus each final policy are evaluated on
historical 200000–200039 and fresh replication-development 210000–210039, with
correct Engage and HOLD-input conditions. Qualification seeds remain sealed.

Reports include per-arm original bootstrap gates, immediate-reference effects,
factorial simple effects and interaction, exposure counts, held-out losses,
parameter changes, and episode-paired bootstrap intervals. Training RNG runs
remain separate; repeated environment seeds are not independent replications.
Fresh checkpoint files have format `snowgym.corrective-checkpoint.v0`, explicit
optimizer-step count, deterministic-only execution, and a pinned R1i parent.
These are supervised checkpoints; no PPO compatibility is implied.
