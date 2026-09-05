# R1j matched decoder probe

R1j tests output geometry after R1i found no clear advantage from relative input
features. This experiment is deterministic-only and does not modify production
PPO, the Gym action schema, or the inherited classifier/critic.

The frozen run is complete. [R1j results](../../../reviews/m7b_r1j_results.md)
show no recovery qualification and no clear improvement over the absolute control.

## Controlled matrix

All arms use R1i's absolute entity features, 33,669 new parameters, zero-output
initialization, R1f source checkpoint, and identical fitting conditions.

| Arm | Movement residual | Shot residual |
| --- | --- | --- |
| `absolute` | Pre-tanh absolute target | Pre-tanh absolute target |
| `displacement` | Bounded world displacement | Pre-tanh absolute target |
| `direction` | Pre-tanh absolute target | Normalized direction correction |
| `both` | Bounded world displacement | Normalized direction correction |

Power uses the same pre-sigmoid residual in every arm. Action selection always
comes from the frozen source. All zero-output initializers reproduce the source
exactly, providing an untrained control for the geometric correction priors.
The absolute arm must reproduce R1i's absolute-feature control before the matrix
is interpreted. Neither teacher movement nor nearest-enemy aiming is installed.

## Decoders

For source movement target $y_0$ in world coordinates and new head output $z_m$,

$$
y_m = \mathrm{clip}_{arena}\left(y_0 + 10\tanh(z_m)\right).
$$

The correction is limited to ten world units per axis from the inherited target
on each decision. This limit is an explicit geometric prior, not a desired
combat range or a mission objective. The inherited target remains the anchor;
this probe does not train a displacement policy from scratch.

For a source shot ray $d_0=y_0-x$, form its unit direction $u_0$ and correct it
with the two new shot outputs $z_s$:

$$
u = \frac{u_0+z_s}{\max(\lVert u_0+z_s\rVert,10^{-6})}.
$$

Embed the corrected direction at the inherited ray length. The implementation
adds the change from the reference unit direction to the inherited target;
normalizing the reference identically makes zero residuals cancel exactly in
float32. For a degenerate inherited ray, the reference is zero and a one-world-
unit embedding radius is used. A zero corrected vector stays finite. Targets
outside the arena are clipped along the shooter-to-target ray so independent
coordinate clipping does not rotate the requested direction. An outward ray
at an arena boundary can collapse to a zero-length target; metrics count
undefined predicted rays separately.

Direction-vector normalization is many-to-one, and displacement clipping is
non-invertible at boundaries. These deterministic outputs must not be inserted
into PPO while pretending they retain the inherited tanh likelihood. Sampled
execution raises an error; custom checkpoints have `ppoCompatible: false`.
A later stochastic contract needs explicitly defined distributions and tests.

## Frozen training and evaluation

Use the audited 5,367-state successful teacher reservoir, training seeds
100000–100039. Each arm first fits a disposable copy on R1i's same 32 selected
states for 200 Adam steps at `1e-3`. All four arms must reduce common loss by
at least 50%, retain finite gradients, reach the new encoders, and preserve
source weights before fresh full fits begin.

The shared loss remains move MSE + throw-direction cosine loss + 0.1 throw
endpoint MSE + 0.5 power MSE. Keeping the endpoint term controls the objective
across arms; the direction decoder cannot independently fit radial endpoint
distance. Retain this limitation when interpreting its loss.

Full fitting uses 20 epochs, minibatches of 256, Adam `3e-4`, gradient norm clip
0.5, and paired initialization/shuffle seed 92001. Retain epochs 0/20 and select
only the final epoch. Record approach/contact/fire teacher-agreement metrics,
parameter changes, exact source preservation, and checkpoint reload parity.

Evaluate the source and four final arms on development seeds 200000–200039,
each with correct Engage and HOLD-preview plan inputs: 400 episodes if the gates
pass. The HOLD preview is computed on each visited physical state; trajectories
can diverge. It measures plan-input sensitivity, not HOLD mission qualification.
Keep Engage success/progress, full-battle wins, and rejected actions separate.

Predeclared simple effects and interaction use paired bootstrap intervals with
10,000 draws and seed 770001. These repeatedly used development seeds support
exploratory comparisons, not qualification; intervals are not multiplicity
adjusted. No provider requests, PPO updates, or executor promotion occur here.

## Entry point

From `snowgym/training`, using a fresh output path:

```bash
.venv/bin/python -m snowgym_training.options.decoder_probe \
  --checkpoint runs/m7b_engage_r1f_supervised_probe_v0/epoch-020 \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_r1j_decoder_probe_v0
```

The runner owns a detached batch host. No browser or HTTP server is needed.
Artifacts are immutable and custom checkpoints use
`snowgym.decoder-probe-checkpoint.v0` with source, configuration, optimizer,
file, and semantic-state provenance.

- [Frozen configuration](../configs/m7b_engage_r1j_decoder_probe_v0.json)
- [Decoder implementation](decoder_probe.py)
- [Runner and checkpoint loader](../options/decoder_probe.py)
