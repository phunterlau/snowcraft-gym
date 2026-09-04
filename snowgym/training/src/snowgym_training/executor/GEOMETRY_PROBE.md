# R1i geometry-feature probe

This is an isolated, deterministic-only supervised experiment. The production
executor, Gym action schema, and PPO likelihood implementation remain unchanged.

## Matched architecture

Both arms freeze the entire R1f epoch-20 actor and critic. They add equal-size
trainable entity encoders and separate movement and shot/power residual heads.
The encoders pool living allies/enemies and present projectiles/obstacles for
each fighter. Each head also receives that fighter's v3 state, owning symbolic
directive, and owning/supported physical role rows. There are no raw ID features.

The absolute arm encodes other entities' arena-normalized positions. The
relative arm subtracts the observing fighter's position from entity positions
and present unit controller targets before encoding. Both retain the fighter's
own absolute state and role-state inputs. The whole policy is therefore not
claimed to be translation invariant. Masked mean/max pools preserve entity
ordering symmetry and handle empty groups.

Both arms keep the inherited absolute-coordinate tanh decoder. A zero-output
residual is added to its move and throw pre-tanh means and its power pre-sigmoid
mean. This preserves the source policy exactly at initialization, while allowing
target losses to train the new encoders. With zero output weights, encoder
gradients are initially zero; the first head update opens that path. Tests and
the fitting gate verify encoder gradients after updates. Action logits are
always inherited unchanged on the same input state.

This first probe isolates input geometry between matched new modules. It does
not yet implement enemy-pointer actions, relative-coordinate output decoding,
or a tactical movement reference. Neither arm installs teacher movement,
nearest-enemy auto-aim, a firing-range rule, or another execution override.
Comparisons against the source also include added capacity and a changed loss;
only the absolute-versus-relative comparison controls those factors.

## Loss and fitting gate

On living units, the common supervised loss is

$$
L = L_{move} + L_{direction} + 0.1 L_{throw\ endpoint} + 0.5 L_{power}.
$$

Move/endpoint errors are mean squared errors in normalized action coordinates.
Direction uses world-space rays from the fighter, with the frozen arena's
half-extents $(50,40)$ applied before normalization:

$$
L_{direction} = 1 - \frac{u\cdot v}
{\max(\lVert u\rVert,10^{-6})\max(\lVert v\rVert,10^{-6})}.
$$

Cosines are clamped to $[-1,1]$. Undefined teacher rays are excluded. A zero
predicted ray against a valid teacher ray incurs unit direction loss. Empty
action masks produce zero loss/gradient. Power is supervised only on throws;
move and shot heads receive no gradient from unused action dimensions.

The audited corpus contains 5,367 states from 40 successful teacher episodes,
training seeds 100000–100039. Before full fitting, disposable copies fit the
first 16 states containing any throw and first 16 containing a move but no throw.
Each arm must reduce the combined loss by at least 50% in 200 Adam steps at
`1e-3`, with finite gradients, reachable new encoders, and unchanged source
weights. Failure stops the experiment before full training.

Fresh arms then fit for 20 epochs, batch size 256, Adam `3e-4`, gradient norm
clip 0.5, and paired initialization/shuffle seed 92001. Epochs 0 and 20 are
retained; evaluation uses only the final epoch. Both arms see the same labels,
loss, minibatch order, and optimizer budget. Their learned parameter changes
and full-corpus movement, throw-direction, and power errors are reported.

## Evaluation and boundaries

Evaluate the frozen source and both final arms on development seeds
200000–200039. For each, compare correct Engage input with a HOLD plan preview
computed on each visited physical state, using the existing evaluation helper.
The alternative trajectories can diverge; the preview is state-matched, not a
claim of identical trajectories. Keep mission success/progress, full-battle
wins, rejected actions, contact, and hit metrics separate. These repeatedly
used development seeds are not qualification evidence.

The custom `snowgym.geometry-probe-checkpoint.v0` container binds source
metadata, fitted weights, optimizer state, configuration, and hashes. Reload
must preserve deterministic actions exactly. It is marked `ppoCompatible: false`;
sampled execution raises an error rather than returning an invented likelihood.
Any later stochastic integration needs its own likelihood and resume tests.

R1/M7b/M7c thresholds remain unchanged. Neither small-batch fitting nor oracle
success from R1h qualifies a learned executor. A probe gain must be measured
against the unchanged source and the matched absolute-feature control; HOLD
selectivity remains a separate concern.

## Entry point

From `snowgym/training`:

```bash
.venv/bin/python -m snowgym_training.options.geometry_probe \
  --checkpoint runs/m7b_engage_r1f_supervised_probe_v0/epoch-020 \
  --reservoir runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output runs/m7b_engage_r1i_geometry_probe_v0
```

The command owns a detached batch host. No HTTP server, browser, or provider
API is needed. Choose a new output path for reproduction; artifacts are immutable.

- [Frozen configuration](../configs/m7b_engage_r1i_geometry_probe_v0.json)
- [Model and loss](geometry_probe.py)
- [Runner and checkpoint loader](../options/geometry_probe.py)
