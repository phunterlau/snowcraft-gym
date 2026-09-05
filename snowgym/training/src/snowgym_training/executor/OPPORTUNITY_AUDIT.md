# R1k conditional opportunity audit

The frozen reference is R1i absolute epoch 20. This headless experiment records
every living fighter's teacher and learner action, all geometry heads, public
action masks, controller readiness, independent R1h recommendations, physical
and plan observations, and Python option-tracker state. Recommendation
availability, legality, and physical usefulness are separate measurements.

From the repository root:

```bash
snowgym/training/.venv/bin/python -m snowgym_training.options.opportunity_audit \
  --checkpoint snowgym/training/runs/m7b_engage_r1i_geometry_probe_v0/absolute-epoch-020 \
  --reservoir snowgym/training/runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz \
  --output snowgym/training/runs/m7b_engage_r1k_opportunities_v0
```

The destination must not exist. The frozen configuration pins checkpoint and
reservoir digests, simulator/hash versions, 40 training seeds, and all budgets.
No development, replication, fresh holdout, qualification, or provider data is
collected. Fresh-holdout ancestry auditing remains a prerequisite for R1l.

## Measurements and decision

Teacher episodes are reconstructed with their original plan IDs and production
semantic actions. Every observation tensor and action label must exactly match
the immutable reservoir before conditional labels are accepted. Learner states
have full executed action prefixes and physical/plan/tracker identity hashes.

For each movement, aim, and power channel, select at most 64 hard opportunities,
alternating teacher/learner agreement strata with a cap of four per episode.
Selection requires the learner to invoke the head, a legal action, and an
available recommendation. Angularly undefined predictions are reported
separately. A branch replaces one fighter's one selected head for one decision,
then follows the frozen reference for a total of 30 decisions including the
replacement. A no-replacement branch uses the same prefix. Reset/prefix replay
must reproduce physical, grounded-plan, and option-tracker identities.

Report damage dealt/received, net damage, preferred-range error, and progress
separately. Bootstrap 10,000 draws over episode-mean paired effects; hard-state
selection limits population interpretation. Movement is useful if the net-damage
95% lower bound is positive, or range-error improvement has a positive lower
bound with nonnegative mean net-damage effect. Aim requires a positive
net-damage lower bound. Power is diagnostic and does not gate R1l.

The disposable fit uses hard movement and aim opportunities from seeds
100000–100031, with 100032–100039 held out of this fit. Only the selected unit's
invoked head is supervised; categorical labels remain separate. Use the R1i
loss coefficients, 200 Adam steps at 1e-3, and clip 0.5. The source and inherited
parameters never change; fitted weights are discarded. Require at least 50%
training-loss reduction, lower validation loss, target-encoder gradient
reachability, and finite-difference agreement. These validation episodes may
have appeared in ancestor training and are not called unseen.

`report.json.r1lAllowed` is true only when the fit and movement/aim consequence
checks all pass. Failure archives the complete audit and stops corrective-data
training. It does not close autonomous R1 or authorize a replacement decoder.

## Artifact contract

Versioned compressed JSONL files contain teacher/learner states, living-unit
opportunities, episode prefixes/results, and paired branches. Opportunities
reference the full state by `stateIndex`; the state contains all conditional
outputs and label masks. `hard-fit.json` includes loss traces, selected IDs,
per-component gradient norms/cosines, and physical Jacobians. `manifest.json`
binds source files, checkpoint, dataset, protocol, configuration, and all output
digests. Gzip headers exclude wall time and temporary filenames.

Damage per legal throw is an aggregate ratio, not attributed projectile hit
probability. No browser replay or PPO compatibility is claimed for this dataset.
