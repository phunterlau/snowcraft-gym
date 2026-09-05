# R1j: matched decoder results

Date: 2026-09-04. Status: completed negative development experiment; R1 remains open.

## Result and decision

Neither decoder change improved mean Engage progress over the existing absolute
control under the frozen training budget. All four fitted arms completed zero
of 40 Engage missions and won zero full battles. Retain the R1i absolute-feature,
absolute-decoder model as the development reference; do not promote a new decoder
or restart PPO based on this experiment.

| Fitted arm | Engage success | Contact | At least one hit | Mean progress |
| --- | ---: | ---: | ---: | ---: |
| Absolute control | 0/40 | 100% | 100% | 30.8% |
| Displacement correction | 0/40 | 95% | 97.5% | 25.3% |
| Direction correction | 0/40 | 100% | 100% | 25.5% |
| Both corrections | 0/40 | 100% | 95% | 22.0% |

The frozen source scored 0/40 with 6.6% mean progress. All 400 evaluation
episodes had zero rejected actions and zero full-battle wins. Mission success
and battle wins are separate metrics.

## Controls and fitting

The matrix uses the same absolute entity features, 33,669 new parameters,
initialization, teacher corpus, loss, minibatch order, learning rate, and
20-epoch budget. The complete source actor/critic stays frozen. Power uses the
same learned residual. There are no teacher execution overrides or new
nearest-enemy references.

The displacement decoder bounds corrections to ten world units per axis around
the inherited movement target. The direction decoder corrects a unit ray,
retains inherited ray length, and clips along the ray at arena boundaries.
These correction priors and their optimization geometry are part of the tested
design. Equal learning rates do not imply equal physical target changes per
gradient step across different decoders.

All zero-output initializers preserve source actions exactly. The disposable
32-state fitting gates reduced loss by 97.6%, 95.3%, 72.4%, and 72.3% in arm
order, exceeding the predeclared 50% requirement. New encoders received move,
direction, and power gradients after fitting; unused heads and frozen source
parameters did not. Passing this small fitting test did not predict mission
completion.

Each fresh arm then received 420 Adam steps and 107,340 teacher-state
presentations over 20 epochs. Parameter L2 changes were 4.5520, 5.2784, 4.5874,
and 5.2129. Source tensors remain exact, and custom checkpoint reloads preserve
deterministic actions exactly. The absolute control reproduces R1i's complete
final model state, training loss trace, and paired development records. The
source evaluation also reproduces R1i exactly.

## Teacher-state agreement

| Metric | Absolute | Displacement | Direction | Both |
| --- | ---: | ---: | ---: | ---: |
| Move endpoint RMSE, world units | 4.48 | 4.89 | 4.47 | 4.88 |
| Throw endpoint RMSE, world units | 19.59 | 19.59 | 6.84 | 6.84 |
| Throw-ray mean error, degrees | 19.73 | 19.72 | 26.51 | 26.50 |
| Power RMSE | 0.1232 | 0.1232 | 0.1261 | 0.1261 |

All fitted arms retain the source's 90.31% action-class accuracy on identical
teacher states. No teacher-state predicted throw ray was undefined. The archive
also retains approach/contact/fire breakdowns.

The direction correction improves endpoint error while worsening angular
error relative to the absolute control. Its inherited-radius constraint and
the shared endpoint-loss term change which errors it can fit independently.
This reinforces the need to judge shots by direction and physical outcomes;
endpoint accuracy alone is insufficient. It does not prove that all directional
action distributions are inferior.

## HOLD-input sensitivity

| Arm under HOLD preview | Contact | At least one hit | Engage progress |
| --- | ---: | ---: | ---: |
| Absolute | 0% | 0% | 0% |
| Displacement | 60% | 25% | 3.9% |
| Direction | 5% | 0% | 0% |
| Both | 50% | 27.5% | 3.9% |

The displacement variants weaken the clean HOLD-input separation achieved by
the absolute control. HOLD is previewed on each visited physical state while
the Engage scenario remains the scoring task. This is not a separate HOLD
mission qualification.

## Predeclared paired comparisons

Progress differences in percentage points, with paired-bootstrap 95% intervals:

| Contrast | Difference | Interval |
| --- | ---: | ---: |
| Displacement with absolute shot decoder | −5.5 | −11.3 to +0.5 |
| Direction with absolute move decoder | −5.3 | −12.4 to +1.6 |
| Displacement with direction decoder | −3.5 | −10.1 to +3.0 |
| Direction with displacement decoder | −3.3 | −9.7 to +3.0 |
| Interaction | +2.0 | −7.7 to +11.8 |

The bootstrap uses 10,000 paired draws and seed 770001, as predeclared.
Intervals are unadjusted exploratory comparisons on repeatedly used development
seeds, 200000–200039. No interval supports a clear positive effect. Success
contrasts are all zero with degenerate bootstrap intervals because every arm
failed every mission; that does not establish population-level equivalence.

## Next bounded milestone

Pause further decoder variations. The tested feature and decoder changes have
not closed the gap to the production teacher. The next diagnostic should
measure per-head errors and label coverage on learner-visited **training**
states, using the fixed R1i absolute reference:

1. Compare teacher-state and learner-state movement, ray, and power errors by
   approach/contact/fire phase. Record action-choice disagreement, range,
   readiness, and whether a production teacher label exists for the head the
   learner actually executes. Keep actual teacher actions distinct from
   independently defined conditional movement/shot recommendations.
2. Audit per-component gradient magnitudes alongside coverage. The present
   evidence does not distinguish limited data coverage from optimization or
   representation limits; do not declare covariate shift as the proven cause.
3. If the audit supports it, predeclare one bounded corrective-data fit with
   unchanged architecture, a matched old-data-only control, and retention of
   original teacher samples. Generate data only from training seeds, preserve
   the existing corpus, and retain the HOLD check before further PPO.

The corrective-data run is not implemented by R1j. Its thresholds and budget
must be fixed before fitting. R1/M7b/M7c gates remain unchanged, and neither
the new deterministic decoder checkpoints nor the development reference is a
qualified executor for online LLM commander experiments.

## Evidence and command

- [Design and headless entry point](../src/snowgym_training/executor/DECODER_PROBE.md)
- [Report and paired contrasts](../runs/m7b_engage_r1j_decoder_probe_v0/report.json)
- [Manifest](../runs/m7b_engage_r1j_decoder_probe_v0/manifest.json)
- [Fitting gates](../runs/m7b_engage_r1j_decoder_probe_v0/small-batch-gates.json)

The immutable archive includes all eight epoch-0/20 checkpoints, 400 episode
records, phase metrics, parameter changes, and source/data/checkpoint digests.
No provider API, browser, or PPO training was used.
