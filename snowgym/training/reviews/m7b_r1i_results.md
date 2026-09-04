# R1i: matched geometry-feature fitting results

Date: 2026-09-04. Status: completed supervised development probe; R1 remains open.

## Main finding

Trainable geometry residuals improve physical execution and Engage/HOLD
separation over R1f. The relative-feature arm completes one Engage mission;
neither arm approaches the recovery threshold or wins a full battle. The
experiment does not establish that relative input coordinates outperform the
matched absolute-feature control.

| Controller | Engage success | Contact | At least one hit | Mean Engage progress |
| --- | ---: | ---: | ---: | ---: |
| Frozen R1f source | 0/40 | 87.5% | 52.5% | 6.6% |
| Absolute-feature residual | 0/40 | 100% | 100% | 30.8% |
| Relative-feature residual | 1/40 | 100% | 97.5% | 26.3% |

All three have zero full-battle wins and zero rejected actions. These are 40
paired development seeds, 200000–200039, not untouched qualification seeds.

## Fitting and integrity

Both arms start from exactly the same R1f policy outputs and add 33,669
trainable parameters. The complete inherited actor and critic stay frozen.
Both use the same new module sizes, initial weights, labels, optimizer budget,
minibatch order, and loss. Only the entity-coordinate transformation differs.
The inherited absolute tanh output decoder is retained in both arms.

The disposable 32-state gate reduced total loss by 97.6% for absolute features
and 98.4% for relative features, exceeding the predeclared 50% requirement.
Move, direction, and power losses reached the new encoders after fitting;
unused heads had zero gradient and inherited parameters had no gradient.
This proves fitting on those selected states, not capacity to represent the
entire control policy accurately.

Fresh arms then fit for 20 epochs on the audited 5,367-state teacher reservoir:
420 optimizer steps and 107,340 state presentations per arm. There were no PPO
updates, critic updates, provider calls, or teacher execution overrides. New
parameter L2 changes were 4.5520 and 4.9899. Both final checkpoint reloads
preserve deterministic actions exactly. All frozen source state tensors remain
identical to R1f, and its paired evaluation records reproduce exactly.

## Teacher-state agreement

| Metric | Source | Absolute residual | Relative residual |
| --- | ---: | ---: | ---: |
| Move endpoint RMSE, world units | 6.83 | 4.48 | 4.45 |
| Throw endpoint RMSE, world units | 9.74 | 19.59 | 19.72 |
| Throw-ray mean error, degrees | 45.65 | 19.73 | 18.81 |
| Power RMSE | 0.1264 | 0.1232 | 0.1240 |
| Action-class accuracy | 90.31% | 90.31% | 90.31% |

No measured teacher throw ray was undefined. Action accuracy is unchanged
because logits are frozen on identical observations. Endpoint error rises while
ray error falls: the throw system normalizes the target vector, so distance
along a ray can change without changing its direction. R1i uses a direction-aware
loss in both arms. Its gains over R1f combine that objective change, added
capacity, and a new gradient path; this experiment does not separate those
contributions. Absolute-versus-relative comparisons do control them.

The remaining errors are material. Mean ray error is still about 19 degrees,
and movement RMSE remains about 4.5 world units on teacher states. Better
average imitation is insufficient for reliable mission completion.

## HOLD counterfactual

The evaluator previews a HOLD plan on each visited physical state while keeping
the Engage scenario and scoring definition. It tests response to changed plan
input; it does not establish success on the separate HOLD mission.

| Controller under HOLD input | Contact | At least one hit | Engage progress |
| --- | ---: | ---: | ---: |
| Source | 80% | 25% | 2.0% |
| Absolute residual | 0% | 0% | 0% |
| Relative residual | 20% | 2.5% | 0.1% |

Correct-plan progress exceeds HOLD-input progress on 40/40 paired seeds for
the absolute arm and 39/40 for the relative arm. This is improved plan-input
separation in the development scenario; mission qualification still fails.

## Exploratory paired comparisons

A post-run paired bootstrap with 10,000 draws and seed 760001 gives:

- Absolute minus source progress: +24.2 percentage points, 95% interval
  +18.8 to +29.7.
- Relative minus source progress: +19.7 points, interval +14.0 to +26.1.
- Relative minus absolute progress: −4.5 points, interval −10.7 to +1.8.
- Relative minus absolute success: +2.5 points, interval 0 to +7.5.

These intervals were computed after the run, are not multiplicity-adjusted,
and use development seeds already examined in earlier work. One completion
does not justify selecting or promoting the relative arm. Retain both final
checkpoints and their negative qualification result.

## Design decision and next step

The new target-learning path is functional: it changes only the intended
parameters, improves physical execution, and preserves useful plan sensitivity.
Changing pooled entity coordinates alone provides no clear additional benefit
in this experiment. Keep relative features as an ablation rather than making
them the production default.

The next bounded design should address the output geometry directly:

1. Predeclare a matched decoder probe that predicts fighter-relative movement
   displacement and shot direction, with separate power. Keep the classifier,
   dataset, new-module capacity, and training budget controlled. Report any
   geometric anchor or range prior explicitly and include its untrained control.
2. Preserve the same-state HOLD check, and measure angular and movement errors
   by approach/contact/fire phase. Retain the current absolute-coordinate
   decoder as the control. Do not assume a nearest-enemy reference represents
   every mission.
3. Before stochastic use, define a versioned action distribution and correct
   transformed log probabilities for the new decoder. Test boundary behavior,
   unused dimensions, stored/recomputed likelihoods, and exact resume before
   any PPO continuation.

This experiment remains deterministic-only. The proposed decoder is not
implemented here. Current R1, M7b, and M7c thresholds remain unchanged, and
online LLM-over-learned-fighter experiments remain gated.

## Evidence

- [Architecture, loss, boundaries, and command](../src/snowgym_training/executor/GEOMETRY_PROBE.md)
- [Report](../runs/m7b_engage_r1i_geometry_probe_v0/report.json)
- [Manifest](../runs/m7b_engage_r1i_geometry_probe_v0/manifest.json)
- [Small-batch gate results](../runs/m7b_engage_r1i_geometry_probe_v0/small-batch-gates.json)
- [Absolute final checkpoint](../runs/m7b_engage_r1i_geometry_probe_v0/absolute-epoch-020/checkpoint.json)
- [Relative final checkpoint](../runs/m7b_engage_r1i_geometry_probe_v0/relative-epoch-020/checkpoint.json)

The archive includes epochs 0/20, all 240 evaluation episode records, full
teacher-agreement metrics, parameter-change evidence, and source/data/checkpoint
digests. Outputs are immutable. No browser or provider API was used.
