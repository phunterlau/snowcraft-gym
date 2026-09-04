# R1f supervised probe: results and next decision

2026-09-04. Measurement repair: `e352e0e`. Probe implementation and frozen
configuration: `8f7d4f9`. All evaluations were headless; no provider calls or
qualification seeds were used.

## Outcome

The final 20-epoch supervised-only probe did not reproduce successful Engage
trajectories. Training-seed and development-seed success both remained 0/40.
Development performance regressed from the R1e source checkpoint. The model
also became less selective between Engage and the counterfactual HOLD plan.
R1 remains open, R2 remains blocked, and no additional PPO continuation was run.

The useful finding is a mismatch between improvement in the BC coordinate
objective and deterioration in physical throw direction. The experiment does
not establish a hard capacity limit for the architecture, and it does not
isolate PPO's contribution through a matched PPO-versus-BC comparison.

## Frozen experiment

The source was R1e update 200. Training used all 5,367 successful-teacher states
from 40 training seeds, fresh Adam at `3e-4`, minibatches of 256, and 20 epochs
(420 optimizer steps; 107,340 state presentations). Only the Stage-1 actor
residual/v3-adapter modules could change. The critic and inherited actor
parameters were frozen. The original BC weights were retained: action 1,
target 5, power 0.5, and throw-class weight 5.

There were zero PPO, entropy, initializer-KL, or critic updates. The objective
was BC alone. Epochs 0, 10, and 20 were retained; epoch 20 was the predeclared
final result. No extra epochs or checkpoint selection followed the outcome.

The arena, random Red controller, 5v5 roster, 200-decision option horizon,
300-decision battle horizon, and deterministic evaluation policy were unchanged.

## Closed-loop results

| Split / checkpoint | Contact | At least one hit | Mean Engage progress | Engage success | Battle wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, source R1e | 95% | 87.5% | 20.8% | 0/40 | 0/40 |
| Training, final probe | 72.5% | 42.5% | 7.2% | 0/40 | 0/40 |
| Development, source R1e | 95% | 70% | 14.0% | 0/40 | 0/40 |
| Development, final probe | 87.5% | 52.5% | 6.6% | 0/40 | 0/40 |

The training rows are newly evaluated on the reservoir's 40 seeds
`100000–100039`. The source-development row comes from the preserved R1e
artifact; final development uses the same seeds `200000–200039`. Training
and development are reported separately.

Final development controls:

| Plan/policy condition | Contact | At least one hit | Mean Engage progress | Engage success |
| --- | ---: | ---: | ---: | ---: |
| Correct Engage | 87.5% | 52.5% | 6.6% | 0/40 |
| HOLD counterfactual (`shuffled`) | 80% | 25% | 2.0% | 0/40 |
| Original initializer | 0% | 0% | 0% | 0/40 |

All conditions have zero rejected actions. The source R1e HOLD control had
zero contact and hits. The new HOLD behavior weakens the earlier evidence of
selectivity, although a correct-versus-HOLD behavioral difference remains.
HOLD is a single wrong-plan control, not a randomized distribution of plans.

## Teacher-state diagnostics

These are in-sample, living-unit measurements against teacher labels. The
dataset contains 26,815 living-unit decisions, including 1,162 throw labels.
Coordinates are converted using the 100-by-80 arena. Target and ray errors
select the teacher's action branch even when classification disagrees.

| Metric | Before fitting | After 20 epochs |
| --- | ---: | ---: |
| Overall action agreement | 88.33% | 90.31% |
| Throw-class agreement | 72.81% | 75.65% |
| Hold-class agreement | 0.80% | 36.98% |
| Move-target RMSE | 10.06 units | 6.83 units |
| Throw-target RMSE | 18.71 units | 9.74 units |
| Mean throw-ray angular error | 31.81 degrees | 45.65 degrees |
| Throw-power RMSE | 0.1130 | 0.1264 |

Mean training BC loss decreased from 0.46494 during epoch 1 to 0.33547 during
epoch 20. These are epoch averages over changing parameters, not fixed-model
validation losses. Epoch 10 already exhibited lower throw-coordinate error
with worse angular error; it was retained for diagnosis and never substituted
for the final checkpoint.

The phase partition is explicit: fire means a teacher throw label; contact
means another action within 9 world units of a living enemy; approach covers
the remaining living labels. At epoch 20:

- Approach: 20,795 labels, 91.44% class agreement, 6.61-unit move RMSE.
- Contact: 4,858 labels, 89.01% class agreement, 8.23-unit move RMSE.
- Fire: 1,162 labels, 75.65% class agreement, 45.65-degree mean ray error.

The model still predicted move for 283 teacher throw labels. Of 503 teacher
hold labels, 311 became moves, 6 became throws, and 186 were classified as hold.
These classification errors coexist with the direction/power errors.

## Interpretation

1. **The new actor paths can change and optimize the supplied loss.** Their
   parameter L2 change from R1e was 3.88022. Inherited heads, other inherited
   actor parameters, and critic changed by exactly zero. This rules out a
   disconnected Stage-1 optimizer as the explanation for this probe.

2. **Absolute aim-coordinate error is an insufficient physical fitting metric.**
   `ThrowSystem.computeThrowDirection` normalizes the displacement from the
   player to the requested aim point; target distance along the same ray does
   not set shot range. Charge controls speed and arc. Two aim points can have
   very different coordinate error but nearly identical throw direction;
   reducing endpoint error can also rotate the ray away from the teacher.
   This probe exhibits the latter metric pattern. See
   [ThrowSystem](../../../src/systems/ThrowSystem.ts).

3. **The current aggregate BC loss can improve while trajectory quality falls.**
   Move labels dominate the pooled target term, while throw weighting applies
   to classification only. This and the direction mismatch are candidates
   for the next controlled intervention. Their individual causal effects
   have not been isolated by this experiment.

4. **Single-plan fitting can alter behavior under other plans.** Shared v3
   adapters and plan residuals changed using Engage-only data. No HOLD labels
   constrained their off-plan outputs. This is a plausible explanation for
   weaker HOLD selectivity; it requires a separate control rather than a claim
   that the policy has learned to interpret HOLD differently.

5. **Removing PPO does not make this configuration sufficient.** The result
   motivates better physical supervision before a longer PPO run. It does
   not prove PPO is harmful: this probe also changes the optimization mixture,
   minibatch size, and optimizer history, and is not a matched causal ablation.

## Recommended next bounded experiment

First run a no-training action-channel intervention on the final probe:
teacher throw direction alone, teacher power alone, and both, with learner
movement/action selection retained. This determines whether better aim would
actually repair hits or whether decisions/movement remain dominant blockers.
Existing intervention infrastructure can support the test without provider
calls or qualification seeds.

Then predeclare one supervised-loss intervention: align the throw-target term
with the executed unit-relative direction while keeping the move term,
architecture, Stage-1 freeze, data, seeds, and training budget fixed. Check
near-zero rays, finite gradients, and gradient scale before training. Report
angle/power errors and correct/HOLD outcomes alongside coordinate MSE. Keep
plan-preservation training changes as a separate experiment.

Continue to require the original R1 success gate. Do not promote a checkpoint
on lower BC loss, improved contact, or the best intermediate epoch. Multi-cluster
target replacement, temporal option observations, and qualification-lineage
repairs from the broader review remain separate unfinished work.

## Evidence and reproducibility

- [Probe artifact directory](../runs/m7b_engage_r1f_supervised_probe_v0/)
- [Machine-readable report](../runs/m7b_engage_r1f_supervised_probe_v0/report.json)
- [Recursive artifact manifest](../runs/m7b_engage_r1f_supervised_probe_v0/manifest.json)
- [Per-seed development evaluation](../runs/m7b_engage_r1f_supervised_probe_v0/development-evaluation.json)
- [Final teacher-state metrics](../runs/m7b_engage_r1f_supervised_probe_v0/teacher-agreement-epoch-020.json)
- [Runner and interpretation guide](../src/snowgym_training/options/README.md)

The manifest records configuration/source/checkpoint/reservoir digests,
simulation/state-hash versions, every retained checkpoint, all epoch metrics,
and hashes for all output files. Historical R1–R1e artifacts were untouched.
The probe uses the shared checkpoint container with an explicit supervised
gate ID; it is not eligible for option-PPO exact resume or automatic promotion.

Implementation verification passed 325 TypeScript tests, the production build,
50 Gym-client tests, and 177 training tests before the run. An additional
archived-result regression checks hashes, seed partitions, all 20 epochs,
frozen critic state, and the negative outcome before the evidence commit.
Final evidence verification passed all 325 TypeScript tests, the build,
50 Gym-client tests, and 178 training tests. The build retains its existing
large-chunk warning; compatibility Gym versions emit deprecation warnings.
