# R1m-S4: frozen exploration and advantage audit

Completed 2026-09-05. Implementation/declaration: `5977ba0`. Evidence:
`runs/m7b_engage_r1m_s4_v0/manifest.json` (21 digest-verified artifacts).
The [declaration](m7b_r1m_s4_declaration.md) was committed before measurement.
No optimization, provider calls, new policy continuations or checkpoint promotion
occurred. Corrected-shot assistance remains part of the archived experiment;
`autonomousQualificationEligible` is false.

## Reconstruction

All nine collection points (three RNGs, updates 1/11/21) reconstructed successfully:
72 trajectories, 41 distinct training seeds, 2,094 learned decisions and 7,874
living MOVE opportunities. These counts do not represent independent environments.
Common-state inference used all 57 training snapshots; 54 contained selected,
valid movement recommendations. Prefix plus learned-window replay required
15,549 simulator decisions, below the declared 25,800 ceiling.

Sampled latents, action arrays, post-action state hashes and rewards matched.
Maximum sampled log-probability error and maximum archived first-minibatch loss
error were both **zero** at every collection point. Snapshot sampling, restored
plan/option/tensor identities and source/archive digests passed. Frozen tails
were digest-verified inputs, not replayed in this audit. Inference left weights
unchanged.

## Exploration and local learning signal

Each row pools descriptive MOVE opportunities within that collection point.
Distance units refer to commanded targets, not distance actually traveled by a
fighter. Radius-three coverage is a latent geometric measure.

| RNG | Update | Mean sampled target shift | Mean recommendation gap | Mean latent distance / sigma | Within radius 3 | Sample-direction / advantage correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 94101 | 1 | 1.002 | 9.089 | 11.78 | 4.7% | 0.006 |
| 94101 | 11 | 0.942 | 9.202 | 12.17 | 6.0% | -0.036 |
| 94101 | 21 | 0.970 | 8.627 | 11.21 | 6.9% | -0.050 |
| 94102 | 1 | 0.970 | 9.818 | 12.82 | 5.1% | -0.032 |
| 94102 | 11 | 0.982 | 7.301 | 9.67 | 8.7% | 0.012 |
| 94102 | 21 | 0.941 | 7.589 | 9.95 | 8.1% | -0.016 |
| 94103 | 1 | 0.974 | 9.015 | 11.88 | 4.1% | 0.057 |
| 94103 | 11 | 0.951 | 8.002 | 10.46 | 7.2% | 0.013 |
| 94103 | 21 | 0.968 | 8.214 | 10.78 | 5.7% | -0.020 |

No sampled coordinate reached the declared saturation threshold. Teacher-helper
agreement passed. The fixed standard deviation generates local target changes
that seldom span the distance to a recommendation. This does not establish
that the entire distance must be traversed in one action, or that reaching the
recommendation improves reward on each audited training state.

The first-minibatch, decision-summed output-score projections toward the teacher
were respectively `1.080, -0.829, -0.133`; `0.213, -0.504, 0.095`; and
`0.350, -0.428, 0.232` for the three RNGs in update order. Their signs are mixed.
These are output-space ascent diagnostics at ratio one, with the actual living
unit/minibatch normalization. They omit shared-parameter coupling and optimizer
effects. Later minibatches in the evidence are frozen-weight counterfactuals,
not reconstructions of sequential optimizer gradients. No significance or causal
claim follows from these descriptive correlations/projections.

## Actual policy movement on identical states

Values first average selected units within each snapshot, then the 54 eligible
snapshots. Positive gap reduction means closer to the teacher recommendation.

| Final RNG | Mean target shift from initializer | Mean recommendation gap reduction | Mean fraction of units closer |
| --- | ---: | ---: | ---: |
| 94101 | 0.302 | +0.083 | 65.6% |
| 94102 | 0.705 | +0.236 | 66.8% |
| 94103 | 1.134 | -0.187 | 38.3% |

Two runs moved somewhat closer; the third moved farther. Mean final gaps remain
7.58–8.00 world units. Parameter change in S3 therefore did not produce a
consistent large correction on the common states. All S3 success gates remain
failed; this audit adds no success evaluation.

## Interpretation and next experiment

The evidence supports a scale mismatch between local exploration and the
teacher's target corrections, accompanied by weak/inconsistent local advantage
alignment. It does not distinguish insufficient exploration from poor reward
credit, critic error or recommendations that are unhelpful on individual states.
S2 established an aggregate benefit on different development states.

The next proposed step is a bounded, training-only paired physical probe before
another PPO run: compare equal-budget perturbations at the archived scale and
one predeclared larger scale, with unchanged action choice, corrected shots and
frozen continuation. Pair opportunities and random directions, measure actual
return/damage/progress and rejection rates, and include a teacher-direction
reference explicitly labeled diagnostic. Choose the scale, sample allocation,
continuation horizon and decision rule in a separate declaration before running.
This would test whether broader exploration exposes useful reward variation
without simultaneously changing the critic, reward or decoder. It is not yet
declared or executed; no broad sweep is authorized by these results.

## Verification

Targeted tests cover archived reconstruction, tamper rejection, unchanged weights,
finite boundaries, geometry scaling, unused-action masking, output-score/autograd
agreement and roster-normalized gradient sums. Full implementation and results
gates: 367 TypeScript tests, build, 51 client tests and 272 training tests. The
build retains its existing chunk-size warning. Environment contracts are unchanged.
R1n autonomous qualification and subsequent mission gates remain open.
