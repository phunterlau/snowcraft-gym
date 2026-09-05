# R1k: conditional opportunity audit

## Decision

The frozen R1i absolute reference passed the R1k audit. Proceed to the approved
R1l state-support × label-support factorial. No production policy was trained
or promoted; the reference still completed 0/40 training-seed Engage episodes.

## Evidence

All 5,367 reconstructed teacher states and their action labels exactly matched
the immutable reservoir. Collection then recorded 8,000 learner states and
38,129 living-fighter opportunities on seeds 100000–100039.

| Invoked head | Opportunities | Excluded by teacher-selected-action mask |
|---|---:|---:|
| Move | 29,508 | 1,255 (4.25%) |
| Throw | 3,122 | 1,787 (57.24%) |

This measures action-mask disagreement on visited states. It does not prove
that similar labeled states were absent from the reservoir.

For each of movement, aim, and power, 64 hard opportunities were tested using
paired 30-decision branches. Only one fighter's one selected head changed for
the first decision; later decisions used the frozen reference. Exact reset and
action-prefix replay verified physical, grounded-plan, and option-tracker
identity before every branch. Paired intervals bootstrap episode means and
apply to these selected opportunities.

| Substitution | Net-damage change, health units (95% interval) | Range-error improvement, world units (95% interval) |
|---|---:|---:|
| Movement | +12.61 [0.29, 25.22] | +1.84 [1.04, 2.76] |
| Aim | +34.42 [25.22, 43.91] | +0.08 [-0.20, 0.34] |
| Power | -0.14 [-6.67, 7.15] | -0.11 [-0.25, 0.03] |

Movement's net benefit primarily reflects lower incoming damage. Aim also
increased damage dealt by 26.67 health units [18.48, 35.43]. Different channel
samples need not contain the same opportunities; this is not a paired ranking
of their effect sizes. Movement/aim each cover 23 episodes, power covers 24.

The disposable 200-step hard-state fit reduced combined loss from 2.25677 to
0.04774 (97.88%). Validation loss on eight episodes excluded from this fit
fell from 1.90913 to 0.40414. Encoder gradients were reachable and numerical
physical-Jacobian checks passed. The fitted weights were discarded and the
entire reference checkpoint remained unchanged. These validation episodes are
not asserted unseen by ancestor training.

## Interpretation and limitations

The fixed geometry modules can fit difficult learner-invoked head labels, and
the reference recommendations improve local physical consequences. Conditional
supervision deserves a controlled test. Neither the masking frequency nor the
disposable fitting result establishes autonomous closed-loop improvement.

The experiment uses one open 5v5 scenario, selected high-error opportunities,
one-step interventions, and 3-second branches. Recommendation availability,
legality, and readiness are recorded separately. Power has no independent
positive result here. No fresh holdouts, development/qualification episodes,
PPO updates, provider calls, or browser sessions were used.

## Reproduction

See the [headless command and frozen gate](../src/snowgym_training/executor/OPPORTUNITY_AUDIT.md),
[report](../runs/m7b_engage_r1k_opportunities_v0/report.json),
[hard fit](../runs/m7b_engage_r1k_opportunities_v0/hard-fit.json), and
[manifest](../runs/m7b_engage_r1k_opportunities_v0/manifest.json).

The archive includes compressed states, opportunities, episode action prefixes,
and 192 paired branch records. Source, dataset, checkpoint, and output digests
are retained. Verification: 325 TypeScript tests, production build, 50 Python
client tests, and 238 Python training tests. Existing build-size and legacy Gym
deprecation warnings remain. Training tests must run from `snowgym/training`;
an initial root-directory invocation hit an existing relative-path fixture,
and the prescribed working-directory invocation passed.
