# R1m-S6: duration/scope correction improves progress; primary gate fails

Completed 2026-09-05. The [declaration](m7b_r1m_s6_declaration.md) and tested
implementation were committed as `2a52828` before collection. All planned runs
completed, with no optimization, provider calls, decoder change or promotion.
The primary transfer gate failed; preserve current training defaults.

## Design and reproducibility

Reuse exactly S5's 24 training first-hit states and fixed single-fighter IDs:
100000 and 100002–100024. Keep is the frozen source with corrected shots.
Single/squad arms replace only living source-selected MOVE destinations for one
or the first 30 decisions. Destinations are recomputed from each branch's current
state. The source classifier remains frozen; a dead single fighter is never
replaced. Existing persistent commands and the original option horizon remain
unchanged. All arms are explicitly teacher-assisted and autonomous-ineligible.

All 120 independently repeated branch pairs matched exactly. All 24 Keep and
24 single-1 traces reproduced S5 Keep/teacher action digests, complete physical
hashes and final outcomes. The source checkpoint, S3 implementation digests,
S5 implementation digest, snapshot selection and archives remained unchanged.

The run used 42,972 prefix-plus-continuation simulator decisions, below the
48,000 limit. There were zero rejected actions among 45,105 retained continuation
action results (90,210 including repeats); prefix actions are excluded from those
counts. [The archive](../runs/m7b_engage_r1m_s6_v0/manifest.json) contains 124
manifest-listed artifacts: 120 full compressed branches, selection/prefixes,
configuration, repeat digests and report. Manifest self-digest, inventory, every
branch/repeat digest, report recomputation and exact budget accounting verified.
Repeated executions are determinism checks, not additional environments.

## Primary comparison and all arms

Success differences are percentage points against Keep. Return is full discounted
executor reward from the shared branch state to option termination. Intervals
use 10,000 seed-paired bootstrap resamples with RNG 990001; they are unadjusted
for multiple comparisons. These are 24 previously exposed training environments.

| Arm | Success | Difference [95% CI] | Return difference [95% CI] |
| --- | ---: | --- | --- |
| Keep | 9/24 | 0 | 0 |
| Single, 1 decision | 8/24 | -4.2 [-16.7, +8.3] | -0.0651 [-0.2817, +0.1492] |
| Squad, 1 decision | 9/24 | 0.0 [-16.7, +16.7] | +0.0021 [-0.2789, +0.2816] |
| Single, 30 decisions | 9/24 | 0.0 [-25.0, +25.0] | +0.0166 [-0.3971, +0.4224] |
| Squad, 30 decisions | 12/24 | +12.5 [-4.2, +29.2] | +0.2399 [-0.0449, +0.5283] |

Squad-30 converts four Keep failures into success and loses one Keep success.
It passes the predeclared >=10-point improvement and rejection checks, but its
success and return interval lower bounds are negative. The primary gate therefore
fails. Do not select another contrast or alter thresholds after observing this.

## Duration and scope

Secondary contrasts are descriptive. Positive values favor longer duration or
larger scope; the interaction subtracts the single-fighter duration effect from
the squad duration effect.

| Contrast | Success difference, points [95% CI] | Return difference [95% CI] |
| --- | --- | --- |
| Single: 30 minus 1 | +4.2 [-16.7, +25.0] | +0.0817 [-0.2669, +0.4313] |
| Squad: 30 minus 1 | +12.5 [0.0, +25.0] | +0.2378 [+0.0245, +0.4603] |
| One decision: squad minus single | +4.2 [-12.5, +20.8] | +0.0672 [-0.2168, +0.3499] |
| 30 decisions: squad minus single | +12.5 [-8.3, +33.3] | +0.2233 [-0.1306, +0.5800] |
| Interaction | +8.3 [-16.7, +37.5] | +0.1561 [-0.2616, +0.6408] |

The squad duration contrast has a positive return interval, but this secondary
comparison does not replace the failed primary test against Keep. Neither the
success nor return interaction is reliably positive. A synergy or necessity claim
would exceed this evidence.

Actual correction dose and geometry differ substantially:

| Arm | Mean overridden MOVE targets | Mean range error from 6.5 | Mean occupancy within 9 |
| --- | ---: | ---: | ---: |
| Keep | 0 | 8.153 | 22.8% |
| Single-1 | 1 | 8.171 | 21.6% |
| Squad-1 | 4.21 | 7.920 | 23.5% |
| Single-30 | 21.13 | 7.946 | 24.4% |
| Squad-30 | 100.25 | 5.963 | 33.4% |

Range statistics aggregate living-unit opportunities within each continuation,
then average its seed-level statistic. Different trajectory lengths change
exposure. The 30-decision arms have mean active exposure 29.08 decisions; one
episode in each arm terminates before decision 30. These are not common-duration
samples. More corrected fighters also means more corrected actions, so scope
does not isolate a distinct coordination mechanism.

## Physical progress and remaining failures

For squad-30 versus Keep, paired local outcomes at min(30, termination) improve:

- Damage dealt: +85.0 health [51.7, 120.0].
- Damage received: +10.0 [-8.3, 27.5].
- Mission progress: +0.170 [0.103, 0.240].

At the original final horizon, damage dealt remains higher by +90.8 [46.7,
139.2], and progress by +0.182 [0.093, 0.278]. The paired local comparisons
above are additional descriptive calculations from the saved traces; the
predeclared primary endpoint remains final success and return.

An exploratory read-only inspection of the 12 failed squad-30 traces found all
12 ended by timeout with the entire assigned Blue force alive. Final progress
ranged from 0.12 to 0.76; five failures reached 0.72 or 0.76. None of the squad-30
episodes had already succeeded at its local intervention endpoint. This identifies
completion within the fixed budget as the observed failure mode, rather than
assigned-force elimination. It does not isolate whether the remaining problem
is range, targeting, action timing or the handoff to the frozen source.

## Design feedback and next boundary

The teacher's ability to improve damage/progress transfers to these training
states. The evidence for improved binary completion remains insufficient under
the declared gate. This is more specific than concluding that teacher movement
has no useful consequences.

S5's isolated target perturbations and S6's sustained interventions have different
effects. Sustained squad correction is a promising mechanism to inspect, while
the factorial does not prove that all-fighter coordination is essential. The
large difference in correction dose must remain explicit.

The current executor reward has a terminal mission term, a smaller combat term
and potential shaping. With zero terminal potential, shaping telescopes to a
start-state constant in the full discounted return. Improved partial progress
therefore need not produce a similarly reliable return improvement. This is an
interpretation of the existing objective, not authorization to change it or to
substitute progress for the success qualification gate.

Stop at the declared budget. The next proposed step is a read-only failure/handoff
audit of these archives: compare target-health progress and range before/after
the 30-decision handoff, identify the remaining enemy and action-readiness patterns,
and stratify by the original remaining option budget. Use existing trajectories;
do not add seeds, teacher duration or PPO updates to obtain a passing interval.
Only after that review should a bounded learning or continuation experiment be
declared. No new such experiment is declared here.

Autonomous R1n, other fixed-plan missions and composed-plan qualification remain
open. No learned checkpoint was created by S6.

## Verification

Targeted tests cover exact duration boundaries, stable fighter identity through
casualties, living/MOVE masks, missing-label handling, unchanged action choice
and power, read-only helper behavior, S5 parity, duplicate trajectories, source
immutability, identity tamper rejection, factorial arithmetic, primary-gate logic
and refusal to overwrite outputs. Existing manifest/seed tests also apply.
Before implementation and results commits: 367 TypeScript tests, 51 client tests,
283 training tests and build passed. Existing bundle-size and legacy-Gym warnings
remain. No environment contract changed; browser verification was unnecessary.
