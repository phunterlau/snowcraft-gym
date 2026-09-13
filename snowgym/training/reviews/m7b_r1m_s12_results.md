# R1m-S12: the egocentric frame accounts for the representation gap, not attention

Completed 2026-09-13. The [declaration](m7b_r1m_s12_declaration.md) (committed
`dd01b2e`) and tested implementation were committed before collection. All
three arms (A, A-rel, B) completed all four DAgger rounds. Used 194,958 of
250,000 budgeted simulator decisions. Zero rejected actions across every
arm, split, and round (0/338,590 total actions).

## Headline

**A-rel and B are statistically indistinguishable from each other and both
clearly beat A.** Per the declaration's three-way decision rule, the result
matches **"A-rel ~= B > A": the egocentric frame was the whole story.**
E3 should use `relative=True`-style egocentric geometry; the attention
module (`AttentionGeometryProbe`) is not carried forward, since it does not
earn its added parameters over the free `relative=True` control once that
control is included.

## Controls and final evaluation

All splits are from-reset Engage episodes (a stated deviation from S2/S3/S9's
first-hit-anchored floor/ceiling; see the declaration). Ceiling and floor
are evaluated on the historical-development split only (an implementation
gap against the declaration's text, which said "both development splits";
not corrected after the fact -- see Verification).

| Control | Historical-development (40) |
| --- | ---: |
| Ceiling (teacher-forced, shared) | 40/40 |
| Floor, all three arms (zero-init, identical by construction) | 13/40 |

| Arm | Training (64, in-sample) | Historical-dev (40) | Replication-dev (40) |
| --- | ---: | ---: | ---: |
| A | 52 (81.3%) | 34 (85.0%) | 31 (77.5%) |
| A-rel | 58 (90.6%) | 36 (90.0%) | 38 (95.0%) |
| B | 61 (95.3%) | 36 (90.0%) | **40 (100.0%)** |

Mean progress (continuous, not gated by the fixed 20%-health threshold)
follows the same ordering on both development splits: historical A 0.769 <
A-rel 0.778 < B 0.787; replication A 0.760 < A-rel 0.792 < B 0.803.

## Paired bootstrap comparisons

10,000 resamples, RNG 952001, paired by seed (`bootstrapSeed`/`bootstrapSamples`
from the declaration).

| Comparison | Split | Success Δ | 95% CI |
| --- | --- | ---: | --- |
| A-rel − A | training | +9.4% | [-1.6, +20.3] |
| B − A | training | **+14.1%** | **[+3.1, +25.0]** |
| B − A-rel | training | +4.7% | [-4.7, +14.1] |
| A-rel − A | historical-dev | +5.0% | [-7.5, +17.5] |
| B − A | historical-dev | +5.0% | [-7.5, +17.6] |
| B − A-rel | historical-dev | +0.0% | [-12.5, +12.5] |
| A-rel − A | replication-dev | **+17.5%** | **[+5.0, +32.5]** |
| B − A | replication-dev | **+22.5%** | **[+10.0, +35.0]** |
| B − A-rel | replication-dev | +5.0% | [0.0, +12.5] |
| A − ceiling | historical-dev | -15.0% | [-27.5, -5.0] |
| A-rel − ceiling | historical-dev | -10.0% | [-20.0, -2.5] |
| B − ceiling | historical-dev | -10.0% | [-20.0, -2.5] |
| arm − its own floor | historical-dev | +52.5 to +57.5% | all exclude 0 |

A and A-rel/B separate clearly on the freshest split (replication-dev, both
significant) and on training (B significant, A-rel borderline); on
historical-dev the direction is consistent (A lowest on every measure) but
no individual interval excludes zero at this sample size. B never
separates from A-rel: the training gap (+4.7%) and the historical-dev gap
(0.0%) both have intervals comfortably straddling zero, and the
replication-dev gap's lower bound (0.0%) lands exactly at the boundary --
the closest either representation gets to a significant edge over the
other, and not a convincing one. All three arms remain significantly below
the from-reset teacher-forced ceiling on historical-dev, with A trailing
furthest.

## Interpretation against the predeclared decision rule

- **A ~= A-rel ~= B, both near ceiling:** not met -- A is 5-22.5 points
  behind on every split.
- **A-rel ~= B > A: the frame was the whole story:** met. A-rel and B never
  separate from each other (largest gap +5% with a boundary-zero lower CI);
  both separate from A on training and replication-dev, and point the same
  direction on historical-dev.
- **B > A-rel > A: attention earns its added parameters:** not met -- the
  B-over-A-rel gap is small, inconsistent in significance, and vanishes
  entirely on historical-dev.
- **Both below 60% of ceiling despite held-out heading error < 10 degrees:**
  not met -- A-rel and B are at 90-100% of the 40/40 ceiling on both
  development splits.

**Decision:** carry `relative=True`-style egocentric geometry into E3's
network. Do not carry `AttentionGeometryProbe` forward as E3's
representation; the fighter-query attention pooling this experiment added
on top of the egocentric frame did not earn its parameters once the frame
itself was controlled for. The declared conditional optimizer-seed
replication (triggered only if B beat both A and A-rel on development) does
not trigger, since B does not beat A-rel there; no replication was run.

## Verification and known implementation gaps

- Ceiling and floor were evaluated on the historical-development split only,
  not both development splits as the declaration's text specifies. Given
  the historical-development ceiling is already 40/40 (matching the review's
  independently-reported R1h from-reset ceiling) and both A-rel and B reach
  95-100% success on replication-development directly, a replication-split
  ceiling number would not change the decision; it was not collected
  after the fact rather than spend further budget re-opening a closed run.
- The held-out heading-error diagnostic (`heading_loss`'s `headingErrorDegrees`)
  was computed only during training-time fitting, not on development-split
  states, so the falsifier requiring it is resolved by the success-rate
  margin alone (all comfortably above 60% of ceiling) rather than by a
  measured heading-error number. Also stated in the declaration's list of
  deviations.
- All 338,590 evaluated and collected actions across every arm, split, and
  round were accepted; zero rejections.
- The frozen R1f source's parameters were verified unchanged before and
  after the run (`semantic_state_digest` equality check inside `run()`);
  the run raises before writing any output otherwise.

The implementation gate passed 367 TypeScript, 51 Python client, and 350
Python training tests (including 10 targeted `geometry_representation_probe`
tests) plus build, before collection. `autonomousQualificationEligible:
false` throughout; this is supervised, teacher-labeled fitting, not
autonomous qualification. No provider calls, browser input, or protocol
changes.
