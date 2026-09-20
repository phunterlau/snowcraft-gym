# R1n-i: the model's own deployed throws and movement, read from R1n-h's frozen checkpoints

Collected 2026-09-20 from `fbc1b3e`.
- The [declaration](m7b_r1n_i_declaration.md) was committed at `e98ee7c`,
  amendments A1–A4 at `8b06580`/`fbc1b3e`.
- The implementation is at `8b06580`/`fbc1b3e`.

Run facts:
- **Decisions:** 65,378 (summed from each checkpoint's regenerated
  `episodes.jsonl`'s `totalActions`, which — confirmed against
  `label_error`'s own `units` count on the same data — is the same
  per-decision row count `deployed_view` processes), against a 200,000
  cap (bound 120,000). No `simulatorDecisions` total was written to
  `report.json` itself (a completeness gap in `aggregate()`, not a
  correctness one — see dev notes); this figure is independently derived
  from the archived episode rows, not from the account/budget counter.
- **All 6 reproduction gates passed exactly:** 100/100 worlds matched the
  archive on every checkpoint, 0 mismatches, 0 missing.
- **All 6 label-error rechecks matched the archive exactly:** `maxDelta`
  is `0.0` on every scalar metric, for every checkpoint — the retained
  `part` data and `collect_with_attribution`'s pipeline reproduce
  `label_error`'s already-archived numbers bit-for-bit, not just
  approximately, at full production scale (this had only been checked at
  tiny scale before collection).
- **Manifest:** the top-level manifest (`sha256:7fbde087…`) verified
  after aggregation.
- **Archive:** `runs/m7b_engage_r1n_i_v0/`.

R1n-i trains nothing, selects no checkpoint, and makes no R1 qualification
claim. `autonomousQualificationEligible` stays false throughout.

An [erratum](#erratum-2026-09-20-the-97103-movement-explanation-is-overstated)
at the end demotes headline point 3 (and the matching §5 and PLAN.md
statements) from a demonstrated cause to a measured association, and
qualifies the C aim claim. No result changes.

## Headline

1. **Condition C's failure is a clean, deployed-action aim failure — now
   confirmed on the model's own executed throws, not just teacher-labelled
   opportunities.** All three C seeds reach range in every episode
   (100/100) but rarely select a close-range throw when they do (rate
   0.006–0.018, roughly an order of magnitude below M's 0.146–0.174), and
   on the throws they do select, ground-truth aim error is 120–146° —
   worse than two random directions, matching R1n-g's/the erratum's
   teacher-conditioned finding almost exactly, but this time measured on
   throws C's own policy chose, not on states the teacher happened to
   label THROW.
2. **97101's contact failure is not explained by the three named
   mechanisms (range-reaching, throw-attempt rate, aim accuracy) — a
   real, reported negative result — but the one metric that does differ,
   time spent in range, is more likely a consequence of not landing hits
   than an independent cause.** 97101 reaches range in every episode,
   selects a close-range throw in every episode, and its deployed aim
   error (11.8°) is close to 97103's (10.4°), the seed with 100% contact.
   Its per-episode in-range decision count (median 34, spread 28–62) is
   higher and more variable than 97102's tight 22/23/23 — but a policy
   that lands a hit and finishes quickly accumulates *few* in-range
   decisions before the episode ends, while one that keeps missing
   lingers in range and accumulates more. That pattern is consistent with
   the in-range count being *downstream* of the contact failure rather
   than upstream of it. None of the three named mechanisms explains
   97101; the one measurable difference does not resolve to an
   independent cause either (§5).
3. **97103's finishing failure has a clean, large, and directly
   predicted explanation: it stops moving under threat after first
   contact.** Restricted to the post-first-hit window, 97103 keeps
   moving while a red projectile is nearby only 37.2% of the time, versus
   90.9% for 97102 (the seed that finishes) — more than a 2× gap,
   matching R1n-e's own "keep moving during snowball flight" survival
   mechanism directly, as declaration §4's third prediction named before
   collection.
4. **The two threat-proxy versions (`anywhere` vs. `nearby`, amendment
   A4) are identical on every M seed measured here** — see §3's table.
   For all three seeds' post-contact decisions, every threatening
   decision that had a live red projectile anywhere in the arena also had
   one within `engageRange`; the narrower proxy never excluded a decision
   the wider one counted, on this data. The two measures were worth
   building separately (they need not have coincided), but on these six
   checkpoints the distinction did not end up mattering empirically.
5. **The retained-`part` pipeline is now validated at full production
   scale, not just the tiny test scale checked before collection.**
   Every one of the 6 checkpoints' `label_error` recheck matched the
   archive with `maxDelta = 0.0` exactly — the strongest form of the
   §10-mandated validation this declaration required before trusting any
   new, ground-truth-conditioned number.

## 1. Reproduction gate and label-error recheck (all 6 checkpoints)

| Checkpoint | Reproduction gate | Worlds matched | Label-error `maxDelta` |
| --- | --- | ---: | ---: |
| M-97101 | passed | 100/100 | 0.0 |
| M-97102 | passed | 100/100 | 0.0 |
| M-97103 | passed | 100/100 | 0.0 |
| C-97101 | passed | 100/100 | 0.0 |
| C-97102 | passed | 100/100 | 0.0 |
| C-97103 | passed | 100/100 | 0.0 |

Every checkpoint's regenerated episodes exactly reproduce R1n-h's archived
outcomes, and every checkpoint's re-derived `label_error` exactly
reproduces R1n-h's archived `label-error.json`. Everything below is built
on this same collection call, per checkpoint.

## 2. Contact-failure diagnostic (declaration §3)

| Checkpoint | Episodes ever in range | Episodes w/ close-range throw | Pooled close-range throw rate | Deployed throw count | Deployed aim error | Per-episode in-range decisions (min/median/max) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M-97101 | 100/100 | 100/100 | 0.146 | 632 | 11.81° | 28 / 34 / 62 |
| M-97102 | 100/100 | 100/100 | 0.174 | 400 | 0.77° | 22 / 23 / 23 |
| M-97103 | 100/100 | 100/100 | 0.153 | 705 | 10.35° | 23 / 47 / 64 |
| C-97101 | 100/100 | 29/100 | 0.011 | 92 | 123.16° | 10 / 28 / 45 |
| C-97102 | 100/100 | 34/100 | 0.018 | 187 | 120.25° | 11 / 18 / 34 |
| C-97103 | 100/100 | 20/100 | 0.006 | 124 | 145.66° | 22 / 33 / 55 |

Every checkpoint, M and C alike, reaches range in every episode —
"never gets close" is false for all six, including the three C seeds that
never once land a hit. C's failure is squarely a *decision-and-aim*
failure: it rarely selects a close-range throw at all (rate roughly an
order of magnitude below M's), and when it does, its aim is worse than
random. M's three seeds all reach range, all throw close-range in every
episode, and all aim reasonably (0.8°–11.8°) — 97101 included.

97101's per-episode in-range decision count (median 34, spread 28–62) is
the one number in this table that clearly differs from 97102's tight
22/23/23 — but read together with the finishing-failure table below,
this is more consistent with an effect of not landing hits than a cause
of it: an episode that lands a hit early ends soon after, so a
finishing checkpoint accumulates few in-range decisions almost by
construction, while one still trying accumulates more the longer it
goes unresolved. Range-reaching, throw-attempt rate, and aim accuracy —
the three mechanisms this diagnostic was built to test — do not separate
97101 from its two working siblings; the one number that does differ
does not obviously resolve to an independent explanation either.

## 3. Finishing-failure diagnostic (declaration §3)

| Checkpoint | Episodes with contact | Post-contact decisions | Under threat (anywhere) | Under threat (nearby) | Keeps moving, anywhere | Keeps moving, nearby |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M-97101 | 12 | 288 | 92 | 92 | 0.598 | 0.598 |
| M-97102 | 100 | 1,998 | 22 | 22 | 0.909 | 0.909 |
| M-97103 | 100 | 4,294 | 1,257 | 1,257 | 0.372 | 0.372 |
| C-97101/97102/97103 | 0 | 0 | — | — | — | — |

The `anywhere` and `nearby` columns are identical for every M seed (§4's
headline point 4) — the distance qualifier amendment A4 added never
excluded a decision the world-level proxy counted, on this data. C's
three seeds never make contact at all, so this diagnostic has no
post-contact window to measure for them — consistent with the
contact-failure table above and with R1n-f/R1n-g/the R1n-h erratum's
independent confirmations of C's total floor.

97103's keeps-moving-under-threat rate (0.372) is less than half of
97102's (0.909), restricted to the exact post-first-hit-to-completion
window in each episode. 97101's rate (0.598) sits between the two, but on
far less data (12 contact episodes, 92 threatened decisions, versus
97103's much larger post-contact sample) — read as suggestive, not
conclusive, for 97101 specifically.

## 4. Predictions vs. results (declaration §4)

| Prediction | Result |
| --- | --- |
| If 97101 is a pure approach/movement failure: `minDistance` stays above `ENGAGE_RANGE` in most non-contact episodes, and close-range throw rate is comparable to 97102's when it does reach range | **Partially confirmed, partially not.** It reaches range in *every* episode (stronger than "most"), and its close-range throw rate (0.146) *is* comparable to 97102's (0.174) — but it still makes contact in only 12% of episodes. The prediction's premise (comparable range-reaching and throw rate) holds; its implied conclusion (so contact should be comparable too) does not. |
| If 97101 is instead a decision/aim failure despite reaching range: it rarely selects THROW there, or its aim error is large even when it does | **Not confirmed.** It selects close-range throws in 100/100 episodes, and its deployed aim error (11.8°) is close to 97103's (10.4°), the seed that reaches 100% contact. |
| If 97103's deaths follow R1n-e's movement-during-incoming-threat mechanism: its post-first-hit episodes show measurably less continued movement under incoming projectiles than 97102's | **Confirmed, and by a large margin.** 0.372 vs. 0.909 — 97103 keeps moving under threat less than half as often as 97102, in the exact post-contact window. |
| If none of the above hold cleanly, report the full descriptive breakdown without forcing a single-cause conclusion | **This is what happened for 97101.** Neither declared contact-failure prediction fits cleanly; §5/§6 report that directly rather than picking the closer-sounding one. |

## 5. What this does and does not decide

**Decides:** condition C's total floor is a deployed aim/decision failure
on the model's own executed throws, not an artifact of `label_error`'s
teacher-label conditioning — the same conclusion the erratum reached from
teacher-conditioned data now holds on the model-conditioned view too.
97103 selects MOVE less often under a nearby projectile after contact than
97102 does (association only; see the erratum). Neither the critic (ruled out in the R1n-h erratum)
nor any of range-reaching, throw-attempt rate, or aim accuracy (ruled out
here) explains 97101's contact failure.

**Does not decide:** what *does* explain 97101. This diagnostic's measures
were specifically chosen to test the three named hypotheses in R1n-h's own
results doc and the erratum's reframing, and all three come back
negative for 97101 while positive for 97103. The one number that does
differ (its higher, more variable in-range decision count) is more
consistent with being downstream of the contact failure — an unresolved
engagement lingers — than an independent cause of it, so it is reported
as a difference, not treated as an explanation. Candidate explanations
neither confirmed nor ruled out here: power/range calibration on the
throws it does attempt (not measured — this diagnostic has no
ground-truth "correct power" without assuming the teacher's own label,
which reintroduces the teacher-conditioning problem this diagnostic exists
to avoid); throw *timing* relative to the enemy's own movement (an
instantaneous per-decision aim/range read cannot distinguish "aimed well
at a target that has since moved" from "aimed well and landed"); or a
genuinely different failure mode not covered by either named hypothesis.
This is the natural next diagnostic if the mixture-imitation track
continues rather than moving to M8 — it would need per-decision projectile
trajectory/impact data this run does not have, not just presence.

**Does not decide:** anything about R1's qualification gates, and trains
no PPO. Does not decide whether 97102 should be promoted or used to seed
a future PPO stage — the 2026-09-18 review's warning (declaration §5)
still applies: no selection rule is authorized by this run.

## Verification

- Python training tests (16 for `checkpoint_failure_diagnostic.py`,
  including two live tests — a bit-for-bit `collect_with_attribution`
  equivalence guard against `fi.collect`, and a tiny end-to-end
  declare/run_checkpoint/aggregate/tamper-detection run), python client
  tests, `npm run build`, `npm test`: all clean before collection
  (456/456 training, 51/51 client, 366/367 npm — the one documented R1n-b
  exception).
- All 6 reproduction gates passed exactly at full production scale (§1).
- All 6 label-error rechecks matched the archive exactly (`maxDelta =
  0.0`) at full production scale (§1) — this had only been checked at
  tiny scale before collection; it is now confirmed on the real data the
  headline numbers are built from.
- The multi-enemy guard (`RuntimeError` if more than one enemy is ever
  live in a decision, amendment A4) did not fire on any of the 65,378
  decisions processed.
- Top-level manifest verified with `death_rate_ppo.verify_sealed` after
  aggregation (`sha256:7fbde087…`).

## Erratum (2026-09-20): the 97103 movement explanation is overstated

An external review (`refs/snowgym_handoff_review_and_next_steps_2026-09-20.md`,
local notes, not committed) questioned headline point 3. I re-read
`finishing_failure_summary` and re-pulled the denominators from
`runs/m7b_engage_r1n_i_v0/report.json`; the review's points hold. No sealed
artifact changes and no number changes. This file is not listed in
`runs/*/manifest.json` (checked directly), so an additive edit is safe,
following the R1n-h erratum.

### 1. Unequal, thin exposure

| Checkpoint | Post-contact decisions | Threatened decisions (denominator) | Keeps-moving rate |
| --- | ---: | ---: | ---: |
| M-97101 | 288 | 92 | 0.598 |
| M-97102 | 1,998 | 22 | 0.909 |
| M-97103 | 4,294 | 1,257 | 0.372 |

97102's 0.909 is 20 of 22 decisions. It finishes fast, so it is rarely
exposed at all; 97103 lingers and is exposed 1,257 times. The two rates come
from different policy-induced trajectories, and successful early completion
itself lowers exposure. Decisions within an episode are serially dependent,
so 1,257 is not 1,257 independent trials. Declaration §4's
comparable-exposure condition was not demonstrated.

### 2. "Moving" is MOVE selection, not motion

`moving` is `model_type == ACTION_MOVE` (`checkpoint_failure_diagnostic.py`,
`finishing_failure_summary`). It is neither speed nor displacement: NOOP
preserves the previous movement order, so a unit can keep moving without
selecting MOVE, and MOVE can be unavailable while a unit is stunned or in a
throw phase, so a lower selection rate can follow being hit rather than cause
it. "Nearby red projectile" is a proximity proxy, not a collision-course test.

### 3. Corrected statements

- **Supported:** 97103 selects MOVE less often than 97102 in post-contact
  decisions with a nearby red projectile (37.2% of 1,257 versus 90.9% of 22).
- **Not established:** that reduced physical evasion causes 97103's deaths, or
  that this is the R1n-e "keep moving during snowball flight" mechanism.
  R1n-e measured actual displacement against random Red; this diagnostic did
  not. Headline point 3's "clean, large, directly-predicted cause", §5's
  "specific, large, directly-predicted behavioral cause", and PLAN.md's
  matching bullet are superseded by this paragraph.
- **C's aim finding** (headline point 1) stands as a description of
  learner-selected throws versus the enemy's instantaneous position. It does
  not isolate how much aim, throw selection, or movement each contribute to
  C's floor, and an angle has no distance-independent "healthy" threshold
  (lateral miss grows roughly as d sin(alpha)).
- **97101** is unchanged: still an unexplained negative result.

Consequence: a PPO run that rewards or targets MOVE frequency would rest on
an untested mechanism. Measure displacement and interception first, or do not
build on this finding.
