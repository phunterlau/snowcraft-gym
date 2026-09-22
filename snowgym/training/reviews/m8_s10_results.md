# M8-S10 results: aim-weighted imitation gives one large, unpredicted win and two unchanged failures

Collected 2026-09-22 from `b9bf048` (declaration and implementation committed before training). Declaration:
[m8_s10_declaration.md](m8_s10_declaration.md). Archive: `runs/m8_s10_throw_aim_repair_v0/` (sealed and verified,
`sha256:a18fee16…`, 19 MB). This is an ordinary autonomous training run, **not** teacher-assisted; these are learner
results. `autonomousQualificationEligible` stays false (a development experiment, not an R1 qualification attempt).
576,883 simulator decisions against the 743,200 bound.

## Predictions: 1 of 3 pass/fail predictions holds, and the result runs opposite to what was predicted

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| P1 | 97102 **and** 97103 (S9's aim-related seeds) each gain ≥ 0.10 on scripted-normal | 97102 +0.00, 97103 +0.00 | **Not held** |
| P2 | No seed regresses on scripted-easy by more than 0.15 | 97101 +0.20, 97102 **−0.19**, 97103 +0.01 | **Not held**: 97102 misses its own margin (0.68 < 0.87 − 0.15 = 0.72) by 0.04 |
| P3 | Critic warm-start gate passes for all three seeds | all pass (R² 0.53–0.61) | **Held** |
| P4 (descriptive) | 97101's gain ≤ the smaller of 97102's/97103's gains | 97101 +0.51 vs 0.00 and 0.00 | **Ran opposite**: 97101 gained the most by a wide margin |

The seed my own diagnostic chain (S8) said was *not* primarily an aim problem improved dramatically. The two seeds
S9 specifically implicated in the throw target were unchanged.

## Results (100 worlds per cell, paired against S5's own archived success on the *same* 100 worlds)

| Seed | Normal success (S5 → S10) | Gain | Easy success (S5 → S10) | Team wipe (dev. eval) |
| --- | --- | --- | --- | --- |
| 97101 | 0.00 → **0.51** | **+0.51** | 0.74 → 0.94 | 0.10 (was ≈1.00) |
| 97102 | 0.00 → 0.00 | 0.00 | 0.87 → 0.68 | 1.00 (unchanged) |
| 97103 | 0.00 → 0.00 | 0.00 | 0.89 → 0.90 | 0.99 (unchanged) |

**97101 alone now clears S5's own "viable initializer" bar** (mean success ≥ 0.25 against scripted-normal), on one
seed rather than a seed mean. The development-evaluation worlds (band 4220000, seeds disjoint from the paired-eval
band) give 0.48, close to the paired band's 0.51 for the same checkpoint under the same deterministic policy — this
shows the 0.51 is not a property of those particular 100 paired worlds, not that it would replicate on a retrain.
First hit lands at decision 53, matching the teacher.

**97101's failure mode changed qualitatively, not just its rate.** Team wipe fell from about 1.00 (S5) to 0.10; the
residual failures are now mostly timeouts (0.44), not deaths. This is the first 3v3 learner result in this track
that survives the fight and fails by running out of clock rather than being killed. S8 found 97101's deficit was
specifically throw *timing*, not aim; a policy that now throws more often (recall 0.66 → 0.88) and survives but
sometimes runs out of decisions is consistent with that diagnosis and is a sharper lead than the aim-error question
below. This is an observation, not an explanation — no mechanism is tested here.

## What the label-error read shows, and the dissociation it exposes

`label_error` (measured on the development-normal band, a proximal read, not the success metric itself) shows the
intervention did what it targeted — **aim error fell for every seed** — but two side effects appear that were not
predicted or measured for:

| Seed | Throw aim error (S5 → S10) | Move heading error (S5 → S10) | Throw recall (S5 → S10) |
| --- | --- | --- | --- |
| 97101 | 13.2° → **10.0°** | 14.2° → **32.1°** | 0.66 → **0.88** |
| 97102 | 24.3° → **14.0°** | 11.8° → 14.2° | 0.71 → 0.69 |
| 97103 | 15.5° → **10.4°** | 12.5° → 17.1° | 0.76 → **0.46** |

Aim error dropped 3–10° for every seed, the proximal target of `aimWeight = 5`, working as designed. But **move
heading error more than doubled for 97101** (14° → 32°) and rose for the others, and **throw recall collapsed for
97103** (0.76 → 0.46) while type accuracy fell for all three (worst for 97101, 0.94 → 0.88). Neither was a declared
measure or prediction; both are read directly from the archived `label-error.json` files, not estimated.

**The dissociation is the notable finding.** 97102 and 97103 got materially *better* aim (by label error) and stayed
at exactly 0.00 success; 97101 got *worse* movement and type accuracy and gained +0.51 success. Aim-error
improvement neither predicts nor is necessary for the success gain seen here. This matches a pattern this track has
now hit repeatedly (R1n-h's original label-error framing, R1n-i's finding that aggregate label error hides
behavioral differences): a teacher-relative label metric is not a reliable proxy for deployed outcome, in either
direction.

## What this does not establish

**The 97101 result cannot be attributed to the loss change.** Training data is a fresh draw (declaration §1), so
this run cannot separate "the aim weighting caused the gain" from "this data draw was easier for seed 97101,
independent of the loss." The declaration's pre-declared contingency — a second run at `aimWeight = 1` on these
same fresh bands — is exactly the check that would settle this, and it was not run here. Until it is, 97101's +0.51
is reported as an association with the intervention, not caused by it.

**97102's easy regression is unexplained.** Its move-heading and type-accuracy label errors both worsened only
mildly, not enough by inspection to obviously explain a 19-point easy-success drop; no further diagnostic was run.

**No mechanism is shown** for why aim-weighting helped one seed and not the other two, why movement and type
accuracy degraded as a side effect, or why throw recall specifically collapsed for 97103. The three seeds still
share one round-zero teacher dataset (this run's own, not S5's), so this is not three independent training cohorts.

## Consequence

This is not a repair that can be recommended as-is: it helps one seed by a large margin, is neutral for two, breaks
scripted-easy for one, and its aim-error win does not explain its success win. The two candidates named before this
result, in order:

1. **The pre-declared contingency:** `aimWeight = 1` on these same fresh bands, isolating the loss change from the
   data draw. If 97101 still gains without the reweighting, the loss change did not cause it.
2. If the reweighting is confirmed causal for 97101 specifically, a smaller `aimWeight` (the side effects on
   movement and type accuracy suggest 5 may be too aggressive) or a term that only touches the throw heads without
   drawing gradient away from type/move, addressed in a following declaration.

Neither is authorized here.
