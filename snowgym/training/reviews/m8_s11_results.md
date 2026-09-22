# M8-S11 results: the loss weight caused 97101's gain; the data draw alone did not

Collected 2026-09-22 from `edfdcf4` (declaration and implementation committed before training). Declaration:
[m8_s11_declaration.md](m8_s11_declaration.md). Archive: `runs/m8_s11_throw_aim_isolation_v0/` (sealed and verified,
`sha256:81abfcfb…`, 19 MB, declaration pinned to `edfdcf4`). Ordinary autonomous training, not assisted.
`autonomousQualificationEligible` stays false. 582,698 simulator decisions against the 743,200 bound.

## Correction to the declaration's own framing, found while writing this up

§1 called (S10 − S11) + (S11 − S5) ≈ (S10 − S5) a "sanity check… not required to hold exactly," as if it were an
empirical result that might not agree. It is not: for any three numbers, (a−b)+(b−c) = (a−c) is an algebraic
identity, true by construction, and the archived numbers below sum exactly (0.39 + 0.12 = 0.51) because they must.
This is not evidence of anything and should not have been described as a check. What *is* informative, and is
reported below, is the two individual differences themselves — the loss effect and the data effect separately —
not their trivial sum.

## Classification: **loss-weight-driven**

Seed 97101's scripted-normal success at `aimWeight = 1` on S10's exact fresh data is **0.12** — at or below the
pre-declared 0.20 threshold, and closer to S5's 0.00 than to S10's 0.51. Corroborated on the independent
development-evaluation band (4220000, disjoint seeds): 0.14, consistent with 0.12.

| Effect | Value | Reading |
| --- | --- | --- |
| Data draw alone (S11 − S5) | **+0.12** | The fresh training data gives a small, real, nonzero gain on its own — not inert, but far short of viable |
| Loss weight alone (S10 − S11) | **+0.39** | The aim-weight increase adds roughly three times the data draw's effect, and is necessary to reach S10's 0.51 |

The loss change is not the whole story (the data draw contributes something) but it is the dominant and necessary
component: at `aimWeight = 1`, 97101 does **not** clear S5's own 0.25 viability threshold on this data (0.12 < 0.25);
only the combination does.

## 97102 and 97103 on scripted-normal: null under every condition tested so far

Both seeds score exactly **0.00** at S5 (original data, weight 1), S10 (fresh data, weight 5), and S11 (fresh data,
weight 1) — three different conditions, one consistent null. Neither the data draw nor the aim-weight change moves
either seed on scripted-normal. This is the cleanest result in the run: whatever limits these two seeds, it is not
addressed by either factor tested.

## Scripted-easy: an unplanned finding — the raised aim weight was partly protective against the data draw

This was not classified or predicted (§2 only classified 97101's normal success), but the pattern is large and
consistent enough to report as a finding, read directly from the archived numbers, not inferred:

| Seed | S5 (original data, w=1) | S10 (fresh data, w=5) | S11 (fresh data, w=1) | Data effect (S11−S5) | Weight effect on easy (S10−S11) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 97101 | 0.74 | 0.94 | 0.95 | +0.21 | −0.01 |
| 97102 | 0.87 | 0.68 | **0.27** | **−0.60** | **+0.41** |
| 97103 | 0.89 | 0.90 | **0.43** | **−0.46** | **+0.47** |

These are two different comparisons and should not be run together. **Data effect (S11 − S5, weight held at 1 in
both):** for 97102 and 97103, this fresh training-data draw is badly damaging to easy performance on its own — a
−0.46 to −0.60 drop, far larger than S10's already-reported −0.19 regression for 97102 relative to S5. **Weight
effect on easy (S10 − S11, data held fixed):** raising the aim weight recovers +0.41 to +0.47 of that damage for the
same two seeds — the raised weight is not what caused the easy regression reported in S10; on this data it partly
offset a larger regression that the data draw alone produces. Neither seed is restored to S5's original level
(`easyComparedToS10AndS5`: `backNearS5` is false for both; `restoredVsS10` is also false, since S11 is worse than
S10, not better — i.e., S10's weighted result is the better of the two, just not as good as S5's). For 97101, easy
was never at risk either way (0.94 vs 0.95, both well above S5's 0.74) — the data draw helped it and the weight
change made no material difference.

## Critic and label error

The warm-start gate passes for all three seeds at `aimWeight = 1` (unchanged from S10). Label error at `aimWeight =
1`: 97101's aim error is 12.3° (between S5's 13.2° and S10's 10.0°) and its move error returns to near S5's level
(10.8° vs S5's 14.2°, far below S10's 32.1°) — consistent with S10's move-error increase being a side effect of the
raised weight specifically, not the data draw, since it disappears when the weight reverts on the same data.
97102's throw recall falls further at `aimWeight = 1` (0.53, below both S5's 0.71 and S10's 0.69) — worse
without the weight than with it, the same direction as its easy-success pattern. 97103's recall partly recovers
toward S5's level (0.62, up from S10's collapsed 0.46 but still below S5's 0.76) — the opposite direction from its
own easy-success pattern (where S10's weighted run was closer to S5 and S11 fell further away). Recall does not
track either factor consistently across the two seeds.

## What this establishes and what it does not

**Establishes:** 97101's scripted-normal gain is attributable mainly to the loss-weight change, with a smaller but
real contribution from the training-data draw; the two null seeds (97102, 97103) are unaffected by either factor on
scripted-normal; and the fresh training data itself — independent of the loss change — is the primary driver of
97102's and 97103's scripted-easy regression, which the raised aim weight partly, not fully, offset.

**Does not establish:** why this particular data draw damages 97102's and 97103's easy performance, why the
raised weight is protective there, why 97101 benefits on normal while the other two do not, or whether a smaller
`aimWeight` would recover more of 97102/97103's easy performance while keeping 97101's normal gain. Three seeds
sharing one round-zero dataset per run is still not three independent training cohorts.

## Consequence

The aim-weight change is confirmed causal for one seed and is not a clean win: it helps 97101 substantially on
normal, is neutral for two seeds on normal, and is protective (not harmful, though not sufficient) for those same
two seeds on easy against a data-draw effect this run exposed. No further tuning is authorized here. Two candidates
for a following declaration:

1. **Replicate 97101's result on a second, independent data draw.** 97101's 0.51 is the only 3v3 learner result in
   this track that has ever cleared S5's own viability bar, and this run established the aim-weight change is
   causally responsible for most of it — but one seed on one data draw is not evidence it generalizes. This is
   ranked first because it tests whether the track has a usable initializer at all, which the easy-arm question
   below does not.
2. Diagnose what specifically in this data draw damages 97102's and 97103's easy performance — now the largest
   single effect measured across S10 and S11 combined, but a narrower question about one training-data artifact
   rather than whether the repair generalizes.

If a smaller `aimWeight` is tried in either, it should be evaluated on the same data draw as its comparison point,
not confounded by yet another fresh draw.

## Erratum (2026-09-22, after an external math/RL review): "loss-weight-driven" should read "consistent with a causal effect," not "confirmed"

A review of the S1–S11 arc (see the parallel erratum on `m8_s10_results.md`) found this document overclaims. The
classification rule and its result stand as computed — 0.12 for 97101 at `aimWeight = 1` on S10's exact data,
correctly below the declared 0.20 threshold — but the surrounding prose ("the loss change is causal and necessary,"
"confirmed causal") states more than a comparison of two single runs can support. With no independent replication of
either condition (S10 and S11 are each one training run), the +0.39 point estimate has no variance to be judged
against, and the same caveat applies to the easy-arm "protective effect" (±0.41/+0.47 swings on a metric that could
itself be run-to-run noise at this sample size). **Read every causal claim in this document as "the point estimate
is consistent with," not "confirmed."** The consequence section's own top-ranked next step — replicate before
building further on 0.51 — was the right call for the wrong reason (generalization, not statistical validity); both
reasons hold. This section is additive; no number in this file changes. The aim-weight branch is not extended
further; see [M8-S12](m8_s12_declaration.md) for the representation-level repair pursued instead.
