# M8-S12 results: the enemy-relative throw decoder beats the old decoder in all three independent cohorts

Collected 2026-09-22 from `c3d0e62` (declaration and implementation committed before training). Declaration:
[m8_s12_declaration.md](m8_s12_declaration.md). Archive: `runs/m8_s12_enemy_relative_throw_v0/` (sealed and
verified, `sha256:ec7248ff…`, 20 MB, declaration pinned to `c3d0e62`). Ordinary autonomous training, not
assisted. `autonomousQualificationEligible` stays false. 1,057,889 simulator decisions against the ~1.47M
bound, 6 runs (3 cohorts × {old, new} decoder), no reproduction-gate failures, no rejected actions in any cell.

## Headline: consistent, statistically clear, in every one of 3 independent cohorts

Both success and units-lost fraction favor the new decoder in **all 6 paired comparisons** (3 cohorts × 2 Red
arms), each a world-paired bootstrap over the same 100 worlds:

| Cohort | Arm | Δ success (new − old) | 95% CI | Δ mean `L` (new − old) | 95% CI |
| --- | --- | ---: | --- | ---: | --- |
| 1 | normal | **+0.70** | [0.61, 0.79] | **−0.50** | [−0.57, −0.42] |
| 1 | easy | +0.08 | [0.03, 0.14] | −0.23 | [−0.28, −0.17] |
| 2 | normal | **+0.57** | [0.47, 0.67] | **−0.31** | [−0.37, −0.25] |
| 2 | easy | +0.45 | [0.35, 0.54] | −0.55 | [−0.63, −0.48] |
| 3 | normal | **+0.14** | [0.03, 0.26] | −0.09 | [−0.17, −0.01] |
| 3 | easy | +0.12 | [0.06, 0.19] | −0.24 | [−0.31, −0.17] |

Every interval excludes zero. This is different in kind from S10's single-run result: each row above is a
legitimate, independently-computed statistic (100 world-paired samples within one cohort, real variance), and
the *direction* agrees in all 3 independent cohorts, not just one.

## Absolute numbers, and the viability bar

| Cohort | Decoder | Normal success | Normal `L` | Easy success | Critic R² (gate) |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | new | **0.74** | 0.47 | 1.00 | 0.401 (pass) |
| 1 | old | 0.04 | 0.97 | 0.92 | 0.661 (pass) |
| 2 | new | **0.57** | 0.69 | 1.00 | 0.387 (pass) |
| 2 | old | 0.00 | 1.00 | 0.55 | 0.521 (pass) |
| 3 | new | **0.30** | 0.80 | 1.00 | 0.326 (pass) |
| 3 | old | 0.16 | 0.89 | 0.88 | 0.596 (pass) |

S5's declaration defines two tiers, both reused here unchanged (0.25 is R1n's own `successMinFraction`, not
retuned): **viable** is seed-mean success ≥ 0.25, **reliable across seeds** is *every* seed ≥ 0.25 (otherwise
"viable-but-unreliable"). Cohorts here are independent replicates, not seeds within one run, but the same two
readings apply directly: cohort-mean success is (0.74+0.57+0.30)/3 = **0.537**, clearing viable, and **all three
cohorts individually clear 0.25**, the direct analogue of S5's stricter "reliable" tier. In S5/S10/S11 combined (7
prior old-decoder runs across two recipes), exactly one seed ever cleared 0.25, at 0.51, under a change this track
went on to call unattributable without further work — that run was viable-but-unreliable at best. This result is
**viable and reliable** by S5's own rule, the strongest reading available, and it is earned: even the smallest
cohort gain (cohort 3, +0.14 over its own paired old-decoder run) has a confidence interval excluding zero.

The old decoder's own numbers here (0.04, 0.00, 0.16 on normal) are broadly consistent with S5's original three
seeds (0.005, 0.000, 0.000) and with the general pattern that fresh data draws produce small, noisy, usually-near-zero
old-decoder outcomes (matching S11's finding that data draws alone can occasionally produce a modest positive
value). Cohort 3's old-decoder 0.16 is the highest old-decoder result seen anywhere in this track and is itself a
reminder that data-draw variance is real — which is exactly why the paired, same-cohort comparison (not a comparison
to S5's archived numbers) is the one being reported as the finding.

## What did not need weight tuning

No loss coefficient was changed from 1. The enemy-selection loss and the angle loss are new terms required by the
new action space, not reweighted versions of the old `aim` term. Every run used the identical unweighted-imitation
recipe (fits, rounds, minibatch, learning rate) — only the decoder architecture and its two loss terms differ from
the old arm run in the same cohort.

## Critic

The warm-start gate passes for all 6 runs. The new decoder's critic R² (0.33–0.40) is consistently lower than the
paired old decoder's (0.52–0.66) in every cohort — the new decoder's policy is harder for the critic to predict
returns for, or its behavior is more varied episode-to-episode (both consistent with a policy that reaches
scripted-normal success far more often, so its return distribution covers more of the possible outcomes rather
than concentrating near a floor). This is descriptive; no claim is made about which explanation is correct.

**This reverses the reading R1n-d/R1n-e/S5–S11 relied on: here, the better policy has the lower critic R², in all
three cohorts.** Every prior step in this track treated a higher critic R² (or the gate passing at all) as
reassurance about the actor. That assumption does not hold in this regime — critic R² is not a quality signal for
the actor here, only a diagnostic about how easy returns are to predict. This matters directly for any future PPO
step built on this initializer, since the critic-gate apparatus was built on the opposite assumption at roster 1.

## What this establishes and what it does not

**Establishes:** across 3 independent training-data draws, replacing the throw target's absolute-coordinate
representation with an enemy-relative one (matching how the simulator's throw physics actually works) produces a
large, statistically clear, consistently-directed improvement over the old representation, on identical data, with
no loss-weight tuning. All three new-decoder cohorts individually clear the track's own viability bar.

**Does not establish:** the *typical* size of the effect (0.14 to 0.70 is a wide range across only 3 draws — a
properly powered estimate needs more cohorts, exactly as the declaration anticipated); whether the improvement
holds against random Red (not tested, per S4's finding that random Red is uninformative for weak learners, though
this decoder is no longer weak on normal — worth testing next); why cohort 3's gain is smaller (no per-mechanism
diagnostic was run, deliberately, to avoid repeating the assisted-diagnosis cost before knowing the architecture
change works at all); or anything about PPO, command-following, or roles. This is imitation only.

## Consequence

Unlike every step in the aim-weight branch, this result is the kind the pre-declared "extend to more cohorts if
promising" contingency was written for. Recommended next steps, in order:

1. **Evaluate the existing 3 cohorts' new-decoder checkpoints against random Red.** No training — a paired eval on
   checkpoints that already exist, costing minutes. This answers a different question from effect-size precision:
   whether the improvement is general or a scripted-Red artifact. S4's "random Red is uninformative" finding was
   scoped to weak learners; this decoder is no longer weak on normal, so that scoping may no longer apply, and
   it's cheap enough to check before spending more compute on additional cohorts.
2. **Extend toward the reviewed ≥5-cohort target** (2 more independent cohorts, same recipe, no other change) to
   properly bound the effect-size range before treating 0.74/0.57/0.30 as representative.
3. Only after that, consider whether a bounded PPO continuation from this initializer is worth declaring — S4
   already showed from-scratch PPO has no signal, and this initializer is now unambiguously stronger than any
   before it, but PPO was out of scope for this step, the critic-R²-as-quality-signal assumption no longer holds
   here, and it should get its own declaration.

None of these is authorized here.
