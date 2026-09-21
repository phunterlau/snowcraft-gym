# M8-S8 results: per-channel teacher substitution on the S5 3v3 learners

Collected 2026-09-20 from `944ad68` (declaration `3441138`, crash fix `944ad68`). Declaration:
[m8_s8_declaration.md](m8_s8_declaration.md). Archive: `runs/m8_s8_channel_intervention_v0/` (sealed and verified,
`sha256:33912263…`, 12 MB). **Every arm is teacher-assisted diagnosis; none of these numbers is an autonomous
result**, and `autonomousQualificationEligible` stays false. 238,230 simulator decisions against the 460,000 bound.

**Discarded first attempt.** The first launch crashed at the `aim+power` cell (`planId is invalid`: the rule name entered
the plan id and `+` is not a valid character). It produced no report and I looked at none of its cells; its partial
unsealed archive was deleted, the identifier was sanitized, a test now collects every rule, and the run was relaunched
from the fixed commit.

**Hard gate passed.** The `none` arm, which queries the teacher at every decision without using it, reproduced the S5
archived rows exactly for all three seeds (100/100 worlds each), so querying the teacher does not perturb the simulation.

## Predictions: 4 of 6 hold

Recovery `R = (yield_arm − yield_none) / (yield_teacher − yield_none)`, with the `all` arm on normal Red (the
tensor-path teacher, yield 13.8; the native teacher's 16.9 is reported alongside) as the reference. Values are
seeds 97101 / 97102 / 97103.

| # | Prediction | Values | Verdict |
| --- | --- | --- | --- |
| P1 | `aim`: R ≥ 0.5 | 0.61 / 1.01 / 1.72 | **Held** |
| P2 | `power`: R < 0.25 | 0.07 / −0.03 / −0.04 | **Held** |
| P3 | `timing`: R < 0.25 | 0.74 / 0.35 / −0.19 | **Not held**: strongly seed-dependent |
| P4 | `throw-all`: R ≥ 0.75 | 1.31 / 1.14 / 1.40 | **Held** |
| P5 | `throw-all`: success ≥ 0.25 | 1.00 / 1.00 / 1.00 | **Held** |
| P6 | `move`: success < 0.10 | 0.05 / 0.42 / 0.00 | **Not held**: 97102 reaches 0.42 |

## Results (normal Red, 100 worlds per cell; success under teacher assistance)

| Arm | Seed | Success | Mean `L` | Wipe | Yield per shot | R | Unit-decisions changed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none (S5 reproduced) | 97101 / 97102 / 97103 | 0.00 / 0.00 / 0.00 | 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 6.8 / 3.8 / 7.3 | 0 | 0 |
| aim | 97101 / 97102 / 97103 | 0.24 / 0.99 / 1.00 | 0.87 / 0.15 / 0.08 | 0.76 / 0.01 / 0.00 | 11.1 / 13.9 / 18.5 | 0.61 / 1.01 / 1.72 | 5–6% |
| power | 97101 / 97102 / 97103 | 0.01 / 0.00 / 0.03 | 0.99 / 1.00 / 0.98 | 0.98 / 1.00 / 0.97 | 7.3 / 3.4 / 7.0 | 0.07 / −0.03 / −0.04 | 4.5–6.3% |
| aim + power | 97101 / 97102 / 97103 | 0.45 / 0.73 / 1.00 | 0.75 / 0.44 / 0.10 | 0.55 / 0.27 / 0.00 | 13.1 / 11.6 / 18.4 | 0.90 / 0.78 / 1.70 | 5–7% |
| timing | 97101 / 97102 / 97103 | 0.95 / 0.02 / 0.00 | 0.08 / 0.98 / 1.00 | 0.05 / 0.96 / 1.00 | 12.0 / 7.3 / 6.1 | 0.74 / 0.35 / −0.19 | 2% |
| throw-all | 97101 / 97102 / 97103 | 1.00 / 1.00 / 1.00 | 0.00 / 0.05 / 0.15 | 0.00 / 0.00 / 0.00 | 16.0 / 15.2 / 16.4 | 1.31 / 1.14 / 1.40 | 7–8% |
| move | 97101 / 97102 / 97103 | 0.05 / 0.42 / 0.00 | 0.97 / 0.65 / 1.00 | 0.94 / 0.58 / 1.00 | 4.9 / 8.9 / 5.7 | −0.26 / 0.51 / −0.26 | 61–70% |

Every intervention arm's difference to `none` is paired by world (intervals in `report.json`); the important ones
exclude zero by a wide margin (e.g. `throw-all` +1.00, `aim` on 97102 +0.99 [0.97, 1.00]). Controls: the tensor-path
teacher (`all`) succeeds 100/100 on both arms with yield 13.8 (normal) and 15.7 (random).

## What the data shows

1. **The throw channel is sufficient.** Replacing all of the learner's throw decisions (timing, aim and power) with the
   teacher's, while the learner keeps its own movement and everything else, gives **100% success on all three seeds** and
   loses at most 15% of units, against 0% and 100% wipes with no substitution. Whatever else is imperfect in the
   learners' movement, it is adequate when the throws are right.
2. **Where the throw lands is the main carrier.** `aim` alone (the teacher's throw target where both would throw)
   lifts success to 0.24 / 0.99 / 1.00. Power alone does nothing (≤ 0.03), and `aim + power` is not better than `aim`
   (0.45 / 0.73 / 1.00, and worse on 97102), a non-monotone result that is reported and not explained.
3. **The failure differs by seed.** For 97101, `timing` alone restores success to 0.95 while `aim` alone reaches only
   0.24; for 97102 and 97103 `timing` is inert and `aim` is enough. The three seeds fail through different parts of the
   throw channel, echoing the seed heterogeneity R1n-h found at 1v1.
4. **Movement is not the main cause, but the arm is a blunt test.** Movement endpoint replacement alone reaches
   0.05 / 0.42 / 0.00 success. But it changes 61–70% of unit-decisions against 2–8% for the throw arms, so it is a very
   different dose, and 97102 gets substantial benefit from it. It does not show movement is irrelevant.
5. **Coverage was high.** In 74–100% of learner THROW decisions the teacher also throws (so `aim` and `power` were
   genuinely applied), and in 81–98% of learner MOVE decisions the teacher also moves.

## Correction to S7: execution path was a confound

S7 compared learners, which act through the tensor pipeline, with a native scripted teacher. The tensor-path teacher
(`all` control) is not the native one: it is slower (84 vs about 76 decisions) with lower yield (13.8 vs 16.9), and wider
spacing at first hit (5.97 vs 4.77). Re-reading S7 against the like-for-like control (exploratory, not new predictions):

| Measure (normal Red) | native teacher (S7) | tensor-path teacher | S7 learners |
| --- | ---: | ---: | --- |
| ally spacing at first hit | 4.77 | **5.97** | 5.95 / 5.49 / 5.38 |
| damage per blue projectile | 16.9 | 13.8 | 6.8 / 3.8 / 7.3 |
| blue projectiles per living unit-decision | 0.156 | 0.147 | 0.139 / 0.136 / 0.137 |
| displacement under threat | 0.153 | 0.186 | 0.126 / 0.110 / 0.118 |
| incapacitated share | 0.427 | 0.400 | 0.488 / 0.583 / 0.484 |
| red projectiles per living red unit-decision | 0.034 | 0.047 | 0.083 / 0.108 / 0.080 |
| red yield (blue health lost per red shot) | 17.0 | 14.1 | 18.0 / 19.2 / 18.6 |

- **The S7 claim that learners are more spread than the teacher is withdrawn for normal Red:** like-for-like the
  learners are no more spread (5.4–5.9 vs 6.0). Against random Red the tensor-path teacher spaces at 4.41 vs the
  native 2.90, and the learners (5.7–6.8) are still wider, so a smaller residual remains there.
- **The mobility deficit does not shrink like-for-like; it grows** (learners 0.59–0.68 of the tensor-path teacher's
  displacement under threat, against 0.72–0.82 of the native one), still confounded with being stunned.
- **The per-shot damage lead survives** (13.8 vs 3.8–7.3, a 1.9–3.7× gap), and S8's arms are measured against the
  tensor-path teacher throughout.
- **Re-scored against the tensor-path teacher, S7's predictions go from 0 of 8 to 2 of 8** (P1, wasted damage share;
  P5, red yield); P2–P4, P6–P8 still fail. This is an exploratory re-read; the original S7 scoring against the native
  teacher stands as the pre-declared result, and an erratum has been appended to `m8_s7_results.md`.

## What this establishes and what it does not

**Establishes (teacher-assisted, 100 worlds, three related seeds):** the S5 learners' failure against scripted-normal is
in the throw channel, not in their movement or formation; replacing their throws is sufficient for 100% success, and the
target point (which enemy and where) is the main carrier, with a seed-specific role for throw timing and none for power.

**Does not establish:** necessity (no channel was shown to be required, only sufficient); the split between choosing a
target enemy and aiming geometrically (the target point conflates them); that any of this transfers to an autonomous
repair; behavior against random Red (only the control ran); or why `aim + power` underperforms `aim`. Sustained
substitution changes the visited states, and `aim + power` on seed 97102 is a concrete instance: adding the teacher's
power to the teacher's aim *lowers* success from 0.99 to 0.73 on the same worlds with essentially the same coverage
(0.88 vs 0.91), which the sufficiency framing cannot explain and which is consistent with better early throws moving the
units into states the learner's own later decisions handle worse. The seeds also share one round-zero teacher dataset, and
success under assistance must never be quoted as a learner result.

## Consequence

The failure now has an address: the throw target. The natural next declarations are (a) splitting target choice from aim
geometry with a further substitution (for example replacing only *which enemy* is targeted, or only the offset from it),
and (b) an autonomous repair aimed at the throw target (for instance a decoder that predicts the throw target relative to
the chosen enemy rather than as an absolute arena coordinate, or more weight on throw labels), evaluated on fresh worlds and
never mixed with these assisted numbers. Neither is started here.
