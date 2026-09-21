# M8-S5 results: 3v3 teacher-imitation initializers

Collected 2026-09-20 from `2ac2479` (declaration and implementation committed before
the run). Declaration: [m8_s5_declaration.md](m8_s5_declaration.md). Archive:
`runs/m8_s5_roster_imitation_v0/` (102 artifacts, manifest `sha256:df4cfdcc…`, verified
with `verify_sealed`; the recorded declaration digest matches the committed file). No
policy was selected or promoted; `autonomousQualificationEligible` stays false.

Run facts: 1,124,388 simulator decisions against the 1,523,200 bound (cap 2,000,000).
R1n-h's mixture recipe (condition M, optimizer seeds 97101–97103) ran unchanged at 3v3;
each seed's final `fit-4.pt` was evaluated deterministically on the S4 paired worlds
(seeds 2100000–2100399, 400 per Red arm) and compared world by world with the S4 teacher.

## Pre-declared reading: **not viable**

Seed-mean success against scripted-normal is **0.0017** (0.005, 0.000, 0.000), far below
the declared 0.25. Per the declaration, the response is to diagnose before any PPO step.

## Paired evaluation (400 worlds per cell; Wilson 95% for success)

| Seed | Red arm | Success | Team wipe | Timeout | Mean `L` | Mean decisions | Success − teacher | `L` − teacher |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 97101 | easy | 0.730 [0.68, 0.77] | 0.20 | 0.08 | 0.519 | 151 | −0.270 | +0.506 |
| 97101 | normal | 0.005 [0.00, 0.02] | 0.99 | 0.01 | 0.995 | 115 | −0.993 | +0.958 |
| 97101 | random | 0.065 [0.04, 0.09] | 0.52 | 0.43 | 0.723 | 184 | −0.935 | +0.713 |
| 97102 | easy | 0.885 [0.85, 0.91] | 0.09 | 0.03 | 0.236 | 108 | −0.115 | +0.223 |
| 97102 | normal | 0.000 [0.00, 0.01] | 1.00 | 0.00 | 1.000 | 91 | −0.998 | +0.963 |
| 97102 | random | 0.077 [0.06, 0.11] | 0.77 | 0.15 | 0.847 | 166 | −0.922 | +0.837 |
| 97103 | easy | 0.907 [0.88, 0.93] | 0.07 | 0.02 | 0.282 | 119 | −0.092 | +0.269 |
| 97103 | normal | 0.000 [0.00, 0.01] | 1.00 | 0.00 | 1.000 | 112 | −0.998 | +0.963 |
| 97103 | random | 0.193 [0.16, 0.23] | 0.33 | 0.48 | 0.523 | 185 | −0.807 | +0.513 |

Differences to the teacher are world-paired (learner minus teacher); the world-bootstrap
intervals are in `report.json` and all exclude zero, trivially so for the scripted-normal
cells, where the learner scores about zero against a teacher near one; the exclusion is
arithmetic there, not evidence. Per-seed values are the result; seed
means are descriptive and no interval pools the seeds. `L` is read next to success and
decisions: against scripted-normal the learners end in about 90–115 decisions because
they are wiped, not because they finish.

## What the data shows

- **A clear, consistent pattern across all three seeds:** strong against scripted-easy
  (73–91% success), essentially zero against scripted-normal (blue is wiped in 99–100% of
  episodes), and weak against random Red (6–19% success, with 33–77% wipes and 15–48%
  timeouts). All seeds agree on the ordering, so this is not the 0/100/14% spread of
  R1n-h's 1v1 M; here the spread is small and the failure is uniform.
- **Random Red is not the easy case for these learners.** The teacher loses about 1% of its
  units to random Red, yet these policies are wiped in a third to three quarters of episodes
  and time out in up to 48%. Nothing here explains why.
- **The measurement is corroborated.** `mixture_imitation`'s own 100-world development
  evaluations (bands 3320000–3323000, a different code path) give the same picture: easy
  76 / 93 / 92%, normal 1 / 0 / 0%, random 6 / 5 / 19%. Those numbers are development-only
  and are not the S5 result.
- **Imitation fidelity is imperfect but not collapsed** (development label-error reads, teacher
  labels on each policy's own visited states): action-type accuracy 94–97%, throw recall
  0.66–0.76, throw aim error 13–24°, far-move heading error 12–14°. These do not identify
  which channel limits performance; R1n-i showed aggregate label error can hide a
  behavioral failure, and this run does not repeat that diagnosis.

## Critic warm start at roster 3 (the re-check S4 required)

The warm-start gate passes for all seeds: predictive R² 0.628 / 0.628 / 0.637 against a
time-only baseline of 0.29–0.31 and an untrained critic at −0.20. R1n's roster-1 values were
about 0.07–0.13. `assignedLivingFraction` is now graded and is an input to the critic, and
it plausibly carries information about the eventual outcome; this was not tested, and
whether a critic with this R² helps PPO is not shown. The R1n-d return-predictability
ceiling (0.16–0.24) was measured at roster 1 and is exceeded here, so it does not carry over
to 3v3.

## What this does and does not establish

**Establishes:** at 3v3, R1n-h's mixture-imitation recipe run unchanged yields initializers
that do not reach the teacher against scripted-normal on any of three optimizer seeds, though
they are competent against scripted-easy; the failure is uniform across seeds, and the
critic gate no longer looks like the obstacle.

**Does not establish:** why they fail against scripted-normal or random Red; that more data
or a different recipe would fix it; that 3v3 imitation is intrinsically harder than 1v1 (no
paired 1v1 comparison exists); or anything about PPO. The optimizer seeds share one round-zero
teacher dataset, so uniform failure across them is weaker evidence of a robust property
than three independent training cohorts would be. There is no single-opponent control, so
nothing here attributes the result to the mixture.

## Consequence

The declared response to "not viable" is an R1n-i-style diagnosis of these frozen
checkpoints before any PPO. The retained `fit-0.pt`–`fit-4.pt` per seed, the paired episode
rows, and the per-seed development label-error reads are the starting evidence. The
candidate questions (contact versus finishing against scripted-normal, why random Red
defeats learners the teacher beats, and whether the failure is a data-volume, coverage or
representation limit) are for a new declaration, chosen and ordered there, not here.
