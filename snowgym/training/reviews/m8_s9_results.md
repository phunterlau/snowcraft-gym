# M8-S9 results: splitting the throw target into which enemy vs. angular offset

Collected 2026-09-21 from `8265e92` (declaration and implementation committed before the run). Declaration:
[m8_s9_declaration.md](m8_s9_declaration.md). Archive: `runs/m8_s9_target_split_v0/` (sealed and verified,
`sha256:b388764c…`, 7.9 MB). **Every arm is teacher-assisted diagnosis; none of these numbers is an autonomous result**, and
`autonomousQualificationEligible` stays false. 154,576 simulator decisions against the 300,000 bound.

**Required gates passed (all three seeds).** `none` reproduces the S5 archived rows exactly, and `aim` reproduces S8's archived
`aim` traces episode by episode (every recorded state and action), so this harness is S8's harness. The reported check also came
out as expected: `aim-heading` (the teacher's heading on a shortened ray, distance discarded) gives outcomes identical to `aim` in
every cell, confirming that the discarded target distance is irrelevant to the throw.

## Predictions: 2 of 4 hold, and the two informative ones fail

| # | Prediction | Values (97101 / 97102 / 97103) | Verdict |
| --- | --- | --- | --- |
| P1 | `enemy`: R ≥ 0.5 | 0.26 / 0.83 / 0.16 | **Not held**: true for 97102 only |
| P2 | `offset`: R < 0.5 | 0.64 / 0.49 / 0.43 | **Not held**: 97101 reaches 0.64 |
| P3 | agreement `k_L = k_T` < 0.5 | 0.41 / 0.27 / 0.33 | **Held, uninformative**: chance is about 1/3 (declared beforehand) |
| P4 | median learner offset > teacher's | 6.9° / 9.7° / 6.1° vs 1.7° / 4.0° / 1.7° | **Held** |

## Results (normal Red, 100 worlds per cell; success under teacher assistance)

| Arm | 97101 | 97102 | 97103 |
| --- | --- | --- | --- |
| none (S5 reproduced) | 0.00 | 0.00 | 0.00 |
| `enemy` (teacher's enemy, learner's offset) | 0.10 (R 0.26) | **0.87** (R 0.83) | 0.05 (R 0.16) |
| `offset` (learner's enemy, teacher's offset) | 0.17 (R 0.64) | 0.06 (R 0.49) | **0.75** (R 0.43) |
| `aim` (both; S8 reproduced) | 0.24 (R 0.61) | 0.99 (R 1.01) | 1.00 (R 1.72) |

Success is the fraction of the 100 worlds won; `R` is the fraction of the damage-per-shot gap closed (S8's definition, against the
tensor-path teacher). Every non-`none` difference is paired by world with a bootstrap interval in `report.json`; the headline
cells (`enemy` on 97102 +0.87 [0.80, 0.93], `offset` on 97103 +0.75 [0.66, 0.83]) exclude zero by a wide margin. No arm needed the
fallback for a degenerate ray. Roughly 5–6% of unit-decisions are changed in each arm.

Descriptive (from `none`, over 1,105–1,699 comparable throws per seed, decisions within an episode serially dependent): the learner
chooses the teacher's enemy 41% / 27% / 33% of the time, statistically hard to tell from random choice among three; its median
angular offset from its own chosen enemy is 6.9° / 9.7° / 6.1° against the teacher's 1.7° / 4.0° / 1.7°; and the median gap between
learner and teacher headings is 14.9° / 25.2° / 14.8°.

## What the data shows

Read with the pre-declared interpretation (section 4 of the declaration), the answer is **not one component; it depends on the
seed**:

- **97102 is a target-selection failure.** Giving it the teacher's choice of enemy while keeping its own angular error restores
  0.87 success; giving it the teacher's precision on its own choice restores only 0.06.
- **97103 is an angular-precision failure.** The reverse: teacher precision on its own choice gives 0.75, the teacher's choice of
  enemy with its own error only 0.05.
- **97101 is neither alone.** Both single components give little (0.10, 0.17) and even the combined `aim` gives 0.24; S8 found that
  its throw *timing* alone restores 0.95, so its failure sits mostly outside the target.
- For 97102 and 97103 only the combination `aim` (0.99, 1.00) matches the full teacher target, and each seed's failure is carried
  by one of the two components while the other is nearly irrelevant.

That the three initializations fail through three different parts of the throw channel matches the seed heterogeneity R1n-h found
at 1v1 and S8 found here. It also means a repair aimed at a single component would not be universal.

## What this establishes and what it does not

**Establishes (teacher-assisted, 100 worlds, three related seeds):** the throw-target deficit is not a single mechanism across
seeds; enemy choice explains 97102, angular precision explains 97103, and neither alone explains 97101. The learners' choice of
enemy is indistinguishable from random and their angular error is 3–5× the teacher's, so both components are genuinely imperfect
in the population even though one dominates per seed.

**Does not establish:** necessity; that "chosen enemy" is an internal selection (it is an angular-proximity proxy); why a seed
depends on one component and not the other; that success counts total damage rather than a particular target, so aiming at a
different enemy is not an error in itself (the teacher's rule for picking an enemy is not established here); or anything about an
autonomous repair. Sustained substitution shifts the visited states (S8's non-monotone `aim + power` is the standing example).

## Consequence

The throw target is now decomposed as far as substitution can take it. A repair has to improve both the choice of target enemy
(near-random at present) and the angular precision (3–5× the teacher's), or be judged per seed. The natural next declaration is an
**autonomous** attempt on fresh worlds, never mixed with these assisted numbers: for example a throw head that predicts the
heading relative to a chosen enemy, an auxiliary selection signal for which enemy to target, and more weight on throw labels, all
compared with S5's unchanged initializers on paired, untouched worlds. Which of those is worth building, and whether to first
recover what the teacher's enemy-selection rule is, is left to that declaration.
