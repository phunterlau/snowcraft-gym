# M8-S6 results: archive-only failure breakdown of the S5 3v3 initializers

Computed 2026-09-20 from `71c824f` (declaration and module committed before any number was
computed). Declaration: [m8_s6_declaration.md](m8_s6_declaration.md). Archive:
`runs/m8_s6_breakdown_v0/` (sealed, verified; pins the S4 and S5 manifests). Zero simulator
decisions; the rows are S5's paired evaluations and S4's teacher on the same 400 worlds per
arm. Damage is `finalTargetDamage` out of 300; success needs at least 240. Outcome categories
are descriptive, not causal.

## Predictions, scored

| # | Prediction (made before computing) | Result |
| --- | --- | --- |
| 1 | Contact ≥ 0.9 in every cell except possibly random Red | **Held.** Contact is 0.95–1.00 everywhere except random-Red seed 97102 (0.82) |
| 2 | Against scripted-normal learners die under 200 damage; against random Red many time out in the 200–239 band | **Half held.** Normal: 96–100% of failures are under 200 damage. **Random Red: failed.** Only 10–17% of failures sit in 200–239; most sit in 100–199 (32–52%) or under 100 (28–40%) |
| 3 | Against random Red timeouts carry substantial damage; wipes carry less | **Timeouts held, wipes not checked.** Timeouts with ≥ 100 damage outnumber those under 100 (0.28 vs 0.14, 0.11 vs 0.04, 0.37 vs 0.11). Damage among wipes was not computed, and wipes are the largest category for two seeds (0.52, 0.77), which I did not anticipate |
| 4 | Learners engage at similar or closer range than the teacher | **Held.** Median minimum distance 6.6–7.2 vs teacher 6.5 against normal; 2.7–4.5 vs 6.3 against random |
| 5 | Learners trade much worse than the teacher | **Held.** Damage per blue unit lost against normal: 45 / 18 / 43 vs teacher 2,163 |

## What the breakdown shows

**Against scripted-normal: timely contact, then collapse.** Learners land their first hit at the
same decision as the teacher (median 53–55 vs 52) and at the same range (6.7–7.4 vs 6.5), then deal
far less damage (median 140 / 60 / 120 vs the teacher's 240) and are wiped by about decision 91–112,
while the teacher finishes at 76 with almost no losses. Seed 97102 is weakest: median damage 60,
under one red unit's worth. The failure is in the fight after contact, not in reaching it.

**Against scripted-easy: near-finishing.** 64–68% of the failures for seeds 97101/97103 end in the
200–239 band and 52% for 97102, just short of the 240 needed, though most episodes still succeed
(73–91%).

**Against random Red: late, attritional, and mostly wipes.** First hit comes later than the
teacher's (median decision 99–103 vs 86), damage stalls mid-way, and blue is wiped in 52% and 77%
of episodes for two seeds (33% for the third), the largest single outcome for those two. The
learners also close to a much shorter minimum distance (2.7–4.5 vs the teacher's 6.3), which is
consistent with, but does not show, being caught in close range by random throws.

## What this cannot show

The rows carry no per-unit death times, movement, throw timing or projectile data. This cannot
separate aim, movement, cohesion or target selection, cannot say why random Red defeats the
learners, and cannot say whether the collapse is a data, coverage or representation limit. It
locates the failure (after contact; a mid-fight exchange collapse), which R1n-i-style logic would
call a finishing/survival problem rather than a contact problem, but it does not explain it.

## Consequence

The pattern supports a trace-level diagnosis of the post-contact fight, with two threads: (a) the
scripted-normal exchange collapse after on-time contact, and (b) why random Red produces wipes and
late contact. That would be a new declaration and a real collection (its own module, since
R1n-i's deployed view guards against more than one live enemy, and a budget). Before deciding, the
open choice is between that diagnosis and cheaper, more targeted uses of the existing S5
checkpoints; that choice is left to the next declaration.
