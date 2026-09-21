# M8-S7 results: trace-level diagnosis of the post-contact fight (S5 initializers, 3v3)

Collected 2026-09-20 from `5009175` (declaration and implementation committed before the run).
Declaration: [m8_s7_declaration.md](m8_s7_declaration.md). Archive: `runs/m8_s7_trace_diagnosis_v0/`
(sealed and verified, `sha256:9f8b4196…`, 7.2 MB of gzipped traces). Observation only: 105,892
simulator decisions against the 160,000 bound. `autonomousQualificationEligible` stays false.

**Reproduction gate: passed in all 8 cells** (100/100 worlds each, every compared field equal to the
archived S4/S5 rows), so the traces are the archived episodes, not lookalikes.

## Predictions: none hold as declared

The rule was mechanical (a prediction holds only if all three learner seeds satisfy it against the
teacher cell). **0 of 8 hold.** Several were right in direction and wrong on the margin I chose;
two were wrong in direction. Values are learner seeds 97101 / 97102 / 97103 against the teacher (T).

| # | Arm | Prediction | Learners | Teacher | Verdict |
| --- | --- | --- | --- | --- | --- |
| P1 | normal | wasted damage share ≥ 0.5 and ≥ T + 0.30 | 0.71 / 0.98 / 0.86 | 0.46 | **Not held**: right direction; 97101 misses the margin (0.71 vs 0.76) |
| P2 | normal | ally spacing at first hit ≤ 0.8 T (clumping) | 5.95 / 5.49 / 5.38 | 4.77 | **Not held, wrong direction**: learners are 13–25% *more* spread |
| P3 | normal | blue shots per living unit-decision ≤ 0.75 T | 0.139 / 0.136 / 0.137 | 0.156 | **Not held**: about 11% lower, not 25% |
| P4 | normal | displacement under threat ≥ 0.8 T (null: no deficit) | 0.126 / 0.110 / 0.118 | 0.153 | **Not held**: a deficit exists (0.82, 0.72, 0.77 of T) |
| P5 | normal | red yield ≥ 1.25 T | 18.0 / 19.2 / 18.6 | 17.0 | **Not held**: only 6–13% higher |
| P6 | normal | incapacitated share ≥ 1.25 T | 0.49 / 0.58 / 0.48 | 0.43 | **Not held**: 13–36% higher, two seeds under the margin |
| P7 | random | post-contact nearest-red distance ≤ 0.8 T | 7.30 / 6.50 / 7.92 | 9.40 | **Not held**: right direction; 97103 misses (0.84 vs 0.80) |
| P8 | random | red shots per living red unit-decision within ±25% of T | 0.123 / 0.118 / 0.121 | 0.095 | **Not held**: red throws 24–29% *more* per unit against learners |

My margins were guesses and several were badly placed; this is reported as a set of failed
predictions rather than as near-misses. Note that P4 was a null prediction: its failure means a
deficit was found, not the reverse, and even that is qualified below.

## What the traces show (all normal Red unless stated)

Per-cell means; learners 97101 / 97102 / 97103 against the teacher. Comparisons are across cells with
different, policy-induced states and exposure (a learner that is losing sees a different fight than the
teacher), so they describe, not attribute.

1. **The damage deficit is per shot, not per throw.** Blue projectiles per living unit-decision are close
   (0.136-0.139 vs 0.156, about 11-12% fewer; this agrees with the corrected THROW action share, 0.137-0.138 vs 0.155,
   as it should since each accepted throw spawns one projectile) but damage per blue projectile is **6.8 / 3.8 / 7.3 vs 16.9**, a 2.3–4.5×
   gap. Red's damage per red projectile is about the same (18.0 / 19.2 / 18.6 vs 17.0). Learners throw
   nearly as often and land far less, which points at aim, timing or target choice rather than passivity.
   That is consistent with the aim error S5's label reads found (13–24°) but this run does not measure aim,
   so it is not shown.
2. **Blue converts damage into kills far less:** red kills per episode 0.46 / 0.02 / 0.22 vs 1.31. Success
   only requires 240 total damage, not kills, so dispersal is not itself a failure; the difference is that
   learners deal too little of it.
3. **Red survives and keeps shooting:** red projectiles per living red unit-decision 0.083 / 0.108 / 0.080 vs
   0.034, two to three times more. A plausible reading is that hits stun and suppress the enemy and low-yield
   blue forfeits that suppression. This was not tested.
4. **Blue is lost quickly once hit:** the first blue death comes 21 / 7 / 13 decisions after the first hit, and
   the wipe 62 / 37 / 57 decisions after (the teacher rarely loses a unit; only 10 of its 100 episodes have one).
5. **Blue moves less, including under threat, but not obviously *because of* threat, and part of it is downstream of being hit.** Displacement per
   living-unit decision under a nearby red projectile is 0.72–0.82 of the teacher's, and with no projectile
   near it is 0.97 / 0.71 / 0.95, so the shortfall is mostly general mobility, not evasion-specific. Displacement is also confounded with being stunned (a stunned unit cannot move) and learners are incapacitated
   more of the time (0.49 / 0.58 / 0.48 vs 0.43), so some of this deficit is a consequence of the damage deficit in finding 1, not a
   separate policy choice; the same caveat applies to the random-Red mobility numbers in finding 7. Corrected
   action shares (below) show learners issue MOVE less (0.40–0.51 vs 0.56) and NOOP more (0.35–0.46 vs 0.28).
6. **Learners spread out; the teacher stays tight.** Ally spacing at first hit 5.4–5.9 vs 4.8 against normal,
   and **5.7–6.8 vs 2.9 against random Red**, where the teacher moves as a compact group. The opposite of
   the clumping I predicted. Whether spread makes units easier to pick off is a hypothesis, not a finding.
7. **Random Red:** learners make first contact later (103 / 107 / 100 vs 88), come to closer range than the
   teacher (nearest-red 6.5–7.9 vs 9.4), move much less (displacement about 0.15 vs 0.38), and face red units
   that throw about a quarter more per unit; red's damage per shot against learners is 3.3–4.6 vs 2.2. The
   teacher takes almost no damage there.

## Defect in two extra measures (disclosed; not a declared measure or prediction)

`throwActionShare` and `moveActionShare` in `report.json` were computed over every blue unit's action entry,
including entries for dead units, which are 30–36% of the learners' entries and none of the teacher's. They
deflate the learners' shares and should not be used. The declared measures and all eight predictions use living
units only and are unaffected. The sealed run is left as archived (it pins the module's digest); the corrected,
post-hoc shares over living units, from `options/roster_trace_posthoc.py` on the archived traces, are:

| Arm | Cell | throw | move | noop | living unit-decisions |
| --- | --- | ---: | ---: | ---: | ---: |
| normal | teacher | 0.155 | 0.563 | 0.282 | 7,579 |
| normal | 97101 / 97102 / 97103 | 0.137 / 0.138 / 0.137 | 0.510 / 0.401 / 0.505 | 0.352 / 0.460 / 0.358 | 12,237 / 7,597 / 10,587 |
| random | teacher | 0.100 | 0.694 | 0.206 | 13,683 |
| random | 97101 / 97102 / 97103 | 0.116 / 0.117 / 0.117 | 0.639 / 0.623 / 0.664 | 0.238 / 0.258 / 0.217 | 17,650 / 11,645 / 21,411 |

## What this establishes and what it does not

**Establishes:** on the archived worlds, the S5 learners' post-contact deficit against scripted-normal is
dominated by low damage per shot (2.3–4.5× lower) with nearly normal throw volume; they kill few red units, lose
units quickly after the first hit, are less mobile and more spread than the teacher, and idle more. None of the
eight pre-declared explanations was confirmed as stated.

**Does not establish:** any mechanism. Per-shot effectiveness could be aim, throw timing, power, or target
choice, and the traces here do not separate them; lower mobility and wider spacing are associations in policy-induced
states, not shown causes (mobility is partly a consequence of being stunned); the measures compare cells whose exposure differs; and 100 worlds on one initialization
procedure (three seeds sharing one round-zero dataset) is a small, correlated sample. Per-shot hit attribution was
deliberately not attempted.

## Consequence

The sharpest, most testable lead is per-shot damage. A bounded common-state intervention could ask whether
replacing only the learner's throw aim, timing or power with the teacher's (diagnostic assistance, never
counted as autonomous) restores the damage, and whether restoring mobility changes survival. That is a
separate declaration with its own module and budget, chosen and ordered there.
