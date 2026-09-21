# M8-S7 — trace-level diagnosis of the post-contact fight (S5 initializers, 3v3)

Declared 2026-09-20, before any trace was collected. Collect-and-archive step; observation
only. No training, no intervention, no checkpoint selection or promotion.
`autonomousQualificationEligible` stays false.

## 0. Why

[S6](m8_s6_results.md) located the S5 failure *after contact*: against scripted-normal the
learners land their first hit at the teacher's decision and range (53–55 vs 52), then deal
far less damage and are wiped by about decision 100 (damage per unit lost 18–45 vs the
teacher's 2,163). Against random Red, contact is late and wipes dominate two seeds. The
archived rows carry no per-unit deaths, positions, shots or motion, so S6 could not say what
the fight looks like. This step records it.

## 1. Design

**Cells.** Learners: S5's three final policies (`fit-4.pt`, seeds 97101–97103) × Red arms
`normal` and `random`. Control: the S4 native teacher on the same worlds and arms. 8 cells.
`easy` is skipped (mostly successful; its failures are near-finishing misses).

**Worlds.** The first 100 of S4/S5's paired worlds, seeds **2100000–2100099**, collected in
blocks of 50 exactly as S5's paired evaluation was, so a deterministic policy must reproduce
its archived episode. No new seed band: these are evaluation worlds already used and never
trained on.

**Reproduction gate (required).** For every cell, the traced episode rows (success, blue units
alive, final decision, final target damage, first-hit decision) must equal the archived S5 (or
S4) rows for the same seeds. If any cell fails, the run still seals and reports, the failure is
the finding, and no analysis of that cell is trusted.

**Logged per decision** (raw simulator state, before each decision plus the final state): every
blue and red unit's alive flag, health, position, velocity, stun, throw-phase and throw-cooldown
values; every projectile's id, owner, team, position and velocity; and each blue unit's executed
action type and acceptance. Nothing is inferred from the model's internals.

**Budget.** Bound 160,000 decisions (learners 120,000, teacher 40,000); hard cap 220,000 (aborts
unsealed). Roughly 10–20 minutes at S4's observed rate; the run is one background process that
refuses to overwrite.

## 2. Measures (all computed from the traces, per episode, then aggregated per cell)

`post-contact` means decisions from the first decision that damages any red unit onward.
`R` = 9.0, the R1n reference engagement range.

- **Timeline:** first-hit decision; first blue death and blue wipe decisions relative to first hit.
- **Fire distribution:** *wasted damage share* = damage dealt to red units that are not dead at
  episode end ÷ total damage dealt (undefined if no damage). It separates focused fire, which
  converts damage into kills, from dispersed fire, which does not.
- **Spacing:** mean pairwise distance among living blue units at the first-hit state; mean distance
  from each living blue unit to its nearest living red unit over post-contact decisions.
- **Shot volume and yield:** blue projectiles spawned per living blue unit-decision post-contact;
  red projectiles spawned per living red unit-decision post-contact; *red yield* = blue health lost
  ÷ red projectiles spawned; *blue yield* likewise. Yields are aggregate ratios; **no per-shot hit
  attribution is made** (it is ambiguous with several projectiles in flight).
- **Physical motion:** mean actual displacement per living blue unit-decision (position change, not
  MOVE selection) post-contact, split by whether any red projectile is within `R` of the unit at
  that decision. This addresses the R1n-i erratum's objection that MOVE selection is not motion.
- **Incapacitation:** share of living blue unit-decisions post-contact with stun or throw-phase
  remaining.

Per-episode values are aggregated to a cell mean with a world-bootstrap 95% interval (10,000
resamples, seed 973001) over episodes where the measure is defined. Learner and teacher cells are
compared as separate cells, not pairwise, because death and exposure measures are often undefined
for the teacher, which rarely loses a unit. Seeds are never pooled. Decisions within an episode are
serially dependent, so all inference is over worlds, not decisions.

## 3. Predictions (fixed now; scored mechanically against the report)

A prediction **holds** only if its inequality is satisfied by the point estimate of **all three**
learner seeds (`L_s`) against the teacher cell (`T`) in the stated arm; otherwise it is reported as
not holding, with the seed values. Ratios against a teacher value of zero are "undefined".

| # | Arm | Prediction | Reads as |
| --- | --- | --- | --- |
| P1 | normal | wasted damage share: `L_s ≥ 0.5` and `L_s ≥ T + 0.30` | learners disperse fire instead of converting damage into kills |
| P2 | normal | ally spacing at first hit: `L_s ≤ 0.8 × T` | learners clump |
| P3 | normal | blue projectiles per living unit-decision: `L_s ≤ 0.75 × T` | learners throw too little after contact |
| P4 | normal | displacement under threat: `L_s ≥ 0.8 × T` | no evasion-motion deficit (a *null* prediction) |
| P5 | normal | red yield: `L_s ≥ 1.25 × T` | learners are hit more per red shot |
| P6 | normal | incapacitated share: `L_s ≥ 1.25 × T` | learners spend more time stunned or in throw phase |
| P7 | random | post-contact distance to nearest red: `L_s ≤ 0.8 × T` | learners sit at close range against random Red |
| P8 | random | red projectiles per living red unit-decision: `0.75 × T ≤ L_s ≤ 1.25 × T` | random Red is not more aggressive against learners; the difference is exposure |

## 4. What each outcome would point to (not a decision rule)

P1 holding points at target selection or focus; P2 at cohesion; P3 at throw volume or readiness; a
failing P4 (a real displacement deficit) would revive the movement explanation R1n-i's erratum
demoted; P5/P6 at survivability; P7/P8 at why random Red produces wipes. Several can hold together.
The next step (probably a bounded intervention on a common state, for the mechanism that survives)
is chosen from the scored results, in its own declaration.

## 5. What this cannot show

Observation cannot separate cause from correlate: a policy that is losing also spends more time
stunned. It does not identify which action channel is wrong, does not test any repair, and covers 100
worlds per cell on one initialization procedure (the three seeds share one round-zero teacher
dataset, so uniform findings are weaker than three independent cohorts). Findings hold for these
frozen checkpoints and these arms only.

## 6. Implementation and archive

New module `options/roster_trace.py` and tests; no existing module is edited (the S6-era
`collect_logged` is 1v1-specific, so the loop is reimplemented for three units, following the
`run_block` structure). The declaration pins the S4 and S5 manifests, the declaration digest and
source digests. Sealed archive `runs/m8_s7_trace_diagnosis_v0/`; results in `reviews/m8_s7_results.md`.
Gates before collection: client and training `pytest`, `npm run build`, `npm test` (366/367, the one
documented exception). A failed reproduction gate or unheld prediction is a finding, not patched around.
