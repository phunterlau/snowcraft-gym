# M8-S8 — per-channel teacher substitution on the S5 learners (diagnostic assistance, 3v3)

Declared 2026-09-20, before any intervention run. Collect-and-archive step. **Every arm here is
teacher-assisted diagnosis, never autonomous behavior:** `assistType` records it and
`autonomousQualificationEligible` stays false. Nothing is trained; no checkpoint is selected or promoted.

## 0. Why

[S7](m8_s7_results.md) found the S5 learners' post-contact deficit against scripted-normal is dominated by
damage per blue projectile (3.8–7.3 vs the teacher's 16.9) with nearly normal throw volume (0.137 vs 0.156
per living unit-decision). That points at throw effectiveness but does not say *which channel*: where the
throw lands (target point), how hard (power), or when (whether to throw), versus positioning. S7 explicitly
could not separate them. This step asks the sufficiency question directly: **if only one channel of the learner's
action is replaced by the teacher's, how much of the gap is closed?**

## 1. Design

**Cells.** S5's three final policies (`fit-4.pt`, 97101–97103) × scripted-normal Red × the S7 worlds, seeds
**2100000–2100099**, collected in blocks of 50 like S5's paired evaluation. No new seed band; these worlds are
evaluation-only and were never trained on.

**Per decision, per living blue unit,** the learner acts deterministically (as in S5/S7). The chooser also asks
the plan teacher for its action `T_u = (type, target, power)` in the *same* state (the DAgger label path,
`plan_teacher_tensor_actions_indices`) and applies one rule to the learner's action `A_u`:

| Arm | Rule (applied only to living units) |
| --- | --- |
| `none` | no change; the teacher is still queried (control for the query itself) |
| `aim` | if `A_u` is THROW and `T_u` is THROW: use the teacher's throw target |
| `power` | if `A_u` is THROW and `T_u` is THROW: use the teacher's power |
| `aim+power` | both of the above |
| `timing` | the unit throws exactly when the teacher throws. If `T_u` is THROW and `A_u` is not: throw using the learner's *own* throw-head target and power (deterministic mean). If `A_u` is THROW and `T_u` is not: adopt the teacher's whole action |
| `throw-all` | `timing`, then `aim+power` on the resulting throws |
| `move` | if `A_u` is MOVE and `T_u` is MOVE: use the teacher's move target |
| `all` | the teacher's action for every unit (a control; see below) |

Two limits of "aim": the throw *target point* encodes both which enemy is fired at and where, so target choice
and aim geometry are replaced together; and a replacement only happens where the teacher also throws or moves,
so the **coverage** (share of the learner's THROW / MOVE decisions that are replaceable) is recorded per arm and
seed. Low coverage would make an arm nearly inert, which is itself reported.

**Controls.**
- `none` must reproduce the S5 archived rows for these 100 worlds exactly (success, blue alive, final decision,
  final damage, first-hit decision). This shows that querying the teacher does not perturb the simulation. **It is
  a required gate**: if it fails, every arm is reported as unreliable.
- `all` (the teacher's action for every unit, executed through the same tensor path as every learner action) is run
  once per Red arm, `normal` and `random`, with seed 97101 (it does not depend on the learner). **Amendment
  (2026-09-20, before the real run):** a 50-world pre-run check of `all` on `normal` showed that the tensor-path teacher
  is *not* identical to the native scripted step the S4/S7 teacher used (mean 83 vs 76 decisions; damage per blue
  projectile 14.2 vs 16.9; it still succeeded in 50/50). Its ally spacing at first hit was also 5.97 against the native
  teacher's 4.77, close to the learners' 5.4–5.9. Execution path is therefore a confound in S7's comparisons of learners with
  the native teacher, and a ceiling on what any substitution arm can reach.
  **Consequences, fixed here:** (1) the recovery reference `yield_teacher` in section 2 is the **`all` arm's own
  yield on `normal`** (the ceiling of this path); the native teacher's 16.9 is reported alongside, not used. (2) The `all`
  arms are the like-for-like reference against which S7's non-yield comparisons (spacing, mobility, action mix,
  incapacitation, red shot rate) are re-read in S8's results, as an exploratory correction of S7, not a new
  prediction. (3) The 50-world check used only the `all` arm; no intervention arm was run before this amendment.

**Budget.** Bound 460,000 decisions (3 seeds × 7 arms × 100 worlds × 200, plus two `all` controls), hard cap 600,000 (aborts
unsealed). Teacher queries roughly double the cost per decision, so plan on about an hour of wall time; the run is
one background process that refuses to overwrite, waited on by PID.

## 2. Measures

Per cell, over the 100 worlds: success, units lost fraction `L`, team wipe, timeout, mean decisions (S4's
definitions); S7's per-episode measures from the raw traces, chiefly **damage per blue projectile** (`blueYield`),
red kills, first-death timing, blue projectiles per living unit-decision, and spacing. Recovery of the yield gap
per seed and arm:

`R = (yield_arm − yield_none) / (yield_teacher − yield_none)`

with the `all` arm on `normal` Red (100 worlds, the tensor-path teacher) as the reference. Arms are compared with `none` world by world
(paired), with a world bootstrap (10,000 resamples, seed 973001). Seeds are never pooled.

## 3. Predictions (fixed now; scored mechanically; all three seeds must satisfy each)

These are guesses about sufficiency and interactions, made with less confidence than the direction of S7's
finding, and S7's own eight all failed; I expect some to fail here too.

| # | Prediction | Reads as |
| --- | --- | --- |
| P1 | `aim`: `R ≥ 0.5` | where the throw lands (target point) carries most of the yield deficit |
| P2 | `power`: `R < 0.25` | throw power is not the main deficit |
| P3 | `timing`: `R < 0.25` | when to throw is not the main deficit |
| P4 | `throw-all`: `R ≥ 0.75` | the throw channel as a whole holds nearly all of it |
| P5 | `throw-all`: success ≥ 0.25 against normal Red | throw repair alone would clear S5's own viability threshold |
| P6 | `move`: success < 0.10 | positioning repair alone does not rescue the fight |

Success here counts under teacher assistance only and must never be quoted as an autonomous result.

## 4. What this can and cannot show

It shows which single channel's teacher replacement is *sufficient* to raise yield or success, given the learner's
other channels, on 100 worlds and three related seeds (one round-zero dataset). It does not show necessity, does
not separate target choice from aim geometry, and cannot tell whether a channel would be repaired by more data,
a different decoder or PPO. Sustained substitution shifts the visited-state distribution, and a teacher action is
the teacher's choice for a state the teacher itself would rarely reach; a null result for one channel therefore
does not clear it, and interactions between channels are only probed by `aim+power` and `throw-all`. Random Red is
covered only by the `all` control, not by intervention arms.

## 5. Implementation and archive

New module `options/roster_intervention.py` and tests; no existing module is edited. It reuses
`roster_trace.collect_traced` with a substitution chooser. The declaration pins the S4, S5 and S7 manifests, the
declaration digest and source digests. Sealed archive `runs/m8_s8_channel_intervention_v0/`; results in
`reviews/m8_s8_results.md`. Gates before collection: client and training `pytest`, `npm run build`, `npm test`
(366/367, the one documented exception). A failed gate or unheld prediction is a finding, not patched around.
