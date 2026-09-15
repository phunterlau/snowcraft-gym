# R1n-f — opponent-transfer evaluation of the R1n-e policies (no training)

Declared 2026-09-14, before any declared collection. This run trains nothing.
It evaluates fixed, archived checkpoints against opponents they never met.

The replication that R1n-e's decision rules recommend is **not** this run; it
moves to R1n-g.

## 0. Question, and what this run does not claim

**Question.** R1n-e's policies beat `RandomAgent` red, which aims where blue
stands at the instant of the throw. The archived post-hoc analysis
(`m7b_r1n_e_results.md` §7) showed the learned gain is evasive movement:
blue's displacement during a snowball's flight rose 0.50 → 1.37 units and
red's hit rate within 12 units fell 69.6% → 24.3%. `ScriptedAiAgent` leads its
aim by 0.18 s, seeks cover, strafes, and retreats. **How much of the R1n-e
policy's competence survives against it?**

**Claims.** None about R1 qualification, none about training. The comparator
policies are frozen archive artifacts. `autonomousQualificationEligible`
stays false.

**Why the scripted teacher is the primary comparator.** An exploratory probe
(§10) found both the R1n-c initializers and the R1n-e finals at 0 wins of 40
against scripted red, while the scripted blue teacher won 40 of 40. A
final-minus-initializer difference between two policies that both score zero
is uninformative, so the primary measure is the **gap to the teacher on the
same worlds**, and the failure modes are declared measures rather than
commentary.

## 1. Fixed inputs

- **Initializers:** `runs/m7b_engage_r1n_c_v0/seed-{97101,97102,97103}/fit-4.pt`,
  loaded exactly as R1n-e loaded them, including the σ×0.5 log-std change
  (deterministic execution ignores σ; the load path is kept identical so the
  arms are comparable to the R1n-e archive).
- **Finals:** `runs/m7b_engage_r1n_e_v0/policy-{97101,97102,97103}/update-200.pt`.
- **Teacher:** the existing scripted path, `make_wrapper(..., scripted=True)`
  with `choose=None`. No teacher code changes.
- **Mission:** frozen Engage, `teacher_option_plan("engage")`, horizon 200,
  10 Hz decisions, 1v1, exactly as R1n-c/d/e.
- **Digests:** the declaration records the SHA-256 of every checkpoint read,
  the R1n-c and R1n-e archive manifests, and the implementation files, and
  verifies the E3 pins.

## 2. Arms

The only change between arms is the scenario's red controller.

| Arm | `redController` | `redDifficulty` | Role |
| --- | --- | --- | --- |
| R | `random` | (unused) | The training opponent: within-run reference. |
| E | `scripted` | `easy` | Aim error ×4.5, throws 55% of the time, dodges poorly, never seeks cover. Partial failure: the probe showed timeouts, not deaths. |
| N | `scripted` | `normal` | Full `ScriptedAiAgent`: lead aim, cover, strafe, retreat. |

**`hard` is not run.** `normal` already floors every learned policy in the
probe; `hard` would buy a second copy of the same wall. Budget goes to
400-world resolution on three arms instead.

**How the opponent is varied.** `full_authority_train_v1.scenario` is a
module-level function that `run_block` calls per reset. R1n-f overrides it
within a context manager for the duration of an arm. The archived
implementation of `full_authority_train_v1.py` is not edited; its digest is
recorded and must still match.

## 3. Evaluation protocol

- **Worlds:** a fresh split F, **871000–871399** (400 worlds), identical for
  every arm and comparator, so every comparison is paired by world.
- **Execution:** deterministic (mean actions) only. Outcomes in the probe are
  saturated at 0 and 100; stochastic execution would add budget without
  changing what is measured. This is a declared narrowing of R1n-e's
  two-mode evaluation.
- **Comparators per arm:** 3 initializers, 3 finals, 1 teacher = 7 runs of
  400 episodes; 8,400 episodes in total.
- **Blocking:** 50 worlds per block, as R1n-e evaluated.

## 4. Measures

**Primary, per arm:**
- `successGapToTeacher` and `deathGapToTeacher`: seed-averaged
  (learned − teacher) differences on the same worlds, for the finals and
  (separately) for the initializers, each with a 400-world bootstrap 95%
  interval (10,000 resamples, seed 983001).

**Secondary, per arm:**
- `finalMinusInitializer`: the seed-averaged paired difference, the
  measure R1n-e's primary test used, reported with the floor clause in §5.
- **Failure-mode decomposition.** Every episode gets exactly one label:
  - `win` — mission success, blue alive;
  - `win-but-dead` — mission success, blue dead;
  - `death` — blue dead, no success;
  - `timeout-with-hits` — timed out, blue alive, at least one hit landed;
  - `timeout-no-hits` — timed out, blue alive, no hit landed;
  - `unresolved` — anything else (expected 0).
  `firstHitDecision is not None` defines "landed a hit", as in R1n-c.
- **Mechanism counters**, from per-decision logs: red projectiles spawned per
  episode, their median spawn distance, red's hit rate, blue's displacement
  during each red snowball's flight, blue projectiles per episode and blue's
  hit rate. Hits are attributed to projectiles the way §7 of the R1n-e
  results did; the fraction of health drops left unattributed is reported.

## 5. Decision rules (evaluated on arm N, then arm E)

Precedence, per scripted arm:

| Order | Condition | Outcome |
| --- | --- | --- |
| 1 | Teacher success < 0.80 on that arm | **invalid**: the arm is not a fair reference |
| 2 | Final success < 0.05 **and** initializer success < 0.05 | **no-transfer-floor** |
| 3 | `successGapToTeacher` (finals) ≤ −0.50 | **fails** |
| 4 | `successGapToTeacher` (finals) ≤ −0.10 | **partial** |
| 5 | otherwise | **transfers** |

- **The floor outcome is deliberately distinct** from a small
  final-minus-initializer difference. It says PPO's contribution is *not
  testable* on that arm, because the imitation policy it started from already
  fails there. It is not evidence that PPO harmed transfer.
- **Reported alongside, never changing the outcome:** the failure-mode table,
  the mechanism counters, arm R's rates as a within-run reference, and
  rejection rates.

**What each outcome recommends (authorizing nothing):**

| Outcome on arm N | Recommendation for the next declaration |
| --- | --- |
| transfers or partial | R1n-g replication as already recommended; opponent variety is a later concern. |
| no-transfer-floor | The generalization loss predates PPO. The next declaration should be about the *imitation* stage: train or fine-tune against a mixture of opponents, and hold out an opponent for evaluation. R1n-g replication on the random opponent alone would measure a narrower skill than intended. |
| fails | Same as floor, plus: the R1n-e policy is worse than its initializer against a new opponent, which would make the anchor and σ choices suspect. |
| invalid | Fix the arm before drawing conclusions. |

## 6. Predictions, stated for falsification

- **Arm R, finals:** success 0.88–0.96 and death 0.03–0.11, i.e. R1n-e's
  split-E numbers reproduced on fresh worlds.
- **Arm N:** every learned comparator below 0.05 success; teacher above 0.90;
  outcome `no-transfer-floor`.
- **Arm E:** learned success below 0.10, with `timeout-*` labels on at least
  25% of episodes; deaths lower than on arm N.
- **`finalMinusInitializer`** on both scripted arms within ±0.05.
- **Mechanism:** red's projectiles per episode fall by more than half from
  arm R to arm N (the scripted red only throws inside 9 units), and its
  median spawn distance drops from about 29 units to under 10.
- **Blue's own hit rate falls** on the scripted arms, because red moves with
  purpose rather than wandering.

## 7. Artifacts

Per arm, under `runs/m7b_engage_r1n_f_v0/arm-{R,E,N}/`:
- `{initializer,final}-{seed}/episodes.jsonl` and
  `trajectory-distances.npz`, plus `teacher/…`;
- `shots.npz` per comparator: per red and blue projectile, spawn decision,
  spawn distance, target speed, displacement during flight, and hit flag;
- `summary.json` per comparator.

At the run root: `declaration.json`, `report.json` (all measures, decision
rules, and the reproduction check), and a sealed SHA-256 manifest covering
every file, with per-arm manifests.

## 8. Budget and stopping

- **Bound:** 8,400 episodes × 200 decisions = 1,680,000 decisions, plus
  20,000 for the §9 reproduction check: **1,700,000**. Cap: **2,000,000**.
- The per-arm and total budgets are enforced in code; exceeding either aborts
  the run.
- Restarts follow R1n-e's A9: a crashed, unsealed directory is renamed
  `aborted-*` and kept; sealed directories are never rerun.

## 9. Verification before the implementation commit

- Python training tests, Python client tests, `npm run build`, and `npm test`
  (which may fail only on the accepted R1n-b seed-preflight entry).
- A test that the scenario override actually reaches the batch host: an
  episode collected under arm N must show red projectiles spawning only
  within `ENGAGE_RANGE`, and one under arm R must show the long-range
  profile.
- **Reproduction check, in code and recorded in `report.json`:** R1n-f's own
  collection loop (which must log per-decision state for the teacher arm,
  where no chooser hook exists) reruns 50 worlds of split E
  (870000–870049) for initializer 97101 under arm R and must reproduce the
  archived R1n-e episodes exactly: success, blue alive, and final decision.
  A mismatch stops the run before any arm is collected.
- `auditSeedDocuments` on the serialized configuration, plus a repo-wide
  numeric scan of the new bands.

## 10. Seeds, and disclosure of the exploratory probes

- **Worlds:** 871000–871399. **Reproduction check:** 870000–870049.
  **Bootstrap:** 983001. No training RNG is used; deterministic execution
  draws no samples.
- **Disclosed probes, run before this declaration** (exploratory, archived
  nowhere, and the reason the primary measure is the teacher gap):
  - 40 worlds, 870000–870039, deterministic: against scripted `easy` and
    `normal`, all six learned comparators won 0; the teacher won 40 of 40 on
    both; against random red the six scored 21–37 wins and the teacher 38.
  - 20 worlds, 870000–870019: red's throw profile differs by arm as expected
    (spawn distance about 29 units under random red, 8–9 under scripted),
    which is what established that the override reaches both wrappers.
  - These worlds are **not** reused as split F, apart from the declared
    50-world reproduction check, which compares only against already-archived
    outcomes.
