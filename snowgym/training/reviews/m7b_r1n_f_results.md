# R1n-f: the generalization loss predates PPO — R1n-c's imitation policies already lose to a leading opponent, and PPO does not change that

Collected 2026-09-14 from `491dfc0`.
- The [declaration](m7b_r1n_f_declaration.md) was committed at `3b7a98a`.
- The implementation is at `39116b7`, amendments A1–A3 at the same commit,
  and A4–A6 at `491dfc0`.

Run facts:
- **Attempts:** two. The first ran to completion (declare, reproduction
  check, all three arms, aggregate, all manifests verified) but its
  `declaration.json` inherited an unused `trainSeedBase: 630000` from
  `full_authority_train_v1.configuration()` that collided with R1n-b's
  seed-preflight band — a `npm test` failure unrelated to anything R1n-f
  measures (A6). It was discarded (never committed, never seen outside this
  repository) rather than archived, and the run was re-declared and fully
  re-collected after overriding the unused field. Every arm's contents are
  byte-identical between the two attempts (same manifest digests), which is
  expected: nothing that changed affects the simulation.
- **Decisions:** 1,085,699, against a 2,000,000 cap (bound 1,690,000, A1).
  Reproduction check 6,020 (twice, once per attempt); arms 342,521 (R),
  449,707 (E), 293,471 (N).
- **No training.** All seven comparators per arm (3 R1n-c initializers, 3
  R1n-e finals, the scripted teacher) are frozen; only the opponent varies.
- **Manifests:** the top-level manifest (`sha256:bab5d26c…`, 72 artifacts)
  and all three arm manifests were re-verified after aggregation.
- **Reproduction check (declaration §9):** 50 worlds of split E, initializer
  97101, arm R — 0 mismatches against the archived R1n-e episodes, both
  attempts.
- **Archive:** `runs/m7b_engage_r1n_f_v0/`.

R1n-f makes no R1 qualification claim and trains nothing.
`autonomousQualificationEligible` stays false throughout.

## Headline

1. **Every learned policy loses to the scripted opponent, at both
   difficulties, at full scale (400 worlds, not the 40-world probe).**

   | Arm | Opponent | Teacher success | Learned success (6 comparators) |
   | --- | --- | ---: | --- |
   | R | random (reference) | 92.0% | 53.8–94.8% |
   | E | scripted, easy | 100% | **0.0% (all six)** |
   | N | scripted, normal | 100% | **0.0% (all six)** |

   This is not a PPO effect: **the R1n-c imitation initializers fail
   identically to the R1n-e finals.** Both floor at zero against scripted
   red while the teacher they were cloned from wins every episode.
2. **PPO did not make the scripted-opponent failure worse, and on arm E it
   changed *how* the policies fail.** `finalMinusInitializerSuccess` is
   exactly 0 on both scripted arms — no policy in either state ever wins.
   But on arm E, deaths rose (`finalMinusInitializerDeath` **+5.5 points
   [+2.4, +8.6]**) as timeouts fell by the same amount: PPO trained the
   policies to engage more (as R1n-e's results already showed against
   random red), and against a leading, retreating opponent that same
   tendency converts passive timeouts into deaths without opening any path
   to a win.
3. **Blue's offense, not just its defense, fails completely against
   scripted red.** Pooled across all six learned comparators:

   | Arm | Blue throws | Blue hits | Blue hit rate |
   | --- | ---: | ---: | ---: |
   | R (random red) | 11,968 | 8,413 | 70.3% |
   | E (scripted easy) | 25,008 | **0** | **0.0%** |
   | N (scripted normal) | 5,488 | 20 | 0.4% |

   Against random red, blue's own throws land at close to the teacher's
   rate (70.3% vs 88.9%). Against scripted red, blue almost never lands a
   hit at all — not even once in 25,008 thrown snowballs on the easy arm.
4. **The mechanism is opponent engagement range, not policy skill.** Red's
   shots spawn from a median 27–30 units under random red and 7–9 units
   under scripted red — inside `ENGAGE_RANGE = 9`, the hard cap coded into
   `AISystem.attack`. Scripted red also throws far less often (0.4–7.5 shots
   per episode versus 15–16 under random red): most of an episode is spent
   maneuvering, not exchanging fire, and blue's policies — trained
   exclusively against an opponent that throws constantly from long range —
   have no exposure to that regime.
5. **Arm R (random red) closely reproduces R1n-e's archived split-E
   results on an independent split.** Finals: 91.25–94.75% success,
   4.5–8.75% death (R1n-e archive: 92.0–94.3% success, 5.0–7.8% death).
   `referenceArmReproducesR1nERange` is true. This is the expected
   consistency check, not a new finding.
6. **Projectile attribution stayed exact.** Every red and blue health drop
   in every arm and comparator matched to a specific thrown projectile
   (`redDeathAttributedFraction` / `blueDeathAttributedFraction` = 1.0
   wherever the denominator was nonzero) — the same standard set by the
   R1n-e results §7 appendix, now a declared, tested measure (`shots.npz`
   per comparator) rather than a one-off script.

## 1. Arm R — random red (reproduction reference)

| Comparator | Success | Death | Timeout |
| --- | ---: | ---: | ---: |
| Teacher | 92.0% | 8.25% | 0.0% |
| init-97101 | 53.75% | 46.75% | 0.25% |
| init-97102 | 78.0% | 22.25% | 0.0% |
| init-97103 | 71.0% | 29.5% | 0.0% |
| final-97101 | 94.75% | 4.5% | 0.75% |
| final-97102 | 91.25% | 8.75% | 0.25% |
| final-97103 | 91.75% | 7.75% | 1.0% |

- **Seed-averaged gaps to the teacher (world-bootstrapped, A5):**
  - initializer success: −24.4 points [−28.3, −20.5];
  - final success: +0.6 points [−2.8, +4.0] — statistically indistinguishable
    from the teacher;
  - final death: −1.3 points [−4.7, +2.0].
- **Seed-averaged final − initializer:** success +25.0 points [+22.0,
  +28.0]; death −25.8 points [−28.8, −22.8]. Consistent with R1n-e's
  archived −24.8-point result, on an independent 400-world split and a
  different bootstrap axis than R1n-e used (world-paired here, per-policy
  paired there — both report the same effect).
- **Mechanism:** red throws 14.6–16.0 per episode from a median 27.4–30.3
  units; blue throws 3.87–5.61 per episode from close range, hitting
  69.9–73.6% (initializers) and 68.0–70.0% (finals) — both below the
  teacher's 88.9%, an unclosed gap R1n-f did not set out to explain.
- **Failure-mode labels, pooled over the three finals:** 83 deaths, 5
  timeouts without a hit, 1 timeout with a hit, 1 win-but-dead, 1110 wins,
  0 unresolved, out of 1200 episodes.

## 2. Arm E — scripted red, easy

| Comparator | Success | Death | Timeout (no hits) |
| --- | ---: | ---: | ---: |
| Teacher | 100% | 0.0% | 0.0% |
| init-97101 | 0.0% | 59.5% | 40.5% |
| init-97102 | 0.0% | 75.0% | 25.0% |
| init-97103 | 0.0% | 52.0% | 48.0% |
| final-97101 | 0.0% | 76.25% | 23.75% |
| final-97102 | 0.0% | 84.5% | 15.5% |
| final-97103 | 0.0% | 42.25% | 57.75% |

- **Outcome: `no-transfer-floor-death`** (declaration §5 order 2, amendment
  A4). Both `initializerSuccessMean` and `finalSuccessMean` are 0, below
  the 0.05 floor.
- **Every recorded timeout on this arm is `timeout-no-hits`** — across all
  six comparators and 2,400 episodes, a policy that survives to the horizon
  without dying also never lands a single hit. `timeout-with-hits` is 0
  everywhere on this arm.
- **Failure-mode labels, pooled over the finals:** 812 deaths, 388 timeouts
  without a hit, 0 with a hit, 0 wins, 0 unresolved.
- **Mechanism:** red throws only 9.2–11.5 per episode (far below arm R's
  15+) from a median 6.8–7.9 units; its hit rate is 37–47%, comparable to
  its rate against the teacher (34%) and far below its rate against blue in
  the teacher's hands (34% — red is not tuned harder here, blue is simply
  worse at avoiding it). Blue throws 8.2–12.6 times per episode and lands
  **zero** of 25,008.

## 3. Arm N — scripted red, normal

| Comparator | Success | Death |
| --- | ---: | ---: |
| Teacher | 100% | 0.0% |
| init-97101 | 0.0% | 100% |
| init-97102 | 0.0% | 100% |
| init-97103 | 0.0% | 100% |
| final-97101 | 0.0% | 100% |
| final-97102 | 0.0% | 100% |
| final-97103 | 0.0% | 100% |

- **Outcome: `no-transfer-floor-death`.** Every one of 2,400 learned
  episodes (6 comparators × 400 worlds) ends in death; timeouts are 0.
- **Mechanism:** red's hit rate is 57–86%, comparable to its rate against
  the teacher (99%, at 1.0 throws/episode — the teacher rarely needs more
  than the minimum). Blue throws only 1.1–3.6 times per episode, roughly a
  third of arm E's rate, and lands 20 of 5,488 (0.36%) — **all 20 belong to
  seed 97102** (10 from the initializer, 10 from the final); 97101 and
  97103 land zero, in both their pre- and post-PPO states.

## 4. Decision rules and recommendations

| Arm | Outcome | Recommendation |
| --- | --- | --- |
| E | `no-transfer-floor-death` | The generalization loss predates PPO. Address the imitation stage: train against a mixture of opponents, holding one out for evaluation. A death-dominant floor has no learnable gradient at this difficulty; consider starting the mixture curriculum from an easier opponent. |
| N | `no-transfer-floor-death` | Same. |

- **Order 1 (invalid) does not apply to either arm:** the teacher scores
  100% on both, well above the 0.80 minimum.
- **Order 2 (floor) applies to both,** and amendment A4's failure-mode split
  resolves the same on both arms: death dominates the pooled final labels
  (812 vs 388 on E; 1,200 vs 0 on N) even though the 40-world probe that
  motivated the declaration suggested arm E would be timeout-dominant. That
  probe was directionally right (timeouts are far more common on E than on
  N) but wrong about which failure mode is the plurality at full scale —
  exactly the kind of thing a 40-episode probe is not powered to resolve,
  and why the declared measure runs the full 400.
- **Checks:** `referenceArmReproducesR1nERange` true; no `unresolved`
  episodes in any arm or comparator; projectile attribution exact
  throughout (§Headline 6).

## 5. Declared predictions

| Prediction (§6) | Result |
| --- | --- |
| Arm R finals: success 0.88–0.96, death 0.03–0.11 | Met: 0.9125–0.9475, 0.045–0.0875. |
| Arm N: every learned comparator below 0.05 success; teacher above 0.90; outcome `no-transfer-floor` | Met, with the A4 refinement: `no-transfer-floor-death`. |
| Arm E: learned success below 0.10, with `timeout-*` labels on at least 25% of episodes | Met: 0% success; timeouts are 15.5–57.75% per comparator, all `timeout-no-hits`. |
| `finalMinusInitializer` on both scripted arms within ±0.05 (success) | Met for success (exactly 0 on both). Not stated for death: death shifted +5.5 points on arm E, outside a ±5-point band, though the prediction as declared was about success only. |
| Red's projectiles per episode fall by more than half from arm R to arm N, median spawn distance from ~29 to under 10 | Met: 14.6–16.0 → 5.8–8.8 projectiles/episode; 27.4–30.3 → 8.5–8.7 units. |
| Blue's own hit rate falls on the scripted arms | Met, far beyond what was anticipated: 68–74% (arm R) → 0.0% (arm E) → 0.0–1.0% (arm N), not a partial decline. |

## 6. What this does and does not decide

**Does show:**
- The competence R1n-e measured against `RandomAgent` does not transfer to
  `ScriptedAiAgent` at any tested difficulty, for either the pre-PPO
  imitation policies or the post-PPO finals.
- The failure is present in the imitation stage (R1n-c) and PPO (R1n-e)
  neither fixes nor worsens it in aggregate, though it does shift arm E's
  failure composition from timeout toward death.
- The failure is bilateral: blue's defense (evasion, tuned to a no-lead
  thrower) and blue's offense (throwing accurately at short range against a
  moving, covering target) both collapse against the scripted opponent.
- The scripted opponent's own behavior — short engagement range, low throw
  frequency, cover-seeking — is well outside anything in blue's training
  distribution, which is the most direct explanation available without a
  further declared experiment.

**Does not show:**
- *Why* blue's offense specifically fails (aim, target selection, or
  positioning that never brings it into a good throw angle) — the mechanism
  counters localize the failure to zero landed hits, not its cause.
- Whether a mixture-opponent imitation curriculum (the recommended next
  step) would close this gap, or by how much.
- Anything about `hard` scripted difficulty, which was not run (§2 of the
  declaration; `normal` already floors every comparator).
- Any R1 qualification: R1n-f trains nothing and makes no such claim.

## Verification

- **Gate at `39116b7` and `491dfc0`, before collection:** training 408
  tests passed (18 new); Python client and build passed; `npm test`
  366/367 with only the accepted R1n-b preflight failure.
- **Reproduction check (§9),** both attempts: 50 worlds, 0 mismatches
  against the archived R1n-e episodes, confirming `collect_logged`'s
  reimplementation of the stepping loop does not perturb the RNG draw
  order.
- **After collection:**
  - the top-level manifest and all three arm manifests were re-verified:
    every artifact digest and the inventory;
  - `declaration.json` records `gitCommit 491dfc0`;
  - the implementation, `full_authority_train_v1.py`, and
    `death_rate_ppo.py` digests match the committed files; the E3 pins
    match; the R1n-c source archive (95 artifacts) and the R1n-e final-run
    manifest both verified;
  - `auditSeedDocuments` found no collision in any new JSON file (after
    A6's fix; the discarded first attempt's collision is documented in the
    declaration, not in this archive);
  - `npm test` reports only the accepted failure.
- **Recomputed from the archive:** all tables and figures in this document
  were taken from `report.json`, the per-arm `arm-report.json` files, and
  the per-comparator `summary.json`/`shots.npz` files, not from intermediate
  probe output.
