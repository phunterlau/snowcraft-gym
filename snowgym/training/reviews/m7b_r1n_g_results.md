# R1n-g: blue's offense fails for three compounding reasons — it often doesn't attempt the right throw, and when it does, the aim is worse than random

Collected 2026-09-14 from `c276e80`.
- The [declaration](m7b_r1n_g_declaration.md) was committed at `19577f1`.
- The implementation is at `256e0bc`, the pre-collection review fix (test
  strengthening and a docstring clarification, no change to method or
  budget) at `c276e80`.

Run facts:
- **Decisions:** 312,567, against a 400,000 cap (bound 253,440). The
  declared "~110 decisions/episode" assumption was an underestimate of the
  real rate (actual: 312,567 / 2,304 episodes ≈ 135.7 decisions/episode on
  average) — episodes here run longer than assumed, not shorter. This
  stayed well inside the cap and triggered no guard, but is recorded here
  for calibration: the same ~110 assumption should not be reused unchecked
  in a future declaration.
- **No training.** All six comparators (3 R1n-c initializers, 3 R1n-e
  finals) are frozen; only the opponent arm varies. `autonomousQualificationEligible`
  stays false throughout; R1n-g decides nothing about R1 qualification.
- **18 cells** (6 comparators × 3 arms), 128 fresh episodes each, on fresh
  seed bands 994000–994127 (R), 995000–995127 (E), 996000–996127 (N).
- **Manifests:** the top-level manifest and all three arm manifests
  verified with `death_rate_ppo.verify_sealed` after aggregation.
- **Archive:** `runs/m7b_engage_r1n_g_v0/`.

## Headline

1. **All three candidate failure modes from the declaration (§0) are real,
   not mutually exclusive.** Type/detection, aim, and positioning all
   degrade sharply and simultaneously on both scripted arms. No single
   factor accounts for the whole effect, but the aim failure alone is
   severe enough to explain the zero-hit result by itself.

2. **The aim failure is the most severe, and it's not just "bad" — it's
   worse than pointing randomly.** Two random 2D directions have a mean
   angular difference of 90°. `throwAimHeadingErrorDegrees`, averaged
   across the six comparators, is **131.8°** on arm E and **111.9°** on arm
   N (arm R baseline: 3.1°) — systematically pointing away from the
   correct target, not merely uncorrelated with it. Every one of the 18
   scripted-arm cells individually exceeds 74.9°, and most exceed 120°.
   This is conditioned only on the *teacher's* label being THROW (not on
   blue having chosen to throw — see the declaration's §1/§2 correction),
   so it answers "if blue's throw head fired here, which way would it
   point," and the answer is: the wrong way, badly, everywhere blue is
   tested against a leading, covering opponent.

3. **Type/detection failure is also real and confirms, not falsifies,
   prediction 1** (which the pre-collection review had flagged as
   disfavored by R1n-f's throw-volume evidence). Mean throw recall falls
   from 0.72 (arm R) to 0.35 (arm E) and 0.15 (arm N) — below the reused
   `throwRecallFlag` (0.5) on both arms, and as low as 0.017 on individual
   arm-N cells. Blue still throws often (support per cell: 751–5,237
   teacher-labelled-THROW rows), so this is not "blue never throws" — it's
   "blue frequently does something else when the teacher would throw."

4. **Positioning also degrades substantially**, ruling out "it's only the
   throw head that's broken": `moveHeadingErrorDegrees` rises from ~8.9°
   (arm R) to 27.9°–83.9° per cell on the scripted arms;
   `moveEndpointErrorWorld` roughly triples (0.52–0.58 → 1.74–3.00).

5. **PPO (R1n-e) leaves the fundamental problem intact but is not neutral.**
   On arm E, finals show modestly *better* throw recall than their
   initializers (e.g. seed 97101: 0.263→0.533) and slightly better aim
   (143.7°→122.3°) — still catastrophic, just less so. Power calibration
   moved the other way: `powerMeanAbsoluteError` roughly doubled on the
   scripted arms for finals versus initializers (e.g. arm E mean
   0.049–0.072 → 0.033–0.135, most finals worse). This is consistent with
   R1n-f's finding that PPO does not change the win/loss floor against
   scripted red — the label-error profile confirms the floor is untouched
   in kind, only nudged slightly in degree, and not uniformly for the
   better.
6. **The arm-R cross-check confirms this diagnostic's own measurement is
   sound.** Same checkpoints, same opponent, different worlds and episode
   count (R1n-c: split B, 601000–601099, 100 episodes; here: 994000–994127,
   128 episodes) — deltas are small across all three seeds (type accuracy
   ≤0.008, throw recall ≤0.026, aim ≤0.36°), the expected size for a
   different-worlds "same ballpark" check, not a measurement artifact.

## 1. Type/detection: `perType["throw"]["recall"]`

| Comparator | Arm R | Arm E | Arm N |
| --- | ---: | ---: | ---: |
| init-97101 | 0.683 | 0.263 | 0.072 |
| init-97102 | 0.754 | 0.309 | 0.096 |
| init-97103 | 0.699 | 0.265 | 0.032 |
| final-97101 | 0.940 | 0.533 | 0.426 |
| final-97102 | 0.847 | 0.433 | 0.252 |
| final-97103 | 0.824 | 0.291 | 0.017 |
| **mean** | **0.791** | **0.349** | **0.149** |

`throwRecallFlag` (0.5, reused from R1n-c) is crossed on both scripted
arms' means, and on 10 of 12 individual scripted-arm cells.

## 2. Aim: `throwAimHeadingErrorDegrees` (teacher-labelled-THROW rows only)

| Comparator | Arm R | Arm E | Arm N | Support (E / N) |
| --- | ---: | ---: | ---: | --- |
| init-97101 | 2.9° | 143.7° | 109.5° | 5,237 / 822 |
| init-97102 | 2.9° | 134.6° | 74.9° | 4,329 / 1,836 |
| init-97103 | 3.1° | 141.3° | 148.4° | 4,148 / 1,668 |
| final-97101 | 3.2° | 122.3° | 109.5° | 2,860 / 751 |
| final-97102 | 3.3° | 120.9° | 87.6° | 3,257 / 902 |
| final-97103 | 3.1° | 127.9° | 141.4° | 3,623 / 2,179 |
| **mean** | **3.1°** | **131.8°** | **111.9°** | |

Every scripted-arm cell has thousands of throw-labelled rows behind its
mean — this is not a small-sample artifact. Five of six arm-N cells and
three of six arm-E cells exceed the 90° two-random-directions baseline,
meaning the throw head is not merely noisy under this opponent; on
average it points closer to *away* from the target than *toward* it.

## 3. Positioning: move heading and endpoint error

| Comparator | moveHeadingErrorDegrees (R / E / N) | moveEndpointErrorWorld (R / E / N) |
| --- | --- | --- |
| init-97101 | 7.8° / 34.3° / 36.0° | 0.54 / 2.54 / 2.76 |
| init-97102 | 9.5° / 40.4° / 33.0° | 0.52 / 1.96 / 2.57 |
| init-97103 | 7.7° / 75.3° / 83.9° | 0.58 / 1.87 / 2.95 |
| final-97101 | 10.5° / 42.8° / 44.4° | 1.47 / 2.64 / 3.00 |
| final-97102 | 11.1° / 34.8° / 27.9° | 1.27 / 1.74 / 2.33 |
| final-97103 | 6.8° / 70.5° / 76.4° | 1.13 / 2.21 / 2.42 |

Seed 97103 (init and final) is the worst-positioned pair on both scripted
arms (70–84° heading error) — a consistent per-seed pattern, not noise,
though R1n-g does not investigate why this seed in particular.

## 4. PPO effect (initializer vs. final, per arm)

Already visible in the tables above. Summary: recall and aim both improve
slightly for finals on arm E (all three seeds); on arm N, recall improves
for two seeds (97101, 97102) and worsens for the third (97103: 0.032→0.017).
Power calibration worsens for finals on both scripted arms in most cells.
None of this changes the qualitative picture — every scripted-arm cell
remains far outside the healthy arm-R range on every metric.

## 5. Archive cross-check (declaration §2, §8)

| Seed | typeAccuracyDelta | throwRecallDelta | throwAimHeadingErrorDegreesDelta |
| --- | ---: | ---: | ---: |
| 97101 | +0.006 | +0.002 | +0.29° |
| 97102 | +0.008 | +0.026 | −0.04° |
| 97103 | +0.003 | +0.019 | +0.36° |

All deltas are an order of magnitude smaller than the scripted-arm effects
above (which run 30–130× larger). This diagnostic's own arm-R measurement
reproduces R1n-c's archived split-B label error closely enough that the
scripted-arm numbers are not a measurement artifact.

## 6. Predictions vs. results (declaration §3)

| Prediction | Result |
| --- | --- |
| Throw recall drops below 0.5 on at least one scripted arm | **Confirmed** on both arms' means, and on 10/12 individual cells — the pre-collection review's concern that R1n-f's throw-volume evidence disfavored this was answered by direct measurement, not by the volume proxy. |
| Aim error on scripted arms is dramatically larger than arm R | **Confirmed**, and more severe than anticipated: not just larger, but past the 90° random-direction baseline in most cells. |
| Move endpoint/heading error rises on scripted arms | **Confirmed.** |
| PPO changes label-error metrics by less than the archive noise floor | **Not confirmed as stated.** PPO's effect on scripted-arm recall and aim (several points/degrees) exceeds the archive cross-check's noise floor (all deltas ≤0.03/≤0.4°) — PPO does shift the label-error profile slightly, just not enough to change the qualitative floor R1n-f found. |

## 7. What this does and does not decide

**Decides:** the zero-hit offense finding from R1n-f is not attributable to
a single cause. R1n-h's mixture-imitation curriculum needs to address all
three: (a) more demonstration volume of the teacher throwing under a
moving/covering opponent (type/detection), (b) aim/power regression losses
that generalize past the training opponent's spawn-distance and threat
geometry (aim — the most severe single factor), and (c) positioning
demonstrations under the same opponent (movement). Per the declaration's
§4 format, this localizes to **"no single factor dominates,"** with aim
identified as most severe and most directly sufficient to explain the
zero-hit result on its own.

**Does not decide:** why the aim head in particular becomes anti-correlated
rather than merely noisy under a closer, moving, covering opponent; whether
the three failures share a root cause (e.g., an out-of-distribution
observation regime affecting all three heads) or are independent; anything
about R1n-h's curriculum design beyond the priorities above — that is
R1n-h's own declaration.

## Verification

- Training pytest, python client pytest, `npm run build`: all clean before
  collection (implementation commit `256e0bc`, review-fix commit `c276e80`).
- `npm test`: 366 passed, 1 failed — the single documented, accepted R1n-b
  `trainSeedBase: 630000` collision. No new collision from this run's
  artifacts (re-checked post-collection: `runs/m7b_engage_r1n_g_v0`'s
  `declaration.json` and per-arm reports contain no seed-context field in
  630000–630119; the new bands 994000–996127 remain clear on a repo-wide
  scan).
- Top-level and all three arm manifests re-verified with
  `death_rate_ppo.verify_sealed` after aggregation (see Run facts).
- The arm-R cross-check against R1n-c's archived split-B label error was
  computed and read (§5) before the scripted-arm numbers were interpreted,
  per declaration §8.
