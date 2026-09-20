# M8-S4 results: 3v3 Engage teacher achievability and random-init floors

Collected 2026-09-20 from `78e29d9` (declaration and implementation committed before
collection). Declaration: [m8_s4_declaration.md](m8_s4_declaration.md). Archive:
`runs/m8_s4_roster_baseline_v0/` (23 artifacts, manifest `sha256:4e548a86…`, verified
with `verify_sealed`). Nothing was trained or selected;
`autonomousQualificationEligible` stays false.

Run facts: 417,097 simulator decisions against the 600,000 bound (cap 700,000). 400
world-paired seeds (2100000–2100399; floors use the first 100), three native Red arms.
Wall time was about 35 minutes for about 250 CPU-seconds of the Python process. The
declaration's expectation of "minutes" was wrong by roughly an order of magnitude; the
cause was not measured (no per-cell timing was archived), so it is left unexplained. S5
should plan its budget in wall-clock time, not only decisions.

## Gates (frozen before data)

| Gate | Rule | Result |
| --- | --- | --- |
| Teacher achievable at 3v3 | success ≥ 0.90 in each arm | **Pass**: easy 400/400, normal 399/400, random 400/400 |
| Floors show no free signal | random-init deterministic success ≤ 0.10 vs both scripted arms, all 3 init seeds | **Pass**: 0/100 in all six cells |

## Teacher (400 worlds per arm; Wilson 95% for success, world bootstrap for `L`)

| Red arm | Success | Team wipe | Timeout | Mean `L` | Mean decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 1.000 [0.990, 1.000] | 0.000 | 0.000 | 0.010 [0.005, 0.016] | 133 |
| scripted-easy | 1.000 [0.990, 1.000] | 0.000 | 0.000 | 0.013 [0.007, 0.020] | 76 |
| scripted-normal | 0.998 [0.986, 1.000] | 0.003 | 0.000 | 0.037 [0.026, 0.050] | 77 |

The single teacher failure is seed 2100342 against scripted-normal, a team wipe. The
teacher loses about 1–4% of its units per episode, so the 3v3 ceiling for `L` is close
to zero and its loss measure has room to discriminate a learner from it.
The teacher's low `L` is partly a consequence of finishing quickly (76-133 decisions, against
104-200 for the floors): fast completion means less exposure, the same structure behind
the R1n-i erratum. `L` comparisons in S5 must therefore be read next to success and
episode length, and a slower learner that loses one unit is not straightforwardly worse at
survival than a faster one.

## Random-initialized floors (100 worlds per cell; success is 0/100 everywhere, Wilson upper bound 0.037)

| Init seed | Mode | vs random: timeout / `L` | vs easy: wipe / timeout / `L` | vs normal: wipe / `L` |
| --- | --- | --- | --- | --- |
| 98101 | deterministic | 1.00 / 0.000 | 0.30 / 0.70 / 0.757 | 1.00 / 1.000 |
| 98101 | stochastic | 1.00 / 0.000 | 1.00 / 0.00 / 1.000 | 1.00 / 1.000 |
| 98102 | deterministic | 1.00 / 0.000 | 0.09 / 0.92 / 0.600 | 1.00 / 1.000 |
| 98102 | stochastic | 0.99 / 0.050 | 0.97 / 0.04 / 0.990 | 1.00 / 1.000 |
| 98103 | deterministic | 1.00 / 0.000 | 0.11 / 0.89 / 0.617 | 1.00 / 1.000 |
| 98103 | stochastic | 1.00 / 0.000 | 0.94 / 0.06 / 0.980 | 1.00 / 1.000 |

What the pattern shows, and its limits:

- **Against scripted-normal every random-init policy is wiped in every episode**
  (mean 104–153 decisions). There is no free success signal to bootstrap from.
- **Against random Red nothing happens.** Every random-init cell times out with
  `L` at or near 0, so blue never engages. Random Red alone cannot show a loss-metric
  improvement for an un-initialized learner; it becomes informative only for a policy that
  already reaches contact.
- **Deterministic and stochastic random-init policies differ against easy Red**
  (deterministic often times out with `L` 0.60–0.76, sampled is wiped with `L` ≥ 0.98).
  This is reported, not explained: no diagnostic here separates stalling from
  movement, and three initialization draws are not a distribution.
- The three init seeds are draws of one procedure, reported per seed and not pooled.

## What this establishes and what it does not

**Establishes:** at 3v3 the plan-conditioned teacher is achievable against all three Red
arms (≥ 0.998 success at 400 worlds); a random-init learner has no route to success
against scripted Red; and the 3v3 loss metric `L` is well-behaved across its full range
(about 0.01–0.04 for the teacher, 1.0 for wiped floors) with the definitions frozen
before the data.

**Does not establish:** that 3v3 Engage is learnable, that any initializer works, that
teacher imitation transfers from 1v1, anything about command-following or roles, or
anything about the critic. The teacher here is the native plan controller; its success
says the task is well-posed, not that a learned actor can reach it.

## Consequence for S5

From-scratch PPO at 3v3 has no signal (floors 0/100 against scripted Red, no contact
against random Red), so S5 follows R1n's recipe: teacher imitation as the initializer,
with `L` (mean, world-paired) primary and success co-primary as declared. Before reusing
R1n's critic warm-start gate, re-check it at 3v3: `assignedLivingFraction` is now graded
rather than binary and the R1n-d return-predictability ceiling was measured at roster 1.
