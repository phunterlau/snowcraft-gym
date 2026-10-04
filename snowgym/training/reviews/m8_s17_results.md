# M8-S17 results: S16's left/right command control replicates on the untouched qualification panel

Collected 2026-10-04 from `cc13f65` (declaration and implementation committed before any call on either panel).
Declaration: [m8_s17_declaration.md](m8_s17_declaration.md). Archive: `runs/m8_s17_command_control_qualification_v0/`
(sealed and verified, `sha256:3445c618…`). It pins S16's sealed implementation digest, which matched, and the S12,
S14, S15 and S16 manifests. 3,246,138 simulator decisions against the 6,393,600 bound and 6,400,000 cap:

| Part | Decisions |
| --- | ---: |
| Regression gate | 71,349 |
| Teacher | 193,067 |
| Learned | 2,981,722 |

No training. `autonomousQualificationEligible` stays false.

**Scope note, repeated from the declaration:** left/right only. That was my choice; the user said "proceed" without
choosing between left/right and adding a centre-requested condition.

**Reach, repeated from the declaration:** the qualification panel holds fresh draws of Red's random numbers on the
three spreads S16 already used (d = 10/20/30). The verdict shows robustness to that draw, not geometric
generalization. Only the held-out-geometry secondary (d = 15/25) speaks to geometry. Both of its spreads lie *between* the
development spreads, so it tests interpolation only. Nothing below d = 10 or above d = 30 was tested.

## Gates

- **Regression gate:** exact in all 12 cells.
- **Teacher:** ō(correct) = 1.000 on both panels and at every spread, mirror-identity violations 0, requested flank
  finished in every episode, no units lost. The held-out secondary is therefore valid.

## Verdict: `replicated`

All three S14 finals are `controllable` on the qualification panel under deterministic execution, by S16's rule.
Deterministic mirror-identity violations were 0.0 for every policy, so deterministic Δo = 2·ō(correct) − 1 exactly.

| Final | ō(correct) | Δo [95%] | S16 dev Δo | requested success | units lost | wipe |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| c1 | 0.969 | +0.939 [+0.915, +0.960] | +0.964 | 0.99 | 0.055 | 0.01 |
| c2 | 0.958 | +0.916 [+0.894, +0.938] | +0.927 | 0.94 | 0.109 | 0.06 |
| c3 | 0.973 | +0.945 [+0.922, +0.965] | +0.961 | 0.99 | 0.025 | 0.01 |
| cohort-averaged | — | +0.933 [+0.920, +0.947] | +0.951 | — | — | — |

Every replicated value is slightly below its development value, by 0.011 to 0.025.

**Outside the verdict, qualification panel:**
- **Stochastic finals:** Δo +0.661 / +0.848 / +0.865, all `controllable`.
- **Initializers:** all `controllable`. Deterministic Δo is +0.589 / +0.839 / +0.529; stochastic is +0.585 / +0.818 /
  +0.804.
- **Cohort 2 final, shown separately (S15):**
  - **Spread.** Δo = 1.000 at d = 10 and 20, but drops to +0.748 at d = 30, the largest per-spread drop of any final
    (S16: +0.781).
  - **Option-level outcomes** (losses at the requested-singleton stop, not a complete battle). Highest units lost
    (0.109) and wipes (0.06) of the finals under the correct command.

## Predictions (declaration §6): 3 of 5 held

1. **The teacher gate passes: held.**
2. **Verdict `replicated`: held.**
3. **The cohort-averaged final deterministic Δo lies within ±0.10 of S16's +0.951: held** (+0.933).
4. **The cohort 1 and 3 initializers are insensitive at d = 10 (interval contains 0): FAILED.**
   - Cohort 1: +0.302 [+0.220, +0.377].
   - Cohort 3: +0.164 [+0.078, +0.246].

   Both remain much weaker at d = 10 than at d = 20/30 (+0.69 to +0.74). The pattern that replicates is "weak at the
   narrowest spread", not "insensitive". S16's d = 10 intervals (+0.109 and +0.016, both including 0) reflect
   sampling variation: S16 had 64 worlds per spread, this panel 134.
5. **Secondary: all three finals `controllable` on the held-out spreads: FAILED.** The c1 final is `sensitive`, not
   controllable (next section).

## Held-out geometry secondary (d = 15, 25; outside the verdict)

| Final, deterministic | Class | ō(correct) | Δo [95%] | d15 / d25 Δo | requested success | units lost | wipe |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| c1 | **sensitive** | 0.693 | +0.385 [+0.325, +0.445] | +0.270 / +0.500 | 0.84 | 0.239 | 0.15 |
| c2 | controllable | 1.000 | +1.000 [+1.000, +1.000] | +1.000 / +1.000 | 1.00 | 0.002 | 0.00 |
| c3 | controllable | 1.000 | +1.000 [+1.000, +1.000] | +1.000 / +1.000 | 1.00 | 0.083 | 0.00 |

**The c1 final's held-out weakness is one-sided.**
- **By command.** On the held-out spreads, its deterministic correct-condition mean flank order is 1.00 for
  `rightmost` but only 0.39 for `leftmost`.
- **Under `leftmost`, episode by episode** (200 episodes):

  | Outcome | Episodes |
  | --- | ---: |
  | Right (mirror) flank finished first | 106 (53%) |
  | Left (requested) flank finished first | 60 (30%) |
  | No flank crossed before the stop | 34 (17%), scored 0.5 |

  62 of these 200 episodes ended in a blue wipe.
- **Not a smooth distance effect.** It fails at d = 15, which lies between d = 10 and d = 20, where the same policy
  scores ≥ 0.94 on the qualification panel.
- **On the qualification panel** the same policy follows `leftmost` at 0.94.
- **Other evidence.** Its physical outcomes also degrade on the held-out spreads. Its stochastic version stays
  `controllable` there (+0.704).
- **The c1 initializer, deterministic,** follows the command better on the held-out panel (+0.637, controllable)
  but is physically worse: requested success 0.70 vs 0.84, units lost 0.404 vs 0.239, wipes 0.30 vs 0.15.

This is a post-hoc description of one policy on one secondary panel. No mechanism is tested.

All initializers and all stochastic policies are `controllable` on the held-out spreads. Cohort-averaged held-out
deterministic final Δo is +0.795 [+0.775, +0.815], lowered by c1.

## What this establishes and what it does not

**Establishes:**
- S16's primary finding replicates on the untouched qualification panel, at nearly the development effect size, in
  all three cohorts. On these three geometries, the frozen S14 finals reliably finish the commanded flank first when
  commanded `leftmost` or `rightmost`, against fresh draws of scripted-normal Red.
- On two unseen but interpolated spreads, two of the three finals follow the command in every episode. The c1
  final does not reliably follow `leftmost` there.

**Does not establish:**
- Geometric generalization in general. Two held-out spreads; one final fails one side there.
- The centre axis or other selectors.
- Mid-fight switches, other doctrine fields, missions, rosters, opponents, terrain or local observations.
- A mechanism.
- PLAN.md's per-mission qualification gates.

## Consequence

Per the declaration, `replicated` means the left/right command axis is ready to be built on. The c1 held-out
asymmetry is a reason to keep reporting per-cohort, per-side results rather than a single averaged number. Next
candidates (each needs the user's call and its own declaration):
- **Mid-fight command switches** (review §7): the commander's actual use case.
- **The centre/three-way target axis**, which needs a no-mirror metric designed on development data.
- **The PLAN.md mission forks** (HOLD/WITHDRAW/ADVANCE).
