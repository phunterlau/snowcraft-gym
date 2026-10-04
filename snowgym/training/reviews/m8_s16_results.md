# M8-S16 results: the frozen S14 fighters follow left/right target commands from reset

Collected 2026-10-04 from `dab172a` (declaration and implementation committed before running). Declaration:
[m8_s16_declaration.md](m8_s16_declaration.md). Archive: `runs/m8_s16_command_control_audit_v0/` (sealed and
verified, `sha256:0559e69a…`). 2,689,333 simulator decisions against the 5,068,800 bound and 5,100,000 cap:

| Part | Decisions |
| --- | ---: |
| Regression gate | 71,349 |
| Teacher | 127,077 |
| Learned | 2,490,907 |

No training. Development panel only (192 worlds, 2720000–2720191). The qualification panel 2730000–2730399 is
untouched. `autonomousQualificationEligible` stays false.

## Gates: both passed

- **Runner regression gate: exact.** In all 12 cells (3 cohorts × 2 policies × 2 modes), the new runner, tracker,
  per-world scenario plumbing and preview calls reproduced S14's archived first evaluation block exactly. This was on
  the default layout with `nearest` as both requested and execution plan.
- **Teacher positive control: passed with no exceptions.**
  - The requested flank fell before its mirror in all 384 correct-condition episodes, at every spread
    (`o` = 1.000; correct − other = +1.000).
  - 99% of its early-window throws were aimed at the requested Red.
  - Its centroid moved 7.3 units toward the requested side by decision 40.
  - It lost no units under the correct command.
  - Mirror-identity violations were 0.

## Headline

**All six learned policies are `controllable` under deterministic execution, the declared primary.** "Controllable"
is S16's declared rule: informative, with Δo = `o`(correct) − `o`(other) ≥ 0.40 and a lower 95% bound above 0.
PLAN.md's per-mission qualification gates were not evaluated. Stochastic execution, whose
sampling is independent per condition and task, agrees.

| Policy | det. ō(correct) | det. Δo [95%] | sto. ō(correct) / ō(other) | sto. Δo [95%] | informative |
| --- | ---: | --- | --- | --- | ---: |
| c1 final | 0.982 | +0.964 [+0.935, +0.987] | 0.835 / 0.164 | +0.671 [+0.616, +0.723] | 0.99 |
| c2 final | 0.964 | +0.927 [+0.898, +0.953] | 0.908 / 0.079 | +0.828 [+0.786, +0.867] | 0.95 |
| c3 final | 0.980 | +0.961 [+0.932, +0.984] | 0.944 / 0.069 | +0.875 [+0.837, +0.910] | 0.99 |
| c1 initializer | 0.781 | +0.562 [+0.487, +0.635] | 0.799 / 0.201 | +0.599 [+0.546, +0.652] | 0.76 |
| c2 initializer | 0.901 | +0.802 [+0.755, +0.846] | 0.911 / 0.068 | +0.844 [+0.806, +0.879] | 0.86 |
| c3 initializer | 0.747 | +0.495 [+0.427, +0.562] | 0.876 / 0.103 | +0.773 [+0.734, +0.811] | 0.68 |

Cohort-averaged (per-world mean over cohorts):

| Policy, mode | Δo [95%] |
| --- | --- |
| Final, deterministic | +0.951 [+0.932, +0.967] |
| Final, stochastic | +0.791 [+0.761, +0.820] |
| Initializer, deterministic | +0.620 [+0.577, +0.661] |
| Initializer, stochastic | +0.739 [+0.712, +0.765] |

**How to read the deterministic column.** As declared (§3), deterministic (other, r) replays (correct, m(r)).
The measured identity violations were **0.0 for every deterministic policy and the teacher**: no batch effect
flipped a single flank order. Deterministic Δo is therefore exactly 2·ō(correct) − 1. Deterministic "other",
"canonical" and "shuffled" are re-expressions of the correct cells and are not separate evidence. The independent
evidence is:
- ō(correct) against 0.5;
- the stochastic cells;
- the diagnostics below.

The early contrast `c` (co-primary) agrees in sign and strength in every policy (Δc +0.91 to +1.61, all intervals
far from 0). It carries a positive level bias under this stopping rule: deterministic canonical shows `c` = +0.01 to
+0.17 although its deterministic `o` is exactly 0.5. The bias arises because an episode whose requested flank falls second keeps accumulating
requested-target damage. Δc is still centred at 0 for a command-blind policy, so the contrast remains valid; its
level is not.

## Predictions (declaration §7): 2 of 4 held

1. **The teacher gate passes: held.**
2. **No learned policy is `controllable`: FAILED.** All six are.
3. **At least one learned policy is `sensitive` or `controllable`: held**, trivially, given 2.
4. **Under canonical, every informative learned policy finishes the centre Red first overall in ≥ 50% of episodes:
   FAILED.** Deterministic centre-first rates under canonical:

   | Policy | Centre first under canonical |
   | --- | ---: |
   | c1 final | 1.00 |
   | c2 final | 0.96 |
   | c3 final | 0.33 |
   | c1 / c2 / c3 initializer | 0.42 / 0.55 / 0.46 |

   It failed because of the c3 final and the c1 and c3 initializers.

Prediction 2 was mine. The review's §4 explicitly said that the enemy scorer lacking plan input "is not proof that the
whole actor is command-blind". I predicted command-blindness anyway, and the data contradict it.

## What the diagnostics show (descriptive; no mechanism is established)

**Evidence consistent with control running through movement.**
- **Lateral movement.** The command reaches the actor only as objective geometry (declaration §0). Final policies
  move their centroid 3.1–4.9 units toward the requested side by decision 40 under the correct command. The
  deterministic "other" value is exactly the negative of this, by the mirror identity; it is not separate
  corroboration.
- **Aim.** Early-window throws aimed at the requested Red are 0.63–0.69 of throws under correct vs 0.17–0.26 under
  other. Throws at the mirror Red are 0.04–0.07 vs 0.41–0.52.
- **Proposed path (untested).** Commanded objective geometry → movement toward that flank → the commanded enemy
  becomes the physically favored choice for a scorer that sees only physical features. This audit does not
  separate that path from any other.

**The zero-plan diagnostic is uninformative as a command ablation.** Zeroing `plan_groups` and `plan_role_state`
does not remove only the command: it **collapses the fighters physically**.
- Blue is wiped in 99–100% of deterministic zero-plan episodes, and in 64–100% of stochastic ones.
- Almost no flank crosses (informative fraction 0.00–0.02 deterministic).
- Its `o` is therefore 0.5 by default, and "correct − zero-plan" (≈ +0.25 to +0.48) measures that collapse, not a
  command effect.

Zero-plan does show that these plan channels are load-bearing for basic fighting. It is outside the production
contract and was always labelled a diagnostic.

**Option-level competence holds in the split layout, with the right command.** Losses and wipes here are measured
at the requested-singleton stop: one 100-health target brought to 20 or below. As with S14, this is option
completion, not a complete battle.
- **Finals, correct command.** The requested flank is finished in 0.94–0.99 of episodes, with units lost
  0.02–0.11 and team wipes 0.01–0.05. That is close to the teacher's 1.00 / 0.00 / 0.00.
- **Initializers, correct command.** They are much weaker: units lost 0.25–0.54, wipes 0.16–0.40.
- **Under canonical (the centre commanded).** The deterministic initializers are wiped in 57–75% of episodes; the
  finals in 0–3%.

Like S14/S15, PPO's gain is concentrated in deterministic execution. Stochastic initializers are closer to the
finals (correct `o` 0.80–0.91 vs 0.84–0.94).

**Control is not uniform:**
- **Spread.** The cohort 1 and cohort 3 deterministic initializers are insensitive at d = 10: Δo +0.109
  [−0.016, +0.234] and +0.016 [−0.086, +0.117]. They are strongly controllable at d = 20 and 30. The finals reach
  Δo = 1.000 at d = 10/20 in cohorts 2 and 3, and are lower at d = 30 (0.78–0.92; cohort 2 drops 22 points).
- **A centre command is not followed uniformly.** Under canonical, the execution plan commands the centre
  (`nearest`), and the teacher finishes it first in 100% of episodes. This reading uses the declared secondary
  `centreFirst`, the same data that failed prediction 4, and was broken down after seeing it:
  - **c1 and c2 finals** follow it (1.00 and 0.96). Under flank commands their centre-first rates fall to 0.30 and
    0.05, so for them the centre command does change behavior.
  - **c3 final** finishes the centre first in 0% of episodes at d = 10 and 20, and 100% at d = 30. Its
    centre-first rate is 0.33 under every condition, so the centre command appears to change nothing for it; spread
    geometry decides.
  - **Initializers** combine two failure modes under canonical: no crossing at all in 26–38% of episodes (mostly
    wipes), and a flank first. Among episodes with a crossing, centre-first is 0.63–0.74. At d = 10 it is 0.03–0.16.

  Left/right control therefore does not imply control over every grounded target.
- **Side bias.** Under deterministic canonical, `o` by side favors the right flank: 0.95 for the c2 final and 0.99
  for the c3 final, which goes to a flank first in two-thirds of episodes. The correct command overrides the bias on
  both sides (requested-side `o` 0.93–1.00). The c1 final is milder (0.66).
- **Cohort 2,** flagged in S15 for random-Red finishing, is still controllable (final Δo +0.927). It is, however, the
  weakest final on every measure under the correct command:

  | Measure, correct command | Cohort 2 final | Cohorts 1 and 3 finals |
  | --- | ---: | ---: |
  | Δo | +0.927 | +0.964 / +0.961 |
  | Requested success | 0.94 | 0.99 / 0.98 |
  | Units lost | 0.106 | 0.044 / 0.021 |
  | Team wipe | 0.05 | 0.01 / 0.01 |

  It also has the most horizon stops under the other command (36 of 384).

## What this establishes and what it does not

**Establishes (development panel, frozen policies, scripted-normal Red, reset starts):**
- In a mirrored three-singleton layout new to them, the S14 fighters, and to a lesser degree their S12
  initializers, change *which flank they finish first* according to a valid `leftmost`/`rightmost` CommandPlan.
- This holds at rates close to the plan-aware teacher, in all three cohorts, and in both execution modes.
- Their physical competence holds in that layout under the correct command.

**Does not establish:**
- That the result replicates on untouched worlds. The qualification panel is reserved for exactly that.
- Control from mid-fight states or under command switches.
- Control over other selectors (the centre command is not followed by the c3 final, nor reliably by the
  initializers), doctrine fields, missions, rosters, opponents or terrain.
- The mechanism. Movement-mediated selection is the leading hypothesis, but untested.
- Anything under local observations.
- Anything about the value of an LLM commander.

## Consequence

The declared recommendation for "any learned policy controllable" is to **replicate on the reserved qualification
panel (2730000–2730399) before any training**. That replication needs its own declaration, reusing this module
unchanged. The qualification panel can be used only once, so its declaration must fix its scope first: left/right only, or
also a centre-requested condition. A centre-requested condition has no mirror flank, so it needs a different metric,
designed on the development panel first. Further candidates follow from the gaps above. Each needs its own
declaration and none is authorized here:
- The centre-command gap: a three-way target audit with the centre as a requested target.
- Command switches from intermediate states (review §7).
- The fixed-plan mission forks (HOLD/WITHDRAW/ADVANCE) toward PLAN.md's per-mission gates.

The review's §6C residual target scorer is **not** motivated by this audit for the left/right axis.
