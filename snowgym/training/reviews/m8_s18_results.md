# M8-S18 results: the frozen S14 fighters only partly redirect at mid-fight (k = 60), one of three fully redirects at contact onset, and re-issuing the same command perturbs two of them

Collected 2026-10-04 from `cccd1af` (declaration and implementation committed before any learned-policy call on the
panel). Declaration: [m8_s18_declaration.md](m8_s18_declaration.md). Archive: `runs/m8_s18_command_switch_audit_v0/`
(sealed and verified, `sha256:1d7ddbe1…`). 6,714,507 simulator decisions against the 10,444,800 bound and 10,450,000
cap:

| Part | Decisions |
| --- | ---: |
| Replay gate | 199,051 |
| Teacher | 396,917 |
| Learned | 6,118,539 |

**Rows are not in git.** Per-episode rows (`*.jsonl`, about 49 MB) are kept locally under the archive path, by
user decision (2026-10-04, `runs/.gitignore`). The tracked archive holds `declaration.json`, `manifest.json` (which
still lists every row's digest) and `report.json`. Without the local rows, `verify_sealed` cannot pass on a fresh
clone.

No training. Development panel 2750000–2750191, spreads 20/30. The reserved panel 2760000–2760399 is untouched.
`autonomousQualificationEligible` stays false.

## Gates and checks

- **Replay gate: exact.** In all 12 cells the new runner, in S16 mode, reproduced S16's archived rows.
- **Activation assertions:** never fired. Groundings matched previews, plan versions incremented, and plan tensors
  were refreshed.
- **Teacher: valid at every k.** Δ`c_post` (switch − reactivate) was +1.956 [+1.935, +1.974] at k = 30,
  +1.547 [+1.493, +1.598] at k = 50 and +1.640 [+1.569, +1.709] at k = 60. Re-activating the same command left
  every teacher episode identical to keep: re-activation side effect exactly 0.
- **Prefix identity: holds exactly for the teacher and every deterministic cell. It FAILED for every stochastic
  cell in blocks 2–3.** This is my implementation error. `learned_cells` sets the shared torch seed once per cell,
  but a cell spans three 64-world blocks. After block 1 each branch has consumed a different number of random draws,
  so blocks 2 and 3 start from different random states.
  - **Scope:** exactly 512 mismatches per stochastic policy and k (blocks 2–3 × 2 tasks × 2 branches × 64 worlds),
    and none in block 1.
  - **Consequence:** the stochastic contrasts in `report.json` pair unequal prefixes and apply the keep branch's
    eligibility to the other branches. **They are invalid and are not reported as results here.** Do not cite the
    report's stochastic classes or contrasts (for example "c3-final-stochastic: redirects").
  - **What survives:** the declared primary is deterministic, and its prefix identity holds, so the classification
    below stands.
  - **Fix for any follow-up:** seed per block.

The teacher's eligible set at k = 60 is 96 worlds, all at d = 30 and all with initial command `leftmost`. In the
other (task, spread) combinations a flank was already damaged by k = 60.

## Primary: deterministic switch − reactivate (target-specific redirect)

Thresholds for `redirects` are 0.5·T_k: 0.98 at k = 30, 0.77 at k = 50 and 0.82 at k = 60. T_k comes from the
teacher's own eligible set, so it is a scale reference, not a like-for-like comparison.

| Final, det. | k = 30 Δc_post | k = 50 Δc_post | k = 60 Δc_post | Classes (30 / 50 / 60) |
| --- | --- | --- | --- | --- |
| c1 | +0.842 [+0.728, +0.958] | +0.127 [+0.053, +0.202] | +0.278 [+0.180, +0.378] | partial / partial / partial |
| c2 | +0.688 [+0.635, +0.737] | +0.676 [+0.607, +0.743] | +0.220 [+0.095, +0.339] | partial / partial / partial |
| c3 | +1.617 [+1.560, +1.674] | +0.987 [+0.913, +1.059] | +0.321 [+0.235, +0.404] | **redirects / redirects** / partial |
| cohort-averaged | +1.049 [+0.986, +1.113] | +0.597 [+0.554, +0.639] | +0.286 [+0.227, +0.343] | — |

**Every final's target-specific redirect is positive at every k.** For c2, c3 and the cohort average it weakens
with lateness. c1 does not follow that pattern: +0.842, +0.127, +0.278, dipping at k = 50 and recovering at k = 60.
- At k = 60 (mid-fight), all three finals are only `partial`, at +0.22 to +0.32. The teacher's value at k = 60 is
  +1.640, but its eligible set (d = 30, `leftmost` start only) differs from the finals', so the two are not
  like-for-like.
- At k = 50 (contact onset), only c3 redirects.
- At k = 30 (pre-contact), only c3 redirects; c1 and c2 are `partial`.

Flank order agrees in sign: cohort-averaged Δo is +0.745 at k = 30, +0.478 at k = 50 and +0.212 at k = 60.

**k = 60 eligibility.** The finals have 96 eligible worlds each at k = 60, all at d = 30. At d = 20 a flank was
already hit by then. Mid-fight conclusions therefore cover only the wider spread.

**Initializers** are deterministic-fragile in two-flank engagements: units lost 0.69–1.00 and wipes 0.52–1.00 under
keep or reactivate. Their redirect classes are `partial` or `redirects` with positive intervals throughout. They are
secondary and not interpreted further.

## The re-activation side effect is large for c1 and c2, small for c3 (teacher: exactly 0)

Re-issuing the **same** command at k changes exactly two of the actor's plan inputs: `planAge` (`plan_groups`
offset 37) and `activationDisplacement` (`plan_role_state` offsets 13–14). This was verified after the run by
diffing every plan-tensor offset before and after a same-plan re-activation at k = 50 (teacher-driven, dev seeds
5986000–5986003). No other plan key changed.

The effect of that change:
- **c1 and c2:** at k = 50/60 it shifts fire and flank order toward the *other* flank, B.
- **c3:** small at k = 50/60, except a shift of fire toward A at k = 30.
- **Direction at k = 30 is mixed across metrics.** Δc_post is negative for all three finals (fire moves toward A),
  while c1's and c2's flank order moves toward B.

Reactivate − keep, deterministic finals:

| Final | Δc_post k = 30 / 50 / 60 | Δo k = 30 / 50 / 60 |
| --- | --- | --- |
| c1 | −0.05 / **+0.79** / **+0.56** | +0.38 / +0.59 / +0.58 |
| c2 | −0.14 / +0.30 / +0.34 | +0.13 / +0.17 / +0.25 |
| c3 | −0.39 / +0.02 / +0.07 | −0.02 / +0.06 / −0.01 |

Consequently, much of the **switch − keep** effect is re-commanding rather than target-following. Switch − keep is
the practical comparison for a commander deciding between switching and doing nothing. In flank order (Δo) at k = 60
it is +0.83 for c1, +0.44 for c2 and +0.16 for c3. c1's re-activation alone produces +0.58 of that, and c3, whose
side effect is small, shows the smallest total.

**This matters for the commander design:**
- **Re-issuing is not neutral.** Re-sending an unchanged plan to c1 and c2 is a behavioral intervention, not a
  no-op.
- **Keep-the-plan baselines** must therefore not re-activate.
- **Training implications.** Any switch-trained continuation should include re-activations and randomized activation
  times, so that `planAge` is not tied to the reset.

## Cost of switching (units lost and wipes)

The stopping rule (both flanks crossed) is branch-independent, so losses are comparable across branches.

| | Units lost, switch − reactivate | Wipe, switch − reactivate |
| --- | --- | --- |
| Teacher, k = 50 | +0.415 [+0.385, +0.446] | +0.177 |
| Teacher, k = 60 | +0.812 [+0.760, +0.861] | +0.854 |
| Finals, det., k = 60 | −0.193 / −0.139 / −0.175 | — |
| Cohort-averaged finals, k = 60 | −0.165 [−0.195, −0.134] | — |

- **The teacher pays heavily for a late switch.** At k = 60 it commits fully to crossing the field to the new flank
  and is mostly wiped doing so.
- **The finals lose fewer units after a switch than after a re-activation** at k = 60. They redirect only partially.
- **Why is not established.** Not committing to the costly cross-field move is one reading; re-activation's own
  disruption is another, since c1's keep branch loses 0.40 vs 0.27 for reactivate and 0.07 for switch. The design
  cannot separate them.

## Post-hoc, valid subset: stochastic, block 1 only (64 worlds, prefix identity verified)

This is not declared; it is reported only because block 1's prefixes are exactly shared.

| Policy, sto. | Δc_post k = 30 | k = 50 | k = 60 (eligible worlds) |
| --- | --- | --- | --- |
| c1 final | +0.81 [+0.68, +0.94] | +0.40 [+0.24, +0.55] | +0.35 [+0.10, +0.60] (37) |
| c2 final | +0.69 [+0.56, +0.83] | +0.59 [+0.46, +0.73] | +0.49 [+0.30, +0.68] (35) |
| c3 final | +1.34 [+1.23, +1.44] | +0.78 [+0.66, +0.91] | +0.42 [+0.16, +0.66] (29) |

Every block-1 stochastic redirect is positive. c2 and c3 weaken with lateness; c1 drops from k = 30 to k = 50 and
then holds. The k = 60 entries are below the declared 48-world minimum.

## Predictions (declaration §8): 2 of 5 held

1. **The teacher gate passes at all three k: held.**
2. **All three finals `redirect` at k = 30: FAILED.** Only c3 does; c1 and c2 are `partial`.
3. **Not all three finals `redirect` at k = 60: held.** None do.
4. **The re-activation side effect is small (|Δc| < 0.20): FAILED.** c1 is +0.79 at k = 50, and others are large.
5. **Switching costs units at k = 60 for the averaged finals: FAILED.** The point estimate is −0.165: switching cost
   fewer units than re-activating.

## What this establishes and what it does not

**Establishes:**
- **Redirect is partial.** On the development panel, the frozen S14 finals redirect toward a newly commanded flank
  after a fixed-time switch. The effect is target-specific beyond re-activation and positive at every k. For c2, c3
  and the cohort average it weakens with lateness; c1 dips at k = 50. At k = 60 every final is only `partial`
  (+0.22 to +0.32).
- **Only c3 fully redirects,** and only before or at contact.
- **Re-commanding alone perturbs two of them.** Re-issuing an unchanged command mid-fight substantially changes c1's
  and c2's behavior, through the `planAge` and `activationDisplacement` inputs. The plan-aware teacher is
  unaffected.

**Does not establish:**
- Anything about stochastic execution beyond the post-hoc block-1 subset.
- Mid-fight control at d = 20, where flanks were already engaged by k = 60.
- Scheduled, stale or event-driven switches.
- A mechanism.
- Replication: the reserved panel is unspent.

## Consequence

The declared recommendation for finals that do not redirect at k = 50/60 is that a **switch-trained continuation**
(review §7, "train command changes as their own intervention") becomes the candidate. It would include:
- mid-fight switches and re-activations;
- randomized activation times;
- per-block common-random-number seeding in its evaluation code.

It needs its own declaration and the user's call. A stochastic re-run of S18 with fixed seeding is a cheaper
alternative, if the stochastic contrast is wanted first.
