# M8-S18 — mid-fight command switches: do the frozen S14 fighters redirect when the target command changes?

Declared 2026-10-04, before any learned-policy call on the panels below. Frozen-checkpoint evaluation plus the
plan-aware teacher as positive control. No training. `autonomousQualificationEligible` stays false (development
panel).

## 0. Question and scope

S16/S17 showed the frozen S14 fighters follow a `leftmost`/`rightmost` command **issued at reset**. An LLM commander
issues commands *during* a fight. The user asked for this step: "do mid fight". **Question:** when the command
changes from flank A to flank B at a fixed decision k, from a state reached under A, do the fighters redirect to B?

Scope and limits, stated up front:

- **Fixed-time switches only.** This is the first rung of review §7. Scheduled, stale and event-driven switches are
  deferred.
- **Switch times.** k = 30 is a **pre-contact control** (first hits are around decision 51–56 for the finals).
  k = 50 is contact onset and k = 60 is mid-fight. The mid-fight claim rests on k = 50 and k = 60.
- **Spreads 20 and 30 only.** At d = 10 the three Reds merge into one grounder cluster by k = 50 (dev finding §9),
  so a flank command no longer names a single enemy there.
- **What a switch changes.** A switch activates a new plan. That changes the target and also resets the plan's
  activation state the actor sees: `planAge` (`plan_groups` offset 37) and `activationDisplacement` (`plan_role_state`
  offsets 13–14). A **re-activate** branch separates these two effects (§3).
- **What a switch does not reset.** The Engage option budget and `option_state` are not reset. Only the critic
  reads them; the actor does not. Assignments are recomputed by the server's grounding at activation and stay all
  three blue units.
- **Generalization.** None beyond scripted-normal Red, these two spreads, reset-time layout and fixed switch times.

## 1. Policies

- **Learned:** the same six as S16/S17: S14 `update-200.pt` finals and S14's σ-scaled initializer references,
  cohorts 1–3, loaded by `enemy_relative_throw_ppo_retention.load_policies`. There is no selection. Finals are
  primary and initializers secondary. Cohort 2 is shown separately (S15).
- **Teacher:** the plan-aware teacher (`plan_teacher_tensor_actions_indices`), which executes whatever plan is
  active. It is the positive control, never used at learned-policy runtime.

## 2. Layout, tasks and worlds

- **Layout:** S16's: three singleton Reds at (30, −d), (30, 0), (30, +d), blue at the default spawns, scripted-normal
  Red, 3v3.
- **Development panel:** 192 worlds, seeds 2750000–2750191, d = (20, 30)[i mod 2], 96 per spread.
- **Reserved qualification panel:** 2760000–2760399, untouched; spending it is the user's call.
- **Tasks:** initial command A ∈ {`leftmost`, `rightmost`}, with B the mirror flank. The requested flank for scoring
  is always **B, the post-switch command**, in every branch.

## 3. Branches (per world, A and k), sharing an identical prefix

All branches run the same execution plan A from reset until decision k. At k:

| Branch | Action at k | Role |
| --- | --- | --- |
| keep | nothing | the no-command-change baseline |
| reactivate | activate a new plan with the **same** selector A | isolates "being re-commanded" |
| switch | activate a new plan with selector B | the test |

**Primary contrast: switch − reactivate.** It differs only in the target. **Side effect: reactivate − keep**, the
effect of re-commanding without changing the target. That side effect bears on the commander's keep-the-plan
baseline (selective-repair audit).

**Reference cell (descriptive only):** B-from-start, i.e. execution plan B from reset with the same stopping rule.
It has a different prefix and no eligibility condition, and is never part of a contrast.

**Shared prefix.** Prefixes are identical by construction: same seed, same commands until k, same block composition,
and in stochastic mode the same torch seed for all three branches (common random numbers). This is verified per world
(§6).

## 4. Eligibility at k (computed on the shared prefix, so it is identical across branches)

A (world, task) is eligible at k for a given policy and mode when, at decision k:
1. the episode is still running, blue is alive, and neither flank target has crossed the 20% threshold;
2. previewed `A` and `B` groundings equal the reset-time singletons;
3. **both flank targets are untouched** (full health).

Condition 3 makes the switch a decision taken before either flank is engaged. In dev data, worlds where the old
flank had already taken 60 damage by k = 60 made the switch an unwinnable race. The threshold (untouched) was chosen
on dev seeds and is disclosed.

**Policy-specific eligibility.** Each policy and mode has its own prefix, so eligible sets differ. Counts are
reported per policy, mode and k. Cross-policy (cohort-averaged) numbers use only worlds eligible for **every** policy
in that average. Finals and initializers are never compared on different world sets.

**Minimum to analyse.** 48 eligible worlds, a world being eligible if at least one of its tasks is. Below that, the
(policy, mode, k) cell is reported as `insufficient`.

## 5. Stopping rule and metrics

**Stopping rule** (branch-independent), first of:
1. both flank targets crossed the threshold;
2. no assigned blue unit alive;
3. environment end;
4. decision 200.

**Primary — post-switch flank damage contrast `c_post`:** (D_B − D_A) / (D_B + D_A) over decisions
k+1..min(k+40, stop); 0 if both are zero. Snowballs already in flight at k are identical in all branches, so they
cancel in the contrast.

**Co-primary — flank order `o_B`:** `flank_order(cross_B, cross_A)`, i.e. 1 if B crosses first, 0 if A first, 0.5
for a tie or neither. Because eligibility requires neither flank crossed at k, the order is decided after k. At late
k, `o_B` also reflects whether the redirect can still win the race, or survive it.

**Secondary:**
- decisions from k to B's crossing (censored at the stop);
- blue centroid lateral displacement toward B's side between k and k + 20;
- early aim shares (B / A / centre) in k+1..k+40;
- units lost and team wipe at the stop. These are comparable across branches because the stopping rule is
  branch-independent; switch − reactivate is reported as the **cost of switching**.

## 6. Gates and checks

1. **S16-replay gate (new runner, no switch).** The new runner is run in S16 mode (no switch, S16's stopping rule) on
   S16's development panel, cell `correct`/`leftmost`, for all six policies in both modes (S16's torch seeds). It must
   reproduce S16's archived rows exactly on the S14 row fields plus `flankOrder`, `requestedCross`, `mirrorCross` and
   `stopReason`. Any mismatch stops the run.
2. **Activation assertions (in-run, every activation; a failure stops the run):**
   - `activationObjectives` after activation equal the previewed IDs (B for switch, A for reactivate);
   - the plan version increments;
   - every plan tensor key fed to the actor is refreshed from the post-activation plan observation.
3. **Prefix identity (analysis).** For every (policy, mode, task, k) and every world that reaches k, all three
   branches must agree on:
   - state hash at k;
   - tracker decision;
   - crossed set;
   - flank health;
   - eligibility.

   Tolerance 0: any mismatch is investigated before interpreting.
4. **Teacher gate per k.** On the teacher's eligible worlds, k is **valid** if there are ≥ 48 eligible worlds and the
   teacher's Δc_post (switch − reactivate) has a lower 95% bound ≥ 0.80. Learned results at an invalid k are reported
   as `teacher-invalid` and not classified. If no k is valid, the run stops before learned cells.

## 7. Analysis and classification

**Contrasts.** World-paired bootstrap: 10,000 resamples, seed 988001. A world's value is the mean over its eligible
tasks. Contrasts are computed per policy, mode and k: switch − reactivate (primary), reactivate − keep, and switch −
keep. Each is computed on `c_post` and `o_B`, plus units lost and wipe for the cost.

**Breakdowns:**
- per spread;
- per initial side;
- cohort-averaged finals and cohort-averaged initializers, each on worlds eligible for all three policies in the
  average.

**Classification** of each learned policy, at each valid k, deterministic as primary and stochastic alongside. Let
T_k be the teacher's Δc_post point estimate at k.

| Class | Condition on Δc_post (switch − reactivate) |
| --- | --- |
| redirects | ≥ 0.5·T_k and lower 95% > 0 |
| partial | lower 95% > 0, below 0.5·T_k |
| ignores | 95% interval contains 0 |
| counter | upper 95% < 0 |
| insufficient | < 48 eligible worlds |

The audit authorizes nothing. Recommendation only:
- **Finals redirect at k = 50/60.** Mid-fight switching is usable by a commander. The next rung is a replication on
  the reserved panel, or scheduled and stale switches. Both are the user's call.
- **Otherwise.** A switch-trained continuation (review §7, "train command changes as their own intervention")
  becomes a candidate. It needs its own declaration.

## 8. Predictions (fixed now)

1. The teacher gate passes at all three k.
2. At k = 30 (pre-contact), all three finals classify as `redirects` (deterministic).
3. At k = 60, not all three finals classify as `redirects` (deterministic).
4. The re-activation side effect is small: |Δc_post(reactivate − keep)| < 0.20 for every final at every valid k
   (deterministic point estimates).
5. Switching costs units at mid-fight: for the cohort-averaged finals at k = 60, the units-lost point estimate for
   switch − reactivate is > 0.

## 9. Seeds, budget, development disclosure

- **Stochastic torch seeds:** 988000 + 100·cohort + 20·kIndex + 2·task + policy, with kIndex 0/1/2 for k = 30/50/60
  and 3 for B-from-start. The same seed is used for all three branches of a cell. Values fall in 988100–988363.
- **Replay gate:** S16's seeds.
- **Bootstrap:** 988001.
- A repo-wide scan, including ignored run directories, found 2750000–2769999 and 988000–988999 unused.

**Budget bound** (horizon 200 per episode):

| Part | Calculation | Decisions |
| --- | --- | ---: |
| Replay gate | 12 cells × 192 × 200 | 460,800 |
| Teacher | 192 × 2 tasks × (3 k × 3 branches + 1 reference) × 200 | 768,000 |
| Learned | 192 × 2 × 10 × 6 policies × 2 modes × 200 | 9,216,000 |
| **Total bound** | | **10,444,800** |

Hard cap: **10,450,000**, checked against all four repo-JSON scanners' bands. Expected actual use is about 5–6M.

**Development disclosure:** off-band dev seeds 5986000–5986047, **plan-aware teacher only**, none archived. The
dev runs covered:
- k = 30 / 50 / 60 / 70 with keep vs switch;
- re-activation at k = 60;
- prefix hash checks.

Findings that shaped this declaration:
- **Prefixes:** hash-identical in every case.
- **Teacher switch success by k** (eligible worlds, finishing B first): 100% at k = 30, 94–95% at k = 50, 40–45%
  at k = 60, and too few eligible worlds at k = 70.
- **Why k = 60 fails at d = 20:** the old flank was already at 40 health at k = 60, one hit from falling.
- **Cost at k = 60, d = 30:** with both flanks untouched, the teacher finished B first in 9 of 16 switch episodes;
  blue was wiped in the other 7.
- **Re-activation:** left all 28 teacher episodes identical to keep.
- **d = 10:** never eligible at k ≥ 50, because the Reds merge into one cluster.

These shaped the choices of k, spreads, the untouched-flank condition and the co-primary.

**Teacher-gate calibration.** Before commit, this module's runner was run with the teacher only on the same dev
seeds (5986000–5986047, d = 20/30) at k = 30/50/60. Eligible worlds and Δc_post (switch − reactivate):

| k | Eligible worlds (of 48) | Δc_post [95%] | o_B under switch |
| --- | ---: | --- | ---: |
| 30 | 48 | +1.98 [+1.95, +2.00] | 1.00 |
| 50 | 48 | +1.54 [+1.44, +1.64] | 0.97 |
| 60 | 24 | +1.45 [+1.24, +1.62] | 0.73 |

Prefix mismatches were 0. The 0.80 gate threshold was set before this calibration and is unchanged.

**k = 60 sample size.** From S16's archived d = 20/30 correct-condition rows (same prefix as A from reset), an
upper bound on learned-policy eligibility at k = 60 is about 157–192 worlds of 192 per policy.

**Learned policies in this module's tests.** These runs used off-band test seeds 5987000–5987999 or S16's archived
development panel; neither panel nor the reserved band was touched:
- An end-to-end test ran the cohort 1 initializer and final, both modes, through switches at decisions 4, 6 and 8
  with a 12-decision horizon (seeds 5987100–5987103). Those switches came before any contact. Outcomes were not
  inspected beyond structural assertions.
- A replay test runs the cohort 1 final in S16 mode (no switch) on S16's archived development panel, first 64
  worlds (2720000–2720063).
- A teacher test runs a real k = 30 keep/reactivate/switch on 5987000–5987003. It asserts only that the switch's
  `c_post` exceeds the reactivate branch's.

## 10. Archive and gates

New module `options/command_switch_audit.py`. It reuses `command_control_audit` (S16) helpers unchanged and edits no
existing module. Sealed archive `runs/m8_s18_command_switch_audit_v0/`. Its `declaration.json` pins:
- the S16 implementation digest (checked equal to S16's seal);
- the S12, S14, S15, S16 and S17 manifests;
- the checkpoint digests;
- this declaration and the module.

Before running: the full training suite, client pytest, `npm run build` and `npm test` (only the documented
failure). Live tests use off-band seeds (5987xxx), except the replay test on S16's archived development panel (§9). After archiving and before committing: the full training suite and
`npm test` again, plus the scanner-band check of the archive's JSON. Results go in `reviews/m8_s18_results.md`.
