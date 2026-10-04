# M8-S16 — command-controllability audit of the frozen S14 fighters

Declared 2026-10-04, before any collection on declared seeds. Evaluation of frozen checkpoints plus a plan-aware
teacher positive control. No training. `autonomousQualificationEligible` stays false: this is a development panel.

## 0. Question, and what this can and cannot show

Step B of the 2026-10-04 first-principles review: **from the same physical state, does a valid symbolic command
change what the learned fighters accomplish?** S12–S14 established physical competence on one fixed order. S15
found the cohort-averaged random/easy competence retained, with cohort 2 regressing on random. Command control has
never been measured.

Facts established while designing this step (off-band dev seeds, disclosed in §9):

- **One geometry, one target set.** Every seed in S4–S15 starts from the identical physical state: blue at x = −30,
  Red at x = +30, both at y ∈ {−5, 0, 5}, full health. The three Reds form one grounder cluster (radius 6), so every
  Engage target set the fighters ever saw was *all three Reds*. Those "400 worlds" are 400 draws of Red's random
  numbers from one geometry. **Any layout with separated clusters is new to these fighters physically, not only
  command-wise.**
- **How the command reaches the actor.** The plan tensors encode no selector identity. A command reaches the actor
  only as objective geometry and health: `plan_groups` `objectiveRelative`; `plan_role_state` `objectiveDisplacement`,
  `objectiveDistance`, `objectiveHealth`, `flankAngle`. During S14 training that geometry was always the centroid of
  all Reds, which the enemy rows already carry. The enemy-choice scorer has no plan input at all (S12 design).
- **Which teacher is plan-aware.** The plan-aware teacher (`plan_teacher_tensor_actions_indices`, the imitation label
  source) finishes the commanded singleton first. The built-in `SimpleBlueAgent` (`/step-scripted`) ignores the plan.
  S4's "teacher" was the latter; see the 2026-10-04 erratum in `m8_s4_results.md`.

**Can show:** whether each frozen policy's execution depends on which of two mirrored, valid, distinct enemy
targets is commanded, measured against the commanded target with a stopping rule independent of the execution plan.

**Cannot show:** control on the training geometry (one cluster has no choice to make); control from mid-fight
states (reset-start only; intermediate branches are deferred); control under other doctrine fields, missions or
rosters; generalization beyond scripted-normal Red; anything about LLM commanders.

## 1. Policies

- **Learned:** S14 `update-200.pt` (final) and S14's sigma-scaled initializer reference, cohorts 1–3, loaded by
  `enemy_relative_throw_ppo_retention.load_policies` exactly as S15 loaded them. Six policies.
- **Teacher positive control:** the plan-aware teacher via `plan_teacher_tensor_actions_indices`, the same path that
  produced the imitation labels. It executes whatever plan is active (the execution plan). It is deterministic.

No teacher action, label or override is used at learned-policy runtime.

## 2. Layout and worlds

- 3v3, 100 × 80 open arena, scripted-normal Red, the training scenario otherwise unchanged.
- Blue spawns explicit at the default (−30, −5), (−30, 0), (−30, 5).
- Red spawns at (30, −d), (30, 0), (30, +d): three singleton clusters.
- The spread is d = (10, 20, 30)[i mod 3] for world index i.
- **Development panel:** 192 worlds, seeds 2720000–2720191 (64 per spread).
- **Reserved, untouched qualification panel:** 2730000–2730399. Not used here.
- A repo-wide scan, including ignored run directories, found no use of 2720000–2739999 or 986000–986399.

## 3. Commands and execution conditions

Requested task r ∈ {`leftmost`, `rightmost`}; its mirror m(r) is the other. All other plan fields are the
canonical training doctrine (`teacher_option_plan("engage")`): one `main` group, `balanced` selection, `engage`
mission, `direct` approach, `balanced` posture, `opportunistic` fire, `medium` range, `normal` cohesion. `largest`
and `weakest` are excluded because they tie onto the `rightmost` singleton in this layout.

| Condition | Execution plan selector | Contract |
| --- | --- | --- |
| correct | r | production |
| other | m(r) | production |
| canonical | `nearest` (the centre Red) | production; the trained doctrine |
| shuffled | uniform over {`leftmost`, `rightmost`}, drawn per (world, r) with numpy seed 986002 | production |
| zero-plan | r, with `plan_groups` and `plan_role_state` zeroed in the actor's input | **diagnostic, outside the contract**; learned policies only |

**Requested-target identity.** The requested, mirror and centre IDs come from `preview_plans` on the reset state
(server grounding, no activation). Every world must ground r, m(r) and `nearest` to three distinct single enemies.
A world violating this stops the run with an error; none are expected.

**Shuffled arithmetic.** With two commands, shuffled is a ~50/50 mix of correct and other, so correct − shuffled ≈
½(correct − other). PLAN.md's per-mission gate of "≥ 20 points over shuffled" therefore corresponds to about 40
points over other. The realized mismatch rate is reported.

**Mirror identity (what deterministic conditions can and cannot add).** In deterministic mode, (other, r) runs the
same execution plan from the same reset as (correct, m(r)), with the requested and mirror flanks swapped. The two
trajectories coincide until the first flank crossing. Therefore:
- o(other, w, r) = 1 − o(correct, w, m(r)), and per world Δo = 2·ō(correct) − 1.
- correct − canonical = Δo / 2. Canonical runs one plan for both tasks, so its mean `o` is 0.5.
- Shuffled is a per-(world, task) pick of correct or other cells.

Deterministic "other", "canonical" and "shuffled" are therefore **not independent evidence**: they re-express
ō(correct) against 0.5, and the results will not present them as separate measurements. Independent information
comes from:
- ō(correct) vs 0.5;
- the stochastic cells (task- and condition-specific torch seeds);
- zero-plan;
- the early contrast `c` (windows can differ once one episode stops);
- the aim and lateral diagnostics.

The identity is exact only up to batch effects: stop times differ by condition, so a block's active-row count, and
with it the policy's batched float arithmetic (S15), diverges after the first crossing anywhere in the block.
Canonical's 0.5 is exact up to the same batch effects in deterministic mode, and holds only in expectation under
stochastic execution. **Check:** the report gives, for the teacher and every deterministic learned policy, the
fraction of (world, task) pairs violating each identity. Tolerances, above which the deviation is investigated before
interpreting:
- teacher: 0.02 (server-side per-world actions, so near 0 is expected);
- learned: 0.10.

## 4. Runner, stopping rule and metrics

A new runner resets each world with its execution plan and installs a tracker that continues past execution-option
completion. The server re-resolves the execution selector once its target dies.

**Stopping rule** (independent of the execution plan), first of:
1. the requested target reaches ≤ 20% of its initial health (alive or dead);
2. no assigned blue unit alive;
3. the environment ends;
4. decision 200 (S14's Engage horizon).

Execution-option completion is logged and never stops an episode.

**Primary — flank order `o`** (per episode): whether the requested flank reaches the 20% threshold before its
mirror.
- 1 if the requested flank crosses strictly earlier.
- 0.5 if both cross at the same decision, or neither crosses before the stop.
- 0 if the mirror crosses first.

The centre Red is excluded from `o` on purpose: it is the physical default, and a "requested first overall" metric
would hide partial control behind it.

**Co-primary — early flank contrast `c`:** (D_req − D_mir) / (D_req + D_mir), where D is damage dealt to each flank
over decisions 1..min(100, stop). `c` = 0 when both are zero.

**Secondary:**
- Requested target reached within the horizon, and its decision.
- First decision with damage to the requested target.
- Which enemy crosses first overall.
- Execution-option completion and its decision.
- Units lost and team wipe at the stop.
- Rejected actions.

Units lost, team wipe and decisions at the stop are reported, but are **not comparable across conditions**. The stop
time depends on the condition (an "other" episode runs on until the requested flank falls), so these are not a
command cost.

**Diagnostics (learned policies and teacher):**
- **Aim:** shares of THROW unit-decisions in the early window whose action-target direction is bearing-closest to
  the requested, mirror or centre Red.
- **Lateral movement:** the blue centroid's lateral displacement toward the requested side at decision 40, or at the
  stop if earlier.
- Enemy-logit changes are not measured; the criterion is physical accomplishment.

## 5. Gates (in run order; a failure stops the run before later cells)

1. **Runner regression gate.** Run the new runner, tracker, per-world scenario plumbing and preview calls on the
   *default* layout, with `nearest` as both requested and execution plan. It must reproduce S14's archived first
   evaluation block (2600000–2600063, 64 worlds, default threads) **exactly**: deterministic and stochastic, final and
   initializer, all three cohorts. Rows are compared on every S14 field except `source`. In this layout the stopping
   rule coincides with S14's option termination by construction (requested = activated = all three Reds).
2. **Teacher positive-control gate.** On the 192-world panel, all four production conditions, both tasks. The
   teacher must reach:
   - mean `o`(correct) ≥ 0.90;
   - `o`(correct) − `o`(other) with a lower 95% bound ≥ 0.70;
   - mean `o`(correct) ≥ 0.85 within each spread.

   If the teacher fails, the axis is not a valid command test, and no learned cell runs.

## 6. Analysis and classification

**Pairing and resampling.**
- The resampling unit is the world, keeping both mirrored tasks of a world together.
- Each world's value is the mean over its two tasks.
- Differences are world-paired bootstrap, 10,000 resamples, seed 986001.
- A fixed side bias cancels under this pairing, so only *signed, requested-target* metrics count as control.
  "Behaves differently under the two commands" does not.

**Contrasts.** Per policy and mode: correct − other (primary), correct − shuffled, correct − canonical, and correct −
zero-plan (diagnostic). Each is computed for `o` and `c`.

**Breakdowns.**
- Per spread.
- Per requested side (`leftmost`, `rightmost`) separately, because the dev teacher showed a side asymmetry in blue
  losses.
- Per cohort, final and initializer, plus a cohort-averaged final and a cohort-averaged initializer (per-world mean
  over cohorts).
- Cohort 2 is always shown separately (S15).

**Informativeness guard (per learned policy).** Count the correct-condition episodes in which at least one flank
crosses the threshold before the stop. If that is below 0.50, the policy's command result is labelled
**uninformative**: it cannot reach the targets in this layout, so "insensitive" would be the wrong word.

**Classification** (per informative learned policy, deterministic primary; stochastic reported alongside), on
Δo = `o`(correct) − `o`(other):

| Class | Condition |
| --- | --- |
| controllable | Δo ≥ 0.40 and lower 95% > 0 (≈ PLAN.md's 20 points over shuffled) |
| sensitive | lower 95% > 0, but not controllable |
| insensitive | 95% interval contains 0 |
| counter-sensitive | upper 95% < 0 |

The audit authorizes nothing. Recommendation only:

- **All learned policies insensitive or sensitive.** The conditional extension in review §6C becomes the next
  candidate: a zero-initialized residual enemy scorer with target-membership input, trained on a balanced
  target-command distribution. It needs its own declaration.
- **Any learned policy controllable.** Replicate on the reserved qualification panel before any training.
- **All uninformative.** Physical competence in split layouts comes first.

## 7. Predictions (fixed now, scored afterward)

1. The teacher gate passes.
2. No learned policy is `controllable` under deterministic execution.
3. At least one learned policy is `sensitive` or `controllable` under deterministic execution. Command geometry
   reaches the movement and offset heads, even though the enemy scorer cannot see it.
4. Under canonical, every informative learned policy finishes the centre Red first overall in ≥ 50% of episodes.

## 8. Seeds and budget

- **Stochastic torch seeds.** Set before each learned cell as 986000 + 100·cohort + 20·condition + 2·task + policy,
  with condition index 0–4 in §3's order, task 0 = leftmost, policy 0 = initializer. All values fall in
  986100–986383 and are distinct.
- **Regression gate seeds.** S14's own seeds, as in S15.
- **Block size.** 64 worlds throughout.

Bound (horizon 200 per episode):

| Part | Calculation | Decisions |
| --- | --- | ---: |
| Regression gate | 3 cohorts × 2 policies × 2 modes × 64 worlds × 200 | 153,600 |
| Teacher | 192 worlds × 2 tasks × 4 conditions × 200 | 307,200 |
| Learned | 192 worlds × 2 tasks × 5 conditions × 6 policies × 2 modes × 200 | 4,608,000 |
| **Total bound** | | **5,068,800** |

Hard cap: **5,100,000** simulator decisions.

## 9. Development disclosure

Off-band dev seeds 5982000–5982001 (smoke) and 5982100–5982115 (16 worlds) were used before this declaration, none
archived. They covered:
- three spreads (d = 10, 20, 30);
- `leftmost`, `rightmost` and `nearest`;
- both teacher paths (`SimpleBlueAgent` and the plan-aware teacher);
- two fire doctrines (`opportunistic`, `focus`).

The plan-aware teacher finished the commanded singleton first in 16/16 worlds for every selector and spread. The
spreads (10, 20, 30) were fixed after seeing that data.

The module's tests also ran on split layouts, at off-band test seeds 5984000–5984999:
- **Preview check:** reset only, 5984000–5984002.
- **Teacher test:** 4 worlds, 5984010–5984013, d = 20, correct vs other. It asserts only that the correct order
  exceeds the other.
- **Zero-plan test:** one deterministic `act()` of the cohort-1 final policy on a reset state, 5984020–5984021.
- **End-to-end test:** cohort 1 initializer and final, both modes, all five conditions, 3 worlds, an 8-decision
  horizon, 5984100–5984102. Too short to reach any flank. Its outcomes were not inspected beyond structural
  assertions.

No learned policy was otherwise run on a split layout before this declaration.

## 10. Provenance, archive, gates

New module `options/command_control_audit.py`; no existing module is edited. New sealed archive
`runs/m8_s16_command_control_audit_v0/`. `declaration.json` pins:
- configuration and git commit;
- budget bound;
- the S14 training, S14 probe, S12 and S15 manifests;
- the six checkpoint digests;
- E3 digests;
- this declaration's digest and the module digest.

Per-cell episode rows are written under `gate/`, `teacher/` and `learned/`, with `report.json` and a sealed
manifest.

Before running: client `pytest`, training `pytest`, `npm run build`, `npm test` (only the documented
`trainSeedBase: 630000` failure). `npm test` is re-run after archiving. Results go in `reviews/m8_s16_results.md`.
