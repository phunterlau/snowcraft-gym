# R1n-c: imitation of the plan teacher that keeps its artifacts (1v1 Engage, global decoder)

This protocol is committed before implementation and before any collection.
As in R1n-b, the tested implementation lands in a separate commit. That
commit may amend this file, stating each change, before any collection.
Collection waits for both commits.

R1n-c is step 1 of the D3 branch recommended by
[R1n-b](m7b_r1n_b_results.md): BC then DAgger from the plan teacher into a
full-authority policy that decodes destinations globally, with every model,
optimizer state, and history file retained.

It also measures the precondition the PPO stage needs: whether a critic fits
Monte Carlo returns on the imitation policy's own episodes, where contact
actually happens. **No PPO update runs in R1n-c.**

`autonomousQualificationEligible: false`. Training uses coded-teacher labels.
At runtime the policy acts alone: no corrected shots, no teacher moves.

## 0. Why imitation and PPO get separate declarations

R1's PPO gate requires at least +20 success points over the initializer, and
the teacher ceiling is 89/100. If imitation already reaches about 69% or
more, that gate cannot be met, and a different primary PPO test is needed
(continuous return, death rate, completion time, or a harder task). Review
§8.1 requires a feasibility and power section before any PPO fit, and that
section cannot be written until the initializer's success is known.

So the PPO stage becomes R1n-d, declared after these results. R1n-c's §6
rules state in advance which kind of R1n-d each outcome calls for.

## 1. Fixed from R1n-b

- **Scenario:** 1v1, 100×80 arena, `RandomAgent` red, `maxTicks` 1800, 10 Hz,
  `teacher_option_plan("engage")`.
- **Engage option:** the frozen spec, **horizon 200**.
  - Declined alternative: 216, which is 1.5× the teacher's p95 of 143.8.
  - Reason: all 11 teacher failures in R1n-b were blue deaths, not timeouts,
    so the horizon is not what limits the ceiling. Keeping 200 also preserves
    comparability with R1n-b's contact and critic numbers.
  - The imitation policy's timeout fraction is reported, so a binding horizon
    would show up.
- **Policy:** `FullAuthorityPolicyV1(destination="global")`, randomly
  initialized, with E3's egocentric 227-wide features, per-arm move σ,
  globally calibrated throw σ, and `OptionCentralCritic`.
  - **The local decoder is excluded, not merely deprioritized.** In R1n-b,
    77.1% of teacher MOVE labels lie beyond its 8-unit one-decision reach, so
    it cannot represent them for any latent.
- **Plan input:** uninformative at roster 1. In R1n-b, the plan teacher and
  `SimpleBlueAgent` produced the same outcome on 100/100 seeds, so there is no
  wrong-plan control and no plan-following claim.

## 2. Labels

At every state a collection visits, the plan teacher (`PlanAwareTeamController`
+ `ReactiveUnitPolicy`, read without stepping via
`SelectiveBatchEnv.plan_teacher_tensor_actions_indices`) labels each living
unit:
- action type (NOOP, MOVE, THROW, HOLD);
- normalized target (for MOVE and THROW);
- power (for THROW).

Units the teacher omits are labeled NOOP, which matches the client's
`decode_action`. A label whose type is illegal under `unit_action_mask` is
counted and excluded; none are expected, since the teacher's actions had 0
rejections in R1n-b.

An out-of-band probe (seeds 928000–928015, not a ceiling measurement) sized
the label ranges:
- MOVE/THROW normalized |target| maximum 0.73/0.84, so `atanh` is finite and
  the latents are at most 1.22;
- power between 0.596 and 0.9995;
- type counts: MOVE 1,604, NOOP 174, HOLD 91, THROW 72.

## 3. Loss

Each term is a mean over the living units in the minibatch that carry the
relevant label. A term with no such units contributes 0. Weights are all 1.

- **Type:** cross-entropy on the masked action logits against the teacher's
  type, unweighted.
  - THROW is about 4% of labels. Class weighting was considered and rejected
    so that the objective stays plain.
  - A throw collapse must show up in the per-type held-out recall (§5), not be
    inferred from success alone.
- **MOVE** (MOVE-labeled units): S12's heading form, applied to the decoded
  world destination `tanh(move_raw)·(50,40)` versus the teacher's destination,
  both measured from the fighter's position.
  - Beyond the 2.1-unit slow radius: `1 − cos` of the heading.
  - Within it: squared endpoint error, with the distance clipped at 3 world
    units.
  - Why not endpoint or latent MSE: R1i/R1j/R1l fitted endpoints better with
    no autonomous gain, while S12's heading form reached 38–40/40.
- **THROW aim** (THROW-labeled units): `1 − cos` between the learned aim ray
  `tanh(throw_raw)·(50,40) − own` and the teacher's aim ray. R1g/R1k found
  that aim direction dominates throw value.
- **Power** (THROW-labeled units): `(sigmoid(power_raw) − power)²` on the 0–1
  scale.
  - Logit-space MSE is rejected because labels near 0.9995 reach logit 7.6
    and would dominate.
  - This scale matches how `act()` emits power.

**Parameters trained:** the encoders and the action, move, throw, and power
heads. **Not trained:** `move_log_std`, `throw_log_std`, `power_log_std`
(they stay at their calibrated initial values for R1n-d's exploration) and
the critic.

## 4. DAgger schedule

- **Optimizer seeds:** 97101, 97102, 97103. Each seeds
  `torch.manual_seed(seed)` before constructing its policy, and seeds every
  minibatch permutation as `seed + 1000·fit`.
- **Round 0 (shared):** 128 complete teacher-driven episodes, the teacher
  acting through the same tensor round trip, on seeds 650000–650127. Its data
  does not depend on any weights, so all three optimizer seeds share it.
- **Fits k = 0…4** (per optimizer seed):
  - fit k trains on all data aggregated so far with one Adam optimizer that is
    never reset: lr 3e-4, gradient clip 0.5, minibatch 256, 3,000 steps per
    fit;
  - after fits 0–3, collect round k+1: 128 complete episodes on seeds
    `650000 + 1000·(k+1) + [0, 127]`, with the just-fitted policy acting
    deterministically and the teacher labeling every visited state;
  - the same environment seeds are used across optimizer seeds (paired).
- **Final policy:** the policy after fit 4. No round or seed is selected.
- **Collections** use complete-episode blocks (`run_block`): only unfinished
  worlds step, and there are no replacement resets.

## 5. Measurements

Final policy per optimizer seed. Development split A is 600000–600099,
reserved since E3 and unused until now. Development split B is 601000–601099,
new.

- **Closed loop, deterministic, on A and B:**
  - success;
  - contact fraction;
  - first-hit and completion-decision distributions;
  - blue death fraction;
  - timeout fraction;
  - rejected-action rate.
- **Closed loop, stochastic, on A:** success and contact, sampling from the
  calibrated σ that R1n-d would explore with.
- **Controls on A and B:**
  - the plan-teacher ceiling;
  - the uniform-random legal-action floor.
- **Paired comparisons:** imitation vs ceiling per split, as success
  differences with 95% bootstrap intervals over seeds (10,000 resamples,
  seed 973001).
- **Held-out label error** on split B, with the teacher labeling the imitation
  policy's own deterministic evaluation states (no extra simulation):
  - type accuracy, plus recall and precision per type;
  - MOVE heading error in degrees (units beyond the slow radius) and endpoint
    error (units within it);
  - THROW aim heading error in degrees;
  - power mean absolute error.
- **Critic precondition for R1n-d:**
  - Run R1n-b's Monte Carlo warm start (B3/B4 exactly: the same metrics, the
    same gate `predictiveR2 ≥ 0.25` and `≥ timeOnlyR2 − 0.05`, gate conditions,
    and clock skill) on each final policy's **stochastic** episodes, since
    PPO's behavior policy is stochastic.
  - Episodes: a training fold of 256 on seeds
    `660000 + 1000·i + [0, 255]` and a held-out fold of 128 on seeds
    `670000 + 1000·i + [0, 127]`, for optimizer seed index `i`.
  - Also report the contact fraction and the held-out return variance, so a
    repeat of R1n-b's clock-only degeneracy would be visible.
- **Fit history:** every 25 steps, the total loss, each term, and the
  gradient norm.

## 6. Decision rules (recommendations for R1n-d; they authorize nothing)

Let `S` be the imitation policy's deterministic success on a split, averaged
over the three optimizer seeds, and `C` the ceiling on that split.

| Outcome (both splits) | R1n-d recommendation |
| --- | --- |
| `S ≥ C − 20` points | **Near ceiling.** R1's +20-over-initializer success gate cannot be met. R1n-d should use a continuous primary test (return, death rate, completion time) or a harder task: a second mission or 2v2. |
| Contact ≥ 25% and `S < C − 20` | **PPO has headroom.** R1n-d runs KL-anchored frozen-reward PPO with R1's +20 gate, sized by a power analysis at the measured `S`. |
| Contact < 25% or `S < 25%` | **Imitation failed.** Diagnose using the per-head, per-type held-out errors before any PPO. |

If the two splits disagree, the more conservative row applies.

Flags, each reported:
- **Critic healthy:** the gate passes for all three optimizer seeds.
- **Throw collapse:** held-out THROW recall below 50%.
- **Execution-mode gap:** stochastic success on A below deterministic success
  on A by more than 20 points. R1n-d must then declare its exploration σ.
- **Seed instability:** the spread of success across optimizer seeds exceeds
  30 points on either split.

Prediction, stated for falsification:
- deterministic success ≥ 60% on both splits;
- contact ≥ 80%;
- THROW recall below MOVE recall;
- critic gate passes, with clock skill clearly above 0 now that contact
  occurs.

## 7. Artifacts retained

The S12 lesson is made explicit here. For each optimizer seed:
- `fit-{k}.pt` for k = 0…4: model state and Adam optimizer state after each
  fit; `fit-4.pt` is the final policy;
- `fit-history.json`;
- `dataset.json`: per-round seeds, row counts, per-type label counts,
  illegal-label count, and a `semantic_state_digest` of each round's
  observation and label tensors;
- `episodes.jsonl` and `trajectory-distances.npz` for every collection and
  evaluation;
- `label-error.json`;
- `critic-warm-start.json`, `critic-warm-start-arrays.npz`, and
  `critic-policy.pt` (the final policy with its warm-started critic).

Run-level: `declaration.json` (config, source digests, pinned E3 digests,
git commit), `controls/` (ceiling and floor), `report.json`, and
`manifest.json`.

The raw aggregated observation tensors (about 1 GB per seed) are not
retained. Each round's data can be regenerated: round 0 comes from the
teacher and its seeds, and round k+1 from `fit-k.pt` acting deterministically
on declared seeds. The per-round digests make any regeneration checkable, and
a test verifies that collection under a fixed checkpoint is reproducible.

## 8. Budget

| Item | Upper bound (simulator decisions) |
| --- | ---: |
| Round 0 teacher collection: 128 × 200 | 25,600 |
| Learner collections: 3 seeds × 4 rounds × 128 × 200 | 307,200 |
| Deterministic evaluation: 3 seeds × 2 splits × 100 × 200 | 120,000 |
| Stochastic evaluation on A: 3 × 100 × 200 | 60,000 |
| Ceiling and floor: 2 × 2 × 100 × 200 | 80,000 |
| Critic precondition: 3 × 384 × 200 | 230,400 |
| **Declared cap** (enforced by `account()`) | **900,000** (bound by item: 823,200) |

## 9. Stopping rule

- Fixed: five fits, three optimizer seeds, evaluation of the final policy
  only.
- No retries with a different budget, seed, or loss.
- If a run is aborted by an implementation defect before completion, its
  output is discarded and the abort is reported. The fix is committed before
  a full rerun.
- Output: `runs/m7b_engage_r1n_c_v0/`, which the runner refuses to overwrite.
- No provider calls, browser input, TypeScript or protocol changes, or edits
  to digest-pinned sources. New behavior goes in new modules.

## 10. Verification before the implementation commit

Targeted tests:
- each loss term against hand-computed values, including type masking (a
  THROW-labeled unit never reaches the move term) and zero-count terms;
- the σ parameters and critic are excluded from the imitation optimizer;
- DAgger aggregation and seed bands;
- collection under a fixed checkpoint is reproducible;
- held-out label-error metrics on synthetic cases;
- the decision rules and flags;
- the budget guard;
- a tiny live end-to-end run.

Full gate: `npm test`, `npm run build`, Python client tests, Python training
tests, and a check that no pinned archive source changed.

## 11. Amendments made in the implementation commit, before any collection

- **A1 — floor generator.** The uniform floor (R1n-b's `uniform_floor`:
  uniform legal type, target uniform on [−1, 1]², power uniform on [0, 1])
  draws from `numpy.random.default_rng(981002 + block offset)`. Each
  50-world block gets a fresh generator (981002, then 981052) on each split,
  so any block can be regenerated on its own. R1n-b used a single generator
  across its blocks.
- **A2 — sampling RNG and block sizes.**
  - Stochastic actions come from torch's global generator, seeded by
    `torch.manual_seed(optimizerSeed)` immediately before the policy is
    constructed.
  - Deterministic actions, round collections, and fits draw nothing from it.
    Minibatch order uses its own `torch.Generator`, as in §4.
  - Draws happen in order: the stochastic evaluation on A, the critic
    training fold, then the held-out fold.
  - The critic warm start is R1n-b's `warm_start_critic_mc` unchanged: 10
    epochs, and a minibatch-order generator seeded with the optimizer seed.
  - Block sizes: 64 worlds for rounds and critic folds, 50 for evaluations
    and controls. Stochastic results depend on this order, so the runner is
    the reference.
- **A3 — row precedence within a split.** Imitation failure is checked
  first, then near ceiling, then PPO headroom. The §6 rows overlap only if
  `C < 45`. This applies the "more conservative row" principle within a
  split, not only between splits.
- **A4 — flags are per optimizer seed.**
  - *Execution-mode gap* is raised if any seed's stochastic success on A falls
    more than 20 points below its deterministic success on A.
  - *Throw collapse* is raised if any seed's held-out THROW recall is below
    50%, or undefined because B has no THROW labels.
  - *Seed instability* uses the max − min of deterministic success across
    seeds.
  - Per-seed values are reported.
- **A5 — seed audit, widened after the R1n-b erratum.** Every band in this
  declaration was checked two ways:
  - `auditSeedDocuments` on the serialized configuration: 10 declarations,
    0 collisions with 630000–630119;
  - a numeric scan of every JSON, TypeScript, Python, and Markdown file under
    `snowgym/` and `refs/`: the only matches are fragments of state hashes
    and digests.

  Split A (600000–600099) is the band E3 reserved in `full_authority_train.py`
  and the R1n declaration. It has never been collected.
- **A6 — gate status.** At this commit `npm test` passes 366 of 367. The one
  failure is the selective-repair preflight collision caused by R1n-b's
  archive (see the erratum in `m7b_r1n_b_results.md`). This commit does not
  change it. After collection, `npm test` is run again: the preflight must
  still name only `runs/m7b_engage_r1n_b_v0/declaration.json`, so R1n-c's
  archive adds no collision.
