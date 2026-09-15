# R1n-h: mixture-imitation curriculum, with a single-opponent control, testing transfer to a held-out opponent

This protocol is committed before implementation and before any collection.
As in R1n-c/e/f/g, the tested implementation lands in a separate commit,
which may amend this file (each change labeled and dated) before any
collection. Collection waits for both commits.

`autonomousQualificationEligible: false`. Training uses coded-teacher
labels throughout; at runtime the policy acts alone.

## 0. Why this exists, and the single primary test

R1n-f found that R1n-c's imitation stage — trained only against
`RandomAgent` — scores 0% against `ScriptedAiAgent` at full scale, and that
PPO (R1n-e) neither causes nor fixes this. R1n-g localized the failure to
three compounding causes, of which aim is most severe: mean throw-aim
error against scripted red is 111.9°–131.8°, *worse than two random
directions* (90°), against a 3.1° healthy baseline.

R1n-h tests the roadmap's proposed fix — training against a mixture of
opponents — and is designed to answer one causal question, not several at
once:

**Does exposing the imitation stage to `ScriptedAiAgent` (easy) during
training improve transfer to `ScriptedAiAgent` (normal), an opponent
neither the mixture nor the control condition ever trains on?**

This is the single named primary test (§6). Two things this declaration
deliberately does *not* treat as co-equal primaries, to avoid picking the
best-looking number after the fact:
- **In-mixture held-out-world success** (fresh worlds, same opponents as
  training) is a **precondition check**, not the primary claim: if the
  mixture-trained policy does not clear a floor against the opponent it was
  actually trained on (scripted-easy), the held-out-opponent number is
  uninterpretable and is reported as such.
- **Label-error recomputation on the held-out opponent** (reusing R1n-g's
  own metrics unchanged) is a **secondary, mechanism-level** measure with
  its own stated prediction (§6): it asks whether the *aim-worse-than-random*
  finding specifically improves, independent of whether win/loss success
  moves enough to look like "transfer."

**Why a control condition, not just a before/after comparison against
R1n-c's archive:** R1n-h necessarily trains on fresh seed bands (§7) — R1n-c's
bands cannot be reused without re-deriving R1n-c's exact training
trajectory, which this declaration does not attempt. Comparing a
freshly-trained mixture policy against R1n-c's archived, differently-seeded
initializer would confound "the mixture helped" with "a different set of
training worlds happened to help." R1n-h therefore trains **two
conditions** side by side, on the same fresh seed-generation scheme,
differing only in opponent exposure:
- **Mixture (M):** every round split 64/64 between `RandomAgent` and
  `ScriptedAiAgent` (easy).
- **Control (C):** every round 100% `RandomAgent` — R1n-c's opponent,
  R1n-h's worlds.

Both conditions are evaluated on the same fresh, paired worlds (§5), so the
mixture-vs-control gap isolates the effect of opponent exposure from the
effect of using different training data.

## 1. Fixed from R1n-c/R1n-e (unchanged, imported, not reimplemented)

- **Policy:** `FullAuthorityPolicyV1(destination="global")`, E3's egocentric
  227-wide features, per-arm move σ, globally calibrated throw σ,
  `OptionCentralCritic`. Randomly initialized per condition per seed (not
  warm-started from R1n-c — see §0's confound reasoning).
- **Engage option:** the frozen spec, horizon 200, `teacher_option_plan("engage")`.
- **Loss** (`imitation_loss`), **excluded parameters** (`imitation_parameters`:
  encoders and heads only, log-stds and critic frozen during imitation),
  **fit loop** (`fit`), **DAgger aggregation** (`Aggregate`), **labels**
  (action type, normalized target, power via
  `plan_teacher_tensor_actions_indices`), **held-out label error**
  (`label_error`) — all from `full_authority_imitation.py`, unedited.
- **Critic precondition** (`warm_start_critic_mc`, `predict_values`,
  `critic_metrics`, `critic_gate`, `gate_conditions`,
  `bootstrap_predictive_r2`, `fold_seeds`, `collect_fold`) from
  `full_authority_train_v1.py`, unedited — reused directly for condition C;
  reimplemented (not edited) for condition M's mixture folds (§4).
- **Opponent override** (`opponent_transfer.scenario_override`), unedited —
  the only way any opponent other than the module-level default
  (`RandomAgent`) is ever selected. **Paired-world bootstrap**
  (`opponent_transfer.world_paired_difference`), unedited — reused for
  both the precondition gap and the primary comparison (§6).
- **Optimizer seeds:** 97101, 97102, 97103 — unchanged across every stage of
  this track, so every comparison in this file is paired by seed as well as
  by world.

`full_authority_imitation.py` and `full_authority_train_v1.py` are **not
edited**. Neither is digest-pinned by E3, but both are shared by every
downstream stage (R1n-e/f/g's `declaration.json` files record their
digests); editing either would silently change what those archived runs'
provenance claims mean. R1n-h adds one new module,
`mixture_imitation.py`, that imports from both.

## 2. New: the mixture/control training design

- **`collect_mixture(client, cfg, model, round_seeds_by_arm, source, account)`:**
  for each arm in `{"random": {}, "easy": {"redController": "scripted", "redDifficulty": "easy"}}`,
  wraps `fi.collect(...)` in `scenario_override(arm_spec)` on that arm's own
  seed sub-block, then concatenates the two `(episodes, part)` results
  (`torch.cat` per tensor key, matching `Aggregate.add`'s own concatenation
  pattern) into one round's data. For condition C, the same function is
  called with only the `"random"` arm populated (a 128/0 split, not a
  separate code path) — one collection routine, not two.
- **Round 0 is shared across the three optimizer seeds *within* a
  condition** (unchanged from R1n-c: it does not depend on any weights),
  but is **collected separately per condition**, since the two conditions'
  opponent exposure differs by construction. Two round-0 collections total,
  not six.
- **Rounds 1–4** (DAgger): collected per seed per condition (6 total
  per round, matching R1n-c's "per seed" pattern ×2 conditions), the
  just-fitted policy acting deterministically, the teacher labeling.
- **Round size stays 128 total episodes**, matching R1n-c exactly — for
  condition M this is 64 random + 64 scripted-easy, *not* 128 of each
  (128-of-each would silently double every round's cost against the
  declared budget in §8).

## 3. Evaluation opponents and splits

Three opponents, each with its own fresh 100-world split, **shared across
both conditions** (paired by world for the mixture-vs-control comparison):

| Split | Opponent | Trained on by M? | Trained on by C? | Role |
| --- | --- | --- | --- | --- |
| eval-random | `RandomAgent` | yes | yes | in-mixture check (both conditions trained here) |
| eval-easy | `ScriptedAiAgent` (easy) | yes | no | in-mixture check for M; C's in-distribution-for-R1n-c check |
| eval-normal | `ScriptedAiAgent` (normal) | no | no | **primary**: held-out for both conditions |

`eval-easy` under condition M is the §0 precondition split. `eval-normal`
under both conditions is the primary comparison. A **stochastic** pass on
`eval-random` only (matching R1n-c's `stochasticSplit` precedent, for the
execution-mode-gap flag) is run for both conditions.

**Label error** is recomputed (via `label_error`, unedited) on `eval-normal`
for both conditions — the direct follow-up to R1n-g's finding, asking
whether `throwAimHeadingErrorDegrees` specifically improves for M relative
to C (§6).

**Teacher ceiling and uniform-random floor:** the teacher ceiling is
measured fresh on all three splits (it does not depend on training
condition, so once per split, not once per condition). R1n-f already
measured the teacher at 100% on `ScriptedAiAgent` at both difficulties at
400-world scale on a *different* split (871000–871399); re-measuring on
R1n-h's own `eval-easy`/`eval-normal` worlds is required because the paired
bootstrap methodology (`paired_difference`) needs the ceiling on the exact
same worlds as the policy being compared, not just the same opponent. The
floor (uniform legal-action baseline) is measured once, on `eval-normal`
only — R1n-f/R1n-g already establish that a floor-level result on the
scripted opponents looks like near-total death or timeout, so a floor
control on the other two splits is not needed to interpret the result.

## 4. Critic precondition, per condition

- **Condition C:** `full_authority_train_v1.warm_start_critic_mc`, called
  unchanged, exactly as R1n-c did.
- **Condition M:** `warm_start_critic_mc_mixture` (new, in
  `mixture_imitation.py`) — a controlled reimplementation of
  `warm_start_critic_mc`'s body: `collect_fold` is called twice per fold
  (train, held-out), once per arm under `scenario_override`, using the
  *same* `fold_seeds`-derived ranges split into two contiguous halves by
  arm (train: 256 total → 128 random + 128 easy; held-out: 128 total → 64 +
  64), then the two arms' tensors are concatenated before
  `predict_values`/`critic_metrics`/`critic_gate`/`gate_conditions`/
  `bootstrap_predictive_r2` run **unchanged** on the merged data. Same
  gate thresholds as R1n-c (`gateMinPredictiveR2` 0.25,
  `gateTimeOnlyMargin` 0.05).

## 5. Seed bands (all fresh; repo-wide scan clear)

Reserved block **400000–449999**, scanned two ways: (a) every JSON under
`snowgym/` parsed and walked for integer fields matching
`/seed|partition|allocation/i` (10,474 existing values found repo-wide, in
hundred-thousand buckets 0–2, 6–9, 14 only — 3, 4, 5 entirely unoccupied);
(b) a plain-text scan of `snowgym/training/reviews`, `refs/snowgym_r1n_*.md`,
and `snowgym/PLAN.md` for the 5-digit pattern in this range (three
incidental matches, all inside unrelated decimals like `0.405377`, not
seeds). None of R1n-b's reserved 630000–630119 is touched.

| Purpose | Base(s) | Count | Range(s) |
| --- | --- | --- | --- |
| Round 0/1/2/3/4, condition M | 400000 (+1000/round) | 128 | 400000–400127 … 404000–404127 |
| Round 0/1/2/3/4, condition C | 410000 (+1000/round) | 128 | 410000–410127 … 414000–414127 |
| eval-random (+ ceiling) | 420000 | 100 | 420000–420099 |
| eval-easy (+ ceiling) | 421000 | 100 | 421000–421099 |
| eval-normal (+ ceiling) | 422000 | 100 | 422000–422099 |
| stochastic eval-random | 423000 | 100 | 423000–423099 |
| floor, eval-normal only | 424000 | 100 | 424000–424099 |
| critic M train fold (`fold_seeds`, stride 1000, ×3 seeds) | 430000 | 256/seed | 430000–430255, 431000–431255, 432000–432255 |
| critic M held-out fold | 435000 | 128/seed | 435000–435127, 436000–436127, 437000–437127 |
| critic C train fold | 440000 | 256/seed | 440000–440255, 441000–441255, 442000–442255 |
| critic C held-out fold | 445000 | 128/seed | 445000–445127, 446000–446127, 447000–447127 |

The mixture round/critic ranges above are the *full* per-round or per-fold
range; §2/§4 split each in half by arm (first half random, second half
easy) rather than introducing separate bases per arm — one declared range
per round/fold, not four.

## 6. Decision rules and predictions (stated for falsification)

Let `gap(X, opponent)` be the seed-averaged, world-paired success
difference between condition `X`'s final policy and the teacher ceiling on
that opponent's split (`paired_difference`, 10,000 resamples, seed
973001 — reused from R1n-c/f's bootstrap seed convention, itself already
clear of any reserved band).

**Precondition (must pass before the primary is interpreted):**
`gap(M, eval-easy)` mean is not more negative than R1n-c's own
`failThreshold` precedent (`−0.50`, reused from R1n-f). If it fails, the
outcome is **`precondition-failed`**: mixture training did not even reach
the opponent it trained on, and the held-out comparison below is reported
but flagged uninterpretable.

**Primary:** `gap(M, eval-normal) − gap(C, eval-normal)`. Since both gaps
share the same teacher-ceiling term, this equals M's seed-averaged success
minus C's seed-averaged success on `eval-normal`, directly — computed by
calling `opponent_transfer.world_paired_difference(M_outcomes_by_seed,
C_seed_averaged_outcomes_by_world, cfg, "success")` **unchanged, imported**
(not reimplemented), which bootstraps over the 100 shared worlds, not the 3
seeds (the exact axis R1n-f's amendment A5 fixed). The teacher's own
`eval-normal` ceiling is not part of this call at all — it is only needed
for the precondition gap above.

| Primary result | Outcome | Meaning |
| --- | --- | --- |
| Precondition fails | `precondition-failed` | Mixture training itself did not work at this scale; do not read the held-out number as evidence about transfer. |
| Interval excludes 0, mean > 0 | `transfers` | Scripted-easy exposure improves transfer to scripted-normal, an opponent neither condition trained on. |
| Interval includes 0 | `no-detected-transfer` | No evidence the mixture generalizes past its own training opponents at this scale. |
| Interval excludes 0, mean < 0 | `regresses` | Mixture training makes the held-out opponent worse — would need its own explanation before any further curriculum work. |

**Secondary (mechanism, reported regardless of the primary outcome):**
`throwAimHeadingErrorDegrees` on `eval-normal`, condition M vs condition C.
Predictions, stated for falsification:
- M's aim error on `eval-normal` drops below 90° (better than two random
  directions) — the specific finding R1n-g flagged as worse-than-random
  should no longer hold if scripted-opponent exposure transfers at all.
- C's aim error on `eval-normal` reproduces R1n-g's archived arm-N numbers
  within the same rough magnitude (74.9°–148.4° per cell) — a same-ballpark
  check analogous to R1n-g's own arm-R cross-check (§8 of that
  declaration), confirming fresh seeds did not change the qualitative
  finding for the unchanged (control) condition.
- Throw recall (`perType["throw"]["recall"]`) on `eval-normal` improves for
  M over C, but R1n-g's other finding — that recall alone does not explain
  the zero-hit result — predicts this improvement is smaller in relative
  terms than the aim improvement, if the aim fix is what's actually doing
  the work.

**Flags** (reused from R1n-c's convention, computed per condition):
`throwCollapse` (`throwRecallFlag` 0.5, on `eval-normal`), `executionModeGap`
(stochastic vs deterministic success on `eval-random`, flag 0.20),
`seedInstability` (success spread across the 3 optimizer seeds, flag 0.30,
computed per split).

## 7. Artifacts retained

Per condition, per optimizer seed: `fit-{k}.pt` (k=0…4), `fit-history.json`,
`dataset.json` (per-round seeds, row counts, per-type label counts,
`semantic_state_digest`), `episodes.jsonl`/`trajectory-distances.npz` for
every collection and evaluation, `label-error.json` (on `eval-normal`),
`critic-warm-start.json`, `critic-warm-start-arrays.npz`, `critic-policy.pt`.

Run-level: `declaration.json` (config, source digests including
`mixtureImitationImplementationDigest`, `imitationImplementationDigest`,
`trainImplementationDigest`, pinned E3 digests), `controls/` (ceiling and
floor per split), `report.json` (decision rules, both conditions' gaps,
the primary comparison with its bootstrap interval, the secondary
label-error comparison), `manifest.json` sealed with
`death_rate_ppo.seal`/`verify_sealed` (imported, not reimplemented).

Raw aggregated observation tensors are not retained (same reasoning as
R1n-c §7: each round is regenerable from its declared seeds and the prior
fit's checkpoint).

## 8. Budget

| Item | Decisions (bound) |
| --- | ---: |
| Round 0, ×2 conditions: 2 × 128 × 200 | 51,200 |
| Learner rounds 1–4, ×3 seeds ×2 conditions: 2 × 3 × 4 × 128 × 200 | 614,400 |
| Deterministic eval, ×2 conditions × 3 splits × 3 seeds × 100 × 200 | 360,000 |
| Stochastic eval-random, ×2 conditions × 3 seeds × 100 × 200 | 120,000 |
| Teacher ceiling, ×3 splits × 100 × 200 | 60,000 |
| Floor, eval-normal only × 100 × 200 | 20,000 |
| Critic precondition, ×2 conditions × 3 seeds × 384 × 200 | 460,800 |
| **Total bound** | **1,686,400** |
| **Declared cap** (enforced by `account()`) | **2,200,000** |

For comparison: R1n-c's cap was 900,000 (single condition, single
opponent); R1n-f's bound was 1,690,000 (no training, seven frozen
comparators over 400-world splits); R1n-e's PPO stage used roughly 8.9M.
This budget is affordable at this track's established scale.

## 9. Stopping rule

- Fixed: five fits, three optimizer seeds, two conditions, evaluation of
  each condition's final policy only. No retries with a different budget,
  seed, loss, or mixture ratio.
- If a run is aborted by an implementation defect before completion, its
  output is discarded (not archived) and the abort is reported; the fix is
  committed before a full rerun (R1n-e's and R1n-f's precedent).
- Output: `runs/m7b_engage_r1n_h_v0/`, which the runner refuses to
  overwrite.
- No provider calls, browser input, TypeScript or protocol changes, or
  edits to `full_authority_imitation.py`, `full_authority_train_v1.py`, or
  any E3-pinned source.

## 10. Verification before the implementation commit

- `collect_mixture` on a synthetic 128-seed round produces exactly 64+64
  episodes/rows from the two arms, in the same aggregate shape `Aggregate.add`
  expects (a unit test constructs two tiny `fi.collect`-shaped parts and
  checks the concatenation).
- `warm_start_critic_mc_mixture`'s merged fold sizes match
  `warm_start_critic_mc`'s single-opponent fold sizes exactly (256/128),
  and calling it with an all-random mixture ratio (128/0) reproduces
  `warm_start_critic_mc`'s own output on the same seeds bit-for-bit (a
  regression guard against the reimplementation drifting from the
  original).
- The seed-band table (§5) contains no overlap between any two rows,
  checked programmatically, not just by eye.
- The primary decision rule (§6) is tested on synthetic gap values for all
  four outcomes, including the precondition-failed short-circuit.
- Budget guard raises past the declared cap.
- A tiny live end-to-end run (both conditions, all rounds, both eval
  splits relevant to the precondition and primary, reduced episode counts)
  completes and seals.
- Full gate: `npm test` (accepting only the documented R1n-b exception),
  `npm run build`, Python client tests, Python training tests, and a check
  that `full_authority_imitation.py` is byte-identical to
  `imitationImplementationDigest` in R1n-g's `declaration.json`, and that
  `full_authority_train_v1.py` is byte-identical to
  `trainImplementationDigest` in R1n-f's `declaration.json` (confirming
  this declaration's "not edited" claim in §1, the same way E3's pin check
  confirms the older sources).

## 11. Amendments made in the implementation commit, before any collection

(None yet — this section is filled in as amendments happen, per this
track's convention.)
