# R1n-i — frozen-checkpoint failure diagnosis: contact vs. finishing, deployed actions, no training

Declared 2026-09-18, before any collection. No policy is trained, updated,
fine-tuned, or newly evaluated for success/death here. This reproduces
R1n-h's own archived `eval-normal` episodes on its own seeds, then reads
one level deeper than `label_error` or R1n-h's report ever did, from data
that already exists inside a `record=True` collection call but was
discarded before being written to disk.

## 0. Why this exists, and what it does not decide

R1n-h's mixture condition (M) transfers to `eval-normal` on average (+38.0
points vs. control), but its three optimizer seeds diverge wildly: success
0.00 / 1.00 / 0.14. An erratum to `m7b_r1n_h_results.md` (2026-09-18)
corrected the original characterization of this divergence — label error
is *not* uniform across the three seeds (97102's throw-aim error is ~18×
smaller than 97101's/97103's), and the two failing seeds fail in
different, archive-verifiable ways:

- **97101 is a contact failure.** It damages the target in only 12/100
  episodes despite 0.878 teacher-relative throw recall.
- **97103 is a finishing/survival failure.** It damages the target in
  100/100 episodes, averaging 62.8 damage, but dies in 86/100 before
  finishing.

The erratum also established, by reading `train_condition()` directly,
that M's better critic-warm-start R² cannot be the cause (it is fit
strictly after these eval outcomes are recorded, and R1n-h runs no PPO).

What remains unexplained is *why* an ~18× difference in an aggregate,
teacher-conditioned error metric corresponds to an near-total difference
in win/loss outcome. `label_error` cannot answer this by construction: its
aim/power/move terms are computed only on states where the **teacher**
selects that action type, regardless of what the model itself chose there
(`full_authority_imitation.py:212-256`, unedited, same conditioning R1n-g's
declaration already documented for a different pair of checkpoints). It
therefore cannot say whether the model's own executed throws — the ones
the simulator actually ran, that these win/loss numbers are made of — were
aimed at the real enemy, thrown in range, or thrown at all.

R1n-i's contribution is narrow: recover the per-decision data a
`record=True` collection call already computes but `train_condition()`
discards, and use it to build a model-conditioned, geometry-grounded view
(aim relative to the *actual* enemy position, not the teacher's label; the
model's *own* selected action, not the teacher's) of the same eval-normal
episodes already archived. This is a deeper read of existing behavior, not
new behavior.

**R1n-i decides nothing about R1 qualification, trains nothing, selects no
checkpoint, and authorizes nothing.** `autonomousQualificationEligible`
stays false throughout.

## 1. Scope and comparators

All six R1n-h checkpoints, loaded as `train_condition()`/
`condition_outcomes_by_seed()` already do:

| Condition | Seeds | Checkpoint |
| --- | --- | --- |
| M (mixture) | 97101, 97102, 97103 | `fit-4.pt` |
| C (control) | 97101, 97102, 97103 | `fit-4.pt` |

`fit-4.pt` is used, not `critic-policy.pt`: it is the exact actor state
`train_condition()`'s `eval-normal-deterministic` numbers were computed
from, saved strictly before the critic warm-start step that follows it
(verified in the erratum). Only `eval-normal` is in scope — the split
where the divergence is observed. `eval-easy` and `eval-random` are not
re-collected here; if the reproduction step (§2) suggests they are needed,
that is an amendment, not an assumption made now.

## 2. Method: reproduction first, then new instrumentation

**Step A — reproduction gate.** For each of the 6 checkpoints, re-run
`fi.collect(client, eval_seeds(cfg, "normal"), cfg, model=<frozen
checkpoint>, source=..., block_worlds=cfg["evaluationBlockWorlds"],
account=account, record=True)` under
`scenario_override(EVAL_ARMS["normal"])` (both imported unchanged — `fi`
from `full_authority_imitation.py`, `scenario_override`/`EVAL_ARMS` from
`opponent_transfer.py`/`mixture_imitation.py`), on the **same** seeds
R1n-h already archived — `mixture_imitation.eval_seeds(cfg, "normal")` =
422000–422099. `block_worlds` is pinned to `cfg["evaluationBlockWorlds"]`
= 50, the exact value `train_condition()` used (two blocks of 50 for 100
seeds) — not a free choice. `run_block` requires `wrapper.batch_size ==
len(seeds)` (`full_authority_train_v1.py:163`), so any `block_worlds`
value would technically run without error, but a different chunking could
still change batched floating-point results even though each world's
dynamics don't couple to the others (the same class of "get the batch
composition exactly right or the comparison means something else" issue
`validate_critic_fold_sizes` exists for in `mixture_imitation.py`).
Matching it exactly removes the question rather than needing to argue it
away. This is a reproduction of already-collected worlds under the
frozen, deterministic, `model.eval()` checkpoint, not new data.

**Mandatory gate before Step B is interpreted for any checkpoint:** the
regenerated `episodes.jsonl`'s per-world `success`/`blueAliveAtEnd` must
match the archived `episodes.jsonl` exactly, for all 100 worlds, for that
checkpoint. This is the same "agreement with archived outcomes" discipline
R1n-f's amendment used for its reimplemented stepping loop, applied here
to a reimplemented *collection call* rather than a stepping loop. A
mismatch on any checkpoint halts analysis for that checkpoint and is
reported as a measurement problem, not folded into the finding.

**Step B — recover per-decision detail.** `fi.collect(..., record=True)`
returns `(episodes, part)`, where `part["observation"]` holds the full
per-decision, per-unit input tensors (`ACTOR_KEYS`: `allies`, `enemies`,
`projectiles`, `obstacles`, their masks, and the plan-state tensors — the
actor's actual input) and `part["labels"]` the teacher's per-decision
label. This is exactly what `label_error()` consumes; `train_condition()`
computes `label_error()`'s aggregate summary from it and then deletes
`part` (`del part`, `mixture_imitation.py`). R1n-i is the first stage to
retain and read `part` directly instead of only its aggregate.

From `part["observation"]`, with no new simulation, R1n-i recomputes:

- **The model's own action per decision** — `model(observation,
  with_value=False)`, `action_logits.argmax(-1)` for the chosen type, and
  the same decode calls `label_error`/`imitation_loss` already use:
  `model.decode_move(observation, prediction["move_raw"])` for movement,
  `torch.tanh(prediction["throw_raw"]) * scale` for the throw aim point.
  This is not merely the same *distribution* as Step A — it is a
  recomputation on the **identical tensor object**. In `run_block`, the
  per-step `rows` dict (`full_authority_train_v1.py:176`) is passed
  unmodified to both `choose(wrapper, active, rows, raws)` (the live
  action, line 183) and `stored.append({"observation": rows, ...})` (line
  186), and `collect()` builds `part["observation"]` directly from
  `stored`. So `part["observation"]` **is** what `Labeler.__call__`/
  `model.act` saw live, not a copy or a re-derived approximation — Step B
  re-runs the same frozen, deterministic forward pass on the same input,
  and (confirmed by the Step A reproduction gate) gets the same action.
- **Ground-truth enemy position**, from `observation["enemies"][...,
  2:4]` masked by `observation["enemies"][..., 1] > .5` (the presence
  flag) — the same position-feature convention `FullAuthorityPolicy.
  pair_features` documents (`executor/full_authority_ppo.py`) and
  `label_error`/`imitation_loss` already rely on for `observation
  ["allies"][..., 2:4]` as the unit's own position. This gives an
  aim-error measure relative to the **actual enemy**, independent of
  whether the teacher happened to label that decision THROW — the gap
  the erratum names explicitly.
- **Incoming-threat exposure**, from `observation["projectiles"]`. Feature
  index 1 is a team flag (`-1.0` blue, `1.0` red — `encode_projectile`,
  `snowgym_client/encoding.py:392-393`), and index 2:4 is world position
  (consistent with `pair_features`'s `coordinate=2` convention for this
  tensor). Filtering `observation["projectiles"][..., 1] > 0` masked by
  `observation["projectile_mask"]` isolates red's live projectiles from
  blue's own outgoing throws at every decision, giving a genuine incoming-
  threat signal, not an unattributed count.

Every Step B quantity is reported **alongside**, not instead of, the
already-archived `label_error` output for the same checkpoint. Nothing
here revises R1n-g's or R1n-h's archived numbers.

## 3. Measures, mapped to the two named failure modes

**Contact-failure diagnostic** (targets 97101-style outcomes): for
episodes with no target damage, does the model ever select THROW near an
enemy? Reported as the model's own THROW-selection rate binned by
distance-to-nearest-enemy, and — from the already-archived `episodes.jsonl`
fields `minDistance`/`meanDistance`, no regeneration needed — whether an
episode's minimum approach distance ever reaches `ENGAGE_RANGE` = 9
(`SimpleBlueAgent.ts`, the scripted teacher's own effective throw range),
cross-tabulated against whether that episode ever contains a close-range,
model-selected THROW. This separates "never gets close" (an
approach/movement failure) from "gets close but doesn't throw, or aims
badly when it does" (a decision/aim failure) — two failures `label_error`'s
own conditioning cannot tell apart.

**Finishing-failure diagnostic** (targets 97103-style outcomes): for
episodes with contact, decision-count from `firstHitDecision` to
`finalDecision`/death (already archived, no regeneration needed), and,
from Step B, whether the model keeps moving while a red projectile is
incoming during that window — the mechanism R1n-e's exploratory appendix
already associated with survival (`r1n-e-what-ppo-learned`). R1n-i checks
whether that mechanism is present or absent here; it does not invent a new
hypothesis.

**Reference case:** 97102 (M, success 100/100) receives the identical
treatment on both diagnostics, not because it needs explaining, but
because "close-range throw rate" and "movement under incoming threat" are
only interpretable next to a working policy measured the same way.

**Condition C reference:** all three C seeds (uniformly 0/100 success,
0 contact episodes each, per the erratum's verified table) receive the
contact-failure diagnostic, primarily to check whether C's failure
signature is homogeneous the way M's split is not.

## 4. Predictions, stated for falsification

- If 97101 is a pure approach/movement failure: `minDistance` stays above
  `ENGAGE_RANGE` in most non-contact episodes, and where it does reach
  range, the model's own close-range THROW-selection rate is comparable
  to 97102's.
- If 97101 is instead a decision/aim failure despite reaching range:
  `minDistance` frequently falls at or below `ENGAGE_RANGE`, but the model
  rarely selects THROW there, or its ground-truth aim error (§2 Step B) is
  large even on the throws it does select.
- If 97103's deaths follow R1n-e's movement-during-incoming-threat
  mechanism: its post-first-hit episodes show measurably less continued
  movement under incoming projectiles than 97102's, at comparable
  exposure counts.
- If none of the above hold cleanly, R1n-i reports the full descriptive
  breakdown without forcing a single-cause conclusion — R1n-g's own
  precedent ("three compounding failures, not one") is the reason this is
  stated as a real possibility, not a fallback disclaimer.

## 5. What this explicitly does not decide

- **No checkpoint is selected, ranked, or recommended for promotion or for
  a future PPO stage.** The 2026-09-18 review's warning applies directly:
  any selection rule needs its own prior declaration and a fresh
  evaluation allocation, not a rule inferred after seeing which seed
  scored well here.
- **No new training-cohort or volume-controlled comparison** — R1n-h
  amendment A1's open item (mixture vs. more total volume) stays a
  separate, undecided declaration.
- **No claim about `eval-easy` or `eval-random`** beyond what R1n-h
  already archived.

## 6. Provenance and digest pins

At declare time, cross-checked against the digests already recorded in
earlier declarations, raising `RuntimeError` on any mismatch (the pattern
R1n-h's own `declare()` established):

- `full_authority_imitation.py` against R1n-g's/R1n-h's recorded digest.
- `full_authority_train_v1.py` against R1n-f's recorded digest.
- `mixture_imitation.py` against R1n-h's own `declaration.json`. R1n-h is
  itself now a frozen, imported-not-edited dependency for this stage — the
  same status the standing convention already gives the two files above,
  extended here because R1n-i reuses `mixture_imitation.eval_seeds`,
  `EVAL_ARMS`, and `configuration()` directly.

`opponent_transfer.scenario_override` is reused unchanged for the opponent
arm switch.

## 7. Artifacts retained

Per checkpoint: the regenerated `episodes.jsonl` (Step A), a
`deployed-action-diagnostic.json` (Step B's aggregated measures), and the
reproduction-agreement result. A top-level `report.json` covering all 6
checkpoints' measures and the reproduction-agreement table. A sealed
SHA-256 manifest, reusing `death_rate_ppo.seal`/`verify_sealed` (imported,
not reimplemented).

## 8. Budget and stopping

- 6 checkpoints × 100 episodes × up to 200 decisions (the Engage horizon)
  ≈ 120,000 decisions, a conservative upper bound — R1n-h's own
  `eval-normal` collection used far fewer in practice, since most episodes
  end well before the horizon (contact, death, or teacher-ceiling
  completion).
- Bound: 130,000. Cap: 200,000, enforced in code.
- No training: a single run-wide counter, no per-policy split.

## 9. Seeds and generators

- Reuses `mixture_imitation.eval_seeds(cfg, "normal")` = 422000–422099
  exactly — a reproduction of worlds R1n-h already declared, audited, and
  archived. No new seed band, no new `auditSeedDocuments`/repo-wide scan
  required.
- No bootstrap sampling: descriptive point estimates and per-episode/
  per-decision breakdowns, matching R1n-g's precedent for this kind of
  diagnostic.

## 10. Verification before the implementation commit

- Python training tests, Python client tests, `npm run build`, `npm test`
  (documented R1n-b preflight exception only — no new seeds, so no new
  collision is possible).
- The Step A reproduction gate (§2) is run and must pass — exact per-world
  success/`blueAliveAtEnd` match against the archive, for all 6
  checkpoints — before any Step B number is interpreted.
- A test that re-derives `label_error`'s existing recall/aim numbers from
  the regenerated `part` for at least one checkpoint and confirms they
  match the archived `label-error.json` for that checkpoint, before
  trusting the new ground-truth-conditioned numbers built on the same
  `part` data.

## 11. Amendments

**A1 (2026-09-20, before collection):** implementation found that `fi.collect`'s
`part` (§2 Step B) does not, in fact, retain per-decision world/episode
attribution. `run_block`'s `stored` computes a world index and a
within-episode decision index per step (`full_authority_train_v1.py:186`),
but `fi.collect` discards both when concatenating `stored` into `part` —
it keeps only `observation`/`labels`, pooled across every decision from
every episode with no way to say which row came from which world. This
does not affect the *aggregate* measures §3/§4 use for the contact-failure
diagnostic (in-range throw rate, ground-truth aim error), which never
needed per-episode grouping. It does affect the finishing-failure
diagnostic, which needs each decision's position relative to that
episode's `firstHitDecision`/`finalDecision` window.

Rather than editing `full_authority_imitation.py` (pinned) or leaving the
finishing-failure diagnostic unmeasurable, R1n-i adds
`collect_with_attribution` to `checkpoint_failure_diagnostic.py`: the same
`v1.make_wrapper`/`v1.run_block`/`fi.Labeler` primitives `fi.collect` itself
uses, unchanged, plus `v1.flatten_block`/`v1.concatenate` (imported
unchanged — the same pair `mixture_imitation.collect_fold_mixture` already
reuses for an analogous purpose) to keep the world/decision attribution
`run_block` already computes. This is additive reuse, not a reimplemented
stepping loop, and is verified equivalent to `fi.collect`'s own
`part["observation"]`/`part["labels"]` for the same seeds and model
(bit-for-bit, `torch.equal`) before being trusted for anything new.

A consequence: Step A and Step B no longer need two separate collection
calls per checkpoint. `collect_with_attribution`'s own `episodes` output is
exactly what the Step A reproduction gate checks against the archive, so
one collection call per checkpoint serves both steps — a simplification
from §2's original two-call description, not a scope change. §7's artifact
list is adjusted to match: the reproduction gate and the Step B measures
are written to one `checkpoint-report.json` per checkpoint (containing
`reproductionGate`, `labelErrorRecheck`, `contactFailure`, and
`finishingFailure`), alongside the regenerated `episodes.jsonl`, rather
than two separate files.

**A2 (2026-09-20, before collection, live-test finding):** the reproduction
gate's semantics were tightened during test-writing. As originally
written, comparing regenerated worlds against the full archived set would
read any world not regenerated (e.g. a smaller test slice) as a mismatch.
`reproduction_gate` now checks only the worlds actually regenerated against
their archived counterpart; on the real 100-episode collection this is
unchanged (all 100 are regenerated, so it reduces to a full check).

**A3 (2026-09-20, before collection, live-test finding):** a first version
of the test suite's `tiny()` configuration shrank `optionHorizon` for
speed, alongside `block_worlds` correctly staying pinned to the archived
value per §2. This made the reproduction gate fail for a real checkpoint —
not because of a measurement bug, but because a shorter horizon changes
when episodes time out, exactly the kind of confound §2's `block_worlds`
pinning already guards against for a different knob. `optionHorizon` is
not overridden in any test now; it stays at the archived value (200)
throughout, confirmed by two live tests passing cleanly afterward
(`test_collect_with_attribution_matches_fi_collect_bit_for_bit`,
`test_tiny_end_to_end_run_checkpoint_and_aggregate`).

**A4 (2026-09-20, before collection, advisor review):** four corrections to
`deployed_view`/§3's diagnostics, made before any collection ran:

1. Enemy presence must be gated by `enemy_mask` (padding), not only the
   alive-flag feature — `features()` composes both; the implementation's
   first draft omitted the mask. `deployed_view` now raises `RuntimeError`
   if more than one enemy is ever live in a decision, rather than silently
   summing two positions into a meaningless `enemy_pos`.
2. `deployed_aim_error` is now NaN-filled on decisions with no live enemy,
   structurally, rather than relying on every downstream reader to apply
   `model_throws`'s masking correctly.
3. §3's contact-failure diagnostic could not, as first built, separate
   "never reaches range" from "reaches range but doesn't throw" if one
   checkpoint's in-range decision count is tiny — a pooled rate over a
   handful of decisions would look like any other rate. `contact_failure_
   summary` now reports per-episode counts (how many of the 100 episodes
   ever have an in-range decision, and the same for a close-range throw),
   not only the pooled rate.
4. §3's "keeps moving while a red projectile is incoming" measure had no
   distance check — "incoming" meant only "a live red projectile exists
   anywhere in the arena," a weak proxy for a direct threat to the acting
   unit. `finishing_failure_summary` now reports two honestly distinct
   versions: "anywhere" (the original, no distance check) and "nearby"
   (`red_nearby`, gated by `engageRange`, the same reference distance §3
   already uses for the contact diagnostic). Neither checks a
   projectile's trajectory or time-to-impact.

`deployed_view` is also now chunked over decisions (`chunk=2048`,
matching `label_error`'s own chunking) rather than running one forward
pass over the whole checkpoint's collected rows — R1n-e's collection notes
recorded ~3.6 GB peaks on full-rollout forward passes over comparably
sized row counts, and the host kills low-memory background tasks.
