# M8-S12 — throw-decoder redesign: enemy-relative heading, replacing absolute coordinates

Declared 2026-09-22, before any training. Autonomous training, not assisted. `autonomousQualificationEligible`
stays false (development experiment).

## 0. Why, and why not another loss-weight run

An external math/RL review of the whole S1–S11 arc (prompted by the user; summarized in the errata appended to
`m8_s10_results.md`/`m8_s11_results.md`) found the aim-weight branch (S10/S11) should not be extended: raising one
coefficient in `imitation_loss`'s already-isolated five-term sum moves along a Pareto front rather than adding
information (the predicted explanation for S10's move-error side effect), and a cosine loss against a single
absolute-coordinate teacher label structurally cannot represent throw lead — the target moves during flight, and S9
already measured the teacher aiming 1.7–4.0° off the *instantaneous* enemy position while learners sit at 6–10° off.
S8's `throw-all` (100% success) versus `aim`-alone (0.24/0.99/1.00) is the same fact from the intervention side:
timing, heading and power are a joint choice the absolute-coordinate representation cannot separately supervise.

This step changes the representation instead of the loss weighting: the throw head predicts **which living enemy**
and an **angular offset from that enemy's bearing**, not an absolute arena point. No loss coefficient is tuned; all
terms carry coefficient 1 (the reviewed lesson applies: no further ad hoc reweighting).

## 1. What the simulator and S9 already established (reused, not re-derived)

`ThrowSystem.tryThrow` (read in S9) computes the throw direction as the unit vector from thrower to target point;
speed and arc come from `power` alone; **target distance is discarded**. S9 separately found the learners' target
points sit roughly 4× the teacher's distance from the thrower — physically irrelevant, confirming distance carries
no signal worth predicting. The new head therefore predicts a **direction only**; a fixed nominal placement distance
is used only to encode a Gym action point, never trained on.

## 2. Architecture

New module `options/enemy_relative_throw.py`; no existing module is edited, including `full_authority_ppo_v1.py`
(digest-pinned). `FullAuthorityPolicyV1EnemyThrow` is a fresh `nn.Module`, structured like `FullAuthorityPolicyV1`
itself (own `encoders`, `action_head`, `move_head`, `power_head`, `move_log_std`/`power_log_std`, `critic =
OptionCentralCritic(...)`), binding `FullAuthorityPolicy.pair_features`/`.features`/`.decode_move` unchanged at the
class level — the same reuse-by-binding pattern `FullAuthorityPolicyV1` itself uses for E3's methods, applicable
here because neither method touches `self` beyond `self.encoders`, which this class also defines with the same
shape. Action type, movement and power are **byte-for-byte the same computation** as `FullAuthorityPolicyV1`; only
the throw pathway differs:

- **Enemy scorer:** `Linear(21, 32) → ReLU → Linear(32, 1)`, applied to each living-or-not enemy slot's
  `pair_features(observation, "enemies")` row (the per-slot, thrower-relative feature vector V1 already computes
  and discards after pooling), masked to living enemies (`enemy_mask & alive`) and filled `-1e9` elsewhere — a
  purely geometric scorer, not conditioned on the unit's broader plan/role context. This is a scope reduction,
  stated explicitly: a context-aware scorer is a candidate for later if selection quality specifically limits
  results.
- **Offset head:** `Linear(227, 64) → Tanh → Linear(64, 2)`, the same shape as V1's retired `throw_head`, now
  producing a 2D vector interpreted as `(cos, sin)` of an angular correction, not an absolute coordinate.
- **Composition:** final throw direction = the chosen enemy's bearing, rotated by the offset (2D unit-vector
  composition — `atan2` and rotation done directly on the vectors, never on a scalar angle, so there is no branch
  cut in the forward pass). A world-unit relative position is required for this to be a physically correct
  bearing — **the arena is not square** (100×80, half-extents (50, 40)), so bearings must be computed in world
  units, not the observation's arena-normalized units, or angles distort along the shorter axis. This is checked by
  a dedicated asymmetric-arena unit test (§5).

**Critic-fit randomness.** `mi.warm_start_critic_mc_mixture` seeds its minibatch generator from
`cfg["trainingRngs"][0]`; each (cohort, decoder) run passes its own value (the cohort's optimizer seed, offset for
the new decoder) so this is not silently shared across all 6 runs, the class of undeclared shared randomness S4's
declaration flagged explicitly. This affects only the critic warm-start fit, not the imitation training or the
paired-eval comparison.

`throw_offset_log_std` (a new frozen-during-imitation parameter, named to match `fi.imitation_parameters`'s existing
`not name.endswith("log_std")` filter — reused unchanged, no edit needed) replaces `throw_log_std`; initialized to a
placeholder (`-1.0`), not calibrated, since no PPO exploration uses it yet.

**Everything else is reused unchanged:** `fi.collect`, `fi.Labeler`, `mi.collect_mixture`,
`mi.warm_start_critic_mc_mixture`, `v1.predict_values`, `fi.imitation_parameters` all operate on `model.act()`
returning the same `{action_type, target, power}` schema and `model.critic(...)` returning a scalar value — neither
depends on how `target` was computed internally, so they run against the new class exactly as they do against
`FullAuthorityPolicyV1`, with no reimplementation.

## 3. Loss

`imitation_loss_enemy_relative` (new function): `type_loss`, `moveHeading`, `moveEndpoint`, `power` are copied
unchanged from `fi.imitation_loss` (same masks, same formulas — verified identical by a regression test against
`fi.imitation_loss`'s own values on the shared terms). The throw term is replaced by two, **both required by the
new action space, not a tuning choice**:

- **Enemy-selection loss:** cross-entropy between the model's masked enemy logits and the teacher's implied choice
  — the living enemy whose world-relative bearing is angularly nearest the teacher's labelled throw direction
  (the same "nearest bearing" rule S9 validated as `decompose`, now computed batched in the loss rather than in a
  diagnostic script). Computed only on THROW-labelled rows with a nonzero teacher direction and at least one living
  enemy.
- **Angle loss:** `1 − cos` between (a) the teacher's own labelled direction and (b) the model's offset composed
  with the **teacher's** chosen enemy's bearing (teacher-forced — computed independent of what the model's own
  enemy head predicts, so this term is well-posed from the first gradient step rather than coupled to a discrete,
  initially near-random selection). Same row mask as the enemy-selection loss.

`total = type_loss + moveHeading + moveEndpoint + enemy_loss + angle_loss + power`, coefficient 1 throughout. `fit`
is reimplemented (as S10's was) because `fi.fit` hardcodes a call to the unweighted, throw_raw-shaped
`fi.imitation_loss`.

## 4. Design: independent training cohorts, not repeated weight-inits

The reviewed lesson: "seed" in S5/S10/S11 meant weight-init only, sharing training data; that is not replication.
This step uses **3 independent training cohorts**, each a fresh seed band (round-zero, DAgger rounds, critic
folds) with its own optimizer-init seed, one blue-Engage learner each (not three per cohort). Within each cohort,
the **old** decoder (`FullAuthorityPolicyV1`, `fi.fit`/`fi.imitation_loss` unchanged — S5's own recipe, unweighted)
and the **new** decoder are trained on identical round-zero data, from the SAME optimizer-init seed — a paired
comparison, the same design S10/S11 used, now replicated 3 times independently instead of once.

**This is still underpowered relative to what the review asked for (≥5 independent cohorts) — chosen at 3 for
wall-clock cost (≈2–2.5 hours for the full 3×2 batch), and stated as underpowered, not adequate, here.** A result
that looks directionally promising across 3 cohorts is grounds for extending to more cohorts in a following
declaration; a null or mixed result at n=3 is reported as inconclusive, not as evidence of failure, since 3
cohorts cannot rule out a real but noisy effect either.

Fresh seed bands (audited against every tracked file, no overlap): cohort *i* (1, 2, 3) uses round base
`5{i}00000`, critic train `5{i}10000`, critic held-out `5{i}15000` — i.e. 5100000/5110000/5115000,
5200000/5210000/5215000, 5300000/5310000/5315000. Paired evaluation reuses S4/S5's worlds, seeds 2100000–2100099
(the same 100 worlds S5, S7–S11 all used), against scripted-normal and scripted-easy Red, matching S10/S11's scope.

## 5. Verification before training

Hand-computed unit tests: `rotate`/`decompose`-equivalent batched geometry against known angles; an **asymmetric
arena test** (width ≠ height) confirming bearings and composed directions match world-space angles, not
normalized-space-distorted ones (the bug class identified while designing this); the enemy-selection and angle
losses against a synthetic observation with a known teacher label and known enemy positions, hand-computed
expected loss values; a zero-living-enemy row producing finite (not NaN) outputs and being excluded from both loss
terms; a regression test that the reused terms (`type`, `moveHeading`, `moveEndpoint`, `power`) exactly match
`fi.imitation_loss`'s own values on shared synthetic input. Live: a tiny end-to-end BC+DAgger+critic run at both
decoders; `act()`'s deterministic path exercised through real `fi.collect` calls (the actual code path DAgger and
evaluation use — S5's whole pipeline always calls `deterministic=True`, confirmed by reading `fi.collect`'s default
and `mi.train_condition`'s call sites, so this step implements the stochastic sampling path minimally, not deeply
tested, since nothing in this run exercises it).

## 6. What this does and does not decide

It decides whether an enemy-relative throw representation, trained with the standard unweighted imitation recipe,
narrows the scripted-normal gap on **3 independent cohorts**, compared paired against the old decoder on the same
data. It does not attempt PPO, does not address 3v3 credit assignment (flagged by the review as a separate,
unaddressed structural question — success/`L`/the critic are team-level throughout this track; this step still
supervises per-unit actions directly via imitation, so team-level credit assignment remains out of scope until a
PPO step returns), does not add a context-aware enemy scorer, and does not claim adequate statistical power. A
promising result here is grounds for more cohorts, not for promoting a checkpoint.

## 7. Archive and gates

Sealed archive `runs/m8_s12_enemy_relative_throw_v0/{cohort-1,cohort-2,cohort-3}/{old,new}/`. Declaration pins the
S4, S5 manifests (paired-eval worlds and the R1n imitation recipe reference), E3 digests, and this module's own
digest. Complete gates before training: client and training `pytest`, `npm run build`, `npm test` (366/367, the one
documented exception). Results in `reviews/m8_s12_results.md`.
