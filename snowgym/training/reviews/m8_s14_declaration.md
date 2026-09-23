# M8-S14 — KL-anchored, Monte Carlo PPO continuation from the M8-S12 enemy-relative throw initializer (3v3 Engage)

This protocol is committed before implementation and before any collection, following this repo's standing
convention (R1n-b through R1n-e, S4, S12): the tested implementation lands in a separate commit; that commit may
amend this file, stating each change, before any collection. **This commit contains the declaration only — no
code, no training.** The user asked to declare this step now; whether to implement and run it is a separate,
later decision.

`autonomousQualificationEligible: false`. The initializers were trained on coded-teacher labels; at runtime each
policy acts alone.

## 0. Relation to prior work, and why now

S12 produced the first 3v3 imitation initializer in this track that is viable AND reliable by S5's own two-tier
rule (cohort-mean scripted-normal success 0.537, all three cohorts ≥ 0.25). S4 measured random-init floors (no
PPO was run there — it showed a random-initialized policy has no route to success at 3v3, not that PPO itself
fails); no PPO of any kind has ever run at 3v3 in this track. S12+S13 together are the first point in this track
where a PPO **continuation** from an initializer is worth considering. This is the only PPO-from-imitation precedent anywhere in this repo:
[`m7b_r1n_e_declaration.md`](m7b_r1n_e_declaration.md) (1v1 Engage, `runs/m7b_engage_r1n_e_v0`), and this step
reuses its design wherever it transfers — KL-anchored clipped-surrogate PPO with pure Monte Carlo advantages, a
frozen reference policy, and an outcome-blind calibration procedure for anything that isn't already precedented.
Two things do not transfer and are addressed explicitly below: R1n-e's initializer has real log-probabilities;
S12's `FullAuthorityPolicyV1EnemyThrow.act()` returns a placeholder zero logp (§3). R1n-e had a prior diagnostics
step (R1n-d) measuring σ sensitivity before committing to a recipe; this step has no such precedent and predeclares
an equivalent outcome-blind probe instead of assuming R1n-e's σ×0.5 transfers untested (§6).

**Credit-assignment limitation, carried forward from S12 §6 and not addressed here:** success, `L`, and the critic
are team-level throughout this track. This step supervises the team's shared advantage broadcast to all three
units' per-unit log-probabilities (`ppo_loss`'s existing per-unit ratio machinery, unchanged) — it does not
introduce or claim any form of individual credit assignment. This is a known, stated limitation of the whole M8
track, not something this step attempts to fix.

## 1. Fixed inputs

- **Scenario, roster, option, horizon:** 3v3 Engage, exactly S4–S13 (`optionHorizon = 200`, `gamma =
  0.9976921765`, same as R1n-e's 1v1 value — the Engage option's discount is roster-independent).
- **Reward:** the frozen Engage option reward, read directly from `options/tracker.py` before writing this
  declaration: `executor_reward = mission_reward + 0.1 * combat + shaping`, where `mission_reward` is ±1 on
  success/failure and 0 otherwise, `combat` is clipped normalized damage dealt minus received, and `shaping` is
  potential-based. Identical formula to R1n-e's; confirmed unedited and roster-independent (S3's "existing stack
  runs at 3v3 with zero edits" finding). No reward change is made or bundled here.
- **Initializers:** S12's three independent cohorts' new-decoder checkpoints, `runs/m8_s12_enemy_relative_throw_v0/
  cohort-{1,2,3}/new/fit-4.pt`, indexed by cohort number 1, 2, 3 (not renumbered 0/1/2, to keep provenance
  traceable to S12/S13's own cohort labels). The S12 manifest is verified before loading, exactly as S13 did.
- **Pinned sources:** E3 digests re-checked; `enemy_relative_throw.py` (S12, already sealed) is read, not edited.
  All new behavior goes in a new module, `enemy_relative_throw_ppo.py` (not yet written — implementation is a
  separate, later commit).

## 2. Rollout opponent and the held-out arm — the tightest constraint in this design

S12/S13's imitation recipe trained on `mixture_imitation.MIXTURE_ARMS` (random + easy only); scripted-normal was
the one arm neither decoder ever trained on, which is exactly what made S13's first-draft "generalization" framing
wrong once corrected — random was in-distribution, not held out. That correction directly shapes this design:

**PPO must roll out against scripted-normal, not the random/easy mixture.** The whole point of this step is to
close headroom where headroom exists (S12 normal-arm success 0.74/0.57/0.30, versus easy at 1.00 and random at
0.77–0.87 in S13 — already near-saturated). Rolling out against random/easy, matching the imitation recipe, would
mostly reinforce behavior against opponents the decoder is already strong against and would not directly optimize
the metric this whole track has used as its primary viability bar.

This means **scripted-normal cannot also serve as a held-out generalization arm for this step** — training and
evaluation both draw from the same opponent identity, exactly as R1n-e's own training and evaluation both drew
from the frozen 1v1 Engage environment. The held-out axis here is **world seeds**, not opponent identity: training
and evaluation use disjoint seed bands (§13), and evaluation additionally uses a seed band untouched by any prior
step — not 2100000–2100099, which every step since S4 has read. This mirrors R1n-e's own fresh split (870000–
870399), not a departure from precedent.

The simulator does expose a third scripted difficulty, `redDifficulty: 'hard'` (confirmed by reading
`SnowGymService.ts`'s validation and `PlanRolloutDataset.ts`'s type), which no step in this whole repository has
ever used in `EVAL_ARMS`/`MIXTURE_ARMS`/`ARMS`. It is a genuinely untouched opponent identity and would be the
right instrument for an opponent-generalization check analogous to what S13 mistakenly thought `random` already
was. **It is out of scope here**, deliberately: using an unvalidated, never-before-measured difficulty tier as
the vehicle for this step's primary claim would import a second unknown (is `hard` well-posed at all — does the
teacher even beat it consistently) on top of the first PPO attempt at 3v3. If this step produces something worth
generalization-testing, a `hard`-arm check is a natural, cheap S15-style follow-up (mirroring S13's role for S12),
requiring its own small teacher-achievability check first (mirroring S4's opening move), but that is explicitly
not this step.

**Critic warm-start restart matches the rollout arm.** S12's critic was warm-started on the mixture recipe
(`mi.warm_start_critic_mc_mixture`), which is the wrong distribution for a critic that must predict returns under
σ-scaled rollouts against scripted-normal specifically. This step re-warm-starts the critic with `full_authority_
train_v1.warm_start_critic_mc` (unchanged, R1n-b's single-arm procedure — reused, not `mi`'s mixture variant),
wrapped in `scenario_override(rb.arm_scenario("normal"))`, on the σ-scaled policy's own episodes (§6's final σ,
per the sequencing note there — not §4's initial value), mirroring R1n-e's own re-warm-start of R1n-c's critic
before PPO (declaration §1). Its seed bands are expressed through `fold_seeds`'s own cfg keys, given in full in
§13, and deliberately do **not** reuse `630000` — the seed preflight-reserved value that several archives (R1n-b,
S4, S7, S8, S9) already collide on (§14 flags this to the user; not fixed here).

## 3. The logp blocker, and the new class this step needs

`FullAuthorityPolicyV1EnemyThrow.act()` (S12, `enemy_relative_throw.py:176-201`) returns `torch.zeros_like
(power_mean)` as an explicit placeholder logp — confirmed by reading the code before writing this section. PPO's
clipped surrogate needs a real log-probability of the sampled action under the *current* policy, recomputed each
epoch, for the importance-sampling ratio (`anchored_actor_loss` calls `model.evaluate_latents(...)` for exactly
this). `FullAuthorityPolicyV1EnemyThrow` has no `evaluate_latents` at all.

This step's (future, separate) implementation needs a new class — `FullAuthorityPolicyV1EnemyThrowPPO`, a
subclass of S12's `FullAuthorityPolicyV1EnemyThrow`, not an edit to it (same pattern S12 itself used relative to
E3) — adding:

- **`evaluate_latents(observation, action_type, latent, *, prediction=None, with_value=True)`**, structured exactly
  like `FullAuthorityPolicyV1.evaluate_latents` (`full_authority_ppo_v1.py:103-121`), substituting the throw term:

  ```
  action_categorical = Categorical(logits=prediction["action_logits"])
  enemy_categorical  = Categorical(logits=prediction["enemy_logits"])     # already masked to living enemies in forward()
  move_normal        = Normal(prediction["move_raw"], move_log_std.exp())
  offset_normal      = Normal(prediction["throw_offset_raw"], throw_offset_log_std.exp())
  power_normal       = Normal(prediction["power_raw"], power_log_std.exp())

  type_logp   = action_categorical.log_prob(action_type) * live
  move_logp   = move_normal.log_prob(latent["move"]).sum(-1) * moves
  enemy_logp  = enemy_categorical.log_prob(latent["enemy"]) * throws
  offset_logp = offset_normal.log_prob(latent["offset"]).sum(-1) * throws
  power_logp  = power_normal.log_prob(latent["power"]) * throws
  per_unit_logp = type_logp + move_logp + enemy_logp + offset_logp + power_logp
  ```

  A row where `throws` is true but `enemy_mask` is entirely false cannot occur if `unit_action_mask` already
  disallows selecting THROW with no living enemy (to be confirmed against the TS action-legality rule at
  implementation time, before trusting this); if it can occur, `enemy_categorical` over an all-`-1e9` row is
  uniform and finite (softmax of equal logits), not NaN, so `enemy_logp` stays finite either way — a case worth a
  dedicated test regardless of which branch is true.
- **`act()`** overridden to call this `evaluate_latents` for its returned logp instead of returning zeros, matching
  `FullAuthorityPolicyV1.act`'s structure exactly.

## 4. Exploration and log-std handling

- `move_log_std` and `power_log_std`: R1n-e's exact recipe — the imitation-calibrated value plus `log(0.5)`,
  frozen, excluded from the optimizer.
- `throw_offset_log_std`: **genuinely new territory, not precedented.** S12 left this parameter at its placeholder
  init (`-1.0`), explicitly uncalibrated (S12 declaration §2: "not calibrated, since no PPO exploration uses it
  yet"). `throw_offset_head`'s final layer has no output nonlinearity (`Linear(64, 2)` with `Tanh` only on the
  hidden layer), so `throw_offset_raw`'s scale is whatever BC's cosine loss happened to leave it at, unconstrained.
  Measured directly against the three real S12 checkpoints before choosing a number — and measured on actual
  THROW-labelled rows from a real deterministic rollout against scripted-normal (16 off-band worlds each,
  `optionHorizon = 70` so scripted red is reached and thrown at, live client), **not** on reset-time states: an
  earlier draft of this measurement used `wrapper.reset()` output (t = 0, enemies far away, no unit has yet
  chosen THROW) and was corrected before this number was trusted.

  | Cohort | THROW rows measured | mean ‖offset_raw‖ | std | min | max |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 1 | 294 | 2.452 | 0.089 | 2.344 | 2.806 |
  | 2 | 253 | 3.430 | 0.123 | 3.138 | 3.599 |
  | 3 | 214 | 3.270 | 0.160 | 2.843 | 3.611 |

  At actual decision-time, ‖offset_raw‖ is tightly clustered per cohort (std under 0.16 in every cohort — far
  tighter than the reset-time measurement's spread, which mixed in states where THROW was never a live option). A
  Gaussian perturbation of the raw vector with std σ produces an angular std of approximately `σ / ‖offset_raw‖`
  radians for σ well below the norm. S9 measured the teacher's own aim precision at 1.7–4.0° off the instantaneous
  enemy position; this step targets a **deliberately tight 2.5° angular std** at declaration time (below the
  teacher's own range, since there is no R1n-d-equivalent measurement of how much angular noise this policy
  tolerates before behavior degrades — §6 exists precisely to check that, so the prior should start conservative).
  Using the pooled mean norm across the three measured cohorts (3.051): `σ = 3.051 · tan(2.5°) ≈ 0.1332`, giving
  **`throw_offset_log_std = -2.0`** (one shared constant across cohorts, matching R1n-e's convention of a single
  formula rather than per-policy tuning). Frozen, excluded from the optimizer, same as the other two.
- Enemy-choice: a `Categorical`, not a Gaussian — there is no σ to shrink. Its exploration behavior is whatever BC
  left it at. §6's probe measures its entropy directly rather than assuming it is well-behaved.
- Action-type sampling is unchanged (categorical, as in R1n-e).

## 5. Critic precondition: a sanity stop, not a quality gate (reused from R1n-e, reframed for S12's finding)

**Sanity stop, unchanged from R1n-e's amendment A7:** a cohort's PPO does not run if the re-warm-started critic's
held-out predictive-R² 95% interval's **upper** bound is below 0 (interval missing counts as failing). This is a
one-sided "is the critic worse than a constant" check, not a quality bar.

**Why S12's critic-R² finding does not block this.** S12 found critic R² is *lower* for the better (new-decoder)
policy across all three cohorts (0.33–0.40 for new vs. 0.52–0.66 for old), reversing what R1n-d/S5–S11 assumed
about R² as an actor-quality signal. That reversal is about using R² to *rank actors* — it says nothing about
whether MC-PPO functions correctly with a modest-R² critic, which is a separate, already-answered question:
R1n-e's own critic R² was 0.070–0.127 (below its own 0.25 gate, and below every S12 cohort's 0.33–0.40), and PPO
still learned there, because **advantages are pure Monte Carlo (`gaeLambda = 1`, no bootstrapping/truncation)** —
the critic is a variance-reduction baseline only, and a weak baseline adds variance, not bias. This step keeps
`gaeLambda = 1` for exactly this reason, overriding `ppo.PPOConfig`'s dataclass default of `0.9885140204` — that
default is for bootstrapped GAE elsewhere in the codebase and must not be inherited silently here.

## 6. Required pre-collection probe (the R1n-d this step doesn't otherwise have)

R1n-e's σ×0.5 recipe was chosen after R1n-d measured σ sensitivity for that decoder's move/throw/power heads. No
equivalent measurement exists for this decoder's discrete enemy choice or its newly-calibrated offset scale. Before
any real training collection, an outcome-blind probe (off-band seeds, §13) runs each cohort's initializer **both
deterministically and stochastically at the declared σ (§4)** on the same 64 worlds against scripted-normal (two
runs, same seeds) and reports, without training on it:

- deterministic vs. stochastic success and mean `L` on the same 64 worlds, as a **world-paired bootstrap
  interval** (`fi.paired_difference`, unchanged), not a raw point difference — 64 worlds alone cannot resolve a
  fixed point threshold against R1n-d's own measured 20–41% per-world discordance (implying a paired-success
  standard error around 0.07 at this sample size, large enough that a bare point rule would fire or miss on noise);
- the enemy-choice categorical's mean entropy and its argmax-vs-sampled agreement rate, restricted to
  THROW-labelled, living-enemy rows.

**Decision rule, predeclared:** for a cohort, training does not proceed at the declared σ if either holds: the
**lower bound** of the paired-bootstrap 95% CI for (stochastic − deterministic) success is below −0.15; or mean
enemy-choice entropy is within 0.1 nats of `ln(number of living enemies)` (i.e., close to uniform — the categorical
carries near-zero information). The implementation commit (not this one) amends this file with the measured values
and, if needed, a small fallback grid (candidates: `{0.5×, 0.25×}` of §4's σ for the continuous heads, tried in
order, stopping at the first that passes; for a degenerate enemy categorical, the stated fallback is to stop
sampling that head during rollout collection — enemy choice is read off the current policy's argmax at collection
time, its `enemy_logp` term is excluded from `anchored_actor_loss`'s ratio, and PPO updates only the other four
terms — never a design decided silently after seeing training outcomes) — the same outcome-blind-probe discipline
R1n-e's own A1 amendment used for its learning rate.

**Sequencing, predeclared to avoid a subtle ordering bug:** the σ probe runs first and fixes the σ used for the
rest of the cohort's run; the critic re-warm-start (§2, §5) runs at that final σ, not §4's initial value, since the
critic must predict returns under the distribution PPO will actually roll out with; the §8 learning-rate probe runs
after that, at the final σ, not before — the Gaussian KL anchor scales as `1/σ²`, so calibrating a learning rate
against one σ and then training at a smaller σ (if the fallback grid triggers) would make the anchor's actual
per-step KL larger than what the lr was chosen to tolerate.

## 7. KL anchor

Exact hybrid-policy KL to the frozen initializer, extending `death_rate_ppo.hybrid_kl`'s formula (confirmed
correct in structure by review: given THROW, enemy choice, offset, and power are conditionally independent in this
model, since `throw_offset_head` does not condition on the sampled/chosen enemy):

```
KL = Σ_a p(a) log(p(a)/q(a))                                    [action_type, categorical]
   + p(MOVE) · KL_N(move)
   + p(THROW) · ( Σ_e p(e) log(p(e)/q(e)) + KL_N(offset) + KL_N(power) )
```

where the inner `Σ_e p(e) log(p(e)/q(e))` is the enemy-choice categorical KL (masked to the same living-enemy set
for both `p` and `q`, since the mask is a property of the observation, not the policy — masked slots contribute a
finite 0 to both distributions' log-probabilities identically), and each `KL_N` is the closed-form Gaussian KL
`Σ_dim (μ_θ − μ_init)² / (2σ²)`, valid because σ is identical and frozen between the trained policy and the
reference. A new `hybrid_kl_enemy_relative(model, reference, observation, *, prediction=None)` implements this in
the new module, following `hybrid_kl`'s exact structure.

## 8. Training

Per cohort c ∈ {1, 2, 3}, as an independent run seeded with `torch.manual_seed(98500 + c)` at its start:

- **Rollouts:** 200 updates, 64 complete episodes per update at the declared σ, one 64-world block per update,
  `run_block` semantics (unfinished worlds only step; no replacement resets). Update u uses world seeds
  `5{c}70000 + 64·u + [0, 63]`.
- **Update** (`anchored_ppo_update_enemy_relative`, following `death_rate_ppo.anchored_ppo_update`'s decoupled
  structure exactly, calling the new `evaluate_latents`/`hybrid_kl_enemy_relative` instead of the old-decoder
  versions):
  - **Actor:** 4 epochs, minibatch 512, Adam over actor parameters except the three frozen log-stds, gradient
    clip 0.5. Clipped surrogate (`ppo.ppo_loss`, unchanged, per-unit ratio mode — the default), ratio clip 0.2,
    minibatch-normalized advantages, approximate-KL stop 0.01.
  - **Entropy:** categorical-only on `action_type`, weight 0.01 — matching R1n-e's amendment A2 exactly. Enemy
    choice is explicitly **excluded** from the entropy bonus: adding `p(THROW) · H(enemy)` would push gradient
    mass toward selecting THROW more often purely to raise an entropy term unrelated to the actual objective, a
    failure mode R1n-e's A2 already identified and avoided for the same reason on the Gaussian heads. No entropy
    bonus on the continuous heads either (unchanged from R1n-e).
  - **KL anchor:** `β · KL(π_θ ‖ π_init)`, β = 0.01, §7's formula, averaged over living units.
  - **Actor learning rate:** not fixed in this declaration. R1n-e found empirically that its precedent value
    (3e-4) produced 0.7–4.0 nats of first-step KL against a 0.01 stop — 150-fold overshoot — and settled on 1e-5
    only after an outcome-blind calibration probe. This step predeclares the **identical procedure**, not the
    number: at the σ selected by §6 (§6's sequencing note: this probe runs after the σ probe, at the final σ, not
    before), measure first-Adam-step approximate KL on one 64-episode off-band calibration rollout (seeds
    `5{c}60064–5{c}60127`, §13 — disjoint from §6's own probe band `5{c}60000–5{c}60063`) for each candidate in
    `{3e-4, 1e-4, 3e-5, 1e-5, 3e-6}`, and select the largest candidate whose measured first-step KL is at most the
    movement stop (0.01) for all three cohorts. The critic keeps a fixed lr of 3e-4 regardless (R1n-e's own
    finding: the critic is far less sensitive, and decoupled from the actor's KL stop).
  - **Critic:** 4 epochs of MSE on the rollout's Monte Carlo returns, its own Adam continuing the warm-start
    optimizer, gradient clip 0.5, decoupled from the actor's KL stop.
- **Checkpoints:** after updates 50, 100, 150, 200. Only update 200 is evaluated (no early stopping on outcome,
  per R1n-e's stopping-rule precedent, §12).
- **History per update:** episodes, success, `L`, timeout counts; mean return; actor/critic loss terms;
  approximate KL; KL-stop epoch/step; mean anchor KL to the initializer; gradient norms; rejected actions —
  matching R1n-e's logged fields, substituting `L` (this track's primary metric) for R1n-e's 1v1-specific death
  flag.

## 9. Evaluation

**Worlds.** A fresh split, 2600000–2600399 (400 worlds, matching R1n-e's scale) — confirmed by repo-wide scan to
be untouched by every step from S4 onward (unlike 2100000–2100099).

**Per cohort, on this split:** the initializer and the final policy (update 200), each deterministic; the same two
at the declared σ, stochastic, seeded `984500 + 10·c + m` (m = 0 initializer, 1 final).

**Reported:** success, mean `L` (with interval), team wipe, timeout fraction, mean decisions, and rejected-action
rate — the fields `roster_baseline.summarize` already computes, reused unchanged as this track's standard. Contact
rate and first-hit/completion p50/p95 are **not** currently produced by `summarize` and are not claimed as already
standard; if wanted for comparability with R1n-e, they are new fields a small helper would need to add, to be
decided (and, if added, tested) at implementation time, not assumed here.

**Paired differences, final − initializer, per world**, world-paired bootstrap (`fi.paired_difference`, unchanged):
per cohort, and cohort-averaged (each world's three per-cohort differences averaged, worlds resampled jointly —
matching R1n-e's cross-policy-correlation handling), 10,000 resamples, seed 984001.

## 10. Primary test and decision rules

Continuing this track's own established convention (S4 §1: "primary for later comparisons = mean L, world-paired;
success co-primary"), **not** importing R1n-e's 1v1-specific death-rate substitution (that substitution existed
only because R1n-c's success ceiling made the unmodified R1 gate infeasible — no such ceiling problem exists here;
S12's normal-arm success, 0.30–0.74, has ample headroom below any ceiling).

**Notation** (deterministic execution, cohort-averaged, on the fresh split):
- `ΔL`, interval `[ΔL⁻, ΔL⁺]`: mean-`L` difference (final − initializer);
- `S⁻`: lower bound of the success difference; `T⁺`: upper bound of the timeout difference.

**Non-inferiority (NI):** `S⁻ > −0.05` and `T⁺ < +0.05` — guards against a degenerate "avoid engagement" strategy
that could lower `L` by refusing to fight (raising timeout and cratering success) without this guard.

| Order | Condition | Outcome |
| --- | --- | --- |
| 1 | NI fails, or `ΔL⁻ > 0` | **harm or avoidance** |
| 2 | `ΔL ≤ −0.05` and `ΔL⁺ < 0` | **improved** (primary passes) |
| 3 | `ΔL⁺ < 0` | **improved, below the 5-point threshold** |
| 4 | median per-cohort final anchor KL < 0.01 (§8's stop threshold) | **no effective training** — the run barely moved; says nothing about PPO and 3v3 |
| 5 | otherwise | **no detectable change** |
| — | any cohort's critic sanity stop (§5) fires | **incomplete** — no cohort-averaged claim |

Order 4 (R1n-e's amendment A8, included from the start here rather than added after seeing a null, per review) sits
**below** both "improved" outcomes and **above** "no detectable change" only — matching R1n-e's own statement that
A8 "does not override … improved" (a real improvement with low anchor KL is still an improvement; a genuine
change can happen in relatively few effective steps). It exists only to distinguish "PPO ran and found nothing" from
"PPO barely ran at all" when neither improvement condition is met.

**Predictions**, stated for falsification:
- `ΔL`'s point estimate is negative (improvement direction) and no larger in magnitude than −0.30. This is
  calibrated against S12's own per-cohort normal-arm `ΔL` from the decoder-redesign itself (−0.50, −0.31, −0.09,
  new vs. old), the only comparable effect-size reference this track has: a bounded PPO continuation from an
  already-improved initializer is expected to produce a smaller marginal shift than the representation change
  that produced those numbers, not a larger one;
- the success difference's lower bound `S⁻` clears the non-inferiority guard, i.e. does not regress;
- the approximate-KL stop binds before the last epoch in most updates, exactly as R1n-e found (the lr-selection
  procedure in §8 is designed to produce this);
- median final anchor KL exceeds 0.01 (i.e., order 2 does not fire) — stated as a prediction, not assumed.

**Recommendations for a follow-up step** (authorize nothing):

| Outcome | Recommendation |
| --- | --- |
| Improved | Fresh replication with new training RNGs and a new untouched split; then, separately, a `hard`-arm generalization check (§2) — only after achievability against `hard` is itself measured. |
| Below threshold, or no change | Diagnose from the learning curves (KL-stop binding rate, anchor KL trajectory, return trend) before any longer run. |
| Harm or avoidance | Stop this PPO configuration; revisit σ, the anchor, or the objective under a new declaration. |
| No effective training | The lr-selection procedure (§8) was too conservative for the update budget; a longer run or a less conservative lr band needs its own declaration, not a silent retry here. |
| Incomplete | Resolve the critic sanity failure first. |

## 11. Budget

Following R1n-e's own precedent exactly (its A1/A7 calibration probes were accounted separately from, and run
before, its main §8 budget table): a small **probe budget**, outcome-blind and run first, separate from the
**training budget** that governs the actual PPO run.

**Probe budget, per cohort** (§6's σ probe, run at 1× then possibly two smaller fallback candidates; §8's
lr-calibration probe, one rollout reused across all five lr candidates):

| Item, per cohort | Decisions |
| --- | ---: |
| σ probe, 1× candidate: 2 runs (deterministic + stochastic) × 64 × 200 | 25,600 |
| σ probe fallback, up to 2 more candidates × 25,600 (worst case) | 51,200 |
| lr-calibration probe: 1 rollout × 64 × 200 (reused for all 5 candidates) | 12,800 |
| **Per cohort, worst case** | **89,600** |
| **All three cohorts, worst case** | **268,800** (probe cap 300,000, enforced separately, checked before any training decision is spent) |

**Training budget, per cohort** (unchanged from R1n-e's exact per-policy scale — same horizon, same 64-episode/
200-update rollout shape, same 400-world/2-mode evaluation, the only working precedent for any of these numbers in
this repo):

| Item, per cohort | Decisions |
| --- | ---: |
| Critic re-warm-start: (256 + 128) × 200 | 76,800 |
| Training: 200 updates × 64 × 200 | 2,560,000 |
| Evaluation: 2 policies × 2 modes × 400 × 200 | 320,000 |
| **Per cohort** | **2,956,800** |
| **All three cohorts** | **8,870,400** (cap 9,000,000, enforced by `account()`, matching R1n-e's cap exactly since every input is identical) |

**Combined worst case across both budgets:** 268,800 + 8,870,400 = 9,139,200 decisions. This is not itself enforced
as a single number — the two caps (300,000 and 9,000,000) are checked independently, so a probe-budget overrun
cannot silently eat into the training cap or vice versa. "Bounded" here means what it means throughout this track:
a single declared configuration, hard caps enforced in code, and a fixed stopping rule (§12) — not an arbitrarily
shrunk pilot (a smaller training-update budget risks landing directly in the "no effective training" outcome,
order 4, rather than producing an interpretable result) and not an open-ended or expandable run.

**Wall clock.** S13's real run (evaluation only, no training, same 3v3 roster and horizon) collected 96,474
decisions in 88 seconds — about 1,096 decisions/second for pure collection (simulator step plus forward inference,
no backward pass). At that rate, one cohort's training-budget collection alone (2,956,800 decisions: critic
re-warm, 200 rollout updates, evaluation) is roughly 45 minutes; three cohorts sequentially, roughly 2.25 hours,
before any PPO update (backward pass) time is added. **Update time is not estimated here and is not knowable until
the lr-calibration probe runs**: R1n-e's own update-time estimate swung by 15× between its original declaration
(0.8 s/update, later found to reflect a learning rate so large the KL stop allowed only one optimizer step) and
its measured value after selecting a working learning rate (about 12 s/update, ~40 minutes/policy, ~2.3 hours
total) — the same class of surprise could occur here, and the honest answer before implementation is that the
training-budget wall clock has a collection-only lower bound (~2.25 hours) and an unknown upper contribution from
backpropagation.

## 12. Stopping rule

- Fixed: three cohorts, 200 updates each, evaluation of update 200 only.
- No retries with another budget, seed, σ, β, or update count within this declaration.
- No early stopping on evaluation results.
- The lr-selection probe (§8) and the σ probe (§6) are outcome-blind by construction (they measure KL/entropy/
  success-gap only, never the trained outcome) and their results are recorded as amendments to this file before
  any real collection, per this repo's standing convention.
- If an implementation defect aborts a cohort's run, its output is discarded and the abort is reported; the fix is
  committed before a full rerun of that cohort.
- Output: `runs/m8_s14_ppo_continuation_v0/`, which the runner refuses to overwrite.
- No provider calls, browser input, TypeScript or protocol changes, reward changes, or edits to digest-pinned
  sources (including S12's own sealed `enemy_relative_throw.py`).

## 13. Seeds and generators

**World seeds** (all confirmed, by a repo-wide scan of every JSON under `snowgym/` at declaration time, disjoint
from every existing use — bands are per cohort c ∈ {1, 2, 3}):
- §6 σ probe: `5{c}60000–5{c}60063` (64 seeds);
- §8 lr-calibration probe: `5{c}60064–5{c}60127` (64 seeds, immediately adjacent to but disjoint from the σ probe
  band);
- PPO training rollout: `5{c}70000–5{c}82799` (200 updates × 64 episodes = 12,800 seeds);
- critic re-warm-start, train: `5{c}90000–5{c}90255` (256 seeds), held-out: `5{c}95000–5{c}95127` (128 seeds) —
  expressed through real `full_authority_train_v1.fold_seeds` cfg keys as `trainSeedBase = 5090000, heldOutSeedBase
  = 5095000, seedBandStride = 100000`, called with `rng_index = cohort` (1, 2, or 3), which reproduces these exact
  bands (`5090000 + 100000·1 = 5190000`, etc.) — not an ad hoc numbering scheme outside the function's own
  parameterization;
- evaluation: 2600000–2600399 (400 seeds, shared across cohorts).

**Torch:** `manual_seed(98500 + c)` at each cohort's run start (covers warm-start sampling, PPO sampling, and
minibatch order); stochastic evaluation seeded `984500 + 10·c + m`, giving the exact set {984510, 984511, 984520,
984521, 984530, 984531} for c ∈ {1,2,3}, m ∈ {0,1} — not a contiguous range, but scanned as the conservative
contiguous superset 984510–984531 below.

**Bootstrap:** 984001.

**Collision check, exact bands as declared above (not the earlier draft's approximate ranges — that draft omitted
the lr-calibration probe band entirely and mis-stated the stochastic-evaluation range as 984500–984530 instead of
the true 984510–984531; both are corrected here after review, and the scan below was re-run against the corrected
bands, not the original ones).** A numeric scan of every JSON file under `snowgym/` (excluding `.git`,
`node_modules`, `.venv`, `dist`) found zero collisions across all 15 per-cohort world-seed bands (5 per cohort × 3
cohorts), the evaluation band, the torch seeds (98501–98503), the bootstrap seed (984001), and the stochastic
contiguous superset (984510–984531). The implementation commit re-runs this scan against the serialized
configuration before collection, per this repo's standing `auditSeedDocuments`-equivalent convention.

## 14. Verification before the implementation commit

Targeted tests (to be written when this step is implemented, not in this commit):
- the enemy-relative hybrid KL (§7) is exact: 0 for identical policies, including a row with every enemy slot
  masked; matches a Monte Carlo estimate from sampled actions; correctly gated by `p(THROW)`;
- the new `evaluate_latents`'s logp, on a synthetic batch, matches a hand-computed sum of the five per-unit terms;
- **`ppo.ppo_loss` broadcasts and masks correctly at 3v3** — per-unit logp shaped `(batch, 3)` against a row
  advantage `(batch,)`, active mask `(batch, 3)` — this repo's PPO update has only ever run at roster 1
  (R1n-e); a dedicated test at roster 3 is required before trusting any training run built on it;
- σ×0.5 log-stds (move, power) and the calibrated offset log-std are frozen and excluded from the optimizer;
- `anchored_ppo_update_enemy_relative` reduces to a plain clipped-surrogate update when β = 0 and entropy weight
  = 0, matching `death_rate_ppo`'s own reduction test for the old decoder;
- **the anchor-KL diagnostic is chunked from the start** (`rollout_anchor_kl`'s row-chunked pattern, reused
  directly, not deferred to a post-hoc memory fix the way R1n-e's own A10 amendment had to be) — the per-row
  tensor here is at least 3× larger than R1n-e's (3 units × up to 21 enemy slots per row, versus R1n-e's single
  unit), so the same peak-memory risk R1n-e hit at 1v1 is more likely here if unchunked, not less;
- the world-cluster bootstrap for cohort-averaged paired differences;
- the decision rules (§10), including order precedence and the incomplete/no-effective-training cases;
- both budget bounds (§11) and their independent guards;
- a tiny live end-to-end run (small update count, small episode count) exercising collection, update, checkpoint,
  and evaluation.

**Full gate, and a correction to how this track has been describing it.** Client and training `pytest`, `npm run
build`, and `npm test`, re-checked here rather than assumed: the `npm test` failure this track has been calling
"366/367, the one documented exception" (citing a single `SelectiveRepair.test.ts` collision on
`m7b_engage_r1n_b_v0/declaration.json`) currently lists **five** colliding files at `trainSeedBase 630000`
(`m7b_engage_r1n_b_v0`, `m8_s4`, `m8_s7`, `m8_s8`, `m8_s9`), not one — checked directly against the actual failure
output before writing this section, not assumed from the earlier habitual phrasing. `PLAN.md`'s own stated gate
("passes only if that is the sole failure and names only that file") has therefore not matched the actual failure
since at least S4. This is flagged to the user in this step's report; it is **not** fixed here — S7/S8/S9's reuse
of `630000` may be an intentional reproduction-gate design (mirroring S11's deliberate reuse of S10's bands), and
determining that needs its own look before `PLAN.md`'s gate text is changed. S14 does not add a sixth file: its own
seed bands, given in full in §13, are entirely disjoint from `630000` by construction. A check that no pinned
archive source changed and the seed-collision scan re-run against the real configuration complete the gate.

## 15. Artifacts to be retained

Mirroring R1n-e's layout, adapted to S12's cohort numbering:

- **Run-level:** `declaration.json` (config, S12 manifest digest, E3 pins, implementation digest, git commit),
  `report.json`, `manifest.json`.
- **`cohort-{1,2,3}/`:** `critic-warm-start.json` + arrays, `sigma-probe.json` (§6), `lr-probe.json` (§8),
  `training-history.json`, `training-episodes.jsonl`, `update-{050,100,150,200}.pt`,
  `evaluation/{initializer,final}-{deterministic,stochastic}/episodes.jsonl`, `cohort-report.json`,
  `manifest.json`.
- **Independent runs:** each cohort's run is independent (own client, own RNG seeding at start) and may run as a
  separate process; a final aggregation step verifies the three cohort manifests and writes the run-level report,
  matching R1n-e's `--stage` structure.
