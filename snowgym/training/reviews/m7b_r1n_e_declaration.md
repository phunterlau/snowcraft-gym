# R1n-e: KL-anchored frozen-reward PPO from the imitation policies, with death rate as the primary test (1v1 Engage)

This protocol is committed before implementation and before any collection.
As with R1n-b, R1n-c, and R1n-d:
- the tested implementation lands in a separate commit;
- that commit may amend this file, stating each change, before any
  collection;
- collection waits for both commits.

R1n-e fine-tunes R1n-c's three imitation policies with PPO. It uses what
[R1n-d](m7b_r1n_d_results.md) measured:
- exploration at σ×0.5;
- a critic that can remove only a small share of return variance, so returns
  are pure Monte Carlo and the critic is used only as a baseline.

At the user's decision, the primary test is **blue death rate**.

`autonomousQualificationEligible: false`. The initializers were trained on
coded-teacher labels. At runtime the policy acts alone.

## 0. Relation to the R1 gate, and what the primary test measures

**The R1 gate, verbatim** (`PLAN.md`, R1m): "Require +20 pp assisted
success, positive paired interval, parameter change, and rejection rate
<0.001". R1n is to "pass unchanged R1 gates plus fresh replication".

**Why R1n-e does not use it.** R1n-c showed that +20 success points over
each seed's initializer is infeasible:
- on split B, two of three seeds would need 103 and 98, against a 91
  ceiling;
- on split A, seed 97102 would need 99, against 93.

R1n-e therefore replaces the success gain with a death-rate test. **R1n-e
cannot satisfy the unchanged R1 gate and makes no R1 qualification claim.**
It keeps R1's other three conditions as checks: a paired interval, parameter
change, and a rejection rate below 0.001.

**The primary test is not the training objective.** PPO maximizes the
frozen Engage executor reward (`tracker.py`):
`mission (±1 on success or failure) + 0.1 · combat + shaping`, where
`combat` is the clipped normalized damage dealt minus damage received.

Death enters only in two ways:
- through the −1 mission reward when no assigned unit is alive;
- through the damage-received half of the combat term, at weight 0.1.

Death rate is thus a consequence of the objective, not the quantity being
maximized. No reward change is made or bundled here.

**Why death rate.** R1n-c located the gap to the teacher in deaths: failures
are deaths in worlds the teacher wins. Deterministic death rates were 26–34%
against the teacher's 7–9%. Because a policy could "improve" death rate by
avoiding the fight, the test carries non-inferiority guards on success and
timeouts (§5).

## 1. Fixed inputs

- **Scenario, option, and horizon:** exactly R1n-c and R1n-d. The Engage
  horizon is 200 and γ = 0.9976921765.
- **Initializers:** `runs/m7b_engage_r1n_c_v0/seed-{97101,97102,97103}/fit-4.pt`,
  indexed i = 0, 1, 2. The R1n-c manifest is verified before loading.
- **Exploration:**
  - `move_log_std`, `throw_log_std`, and `power_log_std` are set to their
    calibrated values plus `log 0.5` (R1n-d's `recommendedSigmaScale`);
  - they are frozen and excluded from the optimizer;
  - action types are sampled.
- **Critic:** the `fit-4.pt` `OptionCentralCritic`, as in R1n-d's C0.
  R1n-d found the variants interchangeable and C0 continues R1n-b's recipe.
  - Before PPO it is warm-started with R1n-b's Monte Carlo procedure, unchanged
    (`v1.warm_start_critic_mc`, 10 epochs), on the initializer's own σ×0.5
    episodes.
  - Folds: train `880000 + 1000·i + [0, 255]`, held-out
    `885000 + 1000·i + [0, 127]`.
- **Pinned sources:** E3 digests are re-checked. New behavior goes in new
  modules only.

## 2. Critic precondition: a sanity stop, not a quality gate

R1n-d measured an attainable-R² upper bound of 0.16–0.24 at σ×1, with
`capture` intervals spanning zero. A capture-of-ceiling gate would need a
new ceiling measurement at σ×0.5 (about 235k decisions per policy) and
still could not be estimated precisely enough to gate on. **R1n-e declines
that gate.**

- **Sanity stop:** PPO for policy i does not run if the warm-started critic's
  held-out predictive R² is below 0, meaning it is worse than a constant
  baseline. The warm-start metrics are reported either way.
- **Why the weak critic is acceptable.** Advantages are **pure Monte Carlo**:
  `A = G − V(s)` on complete episodes, with no bootstrapping (λ = 1, no
  truncation). A weak critic adds no bias; it only fails to remove variance.
  The power analysis (§6) assumes near-Monte-Carlo variance.

## 3. Training

Per policy i, as an independent run seeded with `torch.manual_seed(97301 + i)`
at its start:

- **Rollouts:** 200 updates. Each collects **64 complete episodes** with the
  current policy at σ×0.5, in one 64-world block, using `run_block`
  semantics: only unfinished worlds step, and there are no replacement resets.
  - Update u uses world seeds `1400000 + 50000·i + 64·u + [0, 63]`.
  - The run stores the logged latents, log-probabilities, critic values, and
    Monte Carlo returns.
- **Update** (`anchored_ppo_update`, a new function that follows
  `v1.ppo_update`'s decoupled structure):
  - **Actor:** 4 epochs, minibatch 512, Adam lr 3e-4 over the actor
    parameters except the three log-stds, gradient clip 0.5.
  - **Clipped surrogate:** ratio clip 0.2, minibatch-normalized advantages
    (as in E3), entropy weight 0.01 (categorical only, since σ is frozen),
    approximate-KL stop 0.01.
  - **KL anchor:** add `β · KL(π_θ ‖ π_init)` with β = 0.01, averaged over
    living units. `π_init` is a frozen copy of the initializer with the same
    σ×0.5. For the hybrid policy the KL is exact:

    `KL = Σ_a p(a) log(p(a)/q(a)) + p(MOVE)·KL_N(move) + p(THROW)·(KL_N(throw) + KL_N(power))`

    Each `KL_N` is a sum over dimensions of `(μ_θ − μ_init)² / (2σ²)`, which
    holds because the σ values are identical and frozen.
  - **Critic:** 4 epochs of MSE on the rollout's Monte Carlo returns, with
    its own Adam (the warm-start optimizer, continued) and gradient clip 0.5.
    It does not depend on the actor's KL stop.
- **Checkpoints:** model and both optimizer states after updates 50, 100,
  150, and 200. Only update 200 is evaluated.
- **History per update:**
  - episodes, success, death, and timeout counts;
  - mean return; actor loss terms; approximate KL; KL stop epoch and step;
  - mean KL to the initializer; critic loss; gradient norms; rejected
    actions.

## 4. Evaluation

**Worlds.** A fresh evaluation split E, 870000–870399 (400 worlds). Nothing
has used it.

**Per policy i, on E:**
- the initializer and the final policy (update 200), each deterministic;
- the same two at σ×0.5 stochastic, with `torch.manual_seed(977000 + 10·i + m)`
  before each (m = 0 for initializer, 1 for final).

**Reported for each:**
- success, blue death fraction, timeout fraction, contact;
- first-hit and completion p50 and p95;
- rejected-action rate.

**Paired differences, final − initializer, per world:**
- per policy, as the mean with a 95% bootstrap interval over worlds;
- **seed-averaged**, where each world's three per-policy differences are
  averaged and worlds are resampled: 10,000 resamples, seed 978001.

## 5. Primary test and decision rules

**Notation** (deterministic execution, split E):
- `D`, with interval `[D⁻, D⁺]`: the seed-averaged death-rate difference
  (final − initializer);
- `S⁻`: the lower bound of the seed-averaged success difference;
- `T⁺`: the upper bound of the seed-averaged timeout difference.

**Non-inferiority (NI):** `S⁻ > −0.05` and `T⁺ < +0.05`.

| Order | Condition | Outcome |
| --- | --- | --- |
| 1 | NI fails, or `D⁻ > 0` | **harm or avoidance** |
| 2 | `D ≤ −0.05` and `D⁺ < 0` | **survival improved** (primary passes) |
| 3 | `D⁺ < 0` | **improved, below the 5-point threshold** |
| 4 | otherwise | **no detectable change** |

**Checks, always reported; they do not change the outcome:**
- R1-style parameter change (the actor L2 distance from the initializer > 0);
- rejection rate < 0.001 across training and evaluation;
- the number of policies whose per-policy death difference is negative;
- the σ×0.5 stochastic death difference;
- the incomplete case: if any policy's sanity stop (§2) fires, the outcome is
  **incomplete**, and no seed-averaged claim is made.

**Recommendations for R1n-f** (they authorize nothing):

| Outcome | Recommendation |
| --- | --- |
| Survival improved | Fresh replication with new training RNGs and a new untouched split, then a second contrasting mission. |
| Below threshold, or no change | Diagnose from the learning curves (KL-stop binding, anchor KL, return trend) before any longer run. |
| Harm or avoidance | Stop this PPO configuration and revisit σ, the anchor, or the objective under a new declaration. |
| Incomplete | Resolve the critic sanity failure first. |

**Predictions**, stated for falsification:
- `D` between −10 and −3 points;
- the success difference ≥ 0;
- the approximate-KL stop binds before the last epoch in most updates;
- the mean KL to the initializer stays below 1 nat per living unit.

## 6. Feasibility and power

**Discordance.** On R1n-d's D1 worlds, small perturbations of the same
policy flipped the death outcome of 20–41% of worlds (σ×0.25, σ×0.5, or
type sampling, against deterministic). The analysis assumes a per-world death
discordance `d` between final and initializer of 0.3–0.4.

**Standard error.** With n = 400 worlds and 3 policies, and ignoring
cross-policy correlation, `SE(D) ≈ sqrt(d / (3n))`:
- 1.6 points at d = 0.3;
- 1.8 points at d = 0.4.

**Detectable effect.** The 80%-power, α = 0.05 detectable effect is about
4.4–5.1 points. For a true effect of −8 points, the probability that
`D ≤ −5` is about 0.95 at d = 0.4. For a true −6 it is about 0.7.

**Headroom.** Deaths are 26–34% (R1n-c, deterministic, seed-averaged) against
the teacher's 7–9%. A 5-point reduction closes about a fifth of that gap.

Cross-policy correlation on shared worlds widens the interval. The bootstrap
resamples worlds jointly across policies, so it accounts for that.

## 7. Artifacts retained

**Run-level:** `declaration.json` (config, R1n-c manifest digest, E3 pins,
implementation digests, git commit), `report.json`, `manifest.json`.

**`policy-{seed}/`:**
- `critic-warm-start.json`, `critic-warm-start-arrays.npz`, and
  `critic-episodes/`;
- `training-history.json`, and `training-episodes.jsonl` (one row per
  training episode, tagged by update);
- `update-{050,100,150,200}.pt`;
- `evaluation/{initializer,final}-{deterministic,stochastic}/` with
  `episodes.jsonl` and `trajectory-distances.npz`;
- `policy-report.json` and a per-policy `manifest.json`.

**Independent runs.** Each policy's run is independent: it has its own
client and its own RNG seeding at the start. It may therefore run as a
separate process with results unchanged. A final aggregation step, which
needs no simulation, verifies the three policy manifests and writes the
run-level report.

## 8. Budget

| Item, per policy | Upper bound (decisions) |
| --- | ---: |
| Critic warm start: 384 × 200 | 76,800 |
| Training: 200 updates × 64 × 200 | 2,560,000 |
| Evaluation: 2 policies × 2 modes × 400 × 200 | 320,000 |
| **Per policy** | **2,956,800** (cap 3,000,000, enforced by `account()`) |
| **All three** | **8,870,400** (cap 9,000,000) |

## 9. Stopping rule

- Fixed: three policies, 200 updates, evaluation of update 200 only.
- No retries with another budget, seed, σ, β, or update count.
- No early stopping on evaluation results.
- If an implementation defect aborts a policy run, its output is discarded
  and the abort is reported. The fix is committed before a full rerun of that
  policy.
- Output: `runs/m7b_engage_r1n_e_v0/`, which the runner refuses to overwrite.
- No provider calls, browser input, TypeScript or protocol changes, reward
  changes, or edits to digest-pinned sources.

## 10. Seeds and generators

**World seeds:**
- warm-start train: `880000 + 1000·i + [0, 255]`;
- warm-start held-out: `885000 + 1000·i + [0, 127]`;
- training: `1400000 + 50000·i + [0, 12799]`;
- evaluation: 870000–870399.

**Torch:**
- `manual_seed(97301 + i)` at policy start. This covers warm-start sampling,
  PPO sampling, and PPO minibatch order.
- The warm-start critic minibatch generator is seeded 97301 + i
  (`trainingRngs`).
- The stochastic evaluations use `977000 + 10·i + m`.

**Bootstrap:** 978001.

**Collision check.** Before this declaration, a numeric scan of every JSON,
TypeScript, Python, and Markdown file under `snowgym/` and `refs/` found no
use of 870000–870399, 880000–882255, 885000–887127, 97301–97320, or
977000–978999. It found no 7-digit use in 1,400,000–1,539,999; the only hit
in the scanned 1,300,000–1,599,999 range was R1n-d's 1,350,000 budget
constant. The implementation commit also runs `auditSeedDocuments` on the
serialized configuration.

## 11. Verification before the implementation commit

Targeted tests:
- the hybrid KL is exact:
  - it is 0 for identical policies;
  - it matches a Monte Carlo estimate from sampled actions;
  - it is gated by type probability;
- the σ×0.5 log-stds are frozen and outside the optimizer;
- complete-episode rollouts have logged log-probabilities that match
  `evaluate_latents`, and Monte Carlo advantages equal `G − V`;
- `anchored_ppo_update` reduces to `v1.ppo_update`'s actor objective when
  β = 0, with KL-stop and critic decoupling preserved;
- the world-cluster bootstrap for seed-averaged paired differences;
- the decision rules, including precedence and the incomplete case;
- the budget bound (2,956,800 per policy) and guard;
- the aggregation verifies the per-policy manifests;
- a tiny live end-to-end run.

Full gate:
- `npm test` passes with only the accepted known failure;
- `npm run build`;
- Python client tests;
- Python training tests;
- a check that no pinned archive source changed;
- `auditSeedDocuments` on the serialized configuration.

## 12. Amendments made in the implementation commit, before any collection

- **A1 — actor learning rate 1e-5 (was 3e-4), chosen by an outcome-blind
  rule.**
  - **What the probe found.** A throughput probe of one full-size update
    showed that lr 3e-4 with σ×0.5 moves the policy about 1.4–1.7 nats of
    approximate KL in one Adam step. The KL stop (0.01) therefore allows
    exactly one optimizer step per update, and each step overshoots it about
    150-fold. The anchor KL reached 3.4 nats after two updates.
  - **Why it happens.** Adam's step is about lr per weight regardless of
    gradient scale. Summed across a head, that shifts output means by about
    one σ when σ is 0.02–0.025.
  - **The rule.** Take the largest lr in {3e-4, 1e-4, 3e-5, 1e-5, 3e-6}
    whose first Adam step, from each initializer on one 64-episode
    calibration rollout, gives approximate KL on that rollout at most equal
    to the KL stop (0.01) for all three policies. The calibration used
    off-band seeds 934000 + 1000·i + [0, 63]. It measures KL only and no
    outcome.
  - **Measured first-step approximate KL:**

    | lr | 97101 | 97102 | 97103 |
    | --- | ---: | ---: | ---: |
    | 3e-4 | 0.715 | 2.777 | 4.003 |
    | 1e-4 | 0.079 | 0.313 | 0.454 |
    | 3e-5 | 0.0071 | 0.0283 | 0.0413 |
    | **1e-5** | **0.0008** | **0.0032** | **0.0046** |
    | 3e-6 | 0.00007 | 0.00028 | 0.00041 |

    KL scales as lr², so the rule selects **1e-5**.
  - **The critic is unchanged.** It keeps lr 3e-4: R1n-b's warm-start recipe
    and the continued critic optimizer.
  - **Expected effect.**
    - Within an update, approximate KL grows roughly as n² × the first-step
      KL over n consistent steps. The stop is checked on each minibatch
      before its step and fires when the KL is above 0.01. It should
      therefore allow about 4 actor steps per update for 97101 (first-step
      KL 0.0008) and about 2 for 97102 and 97103 (0.0032 and 0.0046). The
      maximum is 4 epochs × ⌈rollout rows / 512⌉ minibatches.
    - §3's "4 epochs" is an upper limit, not what will run. The effective
      amount of learning differs by policy: an experimental asymmetry
      created by the rule, not a defect.
    - The quantity to read is `actorOptimizerSteps` per update in the
      training history.
    - The lr is not re-chosen after this analysis.
  - **Cross-check.** In the same probe, the exact hybrid KL to the initializer
    matched the approximate KL within 3%.
- **A2 — categorical-only entropy, as §3 states.** `evaluate_latents`'
  entropy includes type-probability-weighted Gaussian entropies. At σ×0.5
  these are about −2.5 nats per dimension, so an entropy bonus on them would
  push type probabilities away from MOVE and THROW toward NOOP and HOLD. The
  loss therefore uses `Categorical(logits).entropy()` over living units. A
  test shows the update equals `v1.ppo_update` exactly when the entropy
  weight and β are 0, with the declared KL stop and without it.
- **A3 — rollout contents.**
  - Latents, log-probabilities, and critic values are logged from `act` at
    collection time. The advantage is `G − V` with those logged values.
  - The per-row reward is kept for `v1.ppo_update` compatibility.
  - History reports `meanUndiscountedReturn` (the sum of rewards) and
    `meanDiscountedReturn` (with γ, comparable to the critic targets).
  - After each update the history also records the exact anchor KL on that
    rollout's states, mean approximate KL, and clip fraction.
- **A4 — run layout.**
  - `--stage declare` writes `declaration.json`. `--stage policy --policy i`
    requires it, checks the configuration digest, and refuses to overwrite.
    `--stage aggregate` verifies every policy manifest.
  - The top-level manifest covers the policy manifests.
  - `--stage all` runs the three in order.
- **A5 — sequential execution.** The probe measured about 3 s of collection
  and 0.8 s of update per update: about 15 minutes of training per policy,
  and under 25 minutes with warm start and evaluation. The run uses
  `--stage all`. Independence (§7) still holds.
- **A6 — seed audit.** `auditSeedDocuments` on the serialized configuration
  found 15 declarations and 0 collisions. Test seeds 932000–932402 and probe
  seeds 933000–936063 are unused elsewhere in the repository.
- **A7 — the critic sanity stop uses the interval, not the point estimate.**
  - **Rule.** Policy i's PPO does not run if the **upper** bound of its
    held-out predictive-R² 95% interval is below 0, or if the interval is
    missing. This replaces §2's point rule (R² < 0).
  - **Why.** Archived fits at declaration time were all positive but
    imprecise: R1n-c warm starts gave 0.062, 0.108, and 0.059, with 97103's
    interval at [−0.043, 0.113], and R1n-d's D3 variants were similar. So a
    critic that is fine can yield a point estimate below 0 by chance. With MC
    advantages, a weak baseline adds variance and no bias (§2). The stop
    should fire only on a critic confidently worse than a constant.
  - **Outcome-blind check.** An off-band warm-start probe at σ×0.5 measured
    critic fit only, never the death rate. It used train seeds
    937000 + 1000·i + [0, 255] and held-out seeds 938500 + 1000·i + [0, 127]:

    | Policy | R² | 95% interval |
    | --- | ---: | --- |
    | 97101 | 0.101 | [0.026, 0.150] |
    | 97102 | 0.087 | [−0.019, 0.150] |
    | 97103 | 0.055 | [−0.029, 0.113] |

    About 47–48k decisions and 24 s per policy. The declared warm-start seeds
    (§3) were not touched.
- **A8 — a `no-effective-training` outcome splits the null.**
  - **Rule.** A new order 4 in §5, before "no detectable change": if neither
    harm nor improvement holds (orders 1–3 fail) and the **median across
    policies of the final anchor KL** is below **0.01**, the outcome is
    **no effective training**. The final anchor KL is the exact hybrid KL to
    the initializer on the last update's rollout states.
  - **Why.** After A1's lr reduction, a null death-rate result could mean
    PPO does not reduce deaths, or that the policies barely moved. The
    floor equals the per-update KL stop: below it, 200 updates compounded
    to less than one nominal PPO step.
  - **Scope.** It does not override harm, survival improved, or improved
    below threshold. Per-policy final anchor KLs are always reported.
  - **Recommendation.** This result says nothing about PPO and death rate.
    A larger step budget needs its own declaration.
- **A9 — restart procedure.**
  - **Normal path.** A policy stage refuses to overwrite. If `--stage all`
    fails partway, the remaining work is `--stage policy --policy k` for each
    unfinished k, then `--stage aggregate`, against the same
    `declaration.json`.
  - **A crashed, unsealed policy directory** is renamed
    `aborted-policy-{seed}-{n}` under the run root, not deleted. The top-level
    manifest covers it. The stage then reruns with the same seeds, and the
    results report the restart and its cause.
  - **Sealed policy directories** are never rerun.
- **A10 — memory fix after an aborted first attempt; fresh declaration.**
  - **What happened.** The first collection (declared at `1070890`) was killed
    by the host at low system memory during update 8 of policy 97101. Its
    directory holds `declaration.json` and an unsealed `policy-97101` with
    only the warm-start artifacts.
  - **What was seen before the kill.** Seven lines of training-rollout
    progress at σ×0.5 (successes, deaths, actor steps, anchor KL) and the
    97101 warm-start R² 0.070 [−0.011, 0.110]. No evaluation episode was
    collected.
  - **Cause, measured on off-band seeds 2000000+.**
    - Python's peak RSS was about 3.6 GB per update. It came from the
      post-update anchor-KL diagnostic evaluating `hybrid_kl` on all ~8k
      rollout rows in one forward pass.
    - Collection peaks at about 0.4 GB and the PPO update at about 0.9 GB.
    - The simulator process holds steady at about 250 MB, so there is no
      leak.
  - **Fix.**
    - `rollout_anchor_kl` evaluates the same KL in 512-row chunks and takes
      the row-weighted mean, which equals the full-batch value; a test
      asserts equality.
    - Peak RSS measured after the fix: about 0.8 GB.
    - It changes only a logged diagnostic and A8's `finalAnchorKl`: no RNG
      draw, loss, or update changes. The configuration is unchanged.
  - **Handling.**
    - The attempt is kept at `runs/m7b_engage_r1n_e_v0_aborted_attempt_1`.
      It is moved under the new run root as `aborted-attempt-1` right after
      the new declaration, so the top-level manifest covers it. It is not
      used in any analysis.
    - The run is re-declared at the A10 commit with identical seeds and
      configuration.
    - Stages run as separate processes: declare, policies 0–2, aggregate.
  - **A5's time estimate is revised.** The earlier 0.8 s update came from
    the lr 3e-4 probe, where the KL stop allowed one actor step. At 1e-5 an
    update takes about 12 s (up to 64 actor and 64 critic steps). That is
    about 40 minutes of training per policy and about 2.3 hours in total.
