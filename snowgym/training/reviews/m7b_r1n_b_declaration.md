# R1n-b: E3 contract repair and pre-actor diagnostics

This protocol is committed before implementation and before any collection.
It deviates from the usual single "declare and test" commit. The tested
implementation lands in a separate commit, and collection starts only after
both commits exist. If implementation shows this protocol must change, the
implementation commit amends this file and states the change before any
collection. Once collection has started, the protocol is frozen.

R1n-b follows the stopped [R1n/E3](m7b_r1n_results.md) experiment. The
motivation and branch context are in the local plan note
`refs/snowgym_continuation_plan_2026-09-13.md`.

Scope:
- R1n-b repairs the E3 contract and measures what the next branch needs to
  know.
- **It trains no actor.** It writes no deployable checkpoint and authorizes no
  promotion.
- It does not change the R1/M7c qualification gates.
  `autonomousQualificationEligible` stays `false`.
- It does not authorize any Phase D branch (repaired E3b, approach shaping, or
  BC/DAgger-first). Each needs its own declaration.

## 1. Motivation: source-verified E3 findings

Line references are to E3 as archived at `4b827e6`. The file digests
`sha256:f1b06b98…` (`options/full_authority_train.py`) and `sha256:90204be5…`
(`executor/full_authority_ppo.py`) still match `runs/m7b_engage_r1n_v0/declaration.json`.

| # | Finding | Location |
| --- | --- | --- |
| F1 | The actor KL-stop `break` exits before `critic_optimizer.step()`, so the actor's KL stop blocks critic updates. The value loss is also inside the shared `losses["total"]`. The E3 declaration said the critic is never gated. | `full_authority_train.py:185-194`; `ppo.py:568-574` |
| F2 | Throw sampling and likelihood reuse the per-arm movement log-std. Local's throw σ is 0.25 in latent space vs global's 0.04/0.05, so the arms also differ in throw exploration. | `full_authority_ppo.py:139,160`; `:28-35` |
| F3 | `heldOutRSquared` is explained variance (`1 − Var(V−G)/Var(G)`), which ignores constant bias. | `full_authority_train.py:161-162` |
| F4 | Warm-start targets are GAE returns that bootstrap from the untrained critic at window edges. The held-out target SD was 0.0038. | `full_authority_train.py:99,123-124,136-144`; `m7b_r1n_results.md` |
| F5 | Actor features are 227-wide, the same as S12's `GeometryProbe`. The E3 declaration's "213" is a stale number. | `full_authority_ppo.py:55`; `executor/geometry_probe.py:26` |
| F6 | Neither the actor nor `RoleAwareCentralCritic` reads `option_state`. The critic's only clock is `log1p(tick)/10`. | `ppo.py:469-476`; `options/engage_v1.py:57-80` |
| F7 | E3's "no 1v1 teacher" covered only `recommend_movement`. The plan-aware production teacher (`/plan-teacher-action` = `PlanAwareTeamController` + `ReactiveUnitPolicy`) and the plan-blind `SimpleBlueAgent` (`/step-scripted`) are both reachable through the batch host. | `server/SnowGymService.ts:40,106,238-256`; `python/src/snowgym_client/batch.py:231,425-457` |
| F8 | Every Engage option ends within its 200-decision horizon. Timeout is a terminal −1 failure. | `options/tracker.py:151-156` |

## 2. Implementation contract (B)

All changes go into new modules:
- `executor/full_authority_ppo_v1.py`
- `options/full_authority_train_v1.py`
- `options/full_authority_diagnostics.py`

The two E3 files must not change. A test asserts that their SHA-256 digests
still equal the archived E3 `declaration.json` values. Behavior not listed
below matches E3: scenario, Engage option and horizon, reward, egocentric
227-wide actor features, local/global `decode_move`, and hyperparameters.

- **B1. Separate throw exploration.**
  - `move_log_std` keeps E3's per-arm calibration: local `log(2/8)` on both
    axes; global `(log(2/50), log(2/40))`.
  - A new `throw_log_std` is initialized to `(log(2/50), log(2/40))` in
    **both** arms, because throw aim decodes as `tanh(latent)` in
    arena-normalized space in both.
  - Throw sampling and likelihood use `throw_log_std`. The arms then differ
    only in `decode_move` and `move_log_std`.
  - The aim-noise implication is measured in B7, not tuned here.
- **B2. Decoupled critic.** Actor and critic have separate losses, backward
  passes, and optimizers.
  - The actor loss is the per-unit clipped policy term minus
    `entropyWeight × entropy`, with no value term.
  - The actor stops, as in E3, when a minibatch's approximate KL exceeds 0.01.
  - The critic loss is `mean((V − G)²)`. It runs all `criticEpochs = 4`
    epochs over the rollout on every update, whatever the actor did.
  - This update is implemented and tested here but not run at scale.
- **B3. Critic metrics.** All are computed on held-out rows with population
  variance.
  - `predictiveR2 = 1 − mean((V − G)²) / Var(G)` — **the gating metric**.
  - `explainedVariance = 1 − Var(V − G) / Var(G)` — reported.
  - `timeOnlyR2` — reported and used in the gate. It is the predictive R² of a
    20-bin clock lookup: the training-fold mean of `G` in each 10-decision
    option-decision bin, with the training-fold global mean for empty bins.
  - `untrainedPredictiveR2` — the critic before warm start, reported.
  - A 95% episode-bootstrap interval for `predictiveR2`: 10,000 resamples of
    held-out episodes, seed 981001, reported.
- **B4. Monte Carlo warm start on complete episodes.**
  - Collection runs in blocks. Every world resets with a fresh seed at block
    start. `EngageOptionBatchV1.step_indices` then steps only worlds whose
    option is unfinished, until every option in the block has ended (F8
    guarantees this within 200 decisions).
  - There are no replacement resets inside a block and no bootstrap.
  - Target: `G_t = Σ_{k=t}^{T} γ^{k−t} r_k`, with `γ = 0.9976921765` and E3's
    executor reward (`mission + 0.1·combat + shaping`).
  - The training fold is 4 blocks × 64 worlds (256 episodes). The held-out fold
    is 2 blocks × 64 worlds (128 episodes) on disjoint seeds. All decision rows
    of each episode are kept.
  - The fit matches E3: critic parameters only, Adam `lr = 3e-4`, 10 epochs,
    minibatch 512, gradient clip 0.5.
- **B5. Critic conditioning.** The critic is `OptionCentralCritic(ModelConfig(observation_version=3))`,
  imported unchanged from `executor/movement_ppo.py`, so it reads
  `option_state`. The actor's inputs are unchanged: 227 features and no
  `option_state`.
- **B6. Retained artifacts.**
  - `critic-warm-start.json`: the B3 metrics.
  - `critic-warm-start-arrays.npz`: for every train and held-out row, world
    seed, episode id, fold, option decision, `option_state`, `G`,
    pre-warm-start `V`, and post-warm-start `V`.
  - `initial-policy.pt`: the full model state after warm start. It is
    untrained; it exists so a later declaration can reuse the initialization.
    It is not promotable.
  - `episodes.jsonl` for every C collection. One row per episode: source,
    world seed, success/failed/timedOut, final decision, first-hit decision,
    final `targetDamage`, blue alive at end, rejected and total actions,
    minimum and mean blue–target distance, and distance at first hit. The
    first-hit decision is the first decision whose tracker
    `metrics.targetDamage > 0`, or null.
  - `manifest.json`: SHA-256 of every file.
- **B7. Exploration calibration, measured offline.**
  - State sample: the first 1,024 held-out rows at RNG index 0, for each arm.
  - For each state, draw 64 latents from the initial move distribution and 64
    from the initial throw distribution.
  - Move: report commanded-destination displacement from the mean destination
    in world units (median, p90), plus the fraction of destinations clamped at
    the arena bound (`|q| ≥ 0.999`) or saturated (global: per-axis `|tanh| > 0.99`;
    local: `tanh(‖z‖) > 0.99`).
  - Throw: report the angular deviation of the aim ray from the mean aim ray
    (median, p90 degrees).
  - This measures commanded, not realized, motion. Physical response was
    probed in R1m-S5 and is not re-measured.

## 3. Diagnostics (C)

The scenario is E3's (1v1, 100×80, `RandomAgent` red, `maxTicks` 1800, 10 Hz)
with `teacher_option_plan("engage")` and the 200-decision Engage horizon. The
seeds are fresh and do not overlap any archived experiment. E3's development
seeds 600000–600099 stay reserved for a later declared evaluation.

- **C1a. Teacher precondition (pass/fail).**
  - On one world (seed 620000): reset, activate the plan, and confirm that
    `FrozenEngageTracker` constructs. This requires an `enemy_cluster`
    activation objective with valid `enemyIds` at roster 1.
  - Then step 5 decisions with `plan_teacher_tensor_actions()` and confirm
    every returned action is accepted.
  - On failure, record `teacherPreconditionPassed: false` with the error and
    skip C1b. This is a finding, not something to work around.
- **C1b. Plan-aware teacher ceiling.**
  - Seeds 620000–620099, 2 blocks of 50 worlds. Actions come from
    `plan_teacher_tensor_actions()` and are stepped through
    `EngageOptionBatchV1.step_indices`, the same tensor round trip a learner
    uses.
  - Metrics: success, completion-decision distribution over successes (p50,
    p95), contact rate, first-hit distribution, blue death rate, and
    rejected-action rate.
  - Record: assistance `none` (this is the coded production teacher itself),
    observation version 3, and task = Engage option success.
  - **Label audit (D3 research):**
    - action-type counts;
    - MOVE labels: fraction with `|target| ≥ 0.999` on either axis
      (unrepresentable by the global `tanh` decode), and own-to-destination
      world distance (median, p90, fraction > 8 units, i.e. beyond local
      reach in one decision);
    - THROW labels: saturation fraction, aim distance, and power (p10, median,
      p90);
    - teacher-labeling wall-clock decisions/second, reported as
      hardware-specific.
- **C1c. Plan-blind rule reference.**
  - Same seeds and blocks, stepped by `SimpleBlueAgent` through
    `SnowGymBatchEnv.step_scripted()`. The diagnostics module updates the
    Engage trackers itself, using a local copy of `step_indices`' tracker logic.
  - Worlds whose option has ended keep stepping physically (and count toward
    the budget) but are not scored.
  - Reported as a reference, never as the plan-following ceiling.
- **Excluded:** the historical `runs/ppo_1v1_bc_v0` and `ppo_1v1_easy_bc_v0`
  checkpoints. They use a pre-v3 contract and belong to a lineage with coded
  nearest-enemy throw targeting and a movement prior, and they were qualified
  on battle wins. They are neither an unassisted ceiling nor a full-authority
  label source.
- **C2. Contact floor.**
  - The uniform-random legal-action floor (E3's definition) on seeds
    620000–620099, paired with C1, 2 blocks of 50 worlds, recorded in
    `episodes.jsonl`.
  - Random-init contact statistics come from the C3 collections (384 episodes
    per arm/RNG, 1,152 per arm) with no extra simulation.
- **C3. Corrected warm start.**
  - Arms `local` and `global` × training RNGs 98001/98002/98003.
    `torch.manual_seed(rng)` runs immediately before policy construction.
  - Episode seeds at RNG index `i ∈ {0,1,2}`: training fold
    `630000 + 1000·i + [0, 255]`; held-out fold `640000 + 1000·i + [0, 127]`.
  - Both arms use the same episode seeds at the same RNG index (paired).
  - Each arm/RNG runs once and records the B3 gate outcome. No actor update
    follows.

**Critic gate** (per arm/RNG): `predictiveR2 ≥ 0.25` **and**
`predictiveR2 ≥ timeOnlyR2 − 0.05`. The absolute threshold keeps E3's number,
now applied to a well-posed target. While contact is rare, the Monte Carlo
return is largely a function of time remaining. A critic more than 0.05 worse
than a 20-bin clock lookup is not a usable baseline. The 0.05 is a predeclared
tolerance for finite-sample noise at 128 held-out episodes.

### D research (reported, never gated)

- **D1 (repaired frozen-reward PPO):** random-init contact rate, success rate,
  and first-hit distribution per arm/RNG and pooled per arm.
- **D2 (approach shaping), offline only:**
  - Let `d*` be the teacher's median blue–target distance at THROW decisions
    (C1b). Evaluate a candidate potential `Φ_d(s) = −max(0, d − d*)/D`, with
    `D` the arena diagonal, on teacher, floor, and random-init trajectories.
  - Report the fraction of pre-contact decisions where `γΦ_d(s′) − Φ_d(s)` is
    nonzero, and the point-biserial correlation between an episode's mean
    `Φ_d` over its first 50 decisions and eventual contact (random-init
    episodes).
  - If C1b did not run, `d*` is undefined and these statistics are recorded as
    not computed. No substitute `d*` is chosen after the fact.
  - The reward is not changed. Any shaped reward needs its own declaration.
- **D3 (BC/DAgger-first):** C1a/C1b availability, the label audit, and
  labeling throughput.

## 4. Feasibility and precision

- A contact rate from 1,152 random-init episodes per arm has a 95% half-width
  of at most about 2.9 points (binomial, near p = 0.5; at p = 0.25 it is about
  2.5).
- Teacher success on 100 seeds has a half-width of about 8 points near 80%.
- The held-out critic metrics use 128 independent episodes. Their uncertainty
  is reported by episode bootstrap, not assumed.
- None of these measurements needs policy learning, so no reachable-change or
  SNR calculation applies. Both return in the declaration of whichever Phase D
  branch trains an actor.

## 5. Budget

| Item | Upper bound (simulator decisions) |
| --- | ---: |
| C1a precondition | 200 |
| C1b plan teacher, 100 episodes | 20,000 |
| C1c scripted reference, 2 × 50 worlds × 200 | 20,000 |
| C2 floor, 100 episodes | 20,000 |
| C3, 6 arm/RNG × (256 + 128) episodes × 200 | 460,800 |
| **Declared cap** (enforced by an `account()` guard) | **600,000** |

At E3's measured 2,911 decisions/second (64 worlds, 1v1, real inference),
the C3 bound is under 3 minutes of simulation. Teacher-labeled blocks are
slower and are timed and reported.

## 6. Decision rules

These are applied to the archived results. They produce a recommendation, not
an authorization.

- **Teacher usable:** C1a passes, C1b success ≥ 80/100, and rejected-action
  rate < 0.1%.
- **Critic healthy for an arm:** the gate passes for all three RNGs of that
  arm.
- **Contact adequate for an arm:** pooled random-init contact rate ≥ 25%.
  - Derivation: at about 200 decisions per random-init episode, an
    8,192-decision update spans about 41 episode-equivalents.
  - 25% gives about 10 contact episodes, the smallest count judged able to
    carry a positive-advantage signal per update.

| Outcome | Recommendation |
| --- | --- |
| Critic unhealthy for an arm | No PPO branch for that arm until a declared critic diagnosis. A BC stage that needs no critic may still be proposed. |
| Critic healthy, contact adequate | D1 is informative; D3 is also open if the teacher is usable. |
| Critic healthy, contact inadequate, teacher usable | **D3.** |
| Critic healthy, contact inadequate, teacher not usable | D2, or a separately declared teacher. |

Label-audit interpretation: if more than 20% of teacher MOVE labels lie beyond
the 8-unit local radius, the D3 declaration must say which decoder it imitates
with, or declare a radius change. The label audit alone never changes a decoder.

## 7. Stopping rule

- Every measurement runs once, with no retries using a different budget,
  seed, or threshold.
- If a run is aborted by an implementation defect before it completes, its
  output directory is discarded and the abort is reported in the results. The
  fix is committed before a full rerun. Completed measurements are never rerun.
- Output: `runs/m7b_engage_r1n_b_v0/`, an immutable directory whose runner
  refuses to overwrite.
- No provider calls, browser input, or TypeScript/protocol changes.

## 8. Verification before the implementation commit

Targeted tests:
- the E3 digests are unchanged;
- `throw_log_std` is separate, equal across arms, and used by throw
  likelihood;
- the decoupled update: critic parameters change on an update with
  `klStopped: true`, and the actor loss carries no value gradient;
- exact Monte Carlo returns on synthetic reward sequences;
- the block collector never resets mid-block and steps only unfinished worlds;
- `predictiveR2` penalizes a constant bias that `explainedVariance` ignores,
  and the time-only baseline behaves as specified;
- C1a returns a structured failure instead of raising;
- the budget guard;
- a tiny live end-to-end diagnostics run.

Full gate: `npm test`, `npm run build`, Python client tests, and Python
training tests.
