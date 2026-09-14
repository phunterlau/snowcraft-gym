# R1n-d: diagnostics before PPO — exploration σ, the attainable critic R², and critic capacity (1v1 Engage)

This protocol is committed before implementation and before any collection.
As with R1n-b and R1n-c:
- the tested implementation lands in a separate commit;
- that commit may amend this file, stating each change, before any
  collection;
- collection waits for both commits.

**Why this step exists.** [R1n-c](m7b_r1n_c_results.md) gave *PPO has
headroom* but set two flags, and its §7 left three questions for the PPO
declaration:
- the critic precondition failed for all three seeds (predictive R²
  0.06–0.11 against 0.25);
- sampling from the calibrated σ cost 8–31 success points;
- the primary test was still open.

The first two need measurements that R1n-c's archive cannot give. At the
user's request ("split"), those measurements are this declaration. **No
actor or PPO update runs in R1n-d.**

**Naming.** The PPO stage that R1n-c's declaration and results call "R1n-d"
is now **R1n-e**. R1n-e is declared after these results. Its primary test
and power analysis (death rate, or a smaller success gain sized per seed)
need no new simulation, so they belong to R1n-e.

`autonomousQualificationEligible: false`. The policies were trained on
coded-teacher labels, and at runtime they act alone.

## 1. Fixed inputs

- **Scenario and option:** exactly R1n-c. That means 1v1 on a 100×80 arena,
  `RandomAgent` red, `maxTicks` 1800, 10 Hz, `teacher_option_plan("engage")`,
  the frozen Engage spec with horizon 200, and γ = 0.9976921765.
- **Policies:** R1n-c's final imitation policies,
  `runs/m7b_engage_r1n_c_v0/seed-{97101,97102,97103}/fit-4.pt`, indexed
  i = 0, 1, 2.
  - The R1n-c manifest digests are verified before loading.
  - The actor parameters and the calibrated `move_log_std`, `throw_log_std`,
    and `power_log_std` are used unchanged.
  - The critic weights in `fit-4.pt` are the untrained critic R1n-c started
    its warm start from.
- **Pinned sources:** the pinned E3 digests are re-checked. New behavior goes
  in new modules only.

## 2. D1 — where the stochastic-execution gap comes from, and how σ affects it

**Worlds.** 680000–680099, the same 100 worlds for every policy and mode.
Evaluation uses complete-episode blocks of 50.

**Modes per policy:**

| Mode | Action type | Move / throw / power |
| --- | --- | --- |
| `det` | argmax | mean |
| `type` | sampled | mean |
| `cont1` | argmax | sampled at σ×1 |
| `full1` | sampled | sampled at σ×1 (R1n-c's stochastic mode) |
| `full05` | sampled | sampled at σ×0.5 |
| `full025` | sampled | sampled at σ×0.25 |

- **σ×s** adds `log(s)` to all three log-stds at sampling time only.
- **Mixed modes** follow `FullAuthorityPolicyV1.act`'s sampling and decoding
  exactly, with the type and continuous draws switched independently. A test
  checks that `full1` and `det` reproduce `act` under the same generator state.

**Reported per mode:**
- success, contact, blue death fraction, and timeout fraction;
- the gap `det − mode` in success points, per policy and seed-averaged.

## 3. D2 — an upper bound on attainable critic R², from branched rollouts

**Source episodes.** For policy i, 24 complete stochastic (σ×1) episodes on
world seeds `690000 + 1000·i + [0, 23]`, in one block. Every world's action
and observation row is recorded at every decision.

**Branch states.** Take `s = (episode j, decision k)` for every
k ∈ {0, 25, 50, 75, 100, 125} at which episode j is still unfinished.

**Rollouts.** From each branch state, run m = 8 rollouts:
1. Reset the same world seed.
2. Replay episode j's recorded actions for decisions 0…k−1.
3. From decision k on, sample stochastically at σ×1 until the option ends.
4. Record the Monte Carlo return `G_{s,r}` from decision k. This is the same
   target definition the critic is trained on.

The rollouts for one source episode run as one block, with one world per
(k, r).

**Replay identity.** At decision k, each rollout world's actor observation
and `option_state` must have the same `semantic_state_digest` as the source
row at decision k. Any mismatch aborts the run as an implementation defect.

**Estimator.** With n states and m rollouts per state, all variances unbiased
(ddof = 1):
- `W` = mean over states of the within-state variance of `G_{s,·}`;
- `B` = variance across states of the state means;
- `V = B − W/m` estimates `Var(E[G|s])`;
- `T = V + W`;
- **ceiling `R²* = V / T`**, reported unclipped.

Each quantity is reported per k and pooled over all branch states (equal
weight per state). The 95% intervals come from a cluster bootstrap that
resamples source episodes with all their states: 10,000 resamples, seed
974001.

**Boundary, stated in advance.**
- `RandomAgent` draws from the seeded world RNG, and a replayed prefix
  restores that RNG with the rest of the state. So red's future random draws
  are fixed within a branch, and only blue's sampling varies.
- `R²*` is therefore the attainable R² for a critic that sees *everything*,
  including red's future draws. Conditioning on less can only lower
  explainable variance, so `R²*` is an **upper bound** for any
  observation-based critic. It is not a measurement of irreducible noise in
  general.
- The branch states are a k-grid sample, not the critic's full row
  distribution.

**Critic evaluation on branch states** (critics from D3, policy i):
- `rolloutR2 = 1 − mean_{s,r}(G_{s,r} − v_s)² / mean_{s,r}(G_{s,r} − Ḡ)²`;
- `capture = 1 − (mean_s(v_s − Ḡ_s)² − W/m) / V`, the fraction of attainable
  state-value variance the critic recovers.

## 4. D3 — critic capacity and representation (a lower bound)

**Folds for policy i.** All episodes are stochastic at σ×1 (R1n-c's
behavior), in blocks of 64:
- train: 256 episodes on `684000 + 1000·i + [0, 255]`;
- held-out: 128 episodes on `687000 + 1000·i + [0, 127]`.

Train-fold episodes with index ≥ 204 (the last 52 seeds) are the
**validation** split for C1–C3.

**Variants.** All use Adam with lr 3e-4, minibatch 512, and targets equal to
Monte Carlo returns on complete episodes, as in R1n-b.

| Variant | Critic | Init | Training data | Epochs | Gradient clip |
| --- | --- | --- | --- | --- | --- |
| C0 | `OptionCentralCritic` | `fit-4.pt` critic | all 256 train episodes | 10 (R1n-b/R1n-c replication) | 0.5 |
| C1 | `OptionCentralCritic` | `fit-4.pt` critic | first 204 | ≤ 200, early stop | 0.5 |
| C2 | `EgocentricCritic` | `torch.manual_seed(97201 + 10·i)` | first 204 | ≤ 200, early stop | 0.5 |
| C3 | `EgocentricCritic` | same init as C2 | first 204 | ≤ 200, early stop | none |

- **`EgocentricCritic`** (new, `executor/full_authority_critics.py`):
  - its own encoders with the actor's shapes;
  - E3's `features` transform, the same egocentric 227-wide features the
    actor uses;
  - a masked mean over living blue units, with the 3 `option_state` fractions
    appended;
  - an MLP of 230→256→256→1 with ReLU.
- **Why these variants:**
  - C1 changes only training length;
  - C2 changes the representation, which is world-frame pooled features for
    `OptionCentralCritic`, against egocentric features;
  - C3 removes the 0.5 gradient clip. R1n-c's imitation fits were clipped on
    at least 99% of logged steps; whether the critic warm start was is
    unrecorded.
- **Early stopping:** patience of 10 epochs on validation MSE, restoring the
  best epoch's weights.
- **Minibatch order:** a `torch.Generator` seeded `97201 + 10·i + c` for
  variant c. The C2/C3 initialization uses the global generator at
  `97201 + 10·i`, which is identical for both.
- **Selection:** the variant among C1–C3 with the lowest validation MSE. Its
  held-out predictive R² is `L_i`, a **lower bound** on attainable R². It is
  selected without touching the held-out fold.

**Reported per variant:**
- R1n-b's held-out metrics: predictive R² with bootstrap 95% CI, explained
  variance, time-only R², clock skill, and gate conditions;
- train and validation MSE per epoch, and epochs run;
- `rolloutR2` and `capture` on D2's branch states;
- held-out R² by decision window (0–49, 50–99, ≥ 100).

## 5. Decision rules (recommendations for R1n-e; they authorize nothing)

**Critic**, for each policy i. Let `U_i` be the pooled D2 ceiling with 95% CI
upper bound `U_i⁺`, and `L_i` the held-out R² of the selected variant.

| Per policy | Row |
| --- | --- |
| `L_i ≥ 0.25` | repairable |
| `U_i⁺ < 0.25` | gate unreachable |
| otherwise | bracketed |

Overall, the outcome is *repairable* if all three policies are repairable,
*gate unreachable* if all three are unreachable, and *bracketed* otherwise
(each policy reported).

| Outcome | R1n-e recommendation |
| --- | --- |
| Repairable | Use the selected critic variant; keep the 0.25 gate. |
| Gate unreachable | Re-declare the critic gate against the measured ceiling (for example as `capture`), and justify the advantage estimator (λ). |
| Bracketed | Use the selected variant; state the critic gate as a `capture` of the measured ceiling, and state D2's red-draw boundary. |

**Exploration**, on seed-averaged gaps:
- `recommendedSigmaScale` is the largest s ∈ {1, 0.5, 0.25} with
  `gap(full·s) ≤ 10` points, or none.
- `typeSamplingDominates` if `gap(type) > gap(cont1)` and `gap(type) > 10`.
  R1n-e must then declare how types are sampled (temperature or entropy),
  not only σ.
- R1n-e re-checks its critic at whatever σ it chooses. D3 measures at σ×1
  only.

**Predictions**, stated for falsification:
- `gap(type) > gap(cont1)`;
- `recommendedSigmaScale` = 0.5;
- a pooled ceiling `U` between 0.25 and 0.6, with the k ≤ 50 ceiling below
  the k ≥ 100 ceiling;
- C0 held-out R² ≤ 0.15, replicating R1n-c;
- an egocentric variant (C2 or C3) is selected and beats C1 on held-out R²;
- overall outcome *bracketed*.

## 6. Artifacts retained

- **Run-level:** `declaration.json` (config, R1n-c manifest digest, pinned
  E3 digests, implementation digests, git commit), `report.json`, and
  `manifest.json`.
- **`d1/policy-{seed}/{mode}/`:** `episodes.jsonl` and
  `trajectory-distances.npz`, plus `d1/summary.json`.
- **`d2/policy-{seed}/`:**
  - `source/episodes.jsonl` and `source-actions.npz` (every recorded action);
  - `branch-states.npz` (the observation rows and digests at each branch
    state);
  - `rollouts.jsonl` (one row per rollout: j, k, r, world seed, return,
    success, death, digest check);
  - `ceiling.json`.
- **`d3/policy-{seed}/`:**
  - the fold `episodes.jsonl`;
  - `critic-C{0..3}.pt` (final weights; the best epoch for C1–C3);
  - `critic-C{0..3}.json` (metrics and per-epoch history);
  - `critic-arrays.npz` (held-out seed, episode, decision, return, and each
    variant's value);
  - `selection.json`.

## 7. Budget

| Item | Upper bound (simulator decisions) |
| --- | ---: |
| D1: 3 policies × 6 modes × 100 × 200 | 360,000 |
| D2: 3 × (24 × 200 source + 24 × 6 × 8 × 200 rollouts) | 705,600 |
| D3: 3 × 384 × 200 | 230,400 |
| **Declared cap** (enforced by `account()`) | **1,350,000** (bound by item: 1,296,000) |

Replayed prefix decisions count toward the budget.

## 8. Stopping rule

- Fixed: three policies, six D1 modes, one D2 grid, and four D3 variants.
- No retries with another budget, seed, grid, or variant.
- If a replay-identity mismatch or any implementation defect aborts the run,
  its output is discarded and the abort is reported. The fix is committed
  before a full rerun.
- Output: `runs/m7b_engage_r1n_d_v0/`, which the runner refuses to overwrite.
- No provider calls, browser input, TypeScript or protocol changes, or edits
  to digest-pinned sources.

## 9. Seeds and generators

**World seeds:**
- D1: 680000–680099;
- D2 source episodes: `690000 + 1000·i + [0, 23]`;
- D3 train: `684000 + 1000·i + [0, 255]`;
- D3 held-out: `687000 + 1000·i + [0, 127]`.

**Torch global generator.** `torch.manual_seed` is called immediately before
each collection:
- D1 mode c (0…5): `975000 + 100·i + c`;
- D3 train: `975500 + i`;
- D3 held-out: `975600 + i`;
- D2 source: `975700 + i`;
- D2 rollout block for source episode j: `976000 + 100·i + j`.

**Critic seeds:** `97201 + 10·i (+ c)` (§4). **Bootstrap:** 974001.

**Order per policy:** D1 → D3 folds and critics → D2 source and rollouts →
critic evaluation on the branch states.

**Collision check.** Before this declaration, a numeric scan of every JSON,
TypeScript, Python, and Markdown file under `snowgym/` and `refs/` found no
use of 680000–699999, 974001, 975000–976999, or 97201–97240. The
implementation commit also runs `auditSeedDocuments` on the serialized
configuration.

## 10. Verification before the implementation commit

Targeted tests:
- mixed-mode sampling: `full1` and `det` reproduce `act`; σ scaling reaches
  all three log-stds; the type and continuous switches are independent;
- a live test that a replayed prefix reproduces the source row digests at the
  branch decision, and that a perturbed action fails the check;
- on synthetic data with known variance components, the ceiling estimator is
  unbiased (the `W/m` correction) and `capture` recovers 1 for the true state
  means and 0 for a constant;
- early stopping restores the best validation epoch, and the validation split
  is by episode;
- `EgocentricCritic` masks, shapes, and option-state dependence;
- decision rules and flags; the budget bound (1,296,000) and guard;
- a tiny live end-to-end run.

Full gate:
- `npm test` passes with only the accepted known failure (per `AGENTS.md`);
- `npm run build`;
- Python client tests;
- Python training tests;
- a check that no pinned archive source changed;
- `auditSeedDocuments` on the serialized configuration.

## 11. Amendments made in the implementation commit, before any collection

- **A1 — train MSE per epoch is a running minibatch mean.** Each epoch
  records the size-weighted mean of the minibatch losses computed before
  each optimizer step. A full pass over the training rows would double
  `EgocentricCritic`'s epoch cost (measured: about 8.7 s per epoch with the
  full pass, about 4.3 s without). Validation MSE is still a full pass, and
  early stopping uses only validation MSE.
- **A2 — time-only baseline.** Every variant's held-out metrics use the full
  256-episode train fold for the 20-bin time-only baseline, including C1–C3,
  which fit on its first 204 episodes. The baseline is then identical across
  variants. It is a reference value, not a fit, and it never touches the
  held-out fold.
- **A3 — gradient norms are logged for every variant.** Each epoch records
  the median pre-clip gradient norm and the fraction of steps with norm
  above 0.5. For C3, clipping is a no-op (`max_norm = inf`).
- **A4 — D2 sampling detail.**
  - Rollout worlds draw a sampled action for every active row at every
    decision. During the replayed prefix those draws are discarded and the
    recorded action is sent instead. Each block's draws are therefore fixed
    by its seed (`976000 + 100·i + j`) and block layout.
  - Worlds are laid out in (k ascending, r ascending) order.
  - The source episode's own return at each branch state is retained
    (`sourceReturn`) but excluded from the estimator.
- **A5 — same-seed worlds in one block.** Before implementation, a probe
  confirmed that the batch host accepts several worlds reset to the same seed
  in one block. A replayed recorded stochastic prefix matched the source row
  digest at k = 0, 7, 30, and 60. A live test repeats this and checks that a
  tampered prefix fails.
- **A6 — row precedence in the critic rules.** "Repairable" (`L_i ≥ 0.25`)
  is checked before "gate unreachable" (`U_i⁺ < 0.25`). Both can hold only
  if the estimates contradict each other (`L_i > U_i⁺`). In that case the row
  is *repairable*, and the contradiction is visible in the per-policy values.
- **A7 — seed audit.** `auditSeedDocuments` on the serialized configuration
  found 13 declarations and 0 collisions. The test seeds 931000–931999 are
  unused elsewhere in the repository.
- **A8 — decision windows.** "≥ 100" is implemented as the window [100, end
  of episode].
- **A9 — an in-distribution comparison for `capture`.**
  - **What is added:** each variant's held-out report adds `branchWindowR2`,
    the predictive R² on held-out fold rows within ±2 decisions of each D2
    branch decision (pooled and per k).
  - **Why:** D2's branch states are genuine on-policy stochastic states,
    replayed from real source episodes, but they come from a different world
    band (690000) than the held-out fold (687000). `branchWindowR2` measures
    the critic at the same decisions on the held-out fold, so a low `capture`
    can be read against it.
  - **Status:** it is reported only and enters no rule.
