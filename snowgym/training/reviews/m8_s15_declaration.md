# M8-S15 — does S14's normal-only PPO retain the initializer's random and easy Red performance?

Declared 2026-10-04, before any collection. Ordinary autonomous evaluation of frozen, already-archived checkpoints;
no training, no checkpoint selection. `autonomousQualificationEligible` stays false (development experiment).

## 0. Why, and what this can and cannot show

S14 (`m8_s14_results.md`) trained each S12 initializer with PPO **only against scripted-normal Red**. The S12
initializers were imitation-trained on a 50/50 **random + scripted-easy** mixture
(`mixture_imitation.MIXTURE_ARMS`). S14's "Consequence" item 1 and the 2026-10-04 first-principles review
(`refs/`, local) both ask the same thing before the project changes training objectives: did the normal-only
continuation keep the initializer's competence against the two opponents it was originally trained on?

**Both arms are in the initializer's imitation distribution.** This is a *retention* check inside that opponent
family, not an unseen-opponent generalization test (S13's correction applies here unchanged). PPO itself never
rolled out against either arm, so the final policies have never been optimized against them by reward. A
regression would mean the PPO continuation specialized away from opponents it had already learned; non-inferiority
would mean the S14 checkpoint can stand in for the initializer in later command-control audits without first
losing a known competence.

A `hard`-arm check stays out of scope: it first needs its own teacher-achievability measurement (S14 §2).

## 1. Policies (frozen; no selection)

For each S14 cohort `c ∈ {1, 2, 3}`, exactly two policies:

- **final**: `runs/m8_s14_ppo_continuation_v0/cohort-c/update-200.pt` (S14's last update, the one S14 evaluated).
  Earlier checkpoints (050/100/150) are **not** evaluated, to rule out post-hoc checkpoint selection.
- **initializer**: S14's own `reference`, rebuilt exactly as S14 built it:
  `enemy_relative_throw_ppo.prepare_policy(cfg, c, sigma_scale=0.5)[1]`, from S12's `cohort-c/new/fit-4.pt`. This
  matters for stochastic mode: S14 scaled the move/power log-stds by 0.5 and set the offset log-std to −2.0, so
  S13's plain `ert.build_model` + `fit-4.pt` loader would give the same deterministic actions but a **different
  sampling distribution**. A hard assertion requires every log-std parameter of the loaded final model to equal the
  prepared initializer's (they were frozen during S14 training).

The final model class is `FullAuthorityPolicyV1EnemyThrowPPO`, loaded into a `prepare_policy`-built instance via
`load_state_dict`, then `eval()`. All S14 sources are imported, never edited.

## 2. Fidelity gate (runs first; a failure stops the step)

Before any new world is played, replay S14's first evaluation block — worlds **2600000–2600063** (`blockWorlds`
64, scripted-normal), both policies, both modes, all 3 cohorts — and compare against S14's archived
`evaluation/{initializer,final}-{deterministic,stochastic}/episodes.jsonl` rows for those seeds.

- Deterministic: every row must be **exactly** equal (full JSON object).
- Stochastic: seed torch with S14's own `stochasticEvalSeedBase + 10·c + m` (`m` = 0 initializer, 1 final)
  immediately before collecting the block, as S14 did before its 400-world `collect_cell`. Every row must be
  exactly equal. This is the only end-to-end proof that the sampling distribution matches S14's.

Exact replay depends on batch composition and thread count, not only on seeds: a development check at a 4-world
block reproduced outcomes but differed from S14's archived 64-world rows in the sixth decimal of distances (batched
float math in the policy). The gate therefore uses S14's exact 64-world block and the process-default torch thread
count, which S14's command-line run also used; the report records both.

Any mismatch, in either mode, stops the step before new worlds are collected; the mismatch gets diagnosed and
reported, not worked around. These are reused worlds: their decisions count toward the budget but **never enter
any result table**.

## 3. Evaluation design

- **Worlds:** a fresh split, **2700000–2700399** (400 worlds), shared by both arms, both modes, both policies and
  all three cohorts. A repo-wide scan (`py`/`ts`/`md`/`json` under `snowgym/` and `refs/`, ignored run
  directories included) found no other use of 2700000–2799999. 2100000 (S4–S13) and 2600000 (S14) are not reused
  as result data.
- **Arms:** `random` (`redController: random`) and `easy` (`redController: scripted, redDifficulty: easy`), 3v3,
  exactly `roster_baseline.arm_scenario`.
- **Modes:** `deterministic` and `stochastic`, exactly S14's `act()` paths.
- **Stochastic torch seeds:** `985000 + 100·c + 10·a + m` (985100–985311), with `a` = 0 random / 1 easy and `m` = 0
  initializer / 1 final, set immediately before each 400-world cell. The range 985000–985399 is otherwise unused in
  the repo, apart from this step's own bootstrap seed 985001 (a separate numpy generator).
- **Collection:** `roster_baseline.collect_cell` unchanged, `blockWorlds` 64.

Cells: 3 cohorts × 2 policies × 2 arms × 2 modes = 24 cells of 400 episodes.

## 4. Metrics

Per (cohort, arm, mode, policy): `roster_baseline.summarize` (success, team wipe, timeout, mean units-lost fraction
`L`, mean decisions, rejected-action rate).

Paired differences (final − initializer), world-paired bootstrap, 10,000 resamples, `bootstrapSeed` 985001, for
**every (arm, mode) cell** on four metrics: success, `L`, timeout, team wipe (`blueAliveCount == 0`). Each is
reported per cohort and **cohort-averaged** (per-world mean over the three policies, then paired), exactly S14's
averaging. The machine report must contain all 4 (arm, mode) cells × 4 metrics × (3 cohorts + averaged); a unit
test fails if any is missing. This closes S14's reporting gap, where the aggregate wrote only the deterministic
pairing.

Informative metrics differ by arm, which is fixed here, not chosen afterward:

- **easy:** the initializer is already at 1.00 deterministic success in all three S12 cohorts (100 worlds), so
  success non-inferiority is nearly automatic. `L` and team wipe carry the information.
- **random:** S13 found the initializer's failures are mostly **timeouts** (13–17%, about 150 mean decisions), so the
  timeout margin carries the information there.

## 5. Decision rules (per (arm, mode) cell, on the cohort-averaged differences)

Margin `δ = 0.05` on every metric, the same value S14 used for its success and timeout guards.

- **Non-inferior:** `Δsuccess` lower 95% > −δ, and `ΔL`, `Δtimeout`, `Δwipe` upper 95% < +δ.
- **Regressed:** not non-inferior, and at least one metric is confidently harmful (`Δsuccess` upper 95% < 0, or
  `ΔL`/`Δtimeout`/`Δwipe` lower 95% > 0).
- **Inconclusive:** neither.

A confidently non-zero harmful change that stays inside the margin is still non-inferior; it is listed as a flag,
not hidden.

**Rejection gate:** the final policy's rejected-action rate is below 0.1% (PLAN.md's per-mission gate) in every
cell.

**Per-cohort flag:** a single cohort whose own interval shows harm beyond the margin (`Δsuccess` upper 95% < −δ, or
`ΔL`/`Δtimeout`/`Δwipe` lower 95% > +δ) in any cell.

**Overall outcome:**

| Outcome | Condition |
| --- | --- |
| `retained` | all 4 cells non-inferior, rejection gate passes, no per-cohort flag |
| `retained-with-cohort-regression` | all 4 cells non-inferior and rejection gate passes, but ≥ 1 per-cohort flag |
| `regressed` | any cell regressed, or the rejection gate fails |
| `inconclusive` | otherwise |

The outcome authorizes nothing. Recommendation only: `retained` → use the S14 final/initializer pairs as the frozen
executors for the command-control audit; `retained-with-cohort-regression` → same, with the flagged cohort reported
separately in every later result; `regressed` → any further fighter training must keep random/easy in its rollout
or anchor mixture, and later audits report final and initializer side by side.

## 6. Predictions (fixed now, scored afterward)

1. Overall outcome is `retained`.
2. On `random`, deterministic, cohort-averaged `Δsuccess` > 0 with its interval excluding zero. Rationale: S14's
   final policy finishes normal-arm options in fewer decisions (77–80 vs 98–117), and the initializer's random-arm
   failures are timeouts.
3. On `random`, the stochastic cohort-averaged `Δsuccess` point estimate is smaller than the deterministic one, as on
   normal (S14).
4. On `easy`, deterministic, cohort-averaged `ΔL` point estimate is ≤ 0.

**Cross-check (not a gate):** each cohort's deterministic initializer success on `random` should lie within ±0.12 of
S13's 0.77/0.87/0.79 (different worlds, 100 vs 400). A larger gap gets investigated as a possible loading problem
before any result is written up.

## 7. Budget

`optionHorizon` 200 bounds each episode.

- Fidelity: 3 cohorts × 2 policies × 2 modes × 64 worlds × 200 = 153,600.
- Evaluation: 24 cells × 400 worlds × 200 = 1,920,000.
- Bound 2,073,600; hard cap **2,100,000** simulator decisions, enforced by the account callback.

Expected actual use is roughly 1.0–1.3M (normal-arm episodes ran ~100 decisions, random-arm ~150).

## 8. Provenance, archive and gates

New module `options/enemy_relative_throw_ppo_retention.py`; no existing module is edited. New sealed archive
`runs/m8_s15_ppo_retention_eval_v0/`:

- `declaration.json` pins: configuration, git commit, budget bound, verified manifest digests of S14's training
  root (`m8_s14_ppo_continuation_v0`), S14's probe root (`m8_s14_probe_v0`) and S12's root
  (`m8_s12_enemy_relative_throw_v0`), the SHA-256 of every loaded `.pt` file, E3 digests, this declaration's digest,
  and the module's digest.
- `fidelity/cohort-c/{initializer,final}-{mode}/episodes.jsonl` and `fidelity/report.json`.
- `cohort-c/{arm}/{policy}-{mode}/episodes.jsonl` per cell.
- `report.json` with summaries, every paired difference, the decision rules and simulator decisions; sealed manifest.

Gates before running: client `pytest`, training `pytest`, `npm run build`, `npm test` (only the documented
`trainSeedBase: 630000` preflight failure). `npm test` is re-run after the archive is written, before the archive is
committed. Results go in `reviews/m8_s15_results.md`.
