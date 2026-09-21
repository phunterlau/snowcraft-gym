# SnowGym implementation plan

## Validated repository state

This plan was reconciled against `refs/snowgym_implementation_note.md`, the
supplied next-step RL design, current systems, browser wiring, tests, and build
configuration on 2026-09-01.

| Capability             | Current engine state                                                                                                     | SnowGym decision                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Multiple blue units    | `Game.spawnSquads` already honors `maxPlayers`; bundled maps have three blue spawns, while normal `main.ts` requests one | Reuse it; do not patch spawning                                                                             |
| Projectile attribution | `Snowball` and `SnowballThrown` already carry owner and team; collision rejects same-team hits                           | Reuse it; friendly fire stays disabled                                                                      |
| Blue action submission | Throwing has generic `tryThrow`; movement only accepted selected-unit UI commands                                        | Add one generic per-unit `tryMove` seam                                                                     |
| Red control            | The classic `AISystem` behavior is available through `ScriptedAiAgent`; seeded random is a second opponent               | Select both through the common `TeamController` boundary                                                    |
| Round termination      | Team counts are generic, but blue loss waits for single-hero lives                                                       | Configure the no-respawn scenario with zero reserve lives; redesign only if a later environment requires it |
| Determinism            | `World` owns a seeded RNG, physics uses fixed 60 Hz steps, and status exposes versioned public-state hashes              | Record provenance and exact actions; keep cross-language golden and replay assertions before benchmarking   |
| Headless use           | Most systems are DOM-free, but `Game` constructs renderer/input and owns the private step loop                           | Compose the systems directly in a DOM-free `SnowEnvironment`                                                |
| RL contract            | Canonical reset/step, masked fixed-shape Gym spaces, configurable rosters, terrain observations, and persistent batch worlds are implemented | Keep HTTP as the reference transport and the batch subprocess as the authoritative high-throughput path |

## Milestones

### R1n-i — frozen-checkpoint failure diagnosis: contact vs. finishing, deployed actions, no training (complete 2026-09-20)

**Results** (`training/reviews/m7b_r1n_i_results.md`; archive
`runs/m7b_engage_r1n_i_v0`; 65,378 of 200,000 decisions; manifest
verified): reads R1n-h's own 6 frozen `eval-normal` checkpoints one level
deeper than `label_error` can — the model's own executed actions and
ground-truth (enemy-position-relative) geometry, not teacher-conditioned
opportunities.

- **Condition C's total floor is confirmed as a deployed aim/decision
  failure**, not an artifact of `label_error`'s teacher-conditioning: all
  three C seeds reach range every episode but rarely select a close-range
  throw (rate ~0.01, roughly an order of magnitude below M's ~0.16), and
  aim worse than random (120–146°) on the throws they do select.
- **97103 selects MOVE less often after first contact when a red projectile
  is nearby** (37.2% of 1,257 decisions versus 97102's 90.9% of 22). Whether
  reduced physical evasion causes its finishing failure is untested; see the
  2026-09-20 erratum in `training/reviews/m7b_r1n_i_results.md`.
- **97101's contact failure remains unexplained — a real negative
  result.** It reaches range, throws close-range, and aims well (11.8°)
  in every episode, indistinguishable from its two working siblings on
  the three mechanisms this diagnostic tested. Its higher, more variable
  in-range decision count is the one measurable difference, but reads as
  a likely consequence of not landing hits (a lingering, unresolved
  engagement) rather than an independent cause.
- **Pipeline validated at full production scale:** all 6 reproduction
  gates passed exactly (100/100 worlds each); all 6 `label_error`
  rechecks matched the archive with `maxDelta = 0.0`.
- **Next:** what does explain 97101 is still open — would need
  per-decision projectile trajectory/impact data this run doesn't have,
  not just presence. Otherwise unblocks a checkpoint-selection
  declaration (not yet written) or moving to M8.

Follow `training/reviews/m7b_r1n_i_declaration.md` (amendments A1–A4).
`autonomousQualificationEligible` stayed false throughout; no checkpoint
is selected or promoted by this run.

### R1n-h — mixture-imitation curriculum, with a single-opponent control, testing transfer to a held-out opponent (complete 2026-09-15; transfers, +38 points)

**Results** (`training/reviews/m7b_r1n_h_results.md`; archive
`runs/m7b_engage_r1n_h_v0`; 1,017,713 of 2,200,000 decisions; manifest
verified):

- **Primary result: mixture training transfers to the fully held-out
  opponent.** M (trained on 64 random + 64 scripted-easy per round) minus
  C (trained on 128 random, a same-volume control) on `eval-normal`
  (scripted normal — an opponent *neither* condition ever trains on) is
  **+38.0 points [+35.7, +40.3]**, a 100-world bootstrap interval that
  excludes zero by a wide margin.
- **The precondition and the mechanism both check out.** M's gap to the
  teacher on `eval-easy` (what it actually trains on) is −1.3 points,
  essentially at ceiling. Mean `throwAimHeadingErrorDegrees` on
  `eval-normal` — R1n-g's most severe finding, worse-than-random aim —
  drops from 127.6° (C, reproducing R1n-g's archive) to 8.6° (M, near the
  healthy ~3° baseline), on an opponent M never trained on.
- **Success diverges from label error at the per-seed level, in two
  distinct ways (erratum, 2026-09-18 — see results doc).** M's label error
  is *not* uniform: 97102's aim error (0.71°) is ~18× better than 97101's
  and 97103's (12.58°, 12.53°). `eval-normal` success is 0.00 / 1.00 / 0.14,
  and the two failing seeds fail differently — 97101 rarely makes contact
  (12/100 episodes); 97103 makes contact every episode but dies finishing
  86/100 of them. This is the run's biggest open question, reframed as two
  separable diagnostics (contact failure vs. finishing/survival failure),
  not one label-error/success mismatch.
- **Unplanned side-finding, and ruled out as a cause of the above (erratum,
  2026-09-18):** M's critic gate passes (R² 0.26–0.47) where C's fails (R²
  0.07–0.10, reproducing R1n-c's own known critic-infeasibility finding).
  `train_condition()` warm-starts the critic strictly *after* the reported
  eval outcomes are recorded, and R1n-h runs no PPO, so the critic cannot
  have caused those outcomes — it remains a reported downstream property,
  not a candidate explanation for the seed spread.
- **Caveat (declaration amendment A1):** round size is fixed at 128 total
  episodes, so M's scripted-easy exposure *substitutes* for half its
  random-opponent data rather than adding to it. This design cannot
  separate "the mixture works" from "the mixture works at this training
  volume" — a volume-controlled arm would need its own declaration.
- **Next:** the reframed contact-vs-finishing question (above) is the
  natural next diagnostic — a frozen-checkpoint, opportunity-level read of
  the existing archive, no new training — ahead of either M8 or a PPO stage
  from these checkpoints.

Follow `training/reviews/m7b_r1n_h_declaration.md`. `autonomousQualificationEligible`
stayed false throughout; this decided nothing about R1 qualification.

### R1n-f — opponent-transfer evaluation of the R1n-e policies (complete 2026-09-14; generalization loss predates PPO)

**Results** (`training/reviews/m7b_r1n_f_results.md`; archive
`runs/m7b_engage_r1n_f_v0`; 1,085,699 of 2,000,000 decisions; all manifests
verified; the §9 reproduction check passed with 0 mismatches):

- **Every learned policy loses to `ScriptedAiAgent` red, at full scale
  (400 worlds), at both difficulties.** All six comparators (3 R1n-c
  initializers, 3 R1n-e finals) score 0% success against scripted easy and
  scripted normal, while the scripted teacher wins 100% of both.
- **This is not a PPO effect.** The R1n-c initializers fail identically to
  the R1n-e finals: `finalMinusInitializerSuccess` is exactly 0 on both
  scripted arms. The generalization loss is in the imitation stage.
- **PPO does shift *how* the policies fail on the easier arm:** deaths rose
  +5.5 points [+2.4, +8.6] as timeouts fell by the same amount, consistent
  with R1n-e's own finding that PPO trained more aggressive engagement.
- **Blue's offense collapses, not just its defense.** Against scripted easy,
  all six comparators combined threw 25,008 snowballs and landed **zero**
  hits. Against scripted normal, 5,488 throws landed 20 hits (0.4%), all
  from a single seed (97102).
- **Mechanism:** scripted red only throws inside `ENGAGE_RANGE = 9`
  (median spawn distance 7–9 units, vs 27–30 under the training opponent,
  `RandomAgent`) and throws far less often (1–12 vs 15–16 per episode) —
  entirely outside blue's training distribution.
- **Arm R (random red) reproduces R1n-e's archived split-E result** on an
  independent split: finals 91.25–94.75% success, 4.5–8.75% death (archive:
  92.0–94.3%, 5.0–7.8%).
- **A6:** a first collection attempt completed but was discarded (not
  archived) after the post-collection seed audit found an unused inherited
  config field colliding with R1n-b's band; the run was re-declared and
  fully re-collected. Every arm's contents are byte-identical between
  attempts.
- **Next (recommended):** address the imitation stage directly — train
  against a mixture of opponents, holding one out for evaluation — rather
  than a replication against the random opponent alone, which would confirm
  a narrower skill than intended. R1n-g (below) diagnosed *why* blue's
  offense scores zero hits against scripted red before this curriculum
  (R1n-h) is designed.

Follow `training/reviews/m7b_r1n_f_declaration.md`. R1n-e's policies were
trained against `RandomAgent` red, which aims where blue currently stands;
the archived behaviour analysis showed the learned gain is evasion. R1n-f
evaluates the frozen initializers and finals against `ScriptedAiAgent` red,
which leads its aim, takes cover, and retreats.

- **Arms:** random (reference), scripted easy, scripted normal. Each arm runs
  3 initializers, 3 finals, and the scripted blue teacher on the same fresh
  400-world split F (871000–871399), deterministic execution only.
- **Primary measure:** the seed-averaged success and death gap to the teacher
  on the same worlds, with a 400-world bootstrap interval. An exploratory
  probe found every learned policy at 0 wins of 40 against scripted red while
  the teacher won 40 of 40, so a final-minus-initializer comparison would be
  floor-to-floor; it is kept as a secondary measure with a declared floor
  clause.
- **Declared measures also include** a failure-mode decomposition (death,
  timeout with or without hits, no contact) and projectile-level mechanism
  counters.
- **Budget:** 1,690,000 decisions bound, 2,000,000 cap. No training.
- **Implementation amendments A1-A5** (declaration §11): the budget
  correction above; the scenario override is a restoring context manager
  around `full_authority_train_v1.scenario` (the file itself is unedited);
  the reproduction check passed (50 worlds, 0 mismatches) before any arm
  collected; the floor outcome is split by dominant failure mode
  (`no-transfer-floor-death`/`-timeout`/`-mixed`), since arm E's timeouts
  and arm N's deaths would otherwise share one label; the teacher-gap
  bootstrap resamples the 400 worlds, not the 3 policies.
- **The replication R1n-e recommended is superseded** by this finding — see
  R1n-g and the mixture-imitation curriculum (R1n-h) below.

### R1n-g — why does blue's offense fail against scripted red? A label-error diagnostic (complete 2026-09-14; three compounding failures, aim most severe)

**Results** (`training/reviews/m7b_r1n_g_results.md`; archive
`runs/m7b_engage_r1n_g_v0`; 312,567 of 400,000 decisions; all manifests
verified):

- **All three candidate failure modes are real and compounding — no single
  factor accounts for the whole effect.** Type/detection, aim, and
  positioning all degrade sharply and simultaneously against scripted red.
- **Aim is the most severe, and worse than random.** Mean
  `throwAimHeadingErrorDegrees` (teacher-labelled-THROW rows only) is
  131.8° on the easier scripted arm and 111.9° on the harder one, against a
  3.1° healthy baseline — past the 90° two-random-directions mark in most
  individual cells, meaning the throw head points systematically *away*
  from the correct target, not merely off from it. This alone is severe
  enough to explain R1n-f's zero-hit finding.
- **Type/detection also collapses:** mean throw recall falls from 0.79
  (healthy) to 0.35 and 0.15 on the two scripted arms, crossing R1n-c's own
  `throwRecallFlag` (0.5) — confirming, not falsifying, the declaration's
  first prediction, which a pre-collection probe had flagged as disfavored
  by throw-volume evidence alone.
- **Positioning degrades too:** move heading error rises ~5-9x, move
  endpoint error roughly triples.
- **PPO (R1n-e) nudges but does not fix any of this** — small
  recall/aim improvements on most cells, power calibration slightly worse
  on most — consistent with R1n-f's finding that PPO does not change the
  scripted-red floor.
- **R1n-h's curriculum must address all three:** more demonstration volume
  of the teacher throwing under a moving/covering opponent, aim/power
  regression that generalizes past the training opponent's engagement
  geometry (highest priority), and positioning demonstrations under the
  same opponent.

Follow `training/reviews/m7b_r1n_g_declaration.md`. `autonomousQualificationEligible`
stayed false throughout; this decided nothing about R1 qualification.

### R1n-e — KL-anchored frozen-reward PPO with death rate as primary test (complete 2026-09-14; survival improved, replication pending)

**Results** (`training/reviews/m7b_r1n_e_results.md`; archive
`runs/m7b_engage_r1n_e_v0`; 5,540,370 of 9,000,000 decisions; all manifests
verified; zero rejected actions):

- **Primary test passes.** The seed-averaged deterministic death-rate
  difference on split E is **−24.8 points [−27.9, −21.8]**.
  - Per policy: −36.0, −15.0, and −23.5, all with CIs below 0.
  - Success rose **+24.0 [+21.1, +27.0]**, timeouts +0.5 [+0.1, +1.0], and
    contact rose for every policy. The gain is not bought by avoiding the
    fight.
- **Deterministic finals.** Success is 94.3%, 92.3%, and 92.0%; deaths are
  5.0%, 7.8%, and 7.5%. That is teacher-level, against the teacher's 93/91
  and 7–9% on R1n-c's splits; the teacher was not run on split E.
- **Stochastic σ×0.5.** Death rate −25.3 [−28.5, −22.1].
- **Critic:** warm-start R² was 0.070–0.127 (the 0.25 gate fails), yet
  Monte Carlo PPO learned.
- **Predictions.** `D` overshot the predicted −10 to −3, and the KL stop
  fired before the last epoch in only 37–58 of 200 updates. Success ≥ 0 and
  anchor KL < 1 (maximum 0.615) held. The ingredients were bundled, so no
  single cause is attributed.
- **No R1 claim.** The per-policy success gains are +34.8, +14.3, and +23.0.
- **Next (declared recommendation):** R1n-f, a fresh replication (new
  training RNGs, new untouched split), then a second contrasting mission,
  under its own declaration.

Follow `training/reviews/m7b_r1n_e_declaration.md`. At the user's decision,
the primary test is blue death rate. It replaces R1's +20 success gain, which
R1n-c showed infeasible, so R1n-e makes no R1 qualification claim.

- **Training:** PPO from each of R1n-c's three imitation policies, 200
  updates of 64 complete episodes each.
  - σ is frozen at ×0.5 (from R1n-d).
  - Advantages are pure Monte Carlo (λ = 1, no truncation), with the
    warm-started `OptionCentralCritic` used as a baseline. A sanity stop
    applies only if the upper bound of the held-out R² interval is below 0
    (A7).
  - A KL anchor to the initializer uses β = 0.01, plus E3's clip and a KL
    stop of 0.01.
  - Reward is the frozen Engage reward, and death rate is not the training
    objective.
- **Primary test:** the deterministic death-rate difference (final −
  initializer) on a fresh 400-world split. A world-bootstrap seed-average must
  be ≤ −5 points with its CI below 0. Non-inferiority guards: the success CI
  lower bound must be > −5 and the timeout CI upper bound < +5.
- **Power:** at an assumed 30–40% discordance, the detectable effect is about
  4–5 points.
- **Budget:** 8,870,400 bound (cap 9,000,000). Policy runs are independent
  and may run in parallel.
- **Implementation:** `options/death_rate_ppo.py`, run with
  `python -m snowgym_training.options.death_rate_ppo --output runs/m7b_engage_r1n_e_v0`.
  Pre-collection amendments A1–A10 are in declaration §12.
  - **A1:** the actor lr changed from 3e-4 to 1e-5 by an outcome-blind
    first-step-KL rule. At 3e-4, one Adam step moved the policy 0.7–4.0 nats
    against the 0.01 KL stop. The critic keeps 3e-4.
  - **A7:** the critic sanity stop uses the R² interval.
  - **A8:** a `no-effective-training` outcome applies when the median final
    anchor KL is below 0.01 and there is no death-rate effect.
  - **A9:** the restart procedure.
  - **A10:** the first attempt was killed at low memory during update 8 of
    policy 97101. The cause was a full-rollout anchor-KL diagnostic peaking
    at 3.6 GB. It is now chunked, peaking at 0.8 GB, and the run was
    re-declared. The aborted attempt is kept.

### R1n-d — diagnostics before PPO: exploration σ, attainable critic R², critic capacity (complete 2026-09-14; halve σ, critic outcome bracketed)

**Results** (`training/reviews/m7b_r1n_d_results.md`; archive
`runs/m7b_engage_r1n_d_v0`; 807,532 of 1,350,000 decisions; 96/96 manifest
digests verified; all 3,184 rollouts replay-identical):

- **Exploration.** The stochastic gap is continuous-action noise: 16.3 points
  at σ×1. Type sampling costs 0.3 points. The gap is 8.7 at σ×0.5 and 2.0 at
  σ×0.25, so the recommended scale is 0.5. Policy 97103 is still 20 points
  below deterministic at σ×0.5.
- **Critic ceiling.** The branched-rollout upper bound on attainable R² is
  0.16, 0.22, and 0.24 (CI upper bounds 0.23, 0.34, 0.33). 76–84% of return
  variance is within-state from blue's own sampling. The ceiling rises from
  0.05–0.18 at k = 0 to 0.23–0.33 at k = 100.
- **Critic capacity.** C0–C3 are indistinguishable: held-out R² within 0.011
  for each policy (0.04, 0.11, 0.05). Longer training, egocentric features,
  and removing the clip give no gain. Capture of the upper-bound value
  variance is about 0.2–0.35, with wide intervals.
- **Rules.** 97101 is gate unreachable; 97102 and 97103 are bracketed, so the
  outcome is bracketed. The absolute 0.25 gate is not a meaningful
  precondition.
- **For R1n-e:** use σ×0.5, a critic gate relative to a ceiling measured at
  its own σ, power sized for near-Monte-Carlo advantage variance, and a
  primary test still to be set.

Follow `training/reviews/m7b_r1n_d_declaration.md`. The user chose to split
off a diagnostic before any PPO; the PPO stage R1n-c called "R1n-d" is now
R1n-e. Pre-collection amendments A1–A8 are in declaration §11.

The implementation is `options/pre_ppo_diagnostics.py` plus
`executor/full_authority_critics.py`, run with
`python -m snowgym_training.options.pre_ppo_diagnostics --output runs/m7b_engage_r1n_d_v0`.

- **D1:** decomposes R1n-c's execution-mode gap on 680000–680099 over six
  modes for each of R1n-c's three final policies. The modes vary type
  sampling, continuous sampling, and σ×{1, 0.5, 0.25}.
- **D2:** an upper bound on attainable critic R² from branched rollouts.
  Each branch replays a recorded stochastic prefix, with replay identity
  checked by digest. The grid is 24 source episodes × k ∈ {0, 25, …, 125} ×
  8 rollouts, and the estimator applies an ANOVA correction. Red's future
  draws stay fixed within a branch, so this is an upper bound, not a noise
  measurement.
- **D3:** a lower bound from critic variants on new folds. C0 replicates
  R1n-c; C1 trains longer with early stopping; C2 is an egocentric critic;
  C3 is C2 without a gradient clip. The selected variant is chosen on
  validation MSE.
- **Rules:** critic repairable, gate unreachable, or bracketed; a
  recommended σ scale; a type-sampling flag. They authorize nothing.
- **Budget:** capped at 1,350,000 decisions; no actor or PPO update.

### R1n-c — imitation of the plan teacher that keeps its artifacts (complete 2026-09-13; PPO has headroom, critic precondition failed)

**Results** (`training/reviews/m7b_r1n_c_results.md`; archive
`runs/m7b_engage_r1n_c_v0`; 535,358 of 900,000 decisions; 95/95 manifest
digests verified):

- **Imitation success:** 65/100 on split A and 74/100 on split B
  (deterministic, three seeds; per seed 54–79 on A, 61–83 on B). Teacher
  ceilings are 93 and 91. Paired gaps: −28 (−35 to −21) on A and −17 (−24 to
  −10) on B.
- **Contact is solved** (92–93%, from 0 at random init). Nearly all failures
  are deaths in worlds the teacher wins.
- **Label error:** type accuracy 94–95%. Recall: MOVE 0.97–0.98, THROW
  0.68–0.73, HOLD 0.51–0.62. Aim error about 3°.
- **The critic precondition fails 0/3:** predictive R² 0.06–0.11, against the
  0.25 gate. It is no longer clock-only. Train R² is only about 0.12, and R²
  is ≈ 0 in the first 100 decisions. The archive cannot tell a critic that
  fits too little from returns that are mostly unpredictable from the state.
- **Flags:** execution-mode gap set (stochastic success 43–54; worst gap 31
  points); no seed instability; no throw collapse.
- **Rules:** A gives headroom and B near ceiling, so the conservative outcome
  is PPO has headroom. R1's +20 gate is infeasible on B (it would need 94
  against the 91 ceiling). R1n-d must settle three things before any PPO
  fit: the primary test, critic repair or an estimator choice, and
  exploration σ.

Follow `training/reviews/m7b_r1n_c_declaration.md`. This is step 1 of the D3
branch R1n-b recommended. The pre-collection amendments A1–A6 are in
declaration §11. The implementation is
`options/full_authority_imitation.py`, run with
`python -m snowgym_training.options.full_authority_imitation --output runs/m7b_engage_r1n_c_v0`.

- **Training:** BC then DAgger from the plan teacher into a
  `FullAuthorityPolicyV1` with global decoding, on 1v1 Engage at the frozen
  200-decision horizon (216 recorded and declined, since teacher failures are
  deaths, not timeouts).
- **Rounds:** round 0 from the teacher, then 4 on-policy rounds and 5 fits,
  for optimizer seeds 97101–97103.
- **Losses:**
  - type: cross-entropy;
  - move: S12-style heading;
  - throw aim: heading;
  - power: MSE on the 0–1 scale.
  - Exploration log-stds and the critic stay frozen.
- **Retained:** every fit's weights and optimizer state, fit history, and
  dataset digests.
- **Evaluation:** final fit only, deterministic on development splits
  600000–600099 and 601000–601099, stochastic on the first, against the
  teacher ceiling and uniform floor, with per-head and per-type held-out label
  error.
- **Critic precondition:** a Monte Carlo warm-start gate on each imitation
  policy's own stochastic episodes.
- **Scope:** local decoding is excluded (labels not representable), the plan
  input is uninformative at 1v1, and no PPO runs. The PPO stage is R1n-d,
  declared after these results, because R1's +20-over-initializer gate may not
  be reachable if imitation is already near the 89/100 ceiling.
- **Budget:** capped at 900,000 decisions.

### R1n-b — E3 contract repair and pre-actor diagnostics (complete 2026-09-13; recommends D3 with the global decoder)

**Results** (`training/reviews/m7b_r1n_b_results.md`; archive
`runs/m7b_engage_r1n_b_v0`, 504,771 of 600,000 decisions, 46/46 manifest
digests verified):

- **Critic gate repaired.** The corrected critic passed in all 6 arm/RNG
  configurations, with predictive R² 0.989–0.999 where E3 reported −61 to
  −511. But the passing target is almost entirely a function of time, so the
  gate certifies clock reading only. A future branch must re-verify its critic
  on its own policy's episodes.
- **Untrained fighters never make contact:** 0 hits in 2,304 random-init
  episodes, and the floor was 0/100. Frozen-reward PPO from scratch (D1) has no
  damage-dealing signal. In the global arm the only non-clock signal is being
  hit, which penalizes approach.
- **Local vs global explore differently despite matched commanded noise**
  (about 2.2 world units each): local never came closer than 14.6 world units;
  global came within throw range 6 times and died 9 times.
- **1v1 teacher ceiling:** the plan teacher wins 89/100 with 0 rejections;
  every failure is a blue death. 77% of its MOVE labels are beyond local reach,
  with none saturated. At roster 1 it behaves identically to `SimpleBlueAgent`,
  so the plan input is uninformative there.
- **Horizon:** 1.5× teacher p95 would be 216, versus the frozen 200.

The predeclared rules give D3 for both arms. Read with the label audit, the
recommendation is D3 with the global decoder: an artifact-retaining BC/DAgger
stage, then KL-anchored frozen-reward PPO with a re-warm-started critic, a
second contrasting mission before any plan-following claim, and leave-one-out
ablations. This needs its own declaration.

**Status before collection:** declared and implemented.

Follow `training/reviews/m7b_r1n_b_declaration.md`. The declaration (`365f67f`)
and its tested implementation are committed; pre-collection amendments A1–A8
are listed in declaration §8.

Implementation:
- `executor/full_authority_ppo_v1.py`;
- `options/full_authority_train_v1.py`;
- `options/full_authority_diagnostics.py`, run with
  `python -m snowgym_training.options.full_authority_diagnostics --output runs/m7b_engage_r1n_b_v0`.

Index-selective scripted steps and teacher reads are in a training-side
`SelectiveBatchEnv` subclass. The digest-pinned Python client is unchanged.
Two blockers stand in series: the E3 critic gate measured a near-constant
bootstrapped target, and the frozen Engage reward gives no dense signal
before first contact.

- **B — repair.** New modules only; a test pins E3's archived file digests.
  - A separate `throw_log_std`, globally calibrated in both arms.
  - A decoupled critic with its own loss, backward pass, and optimizer, not
    gated by the actor KL stop.
  - `predictiveR2` as the gating metric; explained variance and a 20-bin
    time-only baseline reported alongside.
  - Monte Carlo warm-start targets on complete episodes.
  - `OptionCentralCritic`, which reads `option_state`.
  - Retained arrays, per-episode rows, and an offline exploration calibration.
- **C — diagnostics** under a 600,000-decision cap:
  - a 1v1 plan-teacher precondition and ceiling, with a label audit;
  - a plan-blind `SimpleBlueAgent` reference;
  - contact rates for the floor and random-init policies;
  - the corrected warm start for 2 arms × RNGs 98001–98003.

  The seeds were declared as fresh bands: 620000–620099, 630000+, and
  640000+. **Erratum:** 620001–620004 and 630000–630119 had already been
  used by orchestration diagnostics in other scenarios. No result is
  affected, but the selective-repair preflight test fails; see the results
  note.
- **D research**, reported only: contact signal for a repaired E3b, offline
  potential statistics for approach shaping, and teacher availability and
  label representability for BC/DAgger-first.

Predeclared rules turn the results into a Phase D recommendation. They
authorize nothing. No actor training, checkpoint promotion, or gate change.
Longer-range sequencing is in the local plan note
`refs/snowgym_continuation_plan_2026-09-13.md`.

### R1n — E3 small complete autonomous skill: 1v1 full-authority PPO (complete; stopped at critic gate 2026-09-13)

Follow `training/reviews/m7b_r1n_declaration.md`: the reviewer handoff's
Experiment E3, and the first real attempt at the long-open R1n autonomous-
Engage milestone. From-scratch `FullAuthorityPolicy` (new module,
egocentric features carrying forward R1m-S12's winning representation; no
frozen source, no corrected shots, no teacher MOVE), full `{NOOP, HOLD,
MOVE, THROW}` authority, a warm-started decoupled critic gated at held-out
R2 >= 0.25, two arms differing only in movement-destination decoding
(local: isotropic R=8 world-unit radius; global: the existing
arena-centered tanh decode), both calibrated to a genuine 2-world-unit
median immediate perturbation via `calibrated_target_log_std`. 1v1 only;
2v2 deferred (throughput was measured only for 1v1: 2,911 decisions/second
at 64 worlds with real inference, supporting the full 12M-decision budget
in about 69 minutes). No teacher ceiling exists for this roster size, so
the only control is a uniform-random legal-action floor. Supervised/BC
assistance is entirely absent in the primary arms;
`autonomousQualificationEligible: false` (a diagnostic, not a
qualification attempt).

**The experiment did not reach its research question.** All six arm/seed
combinations (local/global x seeds 96001/96002/96003) failed the
predeclared critic warm-start gate (held-out R^2 >= 0.25) on their first
and only attempt (R^2 from -510.67 to -61.14, not a marginal miss); per the
stopping rule none was retried, and no actor training or checkpoint ever
ran. Used 142,880 of the 12,500,000-decision budget. A standalone
diagnostic traced the failure to the warm-start gate's own measurement
design, not to the critic being unable to learn: the 64-decision held-out
window is too short relative to the 200-decision episode horizon, so its
GAE-return target is dominated by bootstrapping off the untrained critic's
own near-constant value (target SD 0.0038) rather than real outcomes, and
R^2's denominator collapses toward zero. The clean fix — matching this
review's own R1m-S11/S9 reanalysis, which found Monte Carlo returns on
completed episodes gave a meaningful held-out R^2 (0.18-0.38) where a
bootstrapped return target did not — is to re-measure warm-start fit
against completed-episode Monte Carlo returns, not a fraction-length
bootstrapped window; this is proposed as a separately declared follow-up,
not executed here. The reward-sparsity property recorded in the
declaration (no dense signal before first contact) also held throughout.
Decision: **inconclusive on L vs. G; the critic warm-start gate itself
needs re-derivation before the actor budget can be spent.** See
`training/reviews/m7b_r1n_results.md` for the full results table and
diagnosis.

Implementation gate passed 367 TypeScript, 51 Python client and 372 Python
training tests (including 12 new targeted tests, several exercising the
live batch client and one live end-to-end training run) plus build. No
provider calls, browser input, or protocol changes; no checkpoint
promotion; R1n remains open pending a corrected warm-start gate.

### R1m-S12 — E2 movement representability under dense supervision (complete; egocentric frame wins 2026-09-13)

Follow `training/reviews/m7b_r1m_s12_declaration.md`: the reviewer handoff's
Experiment E2. Three arms share one frozen R1f source, one residual decoder,
and one controller-aware heading loss over a four-round DAgger loop: A
(current absolute `GeometryProbe`), A-rel (`GeometryProbe(relative=True)`,
a free control added to isolate frame from attention), and B (new
`AttentionGeometryProbe`, egocentric pair features pooled by fighter-query
attention). Evaluation is from-reset only (a stated deviation from the
review's first-hit-anchored floor/ceiling; see the declaration).

A-rel and B are statistically indistinguishable from each other (largest
gap +5 points, every interval includes or touches zero) and both clearly
beat A (replication-dev: A 31/40, A-rel 38/40, B 40/40; training: A 52/64,
A-rel 58/64, B 61/64). Ceiling (teacher-forced, from reset) is 40/40; floor
(zero-init) is 13/40 for all three arms identically. Zero rejected actions
across 338,590 evaluated/collected actions. Decision, per the predeclared
rule: **A-rel ~= B > A, the egocentric frame was the whole story.** E3 uses
`relative=True`-style egocentric geometry; `AttentionGeometryProbe` is not
carried forward. Used 194,958 of 250,000 budgeted decisions. See
`training/reviews/m7b_r1m_s12_results.md` for the paired-bootstrap
comparisons and two stated implementation gaps (ceiling/floor evaluated on
one development split, not both; held-out heading error not measured --
neither affects the decision).

Implementation gate passed 367 TypeScript, 51 Python client and 350 Python
training tests plus build. Supervised throughout;
`autonomousQualificationEligible: false`. No provider calls, protocol
changes, or checkpoint promotion.

### R1m-S11 — E1 null-control and replication of S9-full (complete; null indistinguishable from real 2026-09-13)

Follow `training/reviews/m7b_r1m_s11_declaration.md`: the reviewer handoff's
Experiment E1. Reuses `horizon_train` (S9) without editing it; a new module
`horizon_null_train.py` adds a permuted-advantage null arm and live update-1
opportunity capture. Full arm only. Real training RNGs 99302/99303 (new) plus
archived 99301 (reused); null training RNGs 99301/99302/99303 (all new).

All six predeclared measurements land real and null in the same range: dev
success/return gates, update-1 constant-vector destination fit (R2 0.55-0.97
for both), parameter RMS versus lr*sqrt(steps) (ratio 0.92-1.14 for both),
adjusted training-success gain, deterministic training-frame success, and
stochastic historical success. Where one group's extreme exceeds the
other's it goes in both directions (e.g. a null run has both the largest
constant-vector magnitude and a significant *negative* dev-return interval).
No falsifier triggered. Used 468,762 of a corrected 550,000-decision budget
(the first attempt at 450,000, extrapolated from S9's own figures,
undercounted and was discarded before archiving). See
`training/reviews/m7b_r1m_s11_results.md`. Decision: S3/S9 carry no
information about learnability under this configuration; do not run a
further automatic PPO sweep on this movement-residual line. Proceed to E2
and E3 as planned; neither depends on this configuration.

Implementation gate passed 367 TypeScript, 51 Python client and 350 Python
training tests plus build.

### R1m-S10 — frozen late-state learning audit (complete 2026-09-05)

Follow `training/reviews/m7b_r1m_s10_declaration.md`. Audit all S9 selection
records and reconstruct updates 1/11/21 of both RNG 99301 arms. Separate late
state coverage, critic/advantage quality, frozen mean-output scores and cumulative
final-policy geometry. No fitting, new continuations, provider calls or promotion.

All 48 trajectories and six first-minibatch losses reproduced exactly, using
8,399 simulator decisions. Late rows are 59.5% of full's optimizer selections;
late critic explained variance is near zero and 87–91% of advantage variance
is between episodes. Only 3.04% of audited late MOVE recommendations are within
three latent sigma; cumulative final-policy gap reduction is mixed and nearly
zero pooled. See `training/reviews/m7b_r1m_s10_results.md`. Next proposal is a
separately declared critic-only learnability diagnostic; no actor run authorized
by this negative diagnostic and no qualification gate changed.

### R1m-S9 — learnable control horizon (complete; gate failed 2026-09-05)

Follow `training/reviews/m7b_r1m_s9_declaration.md`: matched short/full movement
PPO on S3 first-hit frames, same initialization/control/reward and 240 presented
optimization rows per update. Thirty updates x eight episodes per arm; final-only
paired development gates, conditional optimizer-seed replication. Preserve
actual sample/KL-stop accounting and the 430,000-decision cap. Corrected shots
remain assistance; no teacher MOVE overrides or autonomous promotion.

RNG 99301 completed both 30-update arms. Final historical successes: short 16/38,
full 13/38, initializer 13/38. Replication-development: short/full 17/36,
initializer 16/36. Both gates failed; conditional optimizer replications were
not run. All 262 zero-residual parity episodes matched; source unchanged, zero
final-evaluation rejects. Equal 7,200 optimizer rows yielded 103/125 actual steps
after KL stopping. Used 160,301 simulator decisions. See
`training/reviews/m7b_r1m_s9_results.md`. Next proposal: frozen late-state
coverage/credit/update-direction inspection, not another automatic PPO sweep.

### R1m-S8 — same-state sustained continuation (complete; diagnostic gate passed 2026-09-05)

Follow `training/reviews/m7b_r1m_s8_declaration.md`: reconstruct every S6 squad-30
nonterminal handoff, compare exact source continuation with sustained teacher
MOVE destinations, and repeat both arms. Keep original scoring, horizon,
classifier and corrected shots. Predeclare paired success/return gates and the
19,200-decision cap. No PPO or autonomous promotion follows automatically.

All 46 repeat pairs matched; all 23 source suffixes reproduced S6 exactly.
Sustained movement achieved 21/23 mission successes versus 12/23: +39.1 points
[21.7, 60.9], return +0.7051 [0.3847, 1.0936], no rejected actions. Nine failures
recovered and no source successes were lost. Used 14,624 simulator decisions;
one pre-handoff timeout was excluded by the declared rule. See
`training/reviews/m7b_r1m_s8_results.md`. This is assisted Engage completion on
exposed training states, not battle wins or autonomous qualification. Next:
predeclare sustained learned-movement control with matched initialization and
explicit sample/optimizer budgets; no new PPO defaults or training yet.

### R1m-S7 — frozen handoff/completion inspection (complete 2026-09-05)

Follow `training/reviews/m7b_r1m_s7_declaration.md`: exact replay of S6 Keep and
squad-30 trajectories to inspect range, readiness, action choice, remaining target
health and original budget around the handoff. No new actions, seeds or training.
Separate descriptive temporal associations from causal handoff claims.

All 48 trajectories reproduced exactly in 8,405 decisions. On 17 complete
paired seeds, squad-30 outward MOVE frequency beyond range rose from 4.6% to
45.4% after handoff, while the immediate progress contrast relative to Keep
remained inconclusive. All 12 squad-30 failures timed out with assigned Blue
alive; 11 post-handoff failure tails averaged 11.8% occupancy within range 9.
See `training/reviews/m7b_r1m_s7_results.md` and the digest-bound S7 archive.
Next proposal: predeclare a same-state post-correction source-versus-sustained
teacher-MOVE continuation fork. No new experiment, reward change, or PPO default
change has been run; S6's primary gate and autonomous qualification remain failed.

### R1m-S6 — teacher duration/scope factorial (complete; primary gate failed 2026-09-05)

Follow `training/reviews/m7b_r1m_s6_declaration.md`: same 24 training states,
single versus squad MOVE correction, one versus 30 decisions, unchanged control,
two exact executions per arm. Keep and single-1 must reproduce S5 artifacts.
Test whether sustained teacher benefit transfers; separate duration, scope and
their interaction while reporting actual intervention dose. No PPO or promotion.

All 120 duplicate pairs and 48 S5 reference traces matched. Squad-30 reached
12/24 successes versus Keep's 9/24, but paired success/return intervals included
zero. Local damage and final progress improved; the primary transfer gate failed.
All 12 squad-30 failures were timeouts with the assigned Blue force alive. See
`training/reviews/m7b_r1m_s6_results.md`. Next proposed work is a read-only
failure/handoff audit of the saved trajectories before another learning experiment.

### R1m-S5 — destination-to-motion boundary probe (complete 2026-09-05)

Follow `training/reviews/m7b_r1m_s5_declaration.md`: 24 training-only snapshots,
one fighter/one decision per branch, signed radial/lateral shifts of 1/5 world
units, unchanged and teacher-target references, duplicate exact reconstruction.
Measure simulated motion and full frozen-continuation consequences separately.
No training, action-interface changes or qualification; stop and review evidence
before declaring any intervention.

All 240 duplicate branch pairs and 24 archived controls matched; zero rejected
actions. Mean first-decision position difference rose from 0.02694 to 0.12232
world units at the larger scale, but neither scale nor any arm established a
positive return effect. Radial and lateral sensitivity differed substantially.
See `training/reviews/m7b_r1m_s5_results.md`. Do not widen PPO noise or change the
decoder by default. A matched teacher-duration/coordination diagnostic is proposed
to distinguish sustained/team control from isolated perturbations; not yet declared.

### R1m-S4 — frozen exploration/advantage audit (complete 2026-09-05)

Follow `training/reviews/m7b_r1m_s4_declaration.md`: reconstruct 72 archived S3
trajectories at updates 1/11/21 and compare saved policies on the same 57 training
snapshots. Measure physical exploration, recommendation distance, actual policy
drift and frozen output-space advantage direction separately. No optimization,
new continuation, provider call or promotion; declare any intervention afterward.

All 72 trajectories reconstructed with zero sampled-log-probability and initial
minibatch-loss discrepancy. Mean sampled target shifts were 0.94–1.00 world units
against recommendation gaps of 7.30–9.82; local advantage alignment was weak and
mixed. Final common-state corrections were small/inconsistent. See
`training/reviews/m7b_r1m_s4_results.md`. A paired training-only exploration-scale
physical probe is proposed; its protocol must be declared before execution.

### R1m-S3 — learned short movement recovery (complete; gate failed 2026-09-05)

Follow `training/reviews/m7b_r1m_s3_declaration.md`: training-only first-hit
snapshots, 30 learned movement decisions followed by frozen continuation, three
RNGs × 30 PPO updates. Preserve tail rewards and expose recovery time through a
new trainable input/checkpoint contract. Evaluate final checkpoints on both
already exposed development sets. No shot-assistance removal or autonomous
promotion; R1n remains open. Freeze the protocol before collection.

All three 30-update runs completed on 57 training-only snapshots. Final success
was 14/38, 14/38 and 11/38 on historical development (initializer 13/38), and
19/36, 17/36 and 17/36 on replication-development (initializer 16/36). Every
paired success interval included zero; no run passed the assisted learning gate.
Source preservation, tail-return accounting, likelihoods and resume checks passed.
Stop at the declared budget; do not promote a checkpoint. See
`training/reviews/m7b_r1m_s3_results.md`. A frozen exploration/advantage-alignment
audit is proposed before another PPO intervention; no new training is declared.

### R1m-S2 — post-hit continuation diagnostic (completed 2026-09-05)

Use `training/reviews/m7b_r1m_s2_declaration.md` to isolate brief versus sustained
movement correction after the first hit, with action-choice and combined controls.
Freeze the R1h source and corrected shots; use 40 already exposed development
seeds, five same-state branches, and exact duplicate replay. No PPO, provider
calls, seed resampling, horizon extension or promotion. Reproduce the repaired
R1m baseline before accepting branch evidence. R1n remains open.

All 40 baselines reproduced exactly; 38 nonterminal first-hit states supplied
190 exact duplicate branch pairs. Keep succeeded on 13/38, choice-30 on 16/38,
move-30 on 23/38, both-30 on 17/38 and move-rest on 35/38. The two movement arms
passed the declared further-study criterion; the choice and combined arms did
not. Every branch remains teacher-assisted. See
`training/reviews/m7b_r1m_s2_results.md`. A short learned movement-recovery option
on training-partition snapshots is proposed next, with a new budget/seed/gate
declaration required before training. No automatic PPO or promotion follows S2.

### R1m-S1 — critic-schedule isolation (complete; gate failed 2026-09-05)

Resume the fighter recovery track with the bounded matched experiment in
`training/reviews/m7b_r1m_s1_declaration.md`: three RNGs, coupled versus independent
critic scheduling, 20 updates per arm. Preserve corrected-shot assistance and
the actor KL stop. Verify identical first-rollout actor/optimizer/RNG behavior
before collection, then report final paired development effects. R1m replication
remains failed and R1n remains open. No provider calls or qualification promotion.

All six declared runs completed. Independent scheduling improved common-rollout
critic fitting with exact first-update actor/Adam/RNG parity, but replication-
development success differences were −5, −2.5 and +20 percentage points across
the three RNGs. The mean +4.17-point gain and two negative estimates failed the
predeclared consistency gate. Keep the coupled default and archive the positive
and negative evidence; no budget extension or checkpoint promotion. See
`training/reviews/m7b_r1m_s1_results.md` and `training/runs/m7b_engage_r1m_s1_v0`.

### Selective tactical repair — completed mechanism audit (2026-09-05)

Declaration: `refs/snowgym_selective_repair_review_and_plan.md`. Compare keep,
binding-only refresh, bounded opportunistic fire and full reactivation on the
three archived casualty cases and 40 fresh diagnostic seeds per 5v5/10v10/6v10
cohort (630000–630119). Use the first unchanged casualty predicate within 300
decisions, no resampling, 0/1/2/4/8-second delays and a shared 300-decision
continuation budget. Audit seed allocations first; preserve fixed scoring
targets and independently rerun every continuation. Report paired exploratory
bootstrap intervals (10,000 resamples, RNG 730001), coverage and negative results.
No provider calls, production-default changes, learned repair policy, automatic
promotion or changes to R1n/M7c qualification are included.

Harness implemented: `orchestration/examples/selective-repair-audit.ts` has
read-only seed preflight, immutable compressed traces, isolated interventions,
exact historical/delayed-prefix parity and paired-bootstrap analysis. The harness
gate passed 367 TypeScript, 51 client and 257 training tests plus build. The sealed
run `orchestration/recovery/examples/selective-repair-20260905-v0` has 117 fresh
casualty cases (37/40, 40/40, 40/40), three historical cases and 2,400 branches,
each rerun exactly; 1,046,237 recorded action results had zero rejections.
Binding refresh matched full reactivation in all 600 physical trajectory pairs.
At zero delay it raised 6v10 wins from 0/40 to 5/40 but lowered 10v10 wins from
33/40 to 15/40. Local fire's 5v5 win-rate gain remains uncertain; its condition
never activated in fresh 10v10/6v10 cases. No default was promoted. See
`orchestration/recovery/SELECTIVE_REPAIR_RESULTS.md` for intervals, delay effects,
scope limitations and a proposed separately declared retain/repair gate.

### Side diagnostic — progress-aware commander recovery (2026-09-05)

Preserve the scripted executor and `CommandPlan` output contract. This work
does not qualify a learned fighter or bypass the M7c gate. Astra-low remains
the operational default; compare Luna-low and Astra-medium explicitly.

1. Add optional, versioned, ID-free server evidence: activation-relative force
   health, frozen target completion, readiness, objective range, conservative
   direct-path obstruction, and bounded execution progress. Geometry proxies
   must not be presented as pathfinding, hit probability, or learned affordances.
2. Collect exact-prefix recovery opportunities using predeclared predicates for
   blocked advance, target elimination, recent casualties, and throws without
   damage. Report missing families rather than inventing disturbances or labels.
3. Compare unchanged versus enriched input on identical snapshots, with the
   same prompt/schema/executor. Archive no-change controls and deterministic
   0/1/2/4/8-second delayed activation before any provider comparison.
4. Gate a bounded provider pilot on successful reconstruction and scenario
   coverage. Separate physical execution, plan validity, recovery progress,
   and eventual battle outcome; all outputs remain immutable and digest-bound.

The first delivery is the provider-independent evidence/fixture/latency gate.
Multi-request closed-loop battles, calibrated skill-success estimates, and
event-driven scheduling comparisons follow this gate; they are not implied
by a successful single-opportunity continuation.

First delivery implemented: `orchestration/recovery/` and
`orchestration/examples/commander-recovery-benchmark.ts`. Optional evidence,
exact-prefix reconstruction, semantic tamper checks, old/enriched request
freezing and 0/1/2/4/8-second keep/reactivate controls are implemented. The
four-world discovery scan found three recent-casualty opportunities and no
examples of the other three families. Preserve this negative coverage result;
declare purpose-built scenarios before a provider comparison. No new API calls
were made. Full gates passed: 355 TypeScript tests, 51 Python client tests,
257 Python training tests, and build. Existing pilot request bodies are exact.

### M0 — autonomous blue control and server status (current)

- [x] Canonical `UnitAction` / `TeamAction` types, free of UI inputs
- [x] Explicit `hold` action that cancels stale movement without changing `noop`
- [x] Detached entity observation with deterministic ID ordering
- [x] `TeamController` policy contract
- [x] Simple blue dodge / approach / throw policy
- [x] Validating SnowCraft action adapter
- [x] DOM-free `reset`, `observe`, and `step` lifecycle
- [x] 10 Hz decisions over the existing 60 Hz simulation
- [x] JSON status, reset, step, and autoplay endpoints
- [x] Verified deterministic autonomous completion in the Node integration test
- [x] Verified live HTTP reset/status/step/autoplay flow

Exit criterion: blue units independently move and throw without selection or
human input, a 3v3 match reaches a team-elimination result, and a client can
retrieve the result from the server without a renderer.

### M1 — reproducible environment contract

- [x] Extract a DOM-free simulation composition root with explicit system order.
- [x] Implement `reset(seed)`, `observe(team)`, and blue-team `step(action)`.
- [x] Advance a configurable integer number of physics ticks per policy decision.
- [x] Return terminal-only reward (`+1`, `-1`, `0`), `terminated`, `truncated`, and
      structured `info`; keep diagnostic event rewards separate.
- [x] Migrate the existing red behavior behind the same `TeamController` boundary.
      The scripted red squad now runs through `ScriptedAiAgent`, which re-runs
      the classic `AISystem` per-tick logic and reports its orders as semantic
      actions; full-episode traces are bit-identical to direct AI registration.
- [x] Record scenario, seed, action trace, simulation version, upstream base
      commit, and one public-state hash per replay frame.
- [x] Add exact same-seed/action-sequence state-hash tests and max-tick
      truncation tests.

Exit criterion: a Node test can run and exactly replay 3v3 without DOM, Canvas,
WebGL, Three.js rendering, browser timing, or input state.

### M2 — Gymnasium bridge

- [x] Publish the initial versioned `snowgym.v0` server schema.
- [x] Add fixed-shape numeric action/observation spaces and masks.
- [x] Register `gym.make("SnowGym/Squad-v0")`.
- [x] Add a project-local Python environment and locked dependencies.
- [x] Pass Gymnasium's environment checker against the live server.
- [x] Add a terminal-only scripted-blue demo command.
- [x] Strictly validate mutating request fields and isolate scripted stepping
      from explicit external actions.
- [x] Add optimistic state-hash guards, step idempotency, and machine-readable
      server capability discovery.
- [x] Check all three registered Gym environments, including map-backed v2,
      and support JSON CLI summaries.
- [x] Record versioned visual-replay JSON from detached server state.
- [x] Add a renderer-free CLI/function that builds reproducible M-vs-N examples
      for open arenas or bounded native map spawn pools.
- [x] Replay recordings through the existing Three.js rendering engine without
      coupling the Gym environment to rendering.
- Let the initial Python adapter consume the JSON server for correctness, then
  add a long-lived batch host/direct transport for training throughput.
- Add vectorized environment support over the future batch transport.
- [x] Add a shared TypeScript/Python golden fixture for the versioned public-state
      canonicalization and hash contract.
- Benchmark throughput at 1, 2, 5, 10, 20, and 60 Hz decision rates.

Exit criterion: `gym.make("SnowGym/Squad-v0")` passes Gymnasium's environment
checker and deterministic cross-language fixtures.

### M3 — configurable N-blue versus M-red scenarios (core complete)

- [x] Replace the fixed `THREE_VS_THREE_OPEN` assumption with validated scenario
      configuration: `blueUnits`, `redUnits`, spawn layout, arena dimensions, red
      difficulty, decision rate, seed, and max ticks.
- [x] Generate deterministic non-overlapping spawn layouts when explicit spawns are
      omitted; reject counts that cannot fit the arena.
- [x] Publish fixed roster maxima for `SnowGym/Squad-v1` (eight slots) and
      `SnowGym/Squad-v2` (ten slots); represent smaller N/M configurations with
      unit-presence and legal-action masks so a registered Gym environment's
      spaces never change after construction.
- [x] Extend reset/server configuration, replay metadata, reward/termination logic,
      and observations without breaking `SnowGym/Squad-v0` 3v3 recordings.
- [x] Load the bundled SnowCraft maps as scenario terrain: obstacles affect
      line-of-sight, cover, and collision, spawn points come from the map, and
      obstacles are exposed to policies as a fixed-capacity masked tensor.
- [x] Add a native 10v10 map whose browser JSON and headless registry definitions
      are contract-tested for exact parity.
- [x] Add a deterministic matrix covering 1v1, 1v3, 3v1, 3v3, and maximum-size
      fights, plus invalid counts/spawns and same-seed replay checks.
- [x] Migrate red behavior to the common `TeamController` boundary, then add
      independently selectable scripted, random, learned, or external opponents
      (scripted and random shipped; learned/external remain future).
- Benchmark episode throughput and balance by N/M configuration before training.

Exit criterion: one versioned environment can reset into multiple validated
N-vs-M configurations while retaining fixed Gym spaces, deterministic replay,
team elimination, and renderer-free server status.

### M4 — hierarchical commander (C4 complete)

- [x] Define the bounded `snowgym.command-plan.v0` group action space and strict
      JSON schema without unit IDs, enemy IDs, coordinates, or physical actions.
- [x] Add strict runtime validation for mission/objective compatibility, unique
      fixed roles, allocation bounds, and acyclic support relationships.
- [x] Add deterministic weighted group allocation for 3v3 through 10v10.
- [x] Add team-relative region and symbolic enemy-cluster target resolution.
- [x] Add a trusted host envelope and immutable atomic `PlanStore`.
- [x] Add a synchronous plan-aware controller and reactive per-unit executor.
- [x] Add a deterministic headless 10v10 split-force demonstration whose
      per-role action traces prove distinct group execution and exact replay.
- [x] Add synchronous plan lifecycle triggers for plan expiry, major own-force
      loss, assigned-group elimination, and objective completion.
- [x] Reconcile delayed candidates against the current living roster, repair
      bounded support/target drift, reject invalid candidates without replacing
      the active plan, and atomically activate accepted plans.
- [x] Keep a deterministic one-group fallback and detached lifecycle trace for
      every accepted, repaired, rejected, and fallback activation.
- [x] Add an ID-free, versioned strategic summary and a provider-neutral async
      `CommanderClient` boundary.
- [x] Prove non-blocking operation with a delayed mock commander: one request in
      flight, cooldown-governed trigger coalescing, simulation-tick timeout,
      stale-response reconciliation, and ignored late responses.
- [x] Add deterministic simulated-latency scheduling and a headless 10v10 C3
      demonstration with exact action/state/trace replay coverage.
- [x] Add a server-only `gpt-5.6-luna` Responses API adapter using strict
      structured output, reasoning, `store: false`, and environment-only
      `OPENAI_API_KEY`.
- [x] Gate the provider adapter on mocked error/refusal/timeout tests plus one
      explicitly authorized live headless schema smoke; never include it in the
      deterministic default test suite.
- [x] Run a separately authorized, wall-clock-paced live battle with automatic
      lifecycle requests disabled and a code-enforced limit of exactly one
      external request. The battle continued at 10 Hz, activated Luna's stale
      symbolic plan mid-battle, and completed without rejected physical actions.

Exit criterion: a slow commander can replace validated symbolic group plans
without blocking the 10 Hz physical controller or exposing transient unit
control to the model.

#### Agent monitoring and correction contract

Monitoring is host-owned and hierarchical. The LLM never polls individual
units, waits inside `step`, or corrects physical actions directly.

| Layer                         | Cadence                                          | Status consumed                                                                     | Corrections it may make                                                                                        | Escalation                                                                                    |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Simulation and action adapter | 60 Hz                                            | collision, legality, cooldown, alive state                                          | reject illegal physical actions and advance authoritative state                                                | report action results to the controller trace                                                 |
| Reactive unit policy          | each team decision, normally 10 Hz               | assigned group, live position/health/state, projectiles, current symbolic objective | dodge, choose a legal throw, reacquire targets, restore range/cohesion, or continue mission movement           | never waits for the LLM; future stuck/repeated-rejection summaries go to lifecycle monitoring |
| Plan-aware team controller    | each team decision                               | immutable active-plan snapshot plus current detached observation                    | late-bind enemy clusters, ally-group positions, and map-region objectives while keeping plan membership stable | continue the last valid snapshot during any commander request                                 |
| Plan lifecycle monitor        | each team decision                               | plan age, living assigned units, whole-team loss fraction, and objective progress   | activate deterministic fallback on expiry, major loss, group elimination, or completion                        | schedule a commander request; coalesce duplicate triggers while one is in flight              |
| Plan reconciler               | only when a candidate arrives                    | candidate provenance plus the newest detached observation                           | validate, shrink infeasible groups, repair missing support/enemy objectives, ground, then atomically activate  | reject unsafe drift and retain the old plan or fallback                                       |
| LLM commander                 | event-driven and rate-limited, target 0.1–0.5 Hz | a compact strategic summary captured with source tick/hash, never engine entities   | propose only a strict symbolic `CommandPlan`                                                                   | timeout/error cannot block control; host fallback and later retry policy remain authoritative |

The C3 delayed-commander test must prove this sequence:

```text
observe -> execute current plan -> detect/coalesce trigger -> request asynchronously
        -> keep executing current plan -> receive stale candidate
        -> reconcile against newest observation -> atomic activate or reject
```

C3 tests cover request timeout, duplicate-trigger coalescing, late response
after fallback, invalid response, provider failure, uninterrupted synchronous
control, and deterministic replay of the same mock-latency schedule. Stuck
detection and repeated action-rejection thresholds remain planned explicit,
debounced lifecycle signals; they must be host-computed rather than inferred by
the model.

### M4.1 — trajectory-aware closed-loop commander (C5)

- [x] Add a passive, bounded trajectory monitor over pre-step observation,
      active-plan snapshot, physical action results, and post-step observation.
- [x] Publish an ID-free, versioned group trajectory digest with mission-aware
      progress, health/cohesion trends, action counts, rejection counts, and a
      host-computed stuck fraction.
- [x] Prove that enabling telemetry cannot change physical actions, public-state
      hashes, plan activation, or replay results.
- [x] Add debounced `plan_stalled` and `action_rejection_repeated` signals with
      activation grace periods and recovery hysteresis.
- [x] Separate soft signals, which retain the current plan while replanning,
      from hard lifecycle failures, which activate deterministic fallback.
- [x] Pass the bounded trajectory digest and preceding plan outcome to each
      stateless commander request without exposing unit IDs or raw trajectories.
- [x] Run deterministic multi-request mock battles with exact state, plan, signal,
      latency, and scheduler-trace replay coverage.
- [x] Add an opt-in, wall-clock-paced Luna battle with one in-flight request, a
      code-enforced per-episode call limit, explicit token/latency accounting,
      and uninterrupted fallback on provider failure.
- [x] Pass an explicitly authorized live C5 acceptance battle: host-computed
      trajectory stalls triggered bounded Luna replanning while the executor
      continued at 10 Hz, used two of three permitted requests, activated a
      valid plan, and finished with zero rejected physical actions.
- [x] Record a versioned, ID-free commander trace sidecar bound to the replay's
      final public-state hash; add an optional scrubber-aware plan, aggregate
      trajectory, and lifecycle overlay to the existing replay UI.
- [x] Generalize the deterministic and Luna trajectory runners across validated
      bundled-map blue/red rosters and red difficulties while preserving the
      original 10v10 defaults. Reject invalid map capacities before entering
      the commander loop.
- [x] Add a host-owned economy-of-force opening for outnumbered blue rosters so
      the synchronous executor does not blindly advance into the enemy
      backfield while commander advice is pending; retain the direct opening at
      parity.

Exit criterion: host-computed trajectory evidence can trigger bounded Luna
replanning during an episode while the 10 Hz executor continues synchronously,
and identical mock latency schedules reproduce identical actions, state hashes,
trajectory digests, plans, and lifecycle traces.

Configurable-runner acceptance (2026-09-02): targeted tests cover an
understrength 6v10 trace on `arena6`, a 3v3 provider-neutral run on `arena4`,
default 10v10 compatibility, deterministic replay, and pre-run rejection of a
10-unit roster on a three-spawn map. No provider request is part of this gate.

Understrength-opening acceptance (2026-09-02): deterministic seed 14 on
`arena6` with 6 blue versus 10 easy scripted red completes with a 1–0 blue win,
two simulated-latency mock responses, zero rejected physical actions, and a
replay-bound commander trace. The example demonstrates possibility, not a
held-out win-rate or online-LLM result.

### M5 — multi-agent and research adapters

- [x] Add a two-team PettingZoo ParallelEnv over the same server and simulator,
      with mirrored fixed-capacity observations, simultaneous guarded joint
      actions, zero-sum terminal rewards, and the official Parallel API gate.
- [x] Add deterministic local-visibility, action-delay, and observation-delay
      research profiles around the parallel environment, with authoritative and
      observation-source ticks reported separately.
- [x] Add an optional, fixed-shape, renderer-free semantic raster that respects
      local visibility and retains the entity-tensor observation contract.
- [x] Add a versioned baseline evaluation suite and sequential multi-episode
      benchmark over the reference HTTP transport, separating deterministic
      results from wall-clock performance.
- [x] Add a Gymnasium single-team wrapper with deterministic random/no-op,
      detached learned-callable, and versioned remote-client opponent adapters;
      retain the existing native scripted/random server-controller route.
- [ ] Deferred: add optional rendered pixels only after learned entity-policy
      baselines exist; pixels are never a correctness or default training input.
- [ ] Carried into M6.1: add a long-lived direct/vectorized batch host and
      benchmark it independently from the correctness/reference HTTP transport.

### Reconciled RL phase boundary

The environment and hierarchical commander are now sufficiently complete to
test the central open question: can a neural policy learn SnowGym's fast hybrid
physical control and later execute its slow symbolic plans? Until that is
answered, pixels, new commander verbs, prompt tuning, map expansion, UI polish,
large policies, and self-play leagues are explicitly deferred.

The next phase reuses rather than replaces the current contracts:

- `/step-scripted` already returns the exact teacher `TeamAction` under
  `info.action`; the exporter must convert it back to the fixed Gym tensor
  action with exact round-trip tests.
- `snowgym.replay.v0` already stores observations, semantic actions, state
  hashes, and outcomes; learned rollouts should continue to use it.
- `snowgym.evaluation-suite.v0` already fixes scenarios, seeds, policies, and
  research profiles; checkpoint evaluation should extend this runner rather
  than introduce another scenario format.
- `SnowGymEnv` and `SnowGymSingleTeamEnv` are the Python closed-loop seams;
  `LearnedOpponent` supports learned opposing teams. The simulator must remain
  unaware of Torch.
- HTTP remains the correctness transport. A batch host must compose the same
  `SnowEnvironment`, never reimplement physics or reward logic.

Scaled data collection deliberately waits for the batch host. A small HTTP
corpus is sufficient to validate the trajectory and model contracts first;
100k–1M transition runs are not an M6.0 prerequisite.

### M6.0 — teacher data and behavior-cloning proof

Goal: prove the observation/action/checkpoint loop can learn and execute the
existing scripted blue teacher before adding policy-gradient failure modes.

#### M6.0a — training and trajectory contracts

- [x] Add a separate `snowgym/training/` Python package with its own lockfile,
      tests, and `torch` dependency; keep `snowgym-client` lightweight.
- [x] Define `snowgym.trajectory.v0` as compressed, non-pickle NumPy shards
      plus a JSON manifest. Record observation/action tensors, masks, reward,
      termination/truncation, scenario, seed, tick, teacher name/version,
      simulation/API/hash versions, and pre/post public-state hashes.
      Compute content digests over canonical array names, dtypes, shapes, and
      raw bytes so archive timestamp/compression metadata cannot affect them.
- [x] Add semantic-`TeamAction` to Gym-action inversion. Prove
      semantic -> tensor -> semantic equality within explicit float tolerance,
      including noop/hold, target normalization, power, dead/absent slots, and
      arbitrary roster sizes.
- [x] Export a small renderer-free scripted corpus through the guarded HTTP
      client. Capture the pre-step observation and the exact action returned by
      `/step-scripted`; never infer labels from post-step motion. Persist action
      results and reject a qualifying corpus containing illegal teacher labels.
- [x] Make shard ordering, manifests, split assignment, and content hashes
      reproducible. Training, validation, and held-out evaluation seed sets must
      be disjoint and committed as versioned manifests.
- [x] Measure the scripted teacher and masked-random baseline on the first held-
      out 1v1 suite before using teacher win rate as a learned-policy ceiling.

M6.0a gate: two exports with the same specification have identical manifests,
canonical tensor digests, tensors, and state-hash trajectories; a dataset audit
rejects a changed schema, corrupt shard, illegal label, or seed overlap.

M6.0a acceptance (2026-09-01): two live 20-transition exports produced the
same manifest and dataset digest; a complete held-out episode produced 53
audited transitions. On evaluation seeds 201/202, scripted blue beat native
random red 2/2 with zero rejected actions, while masked-random blue won 0/2
and both episodes truncated as draws. The committed result is
`training/baselines/teacher_1v1_v0.json`.

#### M6.0b — first neural executor

- [x] Implement small shared MLP entity encoders, masked mean/max aggregation,
      global context, and a shared per-ally actor. Do not begin with a
      transformer.
- [x] Emit a masked categorical action type, bounded normalized target, and
      bounded throw power per present ally slot. Invalid action types receive no
      probability mass.
- [x] Train with masked categorical cross-entropy plus target MSE for move/throw
      actions and power MSE for throw actions only. Keep all loss weights in a
      versioned configuration.
- [x] Add deterministic CPU training acceptance, fixed data-loader ordering,
      a one-batch overfit test, finite-gradient checks, and exact checkpoint
      resume. Accelerators may be optional but are not the reproducibility gate.
- [x] Define `snowgym.checkpoint.v0` metadata: git commit, dataset manifest hash,
      SnowGym versions, architecture, optimizer/loss configuration, training
      seed, step, and evaluation suite.
- [x] Add a `TorchPolicy`/`LearnedOpponent` bridge with detached NumPy tensors,
      `eval()`/no-grad inference, action-space validation, and no Torch import in
      simulator code.
- [x] Extend the evaluation runner with checkpoint policies and deterministic
      metrics: win/draw rate, episode length, survivors, and health lost/dealt
      derived from observations. Preserve terminal-only benchmark reward.
- [x] Record learned closed-loop episodes as normal visual replays; browser
      viewing remains optional validation and never model input.

M6.0 exit: on an explicit held-out 1v1 seed suite, the behavior-cloned policy
is reproducible, respects every mask, executes closed-loop, and improves over
the masked-random baseline. Report teacher, random, and learned results together
rather than claiming teacher parity from training loss alone.

M6.0b acceptance (2026-09-01): a 212-transition scripted corpus from train
seeds 11–14 trained the versioned entity policy for 5,000 deterministic CPU
steps. Independent runs produced the same model/optimizer state and checkpoint
digests. On held-out seeds 201/202, learned blue won 2/2 in 54 decisions with
zero rejected actions and no health lost; scripted blue won 2/2 in 53, while
masked-random blue won 0/2 and timed out at 200. The committed checkpoint is
`training/checkpoints/bc_1v1_v0`, the joined report is
`training/evaluations/bc_1v1_v0.json`, and both learned episodes are normal
`snowgym.replay.v0` files under `public/replays/bc_1v1_v0/`. This is a narrow
1v1 behavior-cloning proof, not evidence of broader scenario generalization.

### M6.1 — persistent batch simulation

Goal: remove one-request-per-decision HTTP overhead without creating a second
simulator.

- [x] Add `snowgym/batch/` with a persistent subprocess host that owns multiple
      independent `SnowEnvironment` instances and exposes a versioned handshake,
      batched reset, batched joint/single-team step, close, and error messages.
- [x] Begin with compact framed or newline-delimited messages over stdin/stdout.
      Keep protocol logging on stderr so stdout remains machine-readable; do not
      require native bindings for the first implementation.
- [x] Add a Python `SnowGymBatchEnv`/client with fixed leading batch dimension,
      per-slot seeds/scenarios, independent terminal state, selective reset, and
      explicit failure semantics. Start with 8, then 32, then 64 worlds.
- [x] Preserve per-world deterministic RNG, ID allocation, controller state,
      masks, rewards, termination, and public-state hashes. One failed world
      must not silently advance any other world.
- [x] Add golden HTTP/batch parity over the same version, scenario, seed, and
      exact semantic action sequence. Compare every state hash, reward,
      termination flag, truncation flag, and action result.
- [x] Add `snowgym/training/benchmarks/throughput.py`. Report environment count,
      decision rate, ticks per decision, decisions/sec, simulation ticks/sec,
      wall-clock real-time factor, CPU utilization, payload bytes, and measured
      serialization share for 1, 8, 32, and 64 worlds.
- [x] Document the 64-world limitation if the target machine cannot sustain it;
      never weaken parity or silently omit failed slots to reach a throughput
      number.

M6.1 exit: the trainer directly consumes at least 32 persistent worlds; 8/32
parity is exact against HTTP, 64 works or has an evidence-backed limitation,
and benchmark results clearly separate simulation from serialization cost.

M6.1 acceptance (2026-09-01): the training benchmark consumed 1, 8, 32, and 64
persistent worlds through `SnowGymBatchEnv`; 8/32 direct-service golden tests
and an eight-world live HTTP check matched complete reset/step payloads exactly.
The 20-decision smoke measured approximately 218, 1,544, 4,707, and 6,964
decisions/s respectively. At 64 worlds it measured 41,783 simulation ticks/s,
696x aggregate real time, 1.34 CPU cores, 1.89 MB of protocol payload, and 4.3%
Python JSON serialization share. These are local short-run measurements, not a
cross-machine performance guarantee. The committed machine-readable report is
`training/benchmarks/batch_throughput_v0.json`.

After this gate, scale the versioned teacher corpus toward 100k–1M transitions
only if learning curves or scenario coverage require it.

### M6.2 — centralized hybrid-action PPO

Goal: demonstrate reward-driven improvement without commander, unit-level MARL,
partial observation, or self-play confounds.

- [x] Add a custom Torch PPO implementation for the squad-level actor. The
      joint log probability includes action type for present units, target only
      for move/throw, and power only for throw; masks also apply to entropy.
- [x] Add detached fixed-horizon rollout/GAE buffers with terminal versus
      time-limit truncation handled separately, strict fixed-world tensor
      validation, immutable snapshots, and deterministic time/world flattening.
- [x] Add the deterministic minibatch optimizer loop with global advantage
      normalization, clipped policy/value losses, finite-gradient enforcement,
      gradient clipping, and aggregated KL/entropy/clip diagnostics.
- [x] Add exact rollout-boundary PPO checkpoint/resume metadata, including
      model, optimizer, Torch random state, update/environment-step counters,
      architecture/configuration, training seed, curriculum provenance, and
      semantic state/metadata digests.
- [x] Collect directly from persistent `SnowGymBatchEnv` worlds with a bounded,
      non-reusing episode-seed schedule, selective terminal resets, explicit
      horizon truncation/value bootstrap, and restartable rollout boundaries.
- [x] Add an atomic headless PPO smoke runner that binds a frozen gate, rollout
      geometry, seed cursor, update metrics, and final checkpoint in a hashed
      machine-readable run manifest; prove resumed and uninterrupted updates
      reach the same semantic state.
- [x] Keep canonical evaluation reward at win `+1`, loss `-1`, draw `0`. If
      sparse learning blocks the smoke test, add an opt-in training wrapper with
      potential-based own-minus-enemy health shaping and test that it leaves
      terminal benchmark results unchanged.
- [x] Use a gated curriculum: 1v1 random, 1v1 easy scripted, 3v3 random, 3v3
      scripted, 3v3 terrain, then 5v5 and 10v10. Do not advance a stage without
      fixed held-out evaluation evidence.
- [x] Keep training and evaluation seeds disjoint. Evaluate checkpoint series,
      not only the best checkpoint, against masked-random, native random, and
      scripted baselines using the versioned suite.
- [x] Freeze a headless PPO checkpoint evaluator over each gate's disjoint
      held-out seeds, with deterministic PPO inference, masked-random and native
      scripted-blue comparisons, canonical returns, rejected-action accounting,
      threshold reporting, and a hashed result contract.
- [x] Add an atomic checkpoint-series runner that retains and evaluates every
      predeclared update, records its checkpoint/run/evaluation digests and the
      complete update curve, distinguishes development from qualifying runs,
      and never selects only the best checkpoint.
- [ ] Record learning curves, policy/value losses, entropy/KL, rejected actions,
      throughput, checkpoint provenance, and replay links in a machine-readable
      run manifest.

M6.2 exit: PPO reproducibly solves the defined 1v1-random gate, shows meaningful
improvement against easy scripted 1v1, and exceeds the random-policy baseline
in 3v3-random evaluation. Exact numerical thresholds and seed counts must be
frozen in the evaluation manifest before the qualifying training run.

M6.2 foundation (2026-09-01): the versioned curriculum freezes disjoint
training ranges and eight evaluation seeds for each 1v1-random,
1v1-easy-scripted, and 3v3-random gate before training. The actor-critic now
implements mask-aware categorical actions, tanh/sigmoid continuous heads,
conditional joint log probability and entropy, a centralized value head,
terminal-aware GAE, clipped PPO loss, KL/clip diagnostics, and opt-in health
potential shaping. Rollout storage, optimizer/resume, and qualifying runs remain
open; the M6.2 exit gate is not yet claimed. The fixed-horizon rollout buffer
now snapshots vector-world tensors, rejects inconsistent transitions, computes
terminal-aware GAE only when complete, and flattens time/world axes for the
optimizer without collapsing entity features. PPO updates use a reproducible
seed/update-index permutation and report sample-weighted diagnostics across
every epoch and minibatch. A restricted-load checkpoint test proves that an
update-save-restore-update sequence exactly matches uninterrupted optimizer
state and weights. The live collector now drives persistent batch worlds within
each rollout, selectively resets completed slots, and checkpoints its monotonic
episode-seed cursor. Collection boundaries deliberately truncate and bootstrap
unfinished worlds so exact resume does not depend on unserialized simulator
state. The headless smoke runner now exercises collection, optimization,
manifest writing, and exact resume end to end, while labeling its output as
infrastructure-only evidence. Qualifying training and held-out evaluation runs
remain open. The batch host now also exposes exact-parity `stepScripted` so the
held-out evaluator compares PPO, deterministic masked-random, and the native
scripted blue controller without an HTTP server or renderer. The evaluator
reports the frozen thresholds but does not turn a threshold miss into hidden
checkpoint selection. Training can now explicitly opt into potential-based
health shaping; checkpoints bind the reward mode, manifests retain shaped and
canonical sums separately, and held-out evaluation remains canonical. A
qualifying checkpoint series remains open. The curriculum now freezes all seven
planned gates through map-backed 10v10, with distinct 10,000-seed training
ranges and eight held-out seeds per gate. A live batch test proves every frozen
scenario and roster loads on the authoritative server; no gate is considered
advanced merely because it is defined. PPO now also supports an explicit,
digest-bound behavior-cloning initialization: the BC policy is loaded only at
the first update and its checkpoint/state/dataset provenance survives every
resume. A development series showed why retaining all checkpoints matters: its
BC-initialized update 1 passed 1v1-random at 8/8, while updates 5 and 10 regressed
to draws. A subsequently frozen conservative candidate (`ppo_1v1_bc_v0`) lowers
the learning rate to `3e-5`, uses one PPO epoch and full 200-decision rollouts,
and passed 8/8 at retained updates 1, 5, 10, and 25 in a reproduced development
run. This is tuning evidence, not yet the post-commit qualifying artifact.

M6.2 gate-1 acceptance (2026-09-02): the committed, BC-initialized
`ppo_1v1_bc_v0` qualifying series retains updates 1, 5, 10, and 25. Every
checkpoint won all eight held-out `1v1-random` episodes, while deterministic
masked-random won none; the final checkpoint averaged 60 decisions and all
policies recorded zero rejected actions. The series is bound to source commit
`60459b5`, the frozen series-config digest, curriculum digest, BC checkpoint and
dataset digests, and every child checkpoint/evaluation digest. The semantic
auditor restricted-loads every checkpoint and rejects modified child artifacts.
Only the first curriculum gate is advanced; 1v1 easy scripted and all later
gates remain open. A deterministic headless recorder produced the final
checkpoint's seed-3101 blue win in 60 decisions, and the existing browser/WebGL
replay smoke reached the terminal frame and restarted without UI errors. The
replay is companion evidence; the accepted v0 series manifest remains immutable.
The trainer now distinguishes exact same-gate resume from cross-gate PPO
transfer. Transfer imports model/value weights but resets the optimizer, update
counter, and episode-seed range while binding source checkpoint, state,
curriculum, gate, and update provenance. A gate-2 development transfer from the
accepted gate-1 final checkpoint remained at 0/8 versus easy scripted red
through update 25, while its teacher baseline won 8/8; gate 2 therefore remains
closed and requires a gate-specific training configuration.
The gate-specific `bc_1v1_easy_v0` initializer was regenerated after its source
specification commit and binds 184 audited teacher transitions. It won both
disjoint BC evaluation episodes against easy scripted red, versus 0/2
masked-random, with zero rejected actions. Conservative PPO development from
this initializer passed the frozen eight-seed gate at updates 1, 5, and 10 but
regressed by update 25, so the gate-2 candidate must stop at a predeclared
update 10. `ppo_1v1_easy_bc_v0` now freezes that checkpoint schedule and all
hyperparameters before qualification; no qualifying gate-2 result is claimed
yet.

M6.2 gate-2 acceptance (2026-09-02): the committed
`ppo_1v1_easy_bc_v0` qualifying series retains updates 1, 5, and 10; all three
pass the frozen held-out threshold. The final checkpoint won 5/8 against easy
scripted red versus 0/8 masked-random and 8/8 teacher, averaged 66.75 decisions,
and recorded zero rejected actions. The semantic auditor verifies all nested
checkpoint and evaluation digests. Gates 1 and 2 are advanced; 3v3-random and
all later gates remain closed.

M6.2 gate-3 preparation (2026-09-02): a provenance-bound PPO transfer from the
accepted gate-2 update-10 checkpoint stayed at 0/8 wins at all retained
development updates 1, 5, and 10, matching masked-random; the scripted teacher
won 8/8 in 53 decisions. The committed `teacher_3v3_random_v0` suite therefore
freezes a separate 3v3 corpus with train, validation, BC-evaluation, and PPO
evaluation seeds all disjoint. Its 424-transition audited training export has
dataset digest
`sha256:616494b021b437d6b8b641bae03255ffe525607985169b24e218407a4daf5dff`.
The deterministic `bc_3v3_random_v0` initializer won both BC evaluation
episodes in 105 decisions with all blue units alive, versus 0/2 masked-random
and 2/2 teacher, with zero rejected actions. This is initializer evidence;
gate-3 PPO configuration and qualification remain open.

Gate-3 development then tested exploration explicitly. BC-seeded PPO with the
historical log-std `-1` and learning rate `3e-5` collapsed to 0/8 after update
1; narrowing target/power log-std to `-3` did not by itself prevent the
categorical policy from crossing a fragile argmax boundary. The predeclared
`ppo_3v3_random_bc_v0` stability candidate combines log-std `-3` with learning
rate `1e-8` and passed 8/8 at every retained update 1/5/10 versus 0/8
masked-random. Its deterministic behavior remains 105 decisions, equal to the
BC initializer. This is an honest retention result through the PPO pipeline,
not material reward-driven improvement; qualification remains open.

M6.2 gate-3 acceptance (2026-09-02): the post-commit qualifying
`ppo_3v3_random_bc_v0` series reproduces the frozen development curve. Updates
1, 5, and 10 each won 8/8 held-out episodes in 105 decisions, versus 0/8
masked-random and 8/8 scripted teacher, with zero rejected actions. The semantic
auditor verifies the series and all nested checkpoint/evaluation digests; the
series is bound to source commit `1829e6a` and config digest
`sha256:f793476d5297c6ebc0c542dfd9a9bd662c81de43e05b92a1aac531ba9c57e341`.
The first three gates are advanced and the stated M6.2 held-out comparison is
met. Later 3v3 scripted/terrain and 5v5/10v10 curriculum gates remain open, and
the 3v3 result remains explicitly BC retention rather than improvement over BC.

Gate-4 preparation found that direct gate-3 PPO transfer loses all eight
`3v3-scripted` evaluation episodes. An eight-episode open-loop teacher corpus
audited successfully, but its first BC checkpoint lost both disjoint BC
evaluation episodes; target-precision weighting improved one development
variant to 1/2, while simply doubling the corpus did not close the gap. This is
treated as closed-loop covariate shift rather than grounds for further blind
hyperparameter sweeps. The server now exposes read-only `GET /teacher-action`,
and `snowgym-export-dagger` records those oracle labels on states visited by a
provenance-bound learned checkpoint while learner steps remain state-hash
guarded. Dataset aggregation and a new gate-4 initializer remain open.

The first recovery-only DAgger development checkpoint also lost both held-out
episodes, confirming that learner-state labels must augment rather than replace
expert trajectories. `snowgym-merge-trajectories` now provides deterministic
ordered aggregation with repeated-input integer weighting, compatibility checks,
source-digest provenance, episode remapping, and a full output re-audit. A
mixed expert/recovery gate-4 initializer remains open.

A 2:1 ordered expert/recovery aggregate then succeeded in development. It
contains 1,745 transitions and has portable digest
`sha256:2fc2770ae2385c16adb14cffde01104a5e4165a4f8154d92021d40ae1fa3e7e4`;
the 10,000-step precision-weighted checkpoint won both held-out BC episodes,
with one and two blue survivors respectively. `bc_3v3_scripted_v0` now freezes
that optimization contract before provenance-valid regeneration. Gate-4 PPO
qualification remains open.

The post-commit `bc_3v3_scripted_v0` regeneration binds source commit
`c17751d` and the portable aggregate digest, and reproduces 2/2 held-out BC
wins with zero rejected actions. The baseline, checkpoint, and evaluation are
now committed evidence; `3v3-scripted` still requires its eight-seed PPO series.

The frozen `ppo_3v3_scripted_bc_v0` development series retains 5/8 wins at
updates 1/5/10 versus 0/8 masked-random, with three red-win seeds and mean
119.875 decisions. It uses the same narrow-exploration `1e-8` stability regime
as gate 3 and does not claim improvement over the DAgger initializer.
Post-commit gate-4 qualification remains open.

M6.2 gate-4 acceptance (2026-09-02): the committed
`ppo_3v3_scripted_bc_v0` qualifying series reproduces 5/8 wins at updates 1,
5, and 10, versus 0/8 masked-random and 8/8 teacher, with zero rejected
actions. All three red-win seeds remain visible; no best-checkpoint selection
was performed. The semantic auditor binds source commit `eef57e4`, config
digest `sha256:de8e80f22a1ebe0d96c96583e3f01de586215a14c932dcbb2faf7965eeebc87b`,
and all nested artifacts. `3v3-scripted` is advanced; terrain and larger-roster
gates remain open.

M6.2 gate-5 and gate-6 acceptance (2026-09-02): the committed
`ppo_3v3_terrain_bc_v0` and `ppo_5v5_terrain_bc_v0` qualifying series each pass
all eight held-out episodes at updates 1/5/10, versus 0/8 masked-random, with
zero rejected actions. Gate 5 introduced terrain teacher data; gate 6 added
action-conditioned movement and throw target heads so the shared continuous
head no longer had to fit incompatible target distributions. Both results are
BC-initialized PPO retention gates, not cold-start PPO claims.

Gate-7 initializer acceptance (2026-09-02): the first 10v10 BC initializer
reliably eliminated nine opponents but targeted defeated roster slots because
`enemy_mask` represents slot presence, not life state. Relational selection now
combines that mask with the encoded alive bit. The frozen
`bc_10v10_terrain_relational_v0` code/policy hybrid keeps neural action
selection, exact nearest-living-enemy throw targets, and a final-opponent move
target while retaining the learned movement head as an auxiliary BC objective.
Its provenance-valid checkpoint won both disjoint BC evaluation episodes in
145 decisions with all ten blue units alive and zero rejected actions, versus
0/2 masked-random and 2/2 scripted teacher at 146 decisions. Gate-7 PPO
qualification remains open.

M6.2 gate-7 acceptance (2026-09-02): the committed
`ppo_10v10_terrain_relational_bc_v0` qualifying series passes all eight
held-out episodes at updates 1/5/10 in 145 decisions, versus 0/8 masked-random
and 8/8 scripted teacher, with zero rejected actions. The semantic auditor
binds source commit `448f8ba`, config digest
`sha256:39bacab64fca11007617d0698b782a3baf3023e9215a562f851437876525ff47`,
series digest
`sha256:e5c1b9b540a54c787208a7a4270698846d136132b2207ca9dafc9f3d422e3034`,
and every nested checkpoint/evaluation artifact. All seven centralized PPO
curriculum gates are now advanced. Gate 7 is explicitly BC-initialized PPO
retention; cold-start learning and material reward-driven improvement are not
claimed.

### M7 — plan-conditioned learned executor

Goal: train the fast learned controller to follow the existing slow
`CommandPlan` language without online LLM calls.

- [x] Add a deterministic synthetic plan curriculum that emits only schema-
      valid plans and records grounded assignments and source seeds.
- [x] Add a fixed-size `PlanTensorEncoder` for mission, approach, posture, fire
      policy, preferred range, cohesion, relative objective/group geometry,
      group fractions, support relation, and plan age. JSON remains canonical;
      the tensor is only an RL adapter.
- [x] Train matched executor architectures with and without plan input under
      the same data and optimization budget.
- [ ] Evaluate direct versus flank trajectories, focus versus distributed fire,
      hold/support/withdraw behavior, unseen valid directive combinations, and
      3v3/5v5 training to 10v10 transfer.
- [ ] Keep reflexes, late binding, action validation, lifecycle fallback, and
      target replacement host-owned; a plan never supplies physical actions.

M7 exit: plan-conditioned policies produce reproducibly distinct, intended
behavior under counterfactual plans for the same initial state and outperform
the no-plan ablation on frozen objective-completion metrics.

#### M7a — fighter contract repair

The fighter review in `refs/snowgym_fighter_rl_review_next_milestones.md`
identified four contract blockers before another large training run: hidden
persistent controller state, incomplete physical role summaries, roster-sized
PPO updates, and a terminal objective that does not define mission obedience.
The accepted implementation order is:

- [x] Register `SnowGym/Squad-v3` with actuator-complete unit/projectile and
      decision-horizon fields while preserving v0-v2 tensor contracts.
- [x] Introduce `snowgym.state.v2` over the complete public state and retain a
      v1 verifier for committed replay artifacts.
- [x] Keep the `[3,38]` command tensor stable; add a masked `[3,20]` physical
      role-state tensor and freeze current-position/region anchors at plan
      activation while entity-backed objectives remain late-bound.
- [x] Replace the training default's squad-product PPO clip with masked
      per-unit clipping averaged inside each decision; normalize entropy and
      health shaping by active/initial roster and derive discounting from
      physical-time half-lives.
- [x] Migrate the accepted target-only initializer by zero-extending its unit
      encoder, add shared zero-output plan residuals, and train a separate
      role-aware centralized critic.
- [x] Extend the persistent batch client with the existing simultaneous
      `stepJoint` contract and collect authoritative plan/role tensors after
      every reset and before every policy decision.

M7a exit: v3 state/action transitions and v2 hashes are deterministic, legacy
replays still verify, migrated zero-extension preserves legacy policy outputs,
1/3/5/10-unit PPO statistics are roster-stable, and interrupted plan-aware
collection resumes exactly.

M7a acceptance (2026-09-03): `SnowGym/Squad-v3` exposes complete public
controller state and decision-horizon data through `snowgym.state.v2`; legacy
v1 replays remain independently verifiable. Stable assignments, activation
anchors, `[3,20]` physical role summaries, and mission progress feed a shared
fighter residual and an independent centralized critic. PPO stores and clips
masked per-unit likelihood ratios, retaining squad products only as a
diagnostic ablation. The persistent batch client supports `stepJoint`,
selective plan activation, and fresh plan/role reads. Its immutable plan
schedule binds content and cursor into exact-resume checkpoints. The accepted
target-only policy migrates through a split first-layer seam, so zero-valued v3
extensions reproduce action logits, learned move/throw targets, and power
exactly before the zero-output plan residual is trained.

#### M7b — fixed-plan option PPO

- [x] Add fixed-plan option episodes with separate mission, combat, potential,
      and canonical reward components. Commander scheduling and plan
      replacement remain disabled.
- [x] Validate engage, advance, hold, withdraw, flank, focus/distributed fire,
      and support definitions with the production plan-aware teacher before
      freezing learner thresholds.
- [ ] Train in that order from the migrated target-only initializer using
      staged unfreezing plus decaying BC and initializer-KL anchors.
- [ ] Qualify one checkpoint on 100 untouched paired seeds per mission against
      zero-plan and shuffled-plan controls. Every mission must pass separately;
      aggregate success cannot hide a failed directive.

M7b exit: every mission reaches at least 75% success, exceeds shuffled-plan
success by at least 20 percentage points with a positive paired-bootstrap 95%
lower bound, improves paired mission progress on at least 70% of seeds, retains
physical win rate within 10 percentage points, and rejects fewer than 0.1% of
physical actions under a nontrivial PPO update.

M7b option-contract acceptance (2026-09-03): the v3 fixed-plan batch wrapper
tracks immutable group assignments and anchors, terminates on the eight frozen
mission definitions, and emits mission, combat, shaping, canonical, and
executor reward fields separately. The committed protocol predeclares disjoint
training, 40-seed-per-mission development, 100-seed-per-mission qualification,
plan-generation, and sealed-map partitions. Live subprocess tests prove every
option achievable with the production plan-aware teacher and zero rejected
actions. Support uses a pre-training 4+1, 30-by-20 proof scenario so the
teacher's bounded support offset can enter the unchanged 8–18% band. This gate
does not yet claim a trained or qualified M7b checkpoint.

M7b training-bridge acceptance (2026-09-03): fixed-option rollouts now reset
completed worlds selectively, reactivate the next content-bound plan, store
production-teacher labels for the decaying BC anchor, and preserve separate
reward sums. The option runner migrates the accepted target-only checkpoint,
enforces staged parameter groups, applies BC and initializer-KL decay, supports
exact resume and stage-1-to-stage-2 transfer, and emits immutable checkpointed
manifests. A real two-world Engage update completed with finite nonzero
gradients and zero rejected actions. The deterministic same-state Hold,
Withdraw, and Advance fork has distinct reproducible v2 hash trajectories. A
digest-validating all-mission qualifier implements the predeclared paired
bootstrap and per-mission gates. These are infrastructure results; the ordered
development training and untouched qualification run remain open.

M7b paired-evaluation bridge (2026-09-03): the headless evaluator holds the
intended option tracker fixed while comparing correct grounded tensors,
state-matched previews of a deterministic shuffled valid plan, and the
migrated initializer. It continues each episode through the battle boundary so
physical win retention is measured independently of early option completion.
Every record includes the paired seed, success, progress, win outcome, rejected
actions, and total actions under an audited artifact digest. Staged PPO transfer
can now advance to the next mission while changing to its disjoint training
seed range. No development or qualification threshold is marked passed by this
infrastructure change.

M7b rollout-horizon correction (2026-09-03): research-mode updates now span
the selected option's complete frozen horizon by default. Short rollouts reset
the world before sparse mission outcomes and are therefore accepted only when
explicitly labeled `infrastructure-smoke`. Per-update manifests now retain
learner and teacher action histograms, action/target/power BC components, and
the hybrid policy exploration scales. A 50-update Engage development probe
remained at zero success: its action class converged toward the teacher, while
its move target remained short of the teacher objective and no combat began.
This negative result is diagnostic; it does not advance the ordered-training
or qualification checkboxes.

M7b target-gradient repair (2026-09-03): the negative Engage probe exposed
that the executed move target included the new plan residual while the BC
supervision view excluded it. The supervised move tensor now includes the same
residual, retaining the special last-enemy execution override boundary. A
dedicated gradient test requires nonzero move-residual gradients and zero
gradients in unused action/throw/power rows. In a five-update full-horizon
probe, target BC MSE fell from 0.199 to 0.138; longer development and the
per-mission gates remain open.

M7b Engage development audit (2026-09-03): development evaluation may now
select one or more trained missions, while qualification remains hard-locked
to all eight missions and 100 untouched seeds each. The repaired stage-2
Engage checkpoint was evaluated on all 40 predeclared development seeds.
Correct, shuffled, and initializer conditions each had zero mission success,
zero mean progress, and zero physical wins; each condition executed 60,000
accepted actions with no rejection. The checkpoint is retained only as a
negative diagnostic and is not eligible for M7b promotion.

M7b recovery decision (2026-09-04): the implementation review in
`refs/snowgym_m7b_engage_review_and_recovery_instructions.md` found that the
staged actor cannot train the zero-initialized v3 extension columns, option
timeouts bootstrap reset-state values, temporal option counters can advance
twice, terminal shaping retains a nonzero next potential, and the fresh critic
shares a global gradient clip with the actor. The failed Stage-1/Stage-2
checkpoints and 40-seed evaluation are preserved under
`snowgym/training/runs/m7b_engage_failed_v0/` as negative evidence.

The recovery milestones are now:

- [x] **M7b-R0 — correctness and attribution:** repair the actor-side v3 path,
      option terminal/tracker/PBRS semantics, actor/critic clipping, and
      discount metadata; then run deterministic action-channel interventions,
      teacher/stochastic/deterministic state-distribution exports, and
      read-only component-gradient diagnostics.
- [ ] **M7b-R1 — Engage bootstrap:** choose exactly one learning intervention
      from R0 evidence and pass the frozen 40-seed bootstrap gate. Teacher
      transitions, if selected, supervise only the BC auxiliary path and never
      enter PPO ratios or advantages.
- [ ] **M7b-R2 — Engage promotion:** retain the unchanged full-distance 5v5
      benchmark and pass the stricter development promotion gate with material
      improvement over both shuffled plans and the initializer.
- [ ] **M7b-R3 — ordered single-role options:** advance, hold, withdraw, flank,
      focus, distributed, then support; each mission must pass separately.
- [ ] **M7b-R4 — universal fixed-plan executor:** one checkpoint passes all
      eight development gates without mission-specific checkpoint switching,
      followed by the untouched 100-seed-per-mission qualification.

No new Engage training, later option, M7c composition, or commander experiment
starts until R0 identifies the failed action channel and R1 passes its gate.

During R0, the paired-seed audit also found that the semantic `RandomAgent`
returned Red actions that the simulator never applied. The SnowGym host now
applies that controller once per team decision; scripted Red retains its
per-physics-tick execution. Distinct random-controller seeds therefore produce
distinct state-hash trajectories. This behavior change is identified as
`snowgym.sim.v2`; legacy v1 replay artifacts remain loadable and hash-verifiable.
All R0 artifacts are regenerated after this repair.

Research correction (2026-09-04): all earlier acceptance claims against a
`redController: random` scenario were measured with an inert Red opponent.
Those artifacts remain immutable evidence of the v1 implementation, but they
do not qualify a fighter under v2. Gates 1, 3, 5, 6, and 7 are reopened for
v2 requalification. Scripted-Red gates 2 and 4 remain valid because their
controller path was unchanged. Historical random-opponent replay tests now
assert the observed v2 losses rather than preserving obsolete blue-win claims.

M7b-R0 acceptance (2026-09-04): the no-training 40-seed matrix establishes
that the failed learner contacts on 82.5% of episodes but never hits; teacher
movement alone reaches contact on 100% without hits, teacher action alone hits
on 45%, teacher action plus movement hits on 87.5%, and the full teacher passes
Engage on 100%. The deterministic learner and successful teacher lie outside
the stochastic learner support on 69.84% and 76.46% of states respectively
under the frozen D2 nearest-neighbor threshold. Read-only gradient diagnostics
show a 23.03 critic norm versus a 0.77 actor norm, implying an old shared-clip
scale of 0.0217, plus strong move-BC/initializer-anchor opposition. The frozen
decision selects exactly one R1 intervention: a successful-teacher trajectory
reservoir for BC-only sampling while PPO remains strictly on-policy. Evidence
under `training/runs/m7b_engage_failed_v0/` binds implementation revisions
`733c985` and `2531aa4`, simulation v2, state-hash v2, the failed checkpoint
digest, all 40 development seeds, and nested artifact hashes. No qualification
seed was used.

M7b-R1 first candidate (2026-09-04): the provenance-bound reservoir contains
5,367 successful production-teacher states from 40 training seeds and is mixed
50/50 with learner-state labels only inside the auxiliary BC loss. The frozen
50-update Stage-1 checkpoint reached contact on 28/40 development seeds and hit
on 10/40, but completed Engage on 0/40. After the predeclared Stage-2 updates,
the final checkpoint reached neither contact nor a hit on any of the 40 seeds;
correct, shuffled, and initializer conditions all had 0% success, with zero
rejected actions. The bootstrap gate failed, so R1 remains open and R2/later
missions remain blocked. The negative run is preserved under
`training/runs/m7b_engage_teacher_reservoir_r1_v0/`. A subsequent R1 attempt
must remain within the selected teacher-reservoir intervention and predeclare
its sampling/anchor or unfreezing change before training.

M7b-R1b predeclaration (2026-09-04): resume the exact update-50 Stage-1 R1
checkpoint and change only the unfreezing schedule. Keep the reservoir, 50/50
BC mixture, learner-only PPO transitions, optimizer and RNG state, training and
development seeds, loss weights, exploration, reward, discount, and 200-update
anchor schedule unchanged. Do not open inherited action, target, or power heads;
retain and evaluate updates 50, 75, and 100. Intermediate checkpoints are
diagnostic only. The frozen bootstrap gate applies exclusively to update 100,
so the trajectory cannot be used for post-hoc checkpoint selection.

M7b-R1b result (2026-09-04): keeping Stage 1 active preserved partial behavior
at update 75, with 27/40 contacts and 8/40 hits, but no mission success. By the
predeclared final update 100, contact, hit, progress, mission success, and
physical-win rates were all zero. Rejected-action rate remained zero throughout.
The final bootstrap gate failed. Opening inherited heads at update 50 is
therefore excluded as the sole explanation for R1's collapse; continued
optimization under the unchanged reservoir/anchor schedule can also erase the
partial Stage-1 behavior. The intermediate checkpoint is negative diagnostic
evidence and is not eligible for promotion. R1 remains open and R2 remains
blocked. The digest-bound run is archived under
`training/runs/m7b_engage_teacher_reservoir_r1b_stage1_hold_v0/`.

M7b-R1c predeclaration (2026-09-04): retrain Stage 1 from update 0 and change
only the BC anchor after update 50. Preserve the original linear decay until
the anchor reaches `0.05`, then hold that floor through update 100; keep the
initializer-KL decay unchanged. Require the update-50 model, optimizer, and RNG
state digest to exactly match the original R1 Stage-1 artifact before accepting
later results. The reservoir, 50/50 BC mixture, learner-only PPO data, seeds,
losses, exploration, reward, discount, and benchmark remain fixed. Retain
updates 50, 75, and 100 for diagnosis, with the bootstrap gate applied only to
update 100.

M7b-R1c result (2026-09-04): update 50 reproduced the original model,
optimizer, and RNG state digest exactly. The `0.05` BC floor prevented the
complete locomotion collapse: update 100 retained contact on 26/40 seeds, but
hit only 1/40 and completed 0/40 missions. Update 75 reached 12/40 hits but also
completed no missions. All conditions retained zero rejected actions and zero
physical wins. The final bootstrap gate failed, so R1 remains open. This
isolates anchor decay as a contributor to contact collapse while showing that
anchor magnitude alone does not preserve the successful teacher's coordinated
move/throw state distribution. The digest-bound negative result is archived at
`training/runs/m7b_engage_teacher_reservoir_r1c_bc_floor_v0/`.

M7b-R1d predeclaration (2026-09-04): retain R1c's `0.05` BC floor and change
only the auxiliary BC loss weighting from 50% to 90% successful-teacher
reservoir loss. Both sources retain equal sample counts per minibatch; earlier
wording called this a sample mixture. The R1c learner-state teacher labels become dominated by
movement as the closed loop loses contact, while the reservoir retains the
successful move/throw sequence. PPO ratios, advantages, returns, and value
targets remain exclusively learner-on-policy. Preserve all other optimizer,
seed, reward, exploration, loss, checkpoint, and evaluation settings; retain
updates 50, 75, and 100 and gate only update 100.

M7b-R1d result (2026-09-04): the update-100 checkpoint passed the contact and
rejected-action gates but not mission-success or control-improvement gates.
Across updates 50, 75, and 100, contact increased from 45% to 67.5% to 85%, hit
rate increased from 12.5% to 22.5% to 52.5%, and mean mission progress reached
8.6%. No episode completed Engage or won the full battle. The monotonic recovery
supports continuing the exact R1d trajectory without changing its learning
mechanism. The audited result is archived under
`training/runs/m7b_engage_teacher_reservoir_r1d_reservoir90_v0/`; R1 remains
open until a predeclared final checkpoint passes.

M7b-R1e predeclaration (2026-09-04): change only training duration by resuming
the exact R1d update-100 checkpoint, including model, optimizer, RNG, seed
schedule, and option schedule, and continuing to update 200. Retain the same
Stage-1 freeze, 90% successful-teacher BC mixture, `0.05` BC floor, learner-only
PPO data, losses, exploration, reward, discount, and training seeds. Evaluate
updates 100, 150, and 200 and apply the bootstrap gate only to update 200.

M7b-R1e result (2026-09-04): exact continuation improved update-200 contact to
38/40, hits to 28/40, and mean mission progress to 14%, with zero rejected
actions. It still completed 0/40 Engage missions and won 0/40 battles. The final
bootstrap gate failed, so R1 remains open and R2 remains blocked. Training
duration alone is insufficient under the current Stage-1/reservoir objective.
Before another full PPO run, require a cheaper reservoir-only supervised causal
probe that tests whether the trainable Stage-1 modules can reproduce the full
teacher trajectory in deterministic closed loop and identifies the remaining
action/target/power error by mission phase. Evidence is archived under
`training/runs/m7b_engage_teacher_reservoir_r1e_continue200_v0/`.

M7b-R1f predeclaration (2026-09-04): first repair expanded-initializer identity,
optimizer-derived learning-rate reporting, active plan-PPO loss/KL diagnostics,
and optional KL stopping. Retain historical artifact bytes. Then run one
bounded supervised-only probe from the final R1e checkpoint: Stage-1 actor
modules, fresh Adam at `3e-4`, 20 epochs, minibatches of 256, seed 91001, and
the existing 5,367-state/40-training-seed successful-teacher reservoir. Keep
the R1e BC component weights (action 1, target 5, power 0.5, throw class 5),
with BC as the entire objective and no PPO, entropy, initializer-KL, or critic
training. Preserve checkpoints/teacher-state diagnostics at epochs 0, 10,
and 20; assess only epoch 20 as the final probe. Report action/target/ray/power
errors by approach/contact/fire phase, deterministic closed-loop reproduction
on all 40 reservoir training seeds at epochs 0 and 20, and final paired
correct/HOLD/initializer controls on the existing 40 development seeds. No
qualification seeds, provider calls, threshold changes, or automatic extra
epochs. This capacity/optimization probe does not itself qualify R1 or isolate
PPO's causal contribution. Decide the next intervention from the completed
probe and its failure channels before another PPO run.

M7b-R1f result (2026-09-04): the frozen 20-epoch supervised-only probe failed
to reproduce Engage on either the 40 reservoir training seeds or the 40
development seeds (0/40 on each). Development contact/hits/progress changed
from R1e's 95%/70%/14% to 87.5%/52.5%/6.6%, with zero wins or rejected
actions. The HOLD counterfactual now contacted on 80% and hit on 25% of seeds,
weakening plan selectivity. On teacher states, throw-coordinate RMSE improved
from 18.71 to 9.74 units while mean throw-ray error worsened from 31.81 to
45.65 degrees; power RMSE also worsened. New actor parameters changed by L2
3.88022; inherited actor and critic parameters remained exact. R1 remains
open and no further PPO continuation ran. Next: a no-training throw-direction/
power channel intervention, followed by a separately predeclared physical
throw-loss probe if supported. The result does not establish an architecture
capacity limit or PPO's causal effect. Evidence:
`training/runs/m7b_engage_r1f_supervised_probe_v0/`; interpretation and next
decision: `training/reviews/m7b_r1f_results.md`.

M7b-R1g predeclaration (2026-09-04): freeze the R1f epoch-20 checkpoint and
run a no-training matrix on development seeds 200000–200039: learner,
teacher-style direction, teacher-style power, both, and full production teacher.
Direction/power replacements apply only to learner-selected throws; all other
action fields remain unchanged at that decision. Recompute recommendations
from each arm's current state. For this fixed single-group Engage scenario,
recommend the nearest living enemy (ID tie-break), lead by 0.18 seconds, and
use the production medium-range power rule. Verify recommendations against
actual production-teacher throw labels; never substitute a teacher move target
as a throw label. The recommendation remains defined when the teacher chooses
to move/dodge, and coverage/disagreement is recorded. Direction replacement
includes implicit target selection, so it cannot isolate fine aim from enemy
choice. Preserve all 40 paired records, state-hash trajectories, source digests,
and paired-bootstrap intervals. No fitting, new seeds, threshold changes,
qualification, or automatic architecture changes. Use the completed matrix to
decide whether structured target selection, direction-aware loss, power control,
or action/movement control should be the next design intervention.

M7b-R1g result (2026-09-04): all 200 episodes completed with no training and
zero rejected actions. On 40 paired development seeds, learner/direction/power/
both/full-teacher Engage successes were 0/7/0/10/40. Direction replacement
raised hit rate from 52.5% to 95% and mean progress from 6.6% to 46.7%; adding
power reached 52.7% progress. Power-only gave no supported improvement, and
the incremental benefit of power over direction is uncertain. Direction includes
enemy selection, so this does not isolate selection from fine aim. Both channel
repairs still leave a large gap to the teacher. Next: predeclare a conditional
action-choice versus movement diagnostic, then a separately versioned
target-relative shot ablation with gradient-reachability, likelihood, and plan
selectivity checks before further PPO. No production actor redesign or promotion
occurred. Artifacts: `training/runs/m7b_engage_r1g_throw_channels_v0/`;
[results and design feedback](training/src/snowgym_training/executor/DESIGN_FEEDBACK_R1G.md).

M7b-R1h predeclaration (2026-09-04): freeze R1f epoch-20 and development
seeds 200000–200039 under simulation/state v2. Run a 2×2 no-training matrix:
learned/teacher action choice crossed with learned/teacher movement destination.
All four arms use the R1g direction-plus-power recommendation on executed throws;
add a full production-teacher control. Action choice includes move/throw/hold/noop,
not firing alone. Re-select the learned conditional target head after any type
replacement. A scenario-scoped movement recommendation supplies the production
Engage range/formation/cohesion destination, or immediate dodge destination,
even when the teacher would throw; validate it against actual teacher move labels.
Keep hold/noop persistence semantics unchanged. Before interpreting the matrix,
require shot-only parity with R1g and combined-intervention parity with the full
teacher. Record decision confusion, readiness, range, dodge-threat context,
coverage, replacement counts, actions digests, and state-hash trajectories.
Use paired development intervals for simple effects and interaction; no new
training, provider calls, qualification seeds, architecture change, or promotion.
Run targeted and full milestone tests before implementation and artifact commits.

M7b-R1h result (2026-09-04): all 200 episodes completed without training or
rejected actions. With corrected shots, learned choice/movement scored 10/40;
teacher movement with learned choice scored 40/40; teacher choice with learned
movement scored 11/40; both and the full teacher scored 40/40. All 40 shot-only
trajectories reproduce R1g exactly, and combined restoration reproduces the
full teacher. Movement repair adds 75 success points (paired-bootstrap 95%:
60–87.5), while choice repair with learned movement adds 2.5 (−12.5–17.5).
The movement intervention includes range keeping, formation/cohesion, and dodge
destinations. It identifies a conditional channel bottleneck, not a qualified
learned controller. Next: predeclare a target-relative movement/shot representation
and gradient probe, retaining the inherited classifier initially; test learning
against matched priors and HOLD selectivity before additional PPO. R1 remains
open. Evidence and design decision: [R1h review](training/reviews/m7b_r1h_results.md).

M7b-R1i predeclaration (2026-09-04): bounded supervised representation probe
from frozen R1f epoch-20, using the audited 5,367-state successful teacher
reservoir (training seeds 100000–100039). Compare equal-size absolute-feature
and fighter-relative-feature residual modules; preserve the existing absolute
tanh target decoder to isolate input geometry. Freeze the complete inherited
actor/critic, including action choice. Both zero-output residuals must reproduce
the source exactly. Use separate move and throw/power heads, trainable masked
entity pools, own/support role state and symbolic directives; no teacher or
nearest-enemy execution override. Optimize identical move MSE + throw cosine
loss + 0.1 throw-endpoint MSE + 0.5 power MSE in both arms. Treat undefined rays
explicitly with a finite epsilon-normalized direction loss.
Before full fitting, require each arm to reduce loss by at least 50% on 32
fixed training states (first 16 with any throw, first 16 with move and no throw)
within 200 Adam steps at 1e-3, with finite gradients and exact frozen weights.
Run this gate on disposable copies. Then fit both fresh arms for 20 epochs,
batch size 256, Adam 3e-4, norm clip 0.5, paired seed 92001 and identical
minibatch order. Retain epochs 0/20; do not select intermediate checkpoints.
Record full-reservoir agreement and gradient reachability; evaluate final arms
on 40 development seeds 200000–200039 with correct and same-state HOLD plan
inputs, plus the frozen source baseline. Save custom deterministic-probe
checkpoints and reload parity, provenance, and parameter-change evidence.
No sampled action/PPO likelihood contract change, PPO updates, provider calls,
qualification seeds, or promotion. If the fitting gate fails, archive it and
stop before the 20-epoch run. Require full milestone tests before each commit.

M7b-R1i result (2026-09-04): both disposable fitting gates passed (97.6% and
98.4% loss reduction), followed by 20 supervised epochs per arm and 240 paired
development episodes. Source/absolute/relative Engage success was 0/0/1 out
of 40, with progress 6.6%/30.8%/26.3%; all had zero battle wins and rejected
actions. Both new arms markedly improved HOLD separation. Throw-ray error fell
from 45.65 degrees to 19.73/18.81 while move RMSE fell from 6.83 to 4.48/4.45.
Inherited actor/critic tensors and classifier outputs remain exact, both new
33,669-parameter modules changed, and checkpoint reloads are exact. Relative
features show no clear advantage over the matched absolute control; neither
checkpoint is promoted. Next: a predeclared relative-displacement/shot-direction
decoder probe with matched controls, phase metrics, and HOLD checks; define and
test any changed stochastic likelihood before PPO. Evidence:
[R1i review](training/reviews/m7b_r1i_results.md). R1 remains open.

M7b-R1j predeclaration (2026-09-04): matched 2×2 deterministic decoder probe
from frozen R1f with R1i absolute input features and identical 33,669-parameter
modules. Arms: absolute, displacement, direction, both. Movement replacement
adds a per-axis correction of 10*tanh(residual) world units to the inherited
target, then clips to arena bounds. Shot replacement corrects the inherited
unit ray, renormalizes it, retains inherited ray length (one-unit fallback for
degenerate rays), and clips along that ray to stay inside the arena. Zero
residuals must exactly preserve the inherited targets, logits, and power.
These are geometric correction priors around the source, not teacher/nearest-
enemy tactical references. Power remains the same learned sigmoid residual.
Keep R1i's loss, 32-state disposable gate (50% reduction in 200 steps at 1e-3),
20-epoch fitting budget, batch 256, Adam 3e-4, norm clip 0.5, seed 92001,
5367-state audited training reservoir, and final-epoch-only selection unchanged.
All four gates must pass before fitting fresh arms. Retain epochs 0/20 and all
phase-conditioned agreement metrics; evaluate source and final arms on the same
40 paired development seeds with correct and HOLD-preview inputs. Require
absolute-control parity with R1i before interpretation. Predeclare success and
progress simple effects/interaction with 10,000 paired bootstrap draws, seed
770001. Check boundary/zero-ray gradients, conditional heads, unchanged source,
custom checkpoint reloads, and all milestone gates before committing. No PPO,
provider calls, qualification seeds, production policy change, or promotion.

M7b-R1j result (2026-09-04): all four gates passed, then four 20-epoch fits
and 400 paired development episodes completed. Absolute/displacement/direction/
both scored 0/40 Engage successes each, with progress 30.8%/25.3%/25.5%/22.0%.
All had zero battle wins and rejected actions. Displacement weakened HOLD
separation; direction reduced endpoint error but worsened ray error to 26.5
degrees versus 19.7 for the control. Paired progress intervals support no clear
positive decoder effect. The absolute final model state and evaluation records
exactly reproduce R1i. Retain that model only as a development reference; no
decoder is promoted. Next: audit per-head errors, conditional-label coverage,
and gradient magnitudes on learner-visited training states before predeclaring
a fixed-architecture corrective-data fit with an old-data control. Do not assume
data coverage is the proven cause or restart PPO yet. Evidence:
[R1j review](training/reviews/m7b_r1j_results.md). R1 remains open.

M7b recovery through R1n (approved 2026-09-04): preserve the supplied review
at `refs/snowgym_r1j_review_and_next_rl_experiments.md`. The fully learned
fighter remains the qualification target; runtime teacher geometry is allowed
only in explicitly labeled diagnostic experiments.

- [x] **R1k — conditional opportunity audit:** freeze R1i absolute epoch 20;
      collect all living-unit opportunities on training seeds 100000–100039,
      reconstruct and verify the original teacher reservoir, audit independent
      head labels, gradients, and one-decision/one-fighter substitutions. Replay
      action prefixes from reset and verify physical, plan, and tracker state.
      Use at most 64 opportunities per channel, 30-decision branches, and a
      disposable 200-step hard-state fit (32 training / 8 validation episodes).
      R1l requires useful physical substitutions, at least 50% hard-fit loss
      reduction, finite reachable gradients, and held-out improvement.
- [x] **R1l — matched corrective-data factorial:** original versus augmented
      state support crossed with teacher-selected versus independent conditional
      labels; same R1i weights, 420 Adam steps, batch 256, learning rate 3e-4,
      clip 0.5, unchanged loss, frozen inherited actor. Episode-balanced sampling;
      augmented arms use equal source counts. Run A–D with RNG seed 93001, then
      replicate predeclared D and A with 93002/93003 only if D improves autonomous
      success without rejection regression. Never select the best arm post hoc.
- [x] **R1m — assisted movement PPO (experiment complete; replication gate failed):** first repair frozen option target scoring
      and expose option budget/state. Reproduce exact R1h corrected-shot baseline;
      freeze action choice and shot geometry, train movement and a separate critic.
      Log exact Normal latent movement samples (fixed std 0.02); 8 worlds × 200
      decisions × 100 updates, 4 epochs, minibatch 400, LR 3e-4, clip ratio 0.2,
      separate gradient clips 0.5, movement KL stop 0.01, no BC or entropy bonus.
      Require +20 pp assisted success, positive paired interval, parameter change,
      and rejection rate <0.001; replicate successful runs with two further RNGs.
      Assisted evidence cannot qualify an autonomous executor.
- [ ] **R1n — autonomous Engage:** separately predeclare aiming/composition after
      movement evidence, remove all runtime teacher overrides, and pass unchanged
      R1 gates plus fresh replication before later M7b/M7c work.

Lineage correction: 40 episodes describes the successful-teacher reservoir and
probe evaluation subset, not all prior PPO exposure. R1e's final continuation
records 401 distinct episode seeds through 101600. Preserve historical
development 200000–200039; reserve replication development 210000–210039.
Proposed fresh training-pool teacher/learner holdouts 108000–108039 and
108100–108139 require a complete ancestry/seed audit before use; collisions or
unresolved ancestry fail preflight. No qualification seeds, provider calls,
new commander vocabulary, or broad PPO/decoder sweep is authorized here.
Archive failed gates and stop the affected branch. Run targeted tests and all
four implementation gates (TypeScript, build, Python client, Python training)
before each implementation commit; preserve unrelated working-tree changes.

R1k result (2026-09-04): exact reconstruction verified all 5,367 reservoir
states. The frozen reference produced 8,000 learner states / 38,129 living-unit
opportunities and 0/40 Engage successes. Teacher-action masking excludes 57.24%
of invoked throws and 4.25% of moves. Across 64 hard opportunities per channel,
one-decision aim and movement substitutions had positive paired net-damage
intervals; power did not. Disposable hard fitting reduced loss 97.88%, improved
episode-held-out loss, and passed encoder/Jacobian checks without changing the
reference. R1k passes and permits R1l; autonomous R1 remains open. See the
[R1k report](training/reviews/m7b_r1k_results.md). Fresh-holdout ancestry was
pending at R1k completion and passed in the subsequent R1l preflight.

R1l result (2026-09-05): ancestry passed after recovering the original BC
training dataset with its exact archived digest. All four final-step arms
completed 420 updates and both 40-seed development sets: every arm had 0/40
Engage successes, with zero rejected actions. D did not improve success over A;
the predeclared replication gate failed and no extra optimizer seeds ran.
On fresh development D improved progress over A by 0.048 (paired interval
[0.001, 0.097]), but matched the immediate R1i reference's mean progress.
Conditional fitting has not recovered sustained autonomous combat. Stop this
supervised branch and retain the evidence for diagnosis; do not increase its
data or fitting budget. The separately scoped R1m movement mechanism remains
teacher-assisted and ineligible for qualification. See the
[R1l review](training/reviews/m7b_r1l_results.md).

R1m contract milestone (2026-09-05): additive authoritative activation-target
membership, versioned three-field option state, frozen-target scoring, separate
option-aware critic, movement-only latent density, exact prefix restoration,
selective reset, and immutable partial-collection checkpoints are implemented.
Legacy option scoring is retained explicitly for historical R1h reproduction.
All four test suites and live Gym v0–v3 compatibility passed. This is a contract
gate only; the subsequent 100-update experiment is reported below.
See [scoped movement](training/src/snowgym_training/executor/SCOPED_MOVEMENT.md).

R1m result (2026-09-05): the exact historical R1h source reproduced all 40
trajectories and 10 successes. Repaired option scoring produces a matched
initialization of 13/40 historical and 16/40 fresh-development successes; common
physical prefixes are identical. The exact 20%-health boundary explains the
three newly successful historical episodes. Three predeclared training RNGs
each completed 100 updates with frozen shot assistance. Historical/fresh final
successes were 29/24, 18/18, and 16/18 out of 40. Only the first historical gate
passed; no run passed the fresh-development gate. All had zero rejected actions
and measurable actor change. Stop at the declared budget: the movement effect
is not consistently replicated, and every checkpoint remains ineligible for
autonomous qualification. See the [R1m review](training/reviews/m7b_r1m_results.md).
R1n, aiming/composition budgets, and later missions require a new evidence-based
predeclaration; no extra optimizer seeds, aiming fit, or runtime provider calls
were started.

#### M7c — full-fight fixed-plan composition

- [ ] Hold one-, two-, and three-role plans throughout complete 3v3, 5v5, and
      10v10 battles with casualties, terrain, and target replacement.
- [ ] Compare correct, zero, shuffled, and random-valid plans using disjoint
      paired seeds and per-role mission metrics.
- [ ] Freeze the first checkpoint that passes every predeclared mission and
      physical-competence gate; do not select a best checkpoint post hoc.

M7c exit: one frozen executor retains distinct role behavior and useful plan
effects across roster sizes. Only then may M8 local CTDE and M9 online commander
comparisons advance.

M7 synthetic curriculum foundation (2026-09-02):
`training/plan/SyntheticPlanCurriculum.ts` deterministically samples all five
mission types and the bounded approach/fire vocabularies from explicit source
seeds. Every sample is canonicalized by the production `CommandPlan` parser and
grounded by the production `PlanGrounder`; the resulting record retains the
symbolic plan, stable unit assignments, seed, plan ID, arena/roster provenance,
and optional source-state hash. The pure core performs no file or provider I/O.
Tests prove repeatability, schema validity, complete non-overlapping assignment,
directive coverage, and rejection of unsafe seed ranges or undersized rosters.
The tensor export/data-join path remains next.

M7 plan-tensor foundation (2026-09-02): `training/plan/PlanTensorEncoder.ts`
maps a production `PlanSnapshot` into three stable role slots, each with 38
bounded features and a separate presence mask. It includes role, mission,
approach, posture, fire, range, cohesion and objective-kind one-hots; tactical-
frame-relative objective/group geometry; requested and live assigned fractions;
support relation; and plan age. Raw unit IDs remain host-owned and are not
learnable features. Tests cover exact shape, slot stability, fractions, support,
age normalization, bounds, and counterfactual engage-versus-hold separation on
the same observation. JSON export/data joining and model ablations remain open.

M7 plan-tensor export foundation (2026-09-02):
`training/plan/PlanTensorDataset.ts` resets the authoritative headless
`SnowEnvironment`, binds simulator/hash provenance and the public source-state
hash, aligns every validated curriculum sample with its `[3,38]` tensor, and
computes a canonical SHA-256 dataset digest. Its auditor rejects plan, seed,
shape, value, or digest corruption. The `export-plan-tensors.ts` CLI supports
configurable maps, rosters, environment/plan seeds, counts, safe overwrite, and
machine-readable summaries. Two independent Winter Front 10v10 exports were
byte-identical. The Python data join and paired model ablation remain next.

M7 Python plan-data bridge (2026-09-02): `training/plan_data.py` verifies the
TypeScript semantic digest, converts aligned samples to immutable NumPy
`float32 [samples,3,38]` tensors plus `int8` masks and `int64` source seeds, and
returns detached copies when plan indices are joined to trajectory transitions.
Unit tests cover dtypes, shapes, repeated-plan alignment, immutability, index
validation, bounds and corruption. A live cross-language load reproduced the
exported 10v10 digest exactly. Adding plan inputs to the model and running the
paired with/without-plan ablation remain next.

M7 model adapter foundation (2026-09-02): the shared `EntityPolicy` now has an
opt-in `plan_conditioned` architecture flag. Enabled models encode the masked
`[3,38]` plan rows plus role-presence mask into one global embedding appended to
the unchanged physical entity context; disabled models retain the exact legacy
parameter shape and metadata. `TorchPolicy` requires and shape-checks the plan
tensors only for conditioned checkpoints. Tests prove missing/malformed inputs
fail closed, masked rows are removed from the adapter input, counterfactual
plans change hidden state for the same physical observation, and gradients are
finite. Plan-caused trajectory collection and the paired training ablation
remain open, so the M7 training checklist item is not yet advanced.

M7 plan-caused rollout foundation (2026-09-02):
`training/plan/PlanRolloutDataset.ts` executes every synthetic plan through the
production `PlanGrounder`, `PlanStore`, `PlanAwareTeamController`, and
`ReactiveUnitPolicy`, restarting the authoritative headless environment from
the same seed and public state hash for each counterfactual plan. Every decision
retains the detached observation, semantic action, dynamic plan tensor, reward,
pre/post state hashes, and physical action acceptance. Its semantic auditor
checks curriculum/episode alignment, observation hashes, tensor bounds, action
results, outcomes, and a canonical dataset digest. Tests prove byte-level
repeatability, same-state restarts, plan/action diversity, zero rejected actions,
and rejection of state, tensor, result, or digest corruption. A portable CLI,
Python trajectory conversion, and the paired training ablation remain next.

M7 portable rollout bridge (2026-09-02): `export-plan-rollouts.ts` exposes the
plan-caused collector for configurable map/open arenas, rosters, environment and
plan seeds, decision horizon/frequency, and red difficulty, with guarded
overwrite and machine-readable outcome summaries. Two independent exports were
byte-identical. `snowgym-convert-plan-rollouts` independently verifies the
ECMAScript canonical digest, public observation hashes, continuity, tensor
bounds, and action acceptance before using the shared Gym encoders to emit
audited `snowgym.trajectory.v0` shards. Each transition carries aligned
`plan_groups [3,38]`, `plan_group_mask [3]`, and the plan source seed. The
training loader exposes these arrays only for provenance-marked datasets, and
conditioned training fails closed when they are absent; the no-plan ablation can
consume the exact same shards and ignore the extras. The canonicalizer now also
matches `JSON.stringify` object semantics for omitted `undefined` fields, with
a write/read digest regression test. Paired training configurations and frozen
behavior metrics remain next.

M7 matched-training runner foundation (2026-09-02):
`snowgym-run-plan-ablation` accepts one audited plan trajectory dataset and one
shared architecture/optimization contract, then deterministically trains a
no-plan and plan-conditioned policy whose generated configs differ only by the
plan adapter flag. The result manifest binds the source config and dataset,
both checkpoint/state digests, architectures, and training losses. Its auditor
reloads each restricted checkpoint and rejects unmatched dataset, optimizer,
loss, seed, step, adapter flags, or child digest. Unit tests run the full pair
twice, prove exact reproducibility, and reject checkpoint-metadata corruption.
A frozen non-smoke data/config split and counterfactual trajectory evaluation
remain required before advancing the paired-training checklist item.

M7 counterfactual evaluator foundation (2026-09-02):
`snowgym-evaluate-plan-ablation` loads both restricted child checkpoints and a
separate audited plan trajectory dataset, then reports overall and same-state
first-decision action accuracy. At the latter boundary it cyclically swaps only
the plan tensors while preserving each physical observation and measures
correct-versus-shuffled action NLL, discrete action changes, and target deltas.
The no-plan control is required to remain exactly insensitive by construction;
conditioned sensitivity alone is not treated as following without improved
correct-plan fit. The evaluation artifact binds the ablation and dataset
digests, and its auditor rejects result or provenance drift. Unit tests execute
the full collect/convert/train/evaluate chain and prove the swap cannot affect
the no-plan model. Frozen held-out data, thresholds, and closed-loop objective
metrics remain next.

M7 development experiment freeze (2026-09-02):
`configs/plan_bc_ablation_dev_v0.json` fixes identical 3,000-step CPU BC
budgets for the first no-plan/conditioned comparison before its outputs are
observed. Development data will use Winter Front map-backed 6v6 rollouts with
environment seed 4200, plan seeds 120-143, and an 80-decision horizon;
evaluation uses disjoint environment seed 5200 and plan seeds 600-611 under the
same roster/map/horizon. This run is diagnostic and has no retroactive pass
threshold. Its outcome will determine whether data balance, model capacity, or
metrics need a separately documented development revision before freezing a
disjoint qualification configuration.

M7 development outcome (2026-09-02): the post-freeze run collected 1,920
Winter Front training transitions and 960 disjoint evaluation transitions.
Their dataset digests are respectively
`sha256:d241e366a465b5920e3257896ec0b8c2c6fd0292e4920226cf109ee2f90b7e4f`
and
`sha256:1ebef435923485f7b26c22d892bd1d5147d31f35bab7d98396bd2d2d636c7069`.
The matched result digest is
`sha256:f838b542ab94c1b37ef763afbfc670afd6bb057844882e12bfe30c865845d8df`.
On held-out first decisions, plan conditioning reduced correct-plan target MSE
from `0.272753` to `0.048208`; swapping only plans raised conditioned target
MSE by `0.306098`, while the no-plan target/action deltas were exactly zero.
Overall conditioned action-type accuracy was lower (`0.955035` versus
`0.973090`), and first-decision action types did not vary, making action NLL
non-discriminative there. The evaluator now explicitly records correct and
shuffled target MSE. These are strong target-following development results, not
an M7 exit: a disjoint predeclared qualification suite and closed-loop
directive/objective metrics remain open.

M7 qualification freeze (2026-09-02): before collecting qualification data,
`configs/plan_qualification_v0.json` fixes Winter Front 6v6 training at
environment seed 6200 / plan seeds 1200-1223 and evaluation at disjoint
environment seed 7200 / plan seeds 1800-1811, both with 80-decision horizons.
It binds the exact 3,000-step `plan_bc_ablation_qual_v0` config digest and
requires every check: conditioned target MSE at most `0.10` and half the
no-plan MSE, shuffled-plan target-MSE increase at least `0.10`, mean target
change at least `0.20`, action-accuracy deficit at most `0.03`, and all no-plan
sensitivity measures at most `1e-12`. The qualification tool validates seed
separation and config binding, verifies the evaluation digest, records every
check, and never performs checkpoint selection. Qualification execution is
next; even a pass advances only the paired offline target-following evidence,
not the closed-loop M7 exit.

M7 qualification-v0 outcome (2026-09-02): the frozen post-commit run is
retained as a failure, not retuned. It used 1,920 training transitions (digest
`sha256:e252f624d0b45cf240af1c9c06709c4bf06b268040db48feb03d109611b0a902`)
and 960 disjoint evaluation transitions (digest
`sha256:b388d20c5f27dbee2984fce34df67d2b391b461b3e81333faf9c7bc8f79a3e78`).
Seven of eight predeclared checks passed: conditioned correct-plan target MSE
was `0.064513` versus no-plan `0.239327`, plan swapping added `0.314999` target
MSE and changed targets by `0.369133`, and no-plan sensitivity remained zero.
The action-accuracy check failed: conditioned accuracy `0.875694` trailed
no-plan `0.924132` by `0.048438`, exceeding the frozen `0.03` allowance. The
result digest is
`sha256:9b8a91fcb91657946f75317015d039410f2c0a6e09a7bc21e1273343145189e2`
and qualification digest is
`sha256:2b53b246fbc236770b2c59a2e0bfcec82b278192dd71c4acc673864bb3524c77`.
The gate stays closed. Next development must isolate plan-conditioned target
features from the action classifier or otherwise address multi-task
interference, then freeze a new disjoint qualification rather than modifying
v0.

M7 action-interference isolation foundation (2026-09-02): the optional
`plan_target_only` plus `separate_target_actor` architecture routes physical
context through the unchanged action actor and routes detached physical
features plus the encoded commander plan through a separate target actor.
Target loss cannot update the shared action feature path; action logits are
therefore counterfactually plan-invariant while movement/throw targets and
power remain plan-conditioned. The option requires plan conditioning and
action-conditioned target heads; legacy and fully shared plan-conditioned
checkpoint shapes remain unchanged. The paired runner
strips this plan-only routing flag from its no-plan control. Tests prove exact
action-logit/hidden invariance under counterfactual plans, distinct target-path
representations, configuration rejection, and finite end-to-end training. A
new development run on the retained v0 data is next; qualification v0 remains
permanently failed.

M7 target-only development freeze (2026-09-02):
`plan_bc_ablation_target_only_dev_v1.json` reuses qualification-v0's retained
training/evaluation datasets, seed, 3,000-step budget, loss weights, and physical
architecture, adding only `plan_target_only` to the conditioned branch. This is
an explicit diagnosis on already observed data, not a qualification retry and
has no pass threshold. If it restores action accuracy while preserving target
following, a v1 qualification must be frozen with new disjoint environment and
plan seeds.

M7 target-only development outcome (2026-09-02): the corrected post-commit run
on retained qualification-v0 data produced bit-identical action behavior for
both branches: overall accuracy `0.928299`, first-decision accuracy `1.0`, and
first-decision action NLL `0.00335723`. Conditioned correct-plan target MSE was
`0.058109` versus no-plan `0.242607`; swapping only the plans added `0.352305`
target MSE and changed predicted targets by `0.405377`, while every no-plan
sensitivity stayed zero. The matched result digest is
`sha256:f3925f9512d8339e7aadb790430b5ebee5d7ab7c4cf3fc569fbcaed812a63a10`
and evaluation digest is
`sha256:e4d186c40bf828db756e01a7afccb619a5396a98a50f119697943500f1b231d0`.
This resolves the observed action-head interference on development data. A v1
qualification with new seeds must still be frozen and passed.

M7 qualification-v1 freeze (2026-09-02): before observing any new outputs,
`plan_qualification_v1.json` retains every v0 numerical threshold and binds the
target-only/separate-actor 3,000-step config digest. Training uses new
environment seed 8200 and plan seeds 2400-2423; evaluation uses disjoint seed
9200 and plan seeds 3000-3011. Map, 6v6 roster, red difficulty, sample counts,
and 80-decision horizon remain unchanged so the architecture fix is the only
intentional experimental change. Execution is next; v0 remains failed
regardless of the v1 outcome.

M7 qualification-v1 acceptance (2026-09-02): the post-freeze run passed every
predeclared check on new seeds. The converted training and evaluation dataset
digests are
`sha256:f64e30e6458ca3d9c9a6e110aae7ae3248e6e2715a938ce323732b1e658d61e6`
and
`sha256:4e7f90f323ad8b3a660b7c6ff26cb91d286d2c64b685a612dd1c3e0622bc92c1`.
Both branches produced exactly identical action accuracy (`0.963021`),
first-decision accuracy (`1.0`), and action NLL (`0.000107431`). Conditioned
correct-plan target MSE was `0.044480` versus no-plan `0.264987`; plan swapping
added `0.353375` target MSE and changed targets by `0.372476`, with every
no-plan sensitivity equal to zero. The matched run, evaluation, and
qualification digests are respectively
`sha256:7583be365ffddcb88d40b4bfb3d3c00dca775b4f760fbe5f5453fff44270011b`,
`sha256:780f7c863baf352ff7de46695457114abb404f16593d879ca237abcf20b86fdf`,
and
`sha256:ea93ce6ff71029e932c1ec4bd3493fce83e80f71185fced6f123d5e6e0b3a12d`.
The matched offline training checklist item is advanced. Direct/flank,
focus/distributed, hold/support/withdraw, unseen-combination, roster-transfer,
and closed-loop objective metrics remain open; M7 exit is not yet claimed.

M7 authoritative online-plan bridge (2026-09-02): the guarded server API now
activates schema-valid symbolic plans and emits current fixed-size plan tensors
without advancing physics. Grounding, stable group assignments, late-bound
objective resolution, tactical geometry, living fractions, plan age, reset
invalidation, and state-hash/idempotency protection remain in TypeScript. The
persistent batch host exposes exact per-world parity, and `SnowGymBatchEnv`
validates and stacks `[B,3,38]` plan tensors for the next learned closed-loop
runner. Service, batch-isolation, validation, age, and live subprocess tests
cover the bridge; closed-loop objective evaluation remains next.

M7 closed-loop evaluation freeze (2026-09-02):
`configs/plan_closed_loop_v0.json` fixes a same-seed 6v6 Winter Front comparison
between direct/focus and left-flank/distributed plans before seeing execution
results. `plan_closed_loop.py` runs the qualified conditioned checkpoint and its
matched no-plan control as real blue policies, fetching fresh host-owned plan
tensors at every decision. Its hashed result reports terminal outcomes, rejected
actions, normalized objective progress, first-action target divergence, and
final group-position separation. The one-decision real-subprocess smoke proves
the complete checkpoint-to-authoritative-world path; the frozen full run is
next and thresholds have not been retrofitted.

M7 closed-loop development outcome (2026-09-02): the frozen v0 run completed
four real executions with zero rejected actions. Both policies truncated at the
predeclared 900-tick horizon and neither won, so this is not an M7 exit. In both
direct and flank cases, the conditioned policy finished with four blue survivors
versus two for no-plan. Its objective-progress advantage was `0.194498` and
`0.214476`; first-action target mean absolute deltas were `0.230875` and
`0.235246`, and final group-position distances were `0.270414` and `0.219296`.
The retained result digest is
`sha256:707864e8d39522f1c1f051d9090f104d956ce10bf8e42b657a2fadefe3a2e767`.
The next revision needs longer terminal evaluation plus explicit hold/support/
withdraw and multi-group objective metrics; this v0 result must not be tuned or
reinterpreted as a win-rate qualification.

M7 behavior-suite-v1 freeze (2026-09-02): before running new episodes,
`configs/plan_closed_loop_behaviors_v1.json` fixes same-seed 6v6 hold-current,
withdraw-backfield, and main-plus-reserve-support cases at a 3,600-tick / 600-
decision horizon. This suite is additive to the immutable v0 direct/flank run;
it tests one-group and multi-group behaviors without changing the qualified
checkpoints. Terminal outcomes and per-role behavior metrics remain unseen.

M7 mission-metric foundation (2026-09-02): the closed-loop evaluator now
retains objective distance, progress, start-to-final displacement, and terminal
position separately for each stable role slot. Case comparisons expose per-role
progress and displacement deltas, so hold can be judged by movement from its
activation position and reserve support is no longer averaged into the main
body. Existing result artifacts remain immutable; a new audited metrics result
must use a distinct path.

M7 behavior metrics-v1 outcome (2026-09-02): the immutable rerun at
`evaluations/plan_closed_loop_behaviors_metrics_v1.json` has digest
`sha256:060c33d396221ee3bf4d0b21e93f65462c0ae27c531feaae991585c18128e2d3`.
Conditioning reduced hold displacement by `0.552726`, which is directionally
correct, but also reduced withdraw displacement by `0.427437`. In the two-group
support case, main and reserve objective-progress deltas were `-0.151835` and
`-0.347228`. The model therefore demonstrates a defensive movement-suppression
effect, not yet distinct hold/withdraw/support competence. The next training
revision must balance these missions and qualify per-role behavior rather than
optimizing the aggregate objective-distance metric.

M7 plan-oracle foundation (2026-09-02): the server and persistent batch host
now expose a read-only plan-aware teacher action at the exact current state
hash. It uses the production `PlanAwareTeamController` and reactive unit policy,
does not advance physics, and fails closed until a plan is active. Python
validates per-world state-hash alignment before returning semantic labels. This
is the host-owned oracle seam for the next plan-conditioned DAgger collector;
learner-state collection and retraining remain open.

M7 plan-DAgger collector foundation (2026-09-02):
`export_plan_dagger.py` now runs the qualified plan-conditioned checkpoint in
persistent authoritative worlds, fetches a fresh host-resolved plan tensor and
same-hash production plan-teacher action at every learner-visited state, then
executes only the learner action. It rejects state drift, semantic-action
round-trip differences, physical action rejection, non-plan checkpoints, and
invalid plan episodes. Audited trajectory shards retain aligned plan tensors,
teacher labels, rollout-checkpoint provenance, split seeds, and complete plans.
A real two-decision integration test covers the full path. A frozen multi-
mission collection spec, merged training corpus, and retraining ablation remain
next.

M7 plan-DAgger-v0 freeze (2026-09-02): the catalog-based
`configs/plan_dagger_v0.json` freezes direct, flank, hold, withdraw, and two-
group support templates; ten training episodes and five each for validation and
evaluation use disjoint seeds. All run on 6v6 Winter Front against easy scripted
red with a 1,800-tick horizon. The catalog and episode references are validated
before the batch host starts. Collection results and retraining outcomes remain
unseen; this freeze must be committed before generation.

M7 plan-DAgger-v0 collection (2026-09-02): the post-freeze headless run produced
2,312 training, 1,191 validation, and 1,210 evaluation transitions with dataset
digests `sha256:8d4138dc4eea7a83f7af3273996a026cba2503268e89b468e08a0f490f08d182`,
`sha256:900a2872af7e730f81535b91356c8f70e2a78ef6b9972116191d14efb3d8700e`,
and `sha256:c8497125d4c62e86b2e0bbc92b1b0dae1275db42049c8149073079ffca2c2b85`.
Every direct, flank, hold, and support rollout lost; withdraw reached the frozen
300-decision limit. These are learner-state correction labels, not performance
evidence. Before combining them with expert-state data, the aggregate writer
must retain plan-conditioning metadata and explicitly support independently
seeded source specs.

M7 plan-aware aggregate foundation (2026-09-02): trajectory merging now has an
explicit independent-source mode for combining expert-state and learner-state
corpora. It requires matching split, capacity, simulator versions, and plan-
conditioning status; rejects any seed overlap; retains only common audited
tensor fields; records dropped auxiliaries per source; and preserves plan tensor
visibility for training. A real qualification-plus-DAgger smoke retained 4,232
transitions and both required plan arrays while dropping only the original
`plan_source_seed` auxiliary. Default same-spec merging remains strict.

M7 safe correction-training foundation (2026-09-02): behavior cloning now
distinguishes exact resume from initialization on a new dataset. The optional
`plan-target-path` mode requires the target-only/separate-target architecture
and freezes entity encoders, the physical action actor, and action head while
training only the plan encoder, target actor/heads, and power head. Checkpoints
bind initializer checkpoint/state digests and the new dataset independently. A
real plan-DAgger smoke proves the action head remains bit-identical while the
plan encoder updates. A frozen correction config and post-freeze run are next.

M7 plan-DAgger correction-v0 freeze (2026-09-02): the audited expert-plus-
learner aggregate contains 4,232 transitions with digest
`sha256:297be4717a9f33a44f374c006e6aca73aed804149c2f781ebd69508b79497bdd`.
`configs/plan_dagger_correction_v0.json` fixes qualified-checkpoint
initialization, plan-target-path-only training, seed 84001, 1,500 steps, batch
64, learning rate `0.001`, and the existing matched loss weights. The physical
action path is frozen. Training and closed-loop outcomes remain unseen.

M7 correction-evaluation bridge (2026-09-02): closed-loop evaluation can now
accept an explicit conditioned checkpoint while retaining the original matched
ablation's no-plan control. Results bind both checkpoint and state digests, and
the semantic auditor reloads the supplied override. This avoids copying a
checkpoint or manufacturing a new ablation manifest for post-DAgger evaluation.

M7 plan-DAgger correction-v0 outcome (2026-09-02): the frozen 1,500-step run
produced checkpoint digest
`sha256:54439be3493c17aa5fdd6a5f21f3698c616784e33c9e0a1ce00bf7bcd932cb29`.
All physical action-path tensors remained bit-identical and all 14 permitted
target-path tensors changed. The correction did not pass closed-loop acceptance:
direct ended with 3 blue / 5 red rather than the prior 4 / 4; hold and withdraw
lost sooner; and support regressed from eliminating two red units to eliminating
none. Evaluation digests are
`sha256:3b1a65956f4c3be56b0654091b021d52e6010949eabbba10ac4c6ae4099192e1`
and
`sha256:dfcfbf1418b7171ce37c62ecd5c4cb3dada316ab32dabd2e2c388e3a99585965`.
The checkpoint is retained as failed evidence and must not replace qualification
v1. Target-only learner-state correction cannot fix mission-dependent action
timing; the next architecture must condition action decisions while preserving
a matched no-plan control and explicit action-accuracy bounds.

M7 residual action-adapter foundation (2026-09-02): the optional
`plan_action_adapter` adds a plan-conditioned residual to action logits without
changing the inherited physical actor shape. Its output layer starts at exact
zero; a target-only qualification checkpoint therefore produces bit-identical
initial logits when loaded into the expanded model. The corresponding
`plan-action-target-path` training mode freezes all inherited entity encoders,
the actor, and action head while updating only the new adapter plus the existing
plan/target path. Tests prove zero-init invariance, compatible initialization,
frozen action-head identity, and adapter learning. A matched frozen experiment
and action-accuracy gate remain next.

M7 single-checkpoint offline gate foundation (2026-09-02):
`plan_checkpoint_evaluate.py` applies the existing action, target, and same-state
counterfactual metrics to one checkpoint on any audited aligned plan dataset.
Its semantic result binds checkpoint/state and dataset digests and is independently
auditable. This supplies the action-accuracy safety measurement needed before a
residual adapter can be judged on closed-loop outcomes.

M7 residual action-adapter-v0 freeze (2026-09-02): the qualified checkpoint's
pre-run learner-state baseline has action accuracy `0.631130`, first-decision
accuracy `1.0`, target MSE `0.121232`, and evaluation digest
`sha256:4e97e2f7686f56559819395d6df9b3b2f09b71bfd4c957664ffb8de0a6440789`.
The frozen run uses seed 85001, 1,500 steps, batch 64, learning rate `0.0003`,
qualified initialization, and plan-action-target-path training. Its conjunctive
spec requires action accuracy at least `0.681130`, first-decision accuracy at
least `0.95`, target MSE at most `0.13`, action counterfactual change at least
`0.05`, target sensitivity at least `0.2`, zero rejected actions, direct/flank
blue survivors at least 4 each, hold/withdraw duration at least 267/341, and
support red survivors at most 4. No adapter training outcome has been observed.

M7 residual action-adapter-v0 outcome (2026-09-02): the frozen run is retained
as a failed gate with checkpoint digest
`sha256:4079ba7186995f293731887bd99073a70c0c6c1136d6bb853034a828b0ffd1b3`.
Offline action accuracy improved to `0.854821` and target MSE to `0.018053`,
while hold and withdraw exceeded their duration thresholds and all actions were
accepted. Direct and flank preserved three rather than four blue units, support
eliminated no red units, and the first-decision counterfactual action-change
rate remained zero. Seven of eleven frozen checks passed; qualification digest
`sha256:4482dcd8585932e62dcc5d0605cbc3f1d62a4eb76f2edeae45ee76d29dd42777`
records the failure. Dataset inspection showed every evaluated plan labels all
six first-decision units as `move`, so that first-decision-only action-change
criterion cannot identify correct mission-dependent timing. The next revision
must collect host-generated counterfactual plan/action labels on the same later
learner-visited physical states; v0 thresholds and results remain immutable.

M7 counterfactual plan-preview foundation (2026-09-02): the service and batch
host can now ground an arbitrary schema-valid plan against the current detached
world, return its host-resolved `[3,38]` tensor and production plan-teacher
action, and discard the temporary plan without changing the active plan or
advancing physics. Requests are guarded by the current public-state hash. This
read-only intervention seam enables paired later-state labels without resetting
plan age or corrupting the executed learner trajectory. Dataset export,
counterfactual training loss, and a disjoint v1 gate remain next.

M7 same-state counterfactual DAgger foundation (2026-09-02):
`snowgym.plan-dagger-export.v1` requires each rollout plan to name a distinct
catalog counterfactual. At every truthful learner transition, the collector
uses read-only plan preview to attach an alternate `[3,38]` tensor and complete
production teacher action for the identical pre-step physical state. The
trajectory auditor validates the extra tensors, legal masked actions, numeric
bounds, and explicit same-state provenance. Behavior cloning can apply a
bounded `counterfactualLossWeight` to a second hybrid loss after replacing only
the plan tensor and label; zero or omission preserves all legacy runs. A frozen
collection schedule, an all-transition paired evaluator, and a v1 residual-
adapter experiment remain next.

M7 all-transition counterfactual evaluator foundation (2026-09-02):
`plan_counterfactual_evaluate.py` runs one checkpoint under both authoritative
plan tensors for every paired state. Its audited result reports whether the
teacher action actually changes, accuracy and NLL under each plan, the model's
action-change rate, recall and strict pair accuracy where the teacher differs,
and target error/sensitivity. Checkpoint/state and dataset digests are bound in
the artifact. This separates an uninformative counterfactual set from a model
that ignores informative plan interventions. Frozen data and acceptance
thresholds remain next.

M7 counterfactual DAgger-v1 collection freeze (2026-09-02): before generating
new labels, `configs/plan_counterfactual_dagger_v1.json` fixes ten training,
five validation, and five untouched evaluation episodes on 6v6 Winter Front
against easy scripted red, each with a 1,800-tick horizon. Seeds 14201–14210,
14301–14305, and 14401–14405 are disjoint from each other and v0. Every direct,
flank, hold, withdraw, and support primary is paired with a distinct plan chosen
to contrast offensive timing, defensive posture, or group support. Training
teacher-diversity diagnostics may inform a separately committed optimization
freeze; validation/evaluation metrics must remain unseen until that freeze.

M7 counterfactual DAgger-v1 collection (2026-09-02): the post-freeze run
produced 2,393 training, 1,238 validation, and 1,200 evaluation transitions
with respective digests
`sha256:c3bd756af35908cfe972c43e11f0292ab5cab3dca0e3375e2bed436725749df5`,
`sha256:d0ec0683697f159e2e14983ee4f19c329fd4ae6778f29f3084185d4ebe237576`,
and `sha256:f5598b7fa17f03fce5e2b7af2814504796e7eea8ea16b36c847ebc955efd22fb`.
One training and one validation episode ended when attrition made their
two-group alternate plan inapplicable; both manifests record the bounded stop,
and evaluation completed all episodes. The permitted training-only diagnostic
found 1,538 teacher-changed unit decisions out of 14,358 (`0.107118`), while
the inherited target-only checkpoint remained exactly action-invariant. This
establishes identifiable supervision without inspecting validation/evaluation
metrics. Optimization and acceptance thresholds must be frozen next.

M7 counterfactual residual-adapter-v1 freeze (2026-09-02): before training or
reading validation/evaluation metrics, `plan_action_adapter_v1.json` fixes
qualified-checkpoint initialization, seed 86001, 2,000 steps, batch 64,
learning rate `0.0003`, unchanged hybrid loss weights, and equal weight on the
same-state counterfactual loss. The v1 qualification spec binds all three
corpus digests, initializer, and training-only baseline. Its paired evaluation
requires teacher diversity at least `0.05`, primary/counterfactual accuracy at
least `0.75`, predicted action changes at least `0.05`, changed-teacher recall
at least `0.50`, strict changed-pair accuracy at least `0.35`, and both target
MSE values at most `0.13`. It retains every v0 closed-loop mission threshold.
The qualification runner audits input/result and checkpoint provenance and
applies all checks conjunctively. Training and held-out results remain unseen.

M7 counterfactual residual-adapter-v1 outcome (2026-09-02): the frozen run is
retained as a failed gate with checkpoint digest
`sha256:4e7c4aaf9644ce63179bbecff2ebdca05e3dee64ad24b40efc6bb96a524348f5`.
On untouched evaluation data, primary/counterfactual action accuracy reached
`0.841111`/`0.860139`, target MSE `0.006856`/`0.009023`, and predicted action
change `0.084583`; teacher change was `0.090417`. Changed-teacher recall and
strict pair accuracy were only `0.165899` and `0.084485`, failing their frozen
thresholds. Closed loop, flank preserved five blue, hold lasted 338 decisions,
and support achieved the first learned plan-conditioned blue win at 6–0 with
zero rejected actions. Direct preserved three blue and withdraw lasted 259
decisions, so those checks failed. Ten of fourteen checks passed; qualification
digest is
`sha256:075a610c3f60ac2a77b112a5dd1c67506b991ae29ad56db2cda1f21f0a81a5c0`.
The next revision must weight the identifiable teacher-changed positions rather
than letting the roughly 90% unchanged pairs dominate the counterfactual loss.

M7 changed-action loss foundation (2026-09-02): behavior cloning now accepts a
bounded `counterfactualChangedActionWeight` only for datasets with audited
same-state labels. It adds symmetric primary/alternate cross-entropy solely on
present unit-states where the production teacher action types differ, retaining
the existing full primary and counterfactual hybrid objectives. Empty changed
batches contribute an exact finite zero. Legacy configs are unchanged. A
separately frozen v2 weight/budget and unchanged held-out gate are next.

M7 changed-action-v2 development freeze (2026-09-02): before training,
`plan_action_adapter_v2_dev.json` fixes the v1 architecture, corpus, 2,000-step
budget, batch size, learning rate, full counterfactual weight, and initializer,
changing only the training seed to 87001 and adding changed-action weight
`5.0`. This is a diagnostic on already observed v1 validation/evaluation and
closed-loop suites, not a qualification retry. If it materially improves
changed-teacher alignment without losing v1's support/flank progress, a new
counterfactual corpus with fresh seeds and a separately frozen gate is required.

M7 changed-action-v2 development outcome (2026-09-02): the fixed weight-5 run
produced checkpoint digest
`sha256:00b63d2e8f99e3557211b90b38694bf0a6dbc0333bcd9d587125e6e69148013d`.
On the reused evaluation split, changed-teacher recall rose from v1's `0.165899`
to `0.761905` and strict pair accuracy from `0.084485` to `0.697389`, but
predicted change increased to `0.395417` versus teacher change `0.090417`.
Closed loop preserved five blue on flank and kept withdraw alive for all 600
decisions at 2 blue / 1 red, while direct preserved two, hold lost after 205,
and support lost 0–3 rather than v1's 6–0 win. All actions remained accepted.
The loss fixes alignment but is overweighted. A predeclared intermediate-weight
development sweep is required before spending fresh qualification seeds.

M7 changed-action weight-sweep-v0 freeze (2026-09-02): before running any new
checkpoints, three configs fix weights `1`, `2`, and `3` with identical seed
88001, 2,000-step budget, data, initializer, model, and other losses. The frozen
development checks require paired accuracy at least `0.75`, changed recall/pair
accuracy at least `0.40`/`0.30`, predicted change in `[0.05,0.20]`, and the
existing five mission thresholds. Selection maximizes checks passed, then
changed-pair accuracy, then prefers lower weight. These reused suites select a
development candidate only; fresh data and a new gate remain mandatory.

M7 changed-action weight-sweep-v0 outcome (2026-09-02): the audited sweep
result digest is
`sha256:e83bff1be89aad0345b0b0336ea7a5caa5bbdb5edabe389ac5c5adb18548054b`.
Weights 1, 2, and 3 each passed 7/11 development checks. Weight 1 alone kept
predicted action change under `0.20` (`0.179456`) but narrowly missed changed
recall (`0.398411`); weights 2/3 improved strict pair accuracy to `0.640182`/
`0.708286` while over-changing `0.304389`/`0.373452`. All three missed flank,
hold, and withdraw thresholds. The predeclared tie-break selects weight 3, but
7/11 is not qualification readiness and no fresh seeds are consumed. Scalar
weighting cannot resolve the observed mission tradeoff; the next development
revision must balance supervision by primary mission and group role.

M7 unit-role conditioning foundation (2026-09-02): the Python batch adapter now
maps host-owned plan assignments onto stable current ally slots as fixed
`[B,U,3]` main/maneuver/reserve one-hot tensors, ignoring dead assignment IDs
without exposing them to the learner. `snowgym.plan-dagger-export.v2` retains
both primary and same-state counterfactual unit-role tensors with audited
one-hot, presence, and coverage rules. The optional `plan_role_conditioned`
model path requires the residual plan adapter and injects per-unit role into a
zero-initialized action residual and a separate zero-initialized target
residual, preserving qualified initializer outputs exactly. Tests cover live
batch encoding, preview roles, schema audit, malformed roles, compatible
initialization, and a training step. A fresh role-aware collection freeze and
development run remain next.

M7 role-aware-v0 development freeze (2026-09-02): before collection,
`plan_role_dagger_v2.json` fixes fresh 15201–15210 training, 15301–15305
validation, and sealed 15401–15405 evaluation seeds with the existing balanced
mission-pair schedule. `plan_role_adapter_v0_dev.json` fixes qualified
initialization, seed 89001, the established 2,000-step budget and losses,
changed-action weight `1.0`, and enables only the new unit-role residuals.
Development may inspect training/validation plus the existing closed-loop
suites; the v2 evaluation split remains unopened until a new gate is frozen.

M7 role-aware-v0 development outcome (2026-09-02): after correcting role
encoding to cover living ally slots only, the audited train/validation corpora
contain 2,358/1,152 transitions with digests `173ad8a4...`/`0ba2ad5e...`.
The frozen 2,000-step run produced checkpoint `490f88c0...`. Validation primary
action accuracy improved to `0.863426`, but predicted plan-change rate
`0.227865` still exceeded the teacher's `0.130642`; changed-teacher recall and
strict pair accuracy were only `0.559247`/`0.462901`. Closed loop preserved
four blue in both direct/flank cases but completed neither, while hold,
withdraw, and support all lost; support regressed to 0 blue versus 6 red.
The checkpoint is retained as a failed development result and the sealed
15401–15405 evaluation split remains unopened. The next M7 revision must
condition each unit on its own resolved group directive, not merely its
main/maneuver/reserve category, while retaining the qualified initializer and
same-state counterfactual evaluation.

M7 per-unit directive foundation (2026-09-02): the optional
`plan_unit_directive_conditioned` architecture derives a deterministic
`[B,U,38]` tensor by selecting each living unit's host-resolved group row with
its audited one-hot assignment. The full local directive now enters both the
zero-initialized action and target residuals while the global plan encoder is
retained. It requires role conditioning, adds no new observation schema, and
keeps older checkpoints and v2 corpora compatible. Tests cover configuration
constraints, exact group-row selection, initialized inference, and the
role-aware closed-loop policy bridge. A frozen development config and run on
the retained train/validation corpus are next; sealed evaluation seeds remain
untouched.

M7 per-unit-directive-v0 development freeze (2026-09-02):
`plan_unit_directive_adapter_v0_dev.json` reuses the corrected v2 training and
validation corpora plus the qualified plan-conditioned initializer. It holds
seed 89001, 2,000 steps, batch 64, learning rate `0.0003`, and both
counterfactual weights at `1.0`, exactly matching role-aware-v0. The sole model
change is `plan_unit_directive_conditioned: true`. Only validation and the
existing closed-loop suites may be inspected; evaluation seeds 15401–15405
remain sealed.

M7 per-unit-directive-v0 development outcome (2026-09-02): the frozen run
produced checkpoint `3855cc78...`. Validation changed-teacher recall/pair
accuracy improved over role-aware-v0 to `0.599114`/`0.503876`, while primary
accuracy fell slightly to `0.855758` and predicted plan changes remained high
at `0.239873`. The withdraw case improved materially to the full 600-decision
horizon with five blue versus four red, but direct/flank retained only three/two
blue and support still lost 0–6. This is retained partial evidence, not a
qualification candidate; evaluation seeds remain sealed. Richer directive
features alone do not resolve the mission tradeoff. The next revision must
balance correction supervision across primary missions and assigned roles.

M7 mission/role balancing foundation (2026-09-02): behavior-cloning configs may
now opt into deterministic `plan-mission-uniform` sampling, which selects plan
names uniformly before choosing transitions, and `roleBalancedLoss`, which
applies mean-one inverse-frequency unit weights to action, target, power, and
teacher-changed losses. Defaults preserve every prior run. Weighting covers
observed roles and gives absent roles zero weight; auditing the retained corpus
revealed 11,488 main, 790 reserve, and zero maneuver labels in train (5,754,
335, and zero in validation). Thus balancing can correct main/reserve skew, but
a future curriculum must add a genuine maneuver-group plan before claiming
three-role transfer. Tests cover deterministic mission balance, equal aggregate
weight across observed roles, malformed loss weights, and a real balanced
training step.

M7 balanced-unit-directive-v1 development freeze (2026-09-02):
`plan_unit_directive_balanced_v1_dev.json` holds the v0 directive architecture,
initializer, seed, 2,000-step budget, batch size, optimizer, and loss weights
fixed. Its only changes are `sampling: plan-mission-uniform` and
`roleBalancedLoss: true`. The run may inspect the retained train/validation
corpora and existing closed-loop suites only; evaluation seeds remain sealed.

M7 balanced-unit-directive-v1 development outcome (2026-09-02): the frozen run
produced checkpoint `3106db06...`. Direct improved to five blue versus four red,
hold extended to 317 decisions, and withdraw again reached 600 decisions with
five blue while reducing red to one. However flank fell to one blue versus five
red and support remained a 0–6 loss. Validation primary/counterfactual accuracy
regressed to `0.826678`/`0.798177`, predicted changes rose to `0.280961`, and
strict changed-pair accuracy fell to `0.447398`. The result is retained but not
promoted, and sealed evaluation remains untouched. The next collection must add
real maneuver-group assignments and more multi-group support supervision before
another controlled training run.

M7 multi-group DAgger-v3 collection freeze (2026-09-02):
`plan_multigroup_dagger_v3.json` defines six plan families on the unchanged 6v6
arena: direct, hold, withdraw, a true main+maneuver left flank, main+reserve
support, and main+maneuver+reserve support. Train seeds 15501–15512 and
validation seeds 15601–15606 are disjoint from sealed evaluation seeds
15701–15706. Each plan appears twice in train and once in validation/evaluation;
all pairs use a distinct same-state counterfactual. Only train/validation may be
collected until a later gate is frozen.

M7 multi-group DAgger-v3 collection outcome (2026-09-02): the audited train
and validation splits contain 2,763 and 1,396 transitions with digests
`5bf6a6a2...` and `ed8dceda...`. Primary role counts are 11,838 main, 1,512
maneuver, 952 reserve in train and 5,852/839/439 in validation; counterfactual
roles also cover all three categories. Attrition-bounded episodes are retained
and explicitly marked. No evaluation artifact exists. A training freeze on
this immutable corpus is next.

M7 multi-group-directive-v2 development freeze (2026-09-02):
`plan_multigroup_directive_v2_dev.json` holds every architecture, seed, budget,
optimizer, loss, sampling, and role-balancing setting from balanced v1 fixed.
Only the dataset changes to the immutable v3 train/validation corpus. Existing
closed-loop suites may be inspected; the v3 evaluation split remains sealed.

M7 multi-group-directive-v2 development outcome (2026-09-02): the frozen run
produced checkpoint `7f570c62...`. Validation primary/counterfactual accuracy
was `0.839780`/`0.827841`, but changed-teacher recall/pair accuracy fell to
`0.273247`/`0.237482`. Direct/flank retained four/two blue; withdraw reached
600 decisions at 1–1; support killed one red but still lost with no blue alive.
All actions were accepted. The checkpoint is retained but not promoted and the
v3 evaluation split remains unopened. Since explicit roles and balanced
multi-group data still trade one mission against another, the next architecture
should gate separate residual experts by the resolved unit directive.

M7 directive-expert foundation (2026-09-02): the optional
`plan_directive_experts` architecture replaces the shared action and target
residuals with five independently parameterized experts, hard-selected by the
unit directive's audited engage/advance/hold/withdraw/support one-hot. It
requires full per-unit directives, preserves qualified initializer outputs via
zero-initialized expert heads, and leaves prior checkpoints unchanged. Training
initialization and trainable-path filters recognize the expert parameters.
Tests prove exact zero-residual preservation, configuration constraints, and
different expert routing for two mission directives. A frozen v3-corpus
development run is next.

M7 directive-experts-v3 development freeze (2026-09-02):
`plan_directive_experts_v3_dev.json` holds the v3 corpus, qualified initializer,
seed, 2,000-step budget, balancing, optimizer, and losses fixed from
multi-group-v2. Its sole architecture change is
`plan_directive_experts: true`. Development may inspect validation and existing
closed-loop suites; evaluation remains sealed.

M7 directive-experts-v3 outcome (2026-09-02): checkpoint `c0b66d6f...`
improves validation primary accuracy to `0.875478` and changed-pair accuracy to
`0.353362`, but closed loop still fails: direct/flank retain four/three blue,
and hold, withdraw, and support all lose. The expert checkpoint is retained,
not promoted, and evaluation remains sealed. Supervised action imitation has
now failed across shared, role-aware, full-directive, balanced, multigroup, and
mission-expert variants; the next M7 step is closed-loop plan-conditioned RL
fine-tuning from a retained BC initializer, not another unguided BC variant.

### M8 — unit-level CTDE / MAPPO

Goal: add decentralized execution only after centralized plan-conditioned PPO
works.

**Gate note:** this milestone's own exit criterion is unchanged, but M7c
below states "only then may M8... advance" and M7c is not complete. M8 work
proceeds anyway under the user's 2026-09-14 red-agent roadmap decision
(project memory, `red-agent-roadmap`): "M8 (unit-level MAPPO) lands
regardless" of M7c, validated first against a known-good baseline on blue.
That is a scope decision for the fighter/red-agent research track, not a
claim that M7c's own exit criterion has been met — M7c's checklist below is
unchanged and remains the gate for whatever actually depends on it (M9's
commander comparisons, the roster/composition mission gates).

- [x] Add `SnowGymUnitParallelEnv` as a new PettingZoo environment; retain the
      existing team-level environment and version. **(M8-S1, complete
      2026-09-20 — see below.)**
- [ ] Begin with global actor observations, then local observations, then local
      observations plus latency. Change only one observability condition per
      experiment.
- [ ] Implement parameter-shared unit actors and a centralized critic over
      global state, assignments, and active plan. Execution must use actor-local
      inputs only.
- [ ] Gate 3v3 before 5v5, and fixed rosters before variable roster transfer.

M8 exit: MAPPO beats its unit-random baseline in frozen 3v3 and 5v5 suites and
the command-conditioned shared actor remains valid under local observations.

**M8-S1 — `SnowGymUnitParallelEnv` (complete 2026-09-20; infrastructure,
no training):** `snowgym/python/src/snowgym_client/unit_parallel_env.py`
(new file; the existing `parallel_env.py`/`encoding.py`/`research_env.py`
are read, not edited). Wraps the existing team-level `SnowGymParallelEnv`
by composition, exposing one PettingZoo agent per living unit
(`f"{team}-{slot}"`, both teams symmetric). First cut, matching M8's own
checklist order: global observations only (every living unit-agent on a
team receives that team's full existing team-level observation unchanged;
per-unit egocentric framing stays a model concern, matching how
`FullAuthorityPolicyV1.features()` already does this from a team tensor),
the shared team reward broadcast to every living unit on that team (no
per-unit credit invented), and per-unit termination on death (verified
stable slot identity for a whole episode — dead units stay in their sorted
slot with `alive: false`, never removed or reindexed,
`observations/Observation.ts:112-136`). `pettingzoo.test.parallel_api_test`
passes; a reset-tensor determinism check and a death-and-removal check pass,
all against a fake client (the simulator-backed versions arrived in M8-S2).
Local-visibility/latency restriction (the next M8 checklist item),
parameter-shared training, and the centralized critic are explicitly out
of scope for this stage. See `training/reviews/m8_s1_declaration.md`.
11 new python client tests (was 51, now 62); full training suite (456),
`npm run build`, `npm test` (366/367, the one documented R1n-b exception)
all re-verified clean.

**M8-S2 — unit-env contract repair (complete 2026-09-20; infrastructure, no
training):** an external review of S1 found that a pure timeout returned
`terminated=True` alongside `truncated=True` for every survivor, that
malformed per-unit actions (`action_type=2.7`, a scalar `target`) were
silently coerced into valid ones before the team env's space check, and that
S1's tests used only a fake client. Repaired in
`unit_parallel_env.py`: `terminated = team_terminated or unit_dead`,
`truncated = team_truncated`, per-agent `info["snowgym_unit"]` carries
`team_terminated` / `team_truncated` / `unit_alive` / `unit_died` (trainers
derive the value boundary from `team_terminated`, not from `terminated`), and
every unit action is checked against its action space before merging. New
`tests/test_unit_parallel_env_live.py` runs against a real server subprocess:
same-action state-hash and observation parity with the team env, seed
determinism, pure timeout, a real mid-battle death with stable slot identity,
and a rejected action that never advances the simulator. 14 of the new
fake-client tests and 3 of the 5 live tests fail on S1's code. 81 client tests
(was 62), 456 training, `npm run build`, `npm test` 366/367 (the one
documented R1n-b exception). No seeds. See
`training/reviews/m8_s2_declaration.md`. Next: S3, the versioned
v3/plan/focal-unit bridge with explicit Red action routing, still without a
learning claim.

**M8-S3 — 3v3 plan-conditioned path on the batch transport (complete
2026-09-20; audit, no training):** orientation found the trainer's path is
the batch transport (`SnowGymBatchEnv` + `EngageOptionBatchV1` + plan
observations), not `SnowGymUnitParallelEnv`, which has no plan/role/option
tensors. Decision: the batch path is the trainer's path; the unit env stays
the M8 API-conformance artifact. `FullAuthorityPolicyV1` is already one
shared network applied per living unit slot with egocentric rows (global
inputs, the roadmap's first M8 condition); unit-local sensing is not claimed.
No existing module was edited. New `options/roster_engage.py` (roster-N
scenario over the R1n arena, ally-slot permutation, equivariance gap) and
`tests/test_roster_engage.py`. Established at 3v3 against scripted-normal Red:
all 3 blue units assigned and all 3 red targets activated; option state and
tensor shapes correct; actor and critic equivariant in slot order; `features`
does not mutate the shared observation; `act` likelihood equals its
re-evaluation with zero weight on unused slots; a real mid-option unit death
leaves the living mask, keeps its slot, contributes no likelihood, does not end
the option, and its submitted action is ignored by the simulator. Batch
`step_joint` and the unit env give identical state hashes for the same
two-team actions, and native scripted Red is a different opponent from a joint
step with idle Red (Red routing declared: scripted/random Red runs natively
behind team actions; joint steps are for conformance and self-play). Finding
for S4: with 3 units, random-init blue loses a unit mid-option in every one of
8 worlds (steps 72-88) while the option continues, so 1v1's "death rate"
(the only blue unit died) needs an explicit 3v3 definition before any result
is reported. Not established: that 3v3 Engage is learnable or that the
teacher is achievable at 3v3. 83 client tests (was 81), 465 training (was
456), `npm run build`, `npm test` 366/367 (the one documented exception).
See `training/reviews/m8_s3_declaration.md`. Next: S4, teacher achievability
and random-init floors at 3v3 with a declared seed band, budget and archive.

**M8-S4 — 3v3 Engage metrics, teacher achievability and random-init floors
(complete 2026-09-20; collected and archived, no training):** the 3v3 metrics
were frozen before data: units-lost fraction `L` (the strict generalization of
R1n's death indicator; primary for later comparisons, mean, world-paired),
success, team wipe, timeout. 400 world-paired seeds (2100000-2100399), three
native Red arms, 417,097 decisions. Teacher success 400/400, 399/400 and 400/400
(random, easy, normal; `L` 0.010, 0.013, 0.037) passes the >= 0.90 gate. Random-init
policies (3 init seeds, deterministic and sampled) score 0/100 in every cell, are wiped
in every episode by scripted-normal, and never engage random Red (all time out); the
"no free signal" gate passes. Not established: that 3v3 Engage is learnable or that any
initializer works. Archive `runs/m8_s4_roster_baseline_v0/` sealed and verified. See
`training/reviews/m8_s4_results.md`. Next: S5, teacher imitation as the 3v3 initializer
(R1n recipe), after re-checking the critic warm-start gate at roster 3.

**M8-S5 — 3v3 teacher-imitation initializers (complete 2026-09-20; collected and
archived, imitation only, no PPO):** R1n-h's mixture recipe (condition M) ran unchanged
at roster 3 for three optimizer seeds and each final policy was evaluated on S4's 400
paired worlds against the archived teacher. **Pre-declared reading: not viable**
(seed-mean success against scripted-normal 0.0017 vs the declared 0.25). Per seed: easy
73/89/91%, normal 0.5/0/0% (blue wiped in 99-100% of episodes), random 6.5/7.7/19% (33-77%
wipes). Uniform failure across seeds, corroborated by the pipeline's own development
evaluations. The 3v3 critic warm-start gate passes (R2 0.63 vs time-only 0.29-0.31).
Not established: why it fails, that 1v1 differs, or anything about PPO. 1,124,388
decisions, archive `runs/m8_s5_roster_imitation_v0/` sealed and verified. See
`training/reviews/m8_s5_results.md`. Next: an R1n-i-style diagnosis of the frozen S5
checkpoints, declared before any PPO.

**M8-S6 — archive-only failure breakdown of the S5 initializers (complete
2026-09-20; zero simulator decisions):** predictions fixed before computing, then scored. Against
scripted-normal the S5 learners make contact on time (first hit at decision 53-55 vs the teacher's
52, similar range) and then collapse: median damage 60-140 vs the teacher's 240, wiped by
decision ~91-112, damage per unit lost 18-45 vs 2,163. Against random Red contact is late and
wipes dominate two seeds (52%, 77%); one prediction (random-Red failures cluster at 200-239 damage)
failed. Located after contact, not explained; no per-unit or trace data exists in the rows. Archive
`runs/m8_s6_breakdown_v0/`. See `training/reviews/m8_s6_results.md`. Next: choose a trace-level
diagnosis of the post-contact fight (new module, since R1n-i's deployed view is 1v1) or a more
targeted use of the frozen checkpoints.

### M9 — slow commander over a learned team

Side experiment authorized 2026-09-05: add OpenAI GPT-6 Astra as a secondary
commander backend alongside GPT-5.6 Luna, preserving the existing symbolic
CommandPlan, prompt, grounder, and scripted executor. CLI selection exposes
backend and reasoning; Astra defaults to low ("light"), with medium available.
Run a bounded 16-request 2×2 pilot (Luna/Astra × low/medium) on four frozen
scenario requests, without retries. Record request/response provenance,
latency, token usage, validation/repair/fallback rates, and matched headless
continuation outcomes. Provider latency and zero-delay tactical consequences
are separate measurements. This authorized side comparison does not reopen
LLM-over-learned-fighter experiments or relax the M7c/M9 qualification gates.

Implementation status: CLI selection and matched-snapshot benchmark are ready;
338 TypeScript, 51 Python client, and 257 Python training tests plus the build
passed. The offline fixtures and scripted baselines are archived in
`orchestration/benchmark/examples/luna-astra-20260905-v0/preflight/`.
The user explicitly approved the 16-request payload/budget after the initial
permission block. The live pilot completed with eight Luna and eight Astra
calls, no retries or timeouts. Fifteen plans passed host validation; Astra-medium
had one trailing-whitespace rejection in the trace-only intent summary and
used fallback. All 16 continuations replayed exactly and issued zero rejected
physical actions. Luna-low and Astra-low each won 3/4 cases at mean latencies
of 3.576 s and 3.564 s. Luna-low's remaining case was horizon-censored; no arm
won the 6v10 case. These four-case results do not establish model superiority.
See `orchestration/benchmark/examples/luna-astra-20260905-v0/README.md` and its
immutable `live/` artifacts. The schema/host whitespace mismatch is recorded
for a separately versioned repair; the pilot contract was not changed.

User-selected operational default (2026-09-05): Astra with low reasoning for
the commander client and live CLIs. Explicit Luna selection retains medium
reasoning unless overridden. The matched benchmark axes, archived artifacts,
map generator, schema, prompt, and executor stay unchanged. Review Astra's
planning opportunities separately from this default-selection change; new
provider experiments and learned-fighter qualification remain gated.
The completed review is `orchestration/ASTRA_PLANNING_REVIEW.md`. Default-switch
verification passed 344 TypeScript, 51 client, and 257 training tests plus the
build; explicit pilot request bodies and artifact checksums remain unchanged.

Goal: connect the already-tested asynchronous commander only after freezing a
competent learned executor.

- [ ] Hold the learned executor checkpoint fixed while comparing no commander,
      random valid plan, rule commander, high-level RL, online LLM, static
      LLM-generated doctrine, and distilled commander baselines through the
      same `CommandPlan` IR.
- [ ] Reuse scheduler coalescing, timeout, fallback, stale-plan reconciliation,
      trajectory triggers, token/latency accounting, and replay trace alignment;
      do not rebuild these inside the trainer.
- [ ] Prove commander failure or latency cannot block the learned 10 Hz executor
      and cannot bypass action validation.

M9 exit: real commander plans drive the fixed learned executor with aligned
replay/trace artifacts, while cheaper baselines and provider cost/latency are
reported on the same held-out scenarios.

### M10 — latency and generalization benchmark

M10 deterministic latency-runner foundation (2026-09-02):
`CommanderLatencyBenchmark` executes the trajectory-aware mock commander over
a validated seed-by-latency matrix and emits episode outcomes, request/plan/
signal counts, aggregate metrics, and a semantic digest. The CLI defaults to
0, 100, 250, 500 ms and 1, 2, 4, 8 simulated seconds at 60 Hz, supports M-vs-N
and maps, and refuses accidental overwrite. Tests prove complete matrices,
exact reruns, tamper detection, and preflight rejection. Strategy baselines and
frozen benchmark specs remain next.

- [ ] Sweep simulated commander latency at 0, 100, 250, 500 ms and 1, 2, 4,
      and 8 seconds; run separately authorized real-latency checks only after
      deterministic sweeps pass.
- [ ] Compare reject, activate-unchanged, and reconcile/late-bind handling for
      plans produced from stale state.
- [ ] Compare exact IDs/coordinates against symbolic late-bound groups without
      exposing forbidden physical details to the production commander path.
- [ ] Evaluate 3v3/5v5-trained executors and proportional group plans on 10v10.
- [ ] Report win/draw rate, objective completion, rejection/repair rate,
      trajectory quality, plan validity, token count, and end-to-end latency.

M10 exit: the benchmark quantifies where the slow-command/fast-executor
hierarchy fails and whether late-bound group plans degrade more gracefully than
individual exact assignments.

### Side tool — constrained LLM map generation (G0–G3 implemented)

- [x] Add a dedicated `snowgym/mapgen/` tool and root command using the exact
      `gpt-5.6-luna` model, strict Responses API output, environment-only
      credentials, `store: false`, and a two-attempt draft/repair ceiling.
- [x] Restrict generation to current map elements and effective attributes;
      reject ignored rotation, invalid footprints, blocked/overlapping spawns,
      disconnected engagement space, and maps above fixed Gym capacities.
- [x] Canonicalize accepted geometry and retain request/map hashes, source
      revision, provider request IDs, latency, token use, attempts, and complete
      validation history as immutable artifact lineage.
- [x] Run generated maps directly through the renderer-free environment without
      adding arbitrary file paths to the live server/Gym contract; emit paired
      normal/swapped-spawn evaluations and optional standard visual replays.
- [x] Add bounded suite generation with explicit development/evaluation labels
      and an explicit promotion command that updates browser and headless static
      catalogs only after validation.

Research boundary: generated artifacts enable validity/repair, diversity,
difficulty, side-bias, policy-transfer, and commander-stress experiments. They
do not count as evidence of policy generalization until frozen policies run on
sealed maps, and offline outcomes are not causal claims. New terrain types,
model-authored behavior flags, arbitrary server file loading, and pixels as
agent input remain out of scope.

Map-generator commits require provider-mock, geometry, artifact, replay,
promotion, full TypeScript, build, and Python gates. Live Luna generation is an
explicitly approved acceptance step and never part of `npm test`.

### Immediate commit sequence and gates

1. Training scaffold, trajectory schema, semantic-action inversion, and small
   scripted exporter.
2. Entity model, hybrid losses, deterministic BC trainer, checkpoint schema,
   evaluator, and learned replay.
3. Persistent multi-environment batch host, Python client, exact HTTP parity,
   and throughput benchmark.
4. Centralized PPO and frozen 1v1/3v3 curriculum gates.
5. Synthetic plan generator, directive encoder, and plan-conditioned ablation.
6. Unit-level PettingZoo adapter and MAPPO.
7. Fixed learned executor under commander baselines and latency/generalization
   evaluation.

Each commit is accepted only after targeted tests plus the full milestone gate:

```bash
npm test
npm run build
cd snowgym/python && .venv/bin/python -m pytest -q
```

Since 2026-09-13, `npm test` has one accepted known failure: the
`SelectiveRepair.test.ts` CLI preflight test's seed collision on
`training/runs/m7b_engage_r1n_b_v0/declaration.json`. The gate passes only if
that is the sole failure and names only that file (see `AGENTS.md` and the
R1n-b results erratum).

Training-package commits additionally run their own unit tests, deterministic
CPU smoke, dataset audit, and checkpoint/evaluation replay gate. Provider-backed
LLM calls, large dataset generation, and long training runs are opt-in and are
never part of the default deterministic suite.

## Guardrails

- No RL, Python, transport, or SnowGym imports from `src/`.
- Policies receive observations and return actions; they never hold engine
  entities or mutate world state.
- Canonical benchmark reward stays terminal-only until experiments explicitly
  choose a shaped reward.
- Every change outside `snowgym/` is recorded in `UPSTREAM_PATCHES.md`.
- Human browser behavior must continue to pass the existing test/build/smoke
  suite after each milestone.
- Repository agents should follow the root `AGENTS.md` and the repo-local
  `.agents/skills/snowgym/SKILL.md` guarded workflows.
