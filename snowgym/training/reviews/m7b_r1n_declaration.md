# R1n: a small complete autonomous skill (reviewer E3)

Commit this protocol and tested implementation before collection. This is
Experiment E3 of the reviewer handoff
(`refs/snowgym_fighter_rl_ppo_review_claude_opus_2026-09-12.md`, section 6):
can PPO learn fighter micro from scratch when the task is well posed --
full action authority, no runtime assistance, a persistent critic, and a
budget an order of magnitude larger than R1m-S9's? This is the first real
attempt at the long-open R1n autonomous-Engage milestone every prior R1m
note names. It does not authorize a checkpoint promotion or change the
M7c/autonomous-qualification gate; passing this diagnostic would support
further autonomous-skill work, not close R1 on its own.

## Deviations from the reviewer's E3 sketch, stated up front

- **1v1 only; 2v2 is deferred.** A measured throughput probe (64 worlds,
  a real forward pass in the loop, 1v1 roster) reached 2,911 decisions/second
  -- comfortably enough to run the full declared budget for 1v1 in this
  session. No equivalent measurement was taken for 2v2, and building,
  testing, and running two roster sizes in one pass was judged higher risk
  than shipping one complete, verified result. The scenario roster is a
  single config constant (`ROSTER = 1`); the same code runs 2v2 by changing
  it, as a follow-up not included in this declaration's budget or gates.
- **Network is a new module, not a reused production class.** `HybridActorCritic`
  (`ppo.py`) already implements a full hybrid action head with a learned
  log-std and an existing "global destination" decoder (`target = tanh(raw)`
  in arena-centered space) -- it would have been the natural base. But it
  has no egocentric/relative-frame option (`ModelConfig` has no such flag),
  and R1m-S12 found the egocentric frame, not attention, is what separated
  the winning arms from the current absolute path. Respecting that result
  meant building `FullAuthorityPolicy` (`executor/full_authority_ppo.py`)
  with `GeometryProbe(relative=True)`'s exact pair-feature transform copied
  in (not imported, for the same digest-safety reason as R1m-S11/S12), and
  writing a new `evaluate_latents`. It stores latents explicitly and decodes
  them separately (the convention `movement_ppo.py`/`recovery_ppo.py`
  already use), which keeps arms L and G identical everywhere except one
  `decode_move` method; `HybridActorCritic`'s own likelihood convention
  (recovering the raw latent by inverting the executed action) does not
  extend cleanly to a non-tanh decoder like L's, which is why that class
  was not subclassed. `ppo_loss` (`ppo.py`, generic and unit-masked) and
  `RoleAwareCentralCritic` (the same critic class `OptionCentralCritic`/
  `RecoveryCritic` already wrap) are reused unchanged.
- **No production teacher ceiling.** `recommend_movement`'s docstring states
  it mirrors the frozen teacher "for this open 5v5 scenario only"; there is
  no existing teacher policy for a bespoke 1v1 roster. The only control is a
  uniform-random legal-action floor. This is a real gap against the review's
  "teacher ceiling" control; it is not filled by extrapolating the 5v5
  teacher's numbers, which do not describe a materially different task.
- **A property of the frozen reward, discovered while testing, is recorded
  here rather than worked around:** `combat` and `shaping` (`tracker.py`)
  are both driven by health lost, not by position or distance. Before any
  damage occurs, shaping is exactly zero -- there is no dense signal
  rewarding closing distance, only the sparse terminal mission_reward
  (+1/-1) and the 0.1-weighted combat term once a hit lands. A from-scratch
  policy that cannot yet land a hit receives no gradient toward approaching
  at all. This is unchanged from the frozen reward decomposition the review
  and every R1m note require keeping; it is stated here because it bears
  directly on how hard "PPO from scratch" is for this specific task, and
  because it was verified empirically (full 200-decision episodes were run
  during implementation testing; a real terminal -1.0 fires correctly at
  timeout, confirming the reward path itself is not broken, only sparse
  before first contact).

## Task, authority, and network

1v1 Engage from reset against `RandomAgent` (`redController: "random"`),
frozen 100x80 arena, the unchanged `FrozenEngageTracker` success threshold
(target reduced to <=20% health) and executor reward
(`mission + 0.1*combat + shaping`). Horizon is the existing 200-decision
"engage" `OptionSpec`; there is no teacher completion-time distribution for
1v1 to derive a 1.5x-of-95th-percentile figure from, so the existing engage
horizon is kept as a stated simplification, not re-derived.

Full authority: `{NOOP, HOLD, MOVE, THROW}` (masked by `unit_action_mask`,
so an untrained policy cannot select an illegal action type), movement
destination, throw aim (always the global `tanh(raw)` decode in both arms;
only movement geometry differs), and throw power. No corrected shots, no
teacher MOVE, no BC anchor in either primary arm; `recommend_movement`
labels are not read anywhere in this run. `assistType: "none"`,
`autonomousQualificationEligible: false` (this is a diagnostic, not a
qualification attempt).

Shared network: one `FullAuthorityPolicy` per arm/seed, trained from a fresh
random initialization (`torch.manual_seed(trainingRng)`), no checkpoint
loaded. Egocentric features (own state, four relative-frame mean/max-pooled
entity types, plan directive/role state -- 213-wide, matching R1m-S12's
winning arm exactly) feed an action-type head, a move head, a throw head,
and a power head, plus learned `target_log_std`/`power_log_std`
(state-independent, shared by move and throw). A `RoleAwareCentralCritic`
is the value function.

## Arms (differ only in movement-destination decoding)

- **L (local):** an isotropic decode, `q = clamp(own + R*tanh(||z||)*(z/||z||)/scale, -1, 1)`,
  `R = 8` world units (squashing the latent's norm, not each axis
  independently, which would otherwise reach `R*sqrt(2)` on the diagonal --
  caught by a test before this was declared). `z ~ Normal(mean,
  exp(target_log_std))` with `target_log_std = log(2/R)` on both
  components, a world sigma of exactly 2 units through the decoder's
  near-origin slope.
- **G (global):** `q = tanh(z)`, the same decoder `HybridActorCritic`
  already uses. Its per-axis sensitivity at the origin is the arena's own
  anisotropic half-extent (50, 40), about 6x `L`'s R=8, so matching the
  same 2-unit world sigma needs a smaller, per-axis `target_log_std =
  (log(2/50), log(2/40))` -- computed from this fixed physical scale alone,
  never from returns (`calibrated_target_log_std` in
  `executor/full_authority_ppo.py`).
- **KL target re-derived for learned sigma, kept at 0.01:** the per-unit
  mean-KL stop used throughout R1m (`movementKlStop`) is
  `(Delta mu)^2 / (2 sigma^2)` for a fixed-sigma Gaussian mean shift --
  already scale-invariant in sigma. A KL of 0.01 implies a step of about
  0.14 standard deviations regardless of what sigma is, so the same
  numeric threshold was kept rather than replaced with an unjustified new
  number; PPO's exact log-ratio KL estimator (`ppo_loss`) captures sigma
  changes too, not just the closed-form mean-shift approximation this
  argument uses to size the threshold.

## Critic warm start and gate

Before any actor update: collect a fresh rollout under the random-init
policy (64 worlds, 256 decisions = 16,384 training-fold decisions) and a
disjoint-seed rollout (64 worlds, 64 decisions = 4,096 held-out decisions,
different seed range so no episode crosses the split). Fit the critic alone
(Adam, same learning rate, 10 epochs over the training fold) with no actor
gradient. Require held-out return R^2 >= 0.25; if not met, stop that
arm/seed before any actor update and record it as a critic-gate failure
rather than training from an unvalidated value function. Afterward, the
critic is optimized every update with its own optimizer, independent of the
actor's KL stop (both step together unless the actor's minibatch is
skipped by the KL check, matching the coupled-but-parallel schedule
`ppo_update` implements; the critic optimizer is never gated by the actor's
KL).

## Budget

64 worlds, 128 decisions/update (8,192/update, at the review's declared
minimum), 244 updates: 1,998,848 decisions per arm/seed, plus its critic
warm start (20,480) and final development evaluation (100 seeds,
deterministic, <=200 decisions each). Two arms x three training RNGs
(96001/96002/96003) x (~1.999M + ~0.02M + <=0.02M) plus one shared
uniform-random floor (100 seeds): declared cap 12,500,000 simulator
decisions, enforced by an `account()` guard. At the measured 2,911
decisions/second (64 worlds, real inference), the full budget is
approximately 72 minutes of simulation.

## Measurements

Final-checkpoint deterministic success, mean progress, and rejected-action
rate on 100 development seeds (600000-600099) per arm/seed; the shared
floor's success on 100 separate seeds (610000-610099); each arm/seed's
critic warm-start held-out R^2; per-update KL-stop fraction and mean
reward from training history. No teacher-relative measurement is available
(see deviations); paired comparisons are arm-vs-arm and arm-vs-floor only.

## Prediction and falsifiers (reviewer's; ceiling comparisons removed per the deviation above)

Prediction: L reaches >=70% deterministic success in at least 2 of 3 RNGs;
G trails L by >=20 points at matched budget; critic held-out R^2 >= 0.3
within the first 10% of training (by the time roughly 200,000 decisions
have been collected); KL stops bind on a minority of updates.

- **L ~= G, both succeed:** movement geometry was not the key; keep global
  destinations, since they need no radius hyperparameter.
- **Both < 30% with a healthy critic (R^2 above the 0.25 gate throughout):**
  PPO from scratch is inadequate at this scale under this reward. Given the
  reward-sparsity property recorded above, this is a live possibility, not
  a formality; if it happens, the next proposal is reward shaping toward
  approach/contact (a change to the frozen reward, requiring its own
  declaration) or a BC/DAgger-first curriculum, not a noise or KL sweep.
- **L succeeds only stochastically (deterministic eval well below the
  stochastic training-time success rate):** qualify in the deployed
  (stochastic) execution mode rather than claiming deterministic mastery.

## Stopping rule

Fixed budget, three training RNGs run regardless of the first seed's
result (unlike S9/S11's conditional-replication pattern, since the review
declared "3 training RNGs" outright for E3, not a conditional trigger), and
final-checkpoint gates only -- no intermediate-checkpoint selection. The
critic gate is the only in-run stop; an arm/seed that fails it is recorded,
not retried with a different warm-start budget.

No provider calls, browser input, or protocol changes. 2v2, a teacher-
relative measurement, and reward reshaping are explicitly out of scope for
this declaration.

Before implementation and results commits: targeted
`full_authority_ppo`/`full_authority_train` tests, `npm test`,
`npm run build`, Python client tests, Python training tests.
