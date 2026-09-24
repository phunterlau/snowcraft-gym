# M8-S14 results: PPO saturates scripted-normal on fresh worlds, mostly by repairing the initializer's own deterministic-vs-stochastic gap

Collected 2026-09-23 from `031485a` (probe results, declaration §17, pinned `sigmaScaleByCohort`/
`actorLearningRate`, committed before this collection). Declaration: [m8_s14_declaration.md](m8_s14_declaration.md)
(§§1–15 the original design, §16 the implementation amendments, §17 the probe results). Archive:
`runs/m8_s14_ppo_continuation_v0/` (sealed and verified). 4,066,282 simulator decisions against the 8,870,400
training-budget bound; wall clock 1h34m (18:48–20:22), close to §17's ~1.9h estimate. `withinTrainingBudget:
true`. `autonomousQualificationEligible` stays false.

**Correction to §17: the probe decision count was never actually computed.** §17 stated "45,864" without summing
the real per-episode decision counts. The true figure, summed from the archived `episodes.jsonl`/`rolloutRows`
after the fact: 38,974 (σ probe, 6 cells) + 18,727 (lr-probe rollouts, 3 cohorts) = **57,701**, still comfortably
under the 300,000 probe cap. Corrected here, not silently edited in §17.

## Headline: the primary test passes, but the size of the deterministic effect needs the stochastic numbers next to it

**Primary (declaration §10):** `ΔL` (mean units-lost-fraction, final − initializer, cohort-averaged, deterministic,
world-paired) = **−0.628**, 95% CI **[−0.646, −0.609]**. `ΔL ≤ −0.05` and the interval excludes zero: **order 2,
"improved," passes.** Non-inferiority holds (`S⁻ = 0.412 > −0.05`, `T⁺ = 0.0 < 0.05`). All three cohorts
individually show lower `L` (`cohortsWithLowerL = 3`). `decisionRules.outcome = "improved"`.

**But most of that number is the initializer's own deterministic policy being weaker than its stochastic one —
already visible in S14's own probe stage (§17) — not a comparably large jump in the trained policy's ceiling.**
Every initializer's deterministic success is well below its own stochastic success (0.780 vs 0.887, 0.608 vs
0.963, 0.300 vs 0.762) — the same pattern §17 flagged as unpredicted and left unexplained. PPO training moves
the **deterministic** policy furthest because that is where the initializer had the most room. Recomputed from
the archived evaluation files (declaration §9 says both modes get reported; the code that ran only wrote the
deterministic pairing to `report.json` — fixed here by computing the stochastic pairing directly from the sealed
`episodes.jsonl` files, no new simulation needed):

| Cohort | `ΔL` deterministic | `ΔL` stochastic | `Δsuccess` deterministic | `Δsuccess` stochastic |
| --- | ---: | --- | ---: | --- |
| 1 | −0.427 [−0.460, −0.393] | −0.156 [−0.193, −0.120] | +0.220 [0.180, 0.260] | +0.090 [0.058, 0.125] |
| 2 | −0.663 [−0.693, −0.634] | −0.088 [−0.117, −0.058] | +0.393 [0.345, 0.440] | +0.030 [0.010, 0.050] |
| 3 | −0.794 [−0.826, −0.763] | −0.321 [−0.365, −0.278] | +0.700 [0.655, 0.745] | +0.223 [0.180, 0.268] |
| **cohort-averaged** | **−0.628 [−0.646, −0.609]** | **−0.188 [−0.210, −0.166]** | **—** | **+0.114 [0.094, 0.134]** |

**Both are real** — every stochastic interval excludes zero too, so this is not "deterministic improved, stochastic
didn't." But the deterministic number is roughly 3× the stochastic one, and the framing that matters is: PPO
produced a real, positive, statistically clear gain under genuine exploration noise (`ΔL ≈ −0.19` cohort-averaged),
and a much larger gain in the specific argmax path, largely because that path started from a worse place than the
stochastic policy the imitation initializer actually represents.

## Absolute numbers: scripted-normal is now saturated, in every cohort, deterministically

| Cohort | Initializer det. success / `L` | Final det. success / `L` | Initializer sto. success / `L` | Final sto. success / `L` |
| --- | --- | --- | --- | --- |
| 1 | 0.780 / 0.427 | **1.000 / 0.000** | 0.887 / 0.203 | 0.978 / 0.047 |
| 2 | 0.608 / 0.663 | **1.000 / 0.000** | 0.963 / 0.138 | 0.993 / 0.050 |
| 3 | 0.300 / 0.794 | **1.000 / 0.000** | 0.762 / 0.374 | 0.985 / 0.053 |

Every cohort's final deterministic policy scores 400/400 on the fresh evaluation split (2600000–2600399), zero
units lost, zero team wipes, zero timeouts, zero rejected actions. This was checked before being trusted: 400
distinct seeds confirmed per cohort; the trained policy wins in *fewer* decisions than the initializer
(77–80 mean decisions vs. 98–117), not more, ruling out a stalling/timeout-exploiting explanation; the initializer's
own deterministic numbers (0.780/0.608/0.300) closely reproduce S12's own archived normal-arm success
(0.74/0.57/0.30) on a different world split, a real cross-check that the setup is doing what it says.

**What this does and does not establish.** PPO trained against the same fixed scripted-normal controller it is
evaluated on (declaration §2's own scoping — the held-out axis here is world seeds, not opponent identity).
**Zero losses across 1,200 evaluation episodes (3 cohorts × 400 worlds) against a deterministic, rule-based
opponent is consistent with genuinely good play or with a reliable exploit specific to that controller's fixed
logic, and this data cannot distinguish the two.** The honest claim is "solves scripted-normal on fresh worlds
deterministically," not general competence. Scripted-normal is now saturated (1.000/0.000 in all three cohorts) —
it can no longer measure any further improvement from here, only regression.

## Predictions, scored (declaration §10) — 2 of 4 held

- **`ΔL`'s point estimate is negative and no larger in magnitude than −0.30 (calibrated against S12's own largest
  per-cohort architecture-change `ΔL` of −0.50): FAILED.** The deterministic cohort-averaged value, −0.628, is
  larger in magnitude than the effect the decoder redesign itself produced. (The stochastic value, −0.188, would
  have cleared this bound — but that is context, not a rescue: the prediction named the deterministic metric that
  §10 itself declared primary, and it is scored against that.)
- **The success difference's lower bound clears the non-inferiority guard: held.** `S⁻ = 0.412`, well above the
  −0.05 margin.
- **The approximate-KL stop binds before the last epoch in most updates: FAILED.** Computed from
  `training-history.json`: the stop fired in 14/200 (7.0%), 4/200 (2.0%), and 12/200 (6.0%) of updates per
  cohort — not "most." `actorOptimizerSteps` mostly ran the full available count (up to 56 per update) rather
  than being cut short. The lr-probe's own selected value (1e-5) was, if anything, more conservative than the
  200-update budget needed — training could plausibly have used a less conservative learning rate or fewer
  updates without landing in "no effective training."
- **Median final anchor KL exceeds 0.01 (order 4, "no effective training," does not fire): held.** Final anchor
  KLs are 0.098, 0.117, 0.180 — an order of magnitude above the 0.01 floor.

**Learning curve shape, read directly from the history (not predicted, reported because the KL-stop miss above
raises the question of what 200 updates actually bought):** rollout success (out of 64 episodes per update)
reaches ≥62/64 by update 2 (cohort 2), 11 (cohort 1), or 26 (cohort 3) — a small fraction of the 200-update
budget. The anchor KL to the initializer grows quickly over the first 20–40 updates (from ~0 to roughly 0.05–0.1,
cohort 3 up to ~0.16) and then **oscillates in a noisy band rather than continuing to grow** for the remaining
~160–180 updates (cohort 1: 0.079→0.045→0.076→0.093→0.085→0.07→0.098→0.077→0.088 at every 20th update; similar
pattern in cohorts 2–3). Most of the 200-update budget looks like it was spent confirming an already-converged
policy, not continuing to move it.

## Critic

Re-warm-started at the selected σ=0.5 against scripted-normal (declaration §2), not S12's mixture-trained critic.
Predictive R²: 0.425, 0.429, 0.332 (cohorts 1–3), all gates pass, all broadly consistent with S12's own
new-decoder critic R² (0.401/0.387/0.326) despite the different warm-start distribution. Per S12's finding, this
is **not** read as a quality signal for the actor — advantages are pure Monte Carlo (`gaeLambda = 1`), so a
modest-R² critic adds variance to training, not bias, and says nothing here about whether the *trained* policy is
good.

## Provenance caveat

The real training root's `declaration.json` does **not** pin the probe root's (`runs/m8_s14_probe_v0/`) manifest
digest — unlike S13's pinning of S12's manifest before loading its checkpoints. `sigmaScaleByCohort` and
`actorLearningRate` are trusted here only through `configuration()`'s pinned values and §17's documentation of
where they came from, not a cryptographic link between the two sealed archives. Noted as a gap, not fixed
retroactively (the training root is already sealed).

## Cost

Probe stage: 57,701 decisions (corrected above) against a 300,000 cap. Training stage: 4,066,282 decisions
against the 8,870,400 bound (46% of budget), 1h34m wall clock for all three cohorts sequentially — close to
§17's ~1.9h estimate, on the low side. S4's scripted teacher scored 399/400 against normal Red, but on a
different world split (400000–400399 vs. this step's 2600000–2600399) — not directly paired, mentioned only as
a rough scale reference, not a comparison.

## What this establishes and what it does not

**Establishes:** the enemy-relative throw decoder, already viable and reliable after S12/S13, can be pushed to
saturate scripted-normal deterministically on fresh worlds via a bounded KL-anchored PPO continuation, in all 3
independent cohorts, with a real (not just deterministic-argmax) improvement under genuine exploration noise too
(`ΔL ≈ −0.19` cohort-averaged, stochastic, CI excluding zero). The lr-probe methodology (fixed after two advisor
rounds, §16) produced a working, shared, reproducible learning rate across all three independent cohorts.

**Does not establish:** general competence, since training and evaluation share the same fixed opponent
identity and only world seeds are held out (declaration §2's own scoping, reaffirmed here); whether the
saturation reflects genuinely strong play or a controller-specific exploit — this evaluation cannot tell the two
apart; whether 200 updates was the right budget (the learning curve suggests most of the movement happened in
the first ~30, with the remainder producing little further change in the anchor KL); anything about a `hard`-arm
generalization check (declaration §2 scoped this out deliberately, pending its own teacher-achievability check
first).

## Consequence

1. **Evaluate the existing final/initializer checkpoints against `random` and `easy` Red.** No training needed —
   both arms were in the imitation mixture PPO never trained against directly (declaration §2's rollout-arm
   scoping was normal-only), so this is the cheapest available check for whether PPO's gains are specific to
   scripted-normal or transfer to opponents already in the decoder's training distribution. Needs its own small
   declaration (mirroring S13's role for S12), but no simulator budget beyond the eval itself.
2. Only after that, a `hard`-arm check — genuinely untouched by anything in this track, requiring its own
   teacher-achievability measurement first (declaration §2).
3. A fresh replication (new training RNGs, new untouched split) before treating this cohort-averaged effect size
   as precise, given it is still n=3 and the deterministic/stochastic gap turned out to matter more than expected.

None of these is authorized here.
