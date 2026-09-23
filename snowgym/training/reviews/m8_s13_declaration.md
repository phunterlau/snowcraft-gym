# M8-S13 — does the new decoder's gain hold against random Red?

Declared 2026-09-23, before running the evaluation. Ordinary autonomous evaluation of existing checkpoints; no
training. `autonomousQualificationEligible` stays false (development experiment).

## 0. Why, and why this is cheap

S12's results doc (`m8_s12_results.md`, "Consequence") ranked this first among three possible next steps precisely
because it needs no training: the 3 cohorts' `old`/`new` checkpoints already exist at
`runs/m8_s12_enemy_relative_throw_v0/cohort-{1,2,3}/{old,new}/fit-4.pt`, sealed and digest-verified. This step loads
them and runs one more paired-eval arm — `random` — that S12 did not test (S12's declaration scoped paired eval to
`("normal", "easy")`, matching S10/S11).

S4's declaration found that **random-init, untrained** blue is uninformative against random Red: a random-init
policy never approaches, so every episode times out with `L` near zero regardless of arm (`m8_s4_results.md`,
"Against random Red nothing happens"). That finding is about an untrained floor, not about a trained policy — it
does not by itself predict what a *competent* policy (old or new decoder, both of which reliably approach and
engage on normal/easy) will show against an opponent that rarely fights back. Two distinct outcomes are plausible
and neither is assumed:

- **(a) The new decoder still beats the old on random Red.** Evidence the representation change is a general
  improvement, not an artifact tuned to how scripted Red specifically behaves.
- **(b) Both decoders saturate near a ceiling on random Red** (a weak opponent should be easy for any policy that
  can already beat scripted-normal at 0.30–0.74). If so, this arm is uninformative for a *different* reason than
  S4's floor-saturation finding — a ceiling effect, not a floor effect — and that distinction will be reported
  explicitly rather than treated as a repeat of S4.

This step reports whichever of these (or a mixed pattern) actually occurs; no outcome is preferred over another.

## 1. What is reused unchanged

`FullAuthorityPolicyV1` and `FullAuthorityPolicyV1EnemyThrow` (both E3/S12, digest-pinned, no edits).
`roster_baseline.collect_cell`/`.summarize`/`.write_rows` (S4, unchanged) — `arm="random"` is already a first-class
value in `roster_baseline.ARMS` and `mixture_imitation.EVAL_ARMS` (`{"redController": "random"}`), so no new
scenario plumbing is needed. `enemy_relative_throw.paired_eval_seeds` (S12, unchanged) gives the same 100 worlds
(seeds 2100000–2100099) S4/S5/S7–S12 all used, now evaluated against a third arm. `full_authority_imitation.
paired_difference` (unchanged) computes the world-paired bootstrap mean-difference and 95% CI, the same statistic
S12's headline table used.

## 2. What this step adds

New module `options/enemy_relative_throw_random_red.py`. No existing module is edited. For each of the 3 cohorts
and both decoders, it loads `fit-4.pt` (the final fit already produced and archived by S12) into a fresh model
instance (`enemy_relative_throw.build_model`, unchanged, called with the same `destination`/`localRadius`/
`targetWorldSigma`/`initialPowerLogStd`/`initialOffsetLogStd` config S12 used, then `load_state_dict`), sets
`model.eval()`, and runs `rb.collect_cell(client, paired_eval_seeds(cfg), cfg, "random", model=model,
mode="deterministic", account=account)` — the identical call S12's `train_one` made for `"normal"`/`"easy"`, just
with the third arm and no training in between. Rows are paired by seed (not list position) between `old` and `new`
within each cohort, matching `death_rate_ppo.paired_analysis`'s convention.

## 3. Metrics and what "hold" means

Per cohort: `Δsuccess = paired_difference(new.success, old.success)`, `ΔL = paired_difference(new.unitsLostFraction,
old.unitsLostFraction)`, plus each decoder's raw success/`L`/timeout on the random arm. No gate is predeclared that
turns this into a pass/fail — S12 already established viability from normal/easy; this step is descriptive, adding
a third arm's evidence about generality. The one thing recorded as a genuine prediction, checked afterward: whether
the sign and rough size of `Δsuccess` on random tracks the same cohort's normal-arm `Δsuccess` from S12 (cohort 1
largest gain, cohort 3 smallest) — a sign-and-order check, not a magnitude-matching claim, since a different arm can
reasonably show a different absolute effect size even under outcome (a).

## 4. Archive and gates

New sealed archive `runs/m8_s13_random_red_eval_v0/`, one paired-eval cell per (cohort, decoder) under
`cohort-{1,2,3}/{old,new}/random/episodes.jsonl`, plus `result.json` per cohort with the paired-difference summary.
Declaration pins `runs/m8_s12_enemy_relative_throw_v0`'s manifest (the checkpoints being loaded must match what S12
sealed), E3 digests, and this module's own digest. Gates before running: client and training `pytest`, `npm run
build`, `npm test` (366/367, the one documented exception). Results in `reviews/m8_s13_results.md`.
