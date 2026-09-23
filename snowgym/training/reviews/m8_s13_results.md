# M8-S13 results: an in-distribution consistency check, not a generalization test — and a declaration correction

Collected 2026-09-23 from `291c5e7` (declaration and implementation committed before running). Declaration:
[m8_s13_declaration.md](m8_s13_declaration.md). Archive: `runs/m8_s13_random_red_eval_v0/` (sealed and verified,
`sha256:d0ec9f76…`). No training; loaded S12's 6 already-archived checkpoints
(`runs/m8_s12_enemy_relative_throw_v0/cohort-{1,2,3}/{old,new}/fit-4.pt`) and ran one paired-eval arm, `random`,
that S12's paired eval did not test. 96,474 simulator decisions against a 120,000-decision bound (`pairedEvalWorlds`
100 × `optionHorizon` 200 × 3 cohorts × 2 decoders), under the 200,000 cap. `autonomousQualificationEligible` stays
false (development experiment).

## Correction to the declaration: `random` is not a held-out arm here

The declaration framed this as testing whether the new decoder's advantage "holds against an opponent type neither
decoder's evaluation touched" — treating `random` as unseen, the way S4 treated an untrained floor's exposure to
random Red. **That premise is wrong, caught in review before this doc was committed.** `enemy_relative_throw.
round_seeds` splits every training round's seeds in half: `{"random": seeds[:half], "easy": seeds[half:]}`
(`options/enemy_relative_throw.py:83-87`), and `mixture_imitation.collect_mixture` collects both halves under
`MIXTURE_ARMS` — `{"random": {"redController": "random"}, "easy": ...}` — for every round, including round zero.
So **half of S12's training data for every cohort and both decoders was literally random-Red episodes.** `random`
is in-distribution, not held-out. Scripted-normal is the only arm in this whole track that neither decoder ever
saw during training (`MIXTURE_ARMS` has no `"normal"` key) — and S12 already tested that. This correction lives
here, not as an edit to `m8_s12_declaration.md`, which is pinned and unedited; the declaration for this step is
also left unedited, since its own digest is pinned in `declaration.json` — this results doc is the record of what
was actually measured.

Given that, the right description of this step is: **does the paired new-vs-old advantage stay consistent on an
arm the decoders trained on but were never evaluated against, across all 3 independent cohorts?** — a weaker,
still non-trivial question than "does it generalize."

## Headline: the advantage is consistent on this in-distribution arm too, in all 3 cohorts

| Cohort | New success | Old success | Δ success | 95% CI | New `L` | Old `L` | Δ `L` | 95% CI |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 0.77 | 0.16 | **+0.61** | [0.50, 0.72] | 0.157 | 0.763 | **−0.61** | [−0.69, −0.52] |
| 2 | 0.87 | 0.21 | **+0.66** | [0.56, 0.75] | 0.080 | 0.640 | **−0.56** | [−0.64, −0.48] |
| 3 | 0.79 | 0.11 | **+0.68** | [0.58, 0.78] | 0.140 | 0.833 | **−0.69** | [−0.77, −0.61] |

All three 95% world-paired bootstrap intervals exclude zero. Neither decoder collapses to a floor or saturates a
ceiling (new 0.77–0.87, old 0.11–0.21), so the comparison is not degenerate — it just is not a held-out-opponent
test.

## The predeclared sign-and-order check: sign held, order is not distinguishable

The declaration predicted checking whether `Δsuccess`'s ordering across cohorts tracks S12's held-out normal-arm
ordering (cohort 1 largest, cohort 3 smallest), as a sign-and-order check, not a magnitude-matching claim. The
sign held in all 3 cohorts. The order does not track (random's gains are +0.61/+0.66/+0.68, nearly the reverse of
normal's +0.70/+0.57/+0.14) — but the three CIs here are [0.50,0.72], [0.56,0.75], [0.58,0.78], which overlap
almost completely. **The correct reading is that the order is not distinguishable at n=100 worlds per cohort, not
that it is reversed.** What is a genuine, checkable contrast: the same cohort-3 checkpoint scores +0.68 here
(in-distribution) and only +0.14 on S12's held-out normal arm — a real within-checkpoint gap between the arm it
partly trained on and the arm it never saw, worth keeping in mind for any future held-out claim.

## The old decoder's random-arm numbers, checked against S12's normal-arm numbers directly (both corrected)

Old-decoder success on `random` (0.16/0.21/0.11) is **higher than its held-out normal-arm success in S12**
(0.04/0.00/0.16) in 2 of 3 cohorts, not comparable-to-or-below as an earlier draft of this section incorrectly
stated before verification. Team-wipe tells a consistent story: old-decoder wipe rate is **lower** on random
(0.64/0.48/0.73) than on normal (0.95/1.00/0.84, from `runs/m8_s12_.../cohort-*/old/result.json`) in all 3
cohorts — the opposite of an earlier draft's claim. Both directions make sense given the correction above: the old
decoder trained on random-Red data too, so it is not meeting an unfamiliar opponent here; scripted-normal remains
harder for it than the opponent type baked into half its own training mix.

## Verification: the checkpoint-loading path is faithful

Reloaded cohort-1's `new` checkpoint via `load_checkpoint` and ran it against 20 of S12's own held-out normal-arm
seeds; per-seed success matched S12's archived `paired-eval/normal/episodes.jsonl` exactly (0 mismatches of 20).
`build_model(seed=0)` + `load_state_dict` reproduces the trained policy exactly, not just the model class.

## What this establishes and what it does not

**Establishes:** across all 3 independent training cohorts, the enemy-relative throw decoder's advantage over the
absolute-coordinate decoder is consistent on an in-distribution arm neither decoder's paired eval measured before,
with no collapse or ceiling saturation. The checkpoint-loading path used here is verified faithful to S12's
originals.

**Does not establish:** generalization to an unseen opponent — S12's held-out scripted-normal arm remains the only
evidence of that, and it is the one already reported there. It also does not establish why cohort 3's held-out gap
(+0.68 in-distribution vs. +0.14 held-out) is larger than the other two cohorts' gaps, or anything about PPO, more
cohorts, or the critic-R² reversal flagged in S12 (untouched here — no critic fit happens in an evaluation-only
step).

## Gates

Targeted tests (6) passed before the run. Client `pytest` (all passing) and training `pytest` (all passing, no
collisions) ran once before the run and again **after** collection and archiving — the second pass specifically to
catch the S10/S11-style latent seed-collision trap, where a new archive appearing on disk can trip another step's
exclusion-set scan; no new collision appeared, and this step declares no new seed bands (it only reuses S4/S5/S12's
shared evaluation band), so no exclusion list needed updating. `npm run build` passed cleanly. `npm test` (run
after archiving) is 366/367 — the one accepted, pre-existing failure (`SelectiveRepair.test.ts`'s `trainSeedBase
630000` collision, documented in `AGENTS.md`/`PLAN.md` since before this step), and no new failure.

## Consequence

This closes the cheapest of S12's three ranked next steps, correctly reframed as an in-distribution consistency
check rather than a generalization test. S12's two remaining ranked options are unchanged and still unauthorized
here: extending toward 5+ independent cohorts to bound the (held-out, normal-arm) effect-size range, and — only
after that — a bounded PPO continuation from this initializer, needing its own declaration given the critic-R²
caveat. A genuine held-out-opponent generalization check would need an arm outside `MIXTURE_ARMS` entirely that
S12 also didn't evaluate — scripted-normal already is that arm, and is already reported.
