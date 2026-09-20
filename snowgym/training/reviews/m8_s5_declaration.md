# M8-S5 — 3v3 teacher imitation as the initializer (mixture recipe), evaluated world-paired against S4

Declared 2026-09-20, before any collection. Collect-and-archive step. No PPO, no
checkpoint promotion, no qualification claim; `autonomousQualificationEligible`
stays false.

## 0. Why

[S4](m8_s4_results.md) showed the 3v3 teacher is achievable (≥ 0.998 success) and a
random-init learner has no signal (0/100 everywhere, wiped by scripted-normal, no
contact with random Red). From-scratch PPO therefore has nothing to climb, so the
R1n route applies: teacher imitation as the initializer. R1n-h found that a mixture of
random and scripted-easy opponents transferred to scripted-normal on average (+38
points) while the three optimizer seeds spread widely (0 / 100 / 14%), and the R1n-i
diagnostic left a contact failure unexplained. S5 asks the roster-3 version of the
one question that decides whether a PPO step is worth declaring:

**Does R1n-h's mixture-imitation recipe, run unchanged at 3v3, produce initializers that
get close to the teacher on the S4 paired worlds, on every seed?**

It is a development experiment, not an attribution. There is no single-opponent control
at 3v3 (R1n-h's condition C) and no claim about *why* any seed succeeds or fails.

## 1. Design

**Reuse, not new training code.** `mixture_imitation.train_condition` for condition M
runs unchanged under an outer roster-3 scenario override (probed before declaring: tiny end-to-end run, and at the real 200-decision horizon an
8-world round-zero teacher collection gave 2,490 rows including 113 THROW labels, 4.5%,
against R1n-h's 1v1 round zero at 4.2%, with the imitation fit and critic warm start running). Nothing existing is edited. The round-zero teacher data, DAgger rounds, fit
schedule, optimizer seeds (97101–97103), architecture (`FullAuthorityPolicyV1`,
global decoder) and every hyperparameter are `mixture_imitation.configuration()`'s, so
the only change from R1n-h's M is the roster.

**Training seed bands (fresh, audited against every tracked file; the surrounding bands
contain unrelated stray integers but no overlap):**

- round seeds M: 3300000 + 1000 × round (5 rounds, 128 episodes each, half random, half
  scripted-easy Red);
- critic train 3310000, critic held-out 3315000;
- `mixture_imitation`'s own development evaluations: random 3320000, easy 3321000,
  normal 3322000, stochastic 3323000 (100 worlds each), floor base 3324000 (unused).

**Paired evaluation (the S5 measurement).** Each seed's final policy
(`fit-4.pt`, no selection among fits or seeds) is run deterministically on the S4 paired
worlds, seeds **2100000–2100399**, 400 worlds against each of the three native Red arms.
These worlds were used for S4's teacher and floors and are never used for training,
selection or tuning. Pairing with the archived S4 teacher episodes is therefore by world.

**Authoritative evaluation.** The 400-world paired evaluation above is the S5 result.
`mixture_imitation.train_condition` also runs its own 100-world development evaluations
(bands 3320000-3323000) and a per-seed label-error read; those are development-only, are
archived, and are not quoted as the result. The paired evaluation collects in blocks of 50
worlds, the same chunking S4's teacher used (S5's inherited default is 64), pinned as
`pairedBlockWorlds`, so batched floating-point behavior matches.

**Metrics** are S4's, unchanged: success, units lost fraction `L` (primary loss metric,
mean, world-paired), team wipe, timeout, mean decisions, rejected-action rate. Per seed
and arm: Wilson intervals for proportions; world-paired differences to the S4 teacher
(learner minus teacher, world bootstrap 10,000 resamples, seed 973001) for success and
`L`. Per-seed values are always reported; no interval pools the three seeds, and the
seed mean is descriptive only. `L` is read next to success and mean decisions (fast
completion lowers exposure).

Budget: bound 1,523,200 decisions, hard cap 2,000,000 (aborts unsealed if exceeded; the
cap is deliberately widened to about 30% above the bound because an abort loses the whole
run, and S4 used 70% of its bound). S4's throughput was about 200 decisions per second, so plan on roughly 1.5–2 hours of
wall time; the run is a single background process that refuses to overwrite.

## 2. Pre-declared readings (fixed now)

| Reading | Rule | If true |
| --- | --- | --- |
| **Viable initializer** | Seed-mean success against scripted-normal ≥ 0.25 on the paired worlds | Declare a bounded PPO continuation from **all three** initializers (no selection), matched controls and a fresh eval allocation |
| **Reliable across seeds** | Every seed has success ≥ 0.25 against scripted-normal | Otherwise the run is reported as viable-but-unreliable, and the PPO step must be designed around the spread |
| **Not viable** | Seed-mean success against scripted-normal < 0.25 | Diagnose (R1n-i style) before any PPO |

0.25 is R1n's own "success minimum fraction" (`successMinFraction`), reused, not tuned to
this data. For orientation only, R1n-h's 1v1 M seeds scored 0 / 100 / 14% on scripted-normal;
S5 is not paired with that and does not test against it.

Critic: `mixture_imitation` fits a warm-start critic per seed. Its R² and gate flag are
reported per seed as the roster-3 re-check that S4 required (`assignedLivingFraction` is
now graded). They do not decide viability. If the gate fails at 3v3, the PPO step's critic
design is revisited in its own declaration.

## 3. What this does and does not decide

It decides whether the imitation initializer is worth building on at 3v3 and how reliable
it is across optimizer seeds. It does not decide that 3v3 Engage is learnable by RL, does
not compare the mixture to a single-opponent control, does not explain any seed's
behavior, and says nothing about command-following, roles, or unit-local sensing. The
optimizer seeds share one round-zero teacher dataset, so they are not independent
training cohorts (the 2026-09-18 review's point); the spread is a lower bound on
retraining variability.

## 4. Verification and archive

New module `options/roster_imitation.py` and tests; no existing module is edited. The
declaration pins the S4 archive's verified manifest, the E3 digests, and the source
digests of `mixture_imitation.py`, `full_authority_imitation.py`,
`full_authority_train_v1.py`, `roster_baseline.py`. Gates before collection: client and
training `pytest`, `npm run build`, `npm test` (366/367, the one documented exception).
Sealed with `death_rate_ppo.seal`, verified with `verify_sealed`; results in
`reviews/m8_s5_results.md`. A failed reading is a finding, not patched around; the
failure policy from S4 applies (the run completes, seals and reports; the response is a
new declaration).
