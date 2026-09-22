# M8-S10 — autonomous repair attempt 1: upweight the throw-aim loss term (3v3 imitation)

Declared 2026-09-22, before any training. **This is the first autonomous step since S5** — the resulting policies are
not teacher-assisted and their numbers may be reported as ordinary (if still weak) learner results, unlike every
S7–S9 substitution number. No checkpoint is promoted or qualified; `autonomousQualificationEligible` stays false
(this is a development experiment, not an R1 qualification attempt).

## 0. Why this specific repair, and what was checked before proposing it

S6–S9 located the S5 3v3 imitation initializers' failure against scripted-normal in the throw channel, specifically the
throw target, with the failure mode differing by optimizer seed (97102: which enemy; 97103: angular precision; 97101:
neither alone, more consistent with S8's `timing` finding). S9's consequence section proposed several autonomous repair
candidates; this step is the cheapest of them: increase the imitation loss's weight on throw aim.

**Read before proposing, not assumed:** `full_authority_imitation.imitation_loss` (unedited; read directly). Its
`total` is `type_loss + heading + endpoint + aim + power` — **five terms, each already an isolated mean over only its
own applicable rows** (e.g. `aim` is the mean cosine loss over THROW-labelled rows alone), summed with an implicit
coefficient of 1 each. This is not the "minority class diluted by a shared average" mechanism a first guess might
assume; there is no shared average to dilute. The actual lever is the coefficient on one term in that sum.

**Checked in S5's own archived fit histories** (`condition-M/seed-*/fit-history.json`, seed 97101's final logged step of
each fit): `throwAim` is not converging to a small residual the way `power` and `moveHeading` do — it is 0.03 / 0.07 /
0.17 / 0.09 / **0.24** across the five fits (a net increase over training, not a decrease), while `power` ends at 0.02 and
`moveHeading` at 0.12. `moveEndpoint` is the single largest term (0.82) but S8 found movement is not the located
cause (its `move` arm was weak and used a much larger, blunter substitution than the throw arms; S9 did not test
movement at all). The choice to weight `aim`, not `moveEndpoint`, follows the causal diagnosis (S6–S9), not the raw
loss magnitude, and this is stated explicitly because the two point in different directions.

**What this predicts and does not:** raising `aim`'s coefficient gives the optimizer more gradient pressure to reduce
that term specifically. It does not by itself resolve seed 97101's deficit, which S8 found is not primarily an aim
problem; a null result for 97101 would be consistent with the prior diagnosis, not contrary to it. This is stated as a
prediction, not assumed after the fact.

## 1. Design

**Reuse.** `full_authority_imitation.py`, `mixture_imitation.py`, `full_authority_train_v1.py`,
`roster_baseline.py`, `roster_imitation.py` are read, not edited (digests pinned). `mi.imitation_loss`'s forward
pass, masks, and every term's computation are reused unchanged via direct import; only the linear combination that
becomes `total` for backpropagation changes, in a new function in a new module. `mi.collect_mixture`,
`mi.round_seeds_by_arm`, `mi.warm_start_critic_mc_mixture`, `mi.critic_fold_seeds_by_arm`, `fi.Aggregate`,
`fi.part_summary`, `fi.imitation_parameters`, `fi.label_error` are reused unchanged. The training orchestration
(`train_condition`-equivalent) is reimplemented in the new module because `mi.train_condition` hardcodes a call to
`fi.fit`, which itself hardcodes a call to the unweighted `fi.imitation_loss`; neither accepts a substitute loss.

**Change.** `total = type_loss + heading + endpoint + aimWeight × aim + power`, with **`aimWeight = 5`**. This is a
single value, not a sweep: chosen only so `aim`'s typical gradient contribution is comparable to or larger than
`moveEndpoint`'s at the magnitudes seen in S5's fit histories (roughly 5 × 0.1–0.2 ≈ 0.5–1.0, versus `moveEndpoint`'s
0.8–2.0), without the change also being reported as if it were tuned to a result — it was fixed before any training run
under this repair.

**Held fixed versus S5, for a controlled per-seed comparison:** the same three optimizer seeds (97101, 97102, 97103,
so initial weights are bit-identical to S5's before training begins), the same architecture, the same fit/round/epoch
counts, the same critic warm-start procedure and gate. **Changed versus S5:** the loss weighting (above), and fresh
training-data seed bands (round-zero and DAgger-round worlds, critic-fold worlds) — an independent training-world
cohort, not S5's exact data, per the 2026-09-18 review's recommendation to separate optimizer-seed identity from
training-data identity. Any difference from S5 is therefore attributable to the loss change **and** the fresh data
draw together, not to the loss change in isolation; this is stated as a limit, not resolved here.

**Scope reduction versus `mi.train_condition`** (to bound cost and wall time): only the scripted-normal development
evaluation and its `label_error` read are collected internally (label_error directly measures whether the intervention
reduced throw aim error, the proximal target of the change); the scripted-easy, random, and stochastic-random internal
development reads that `mi.train_condition` also computes are skipped. They are not needed for this step's questions
and are not part of R1n-h's condition-C comparison, which does not exist here either (no single-opponent control; this
step's baseline is S5 itself).

**Fresh training seed bands** (audited against every tracked file in the repository, no overlap): round seeds
4200000 + 1000 × round (5 rounds); critic train 4210000, critic held-out 4215000; development-normal eval 4220000
(100 worlds).

**Baseline values, computed now from the archive (not assumed).** S5's headline normal success (400 worlds) was
0.005 / 0.000 / 0.000 for seeds 97101 / 97102 / 97103; on the 100-world subset used here it is 0.000 / 0.000 / 0.000
(97101's single 400-world win falls outside the first 100). S5's easy success on the 100-world subset is
0.74 / 0.87 / 0.89, close to its 400-world values of 0.73 / 0.885 / 0.907. These subset numbers, not the 400-world
headline, are what "gain vs S5" is measured against.

**Paired evaluation (the authoritative measurement).** Each seed's final policy (`fit-4.pt`) is evaluated
deterministically on the **same 100 worlds** S5, S7, S8, and S9 all used, seeds **2100000–2100099**, against
scripted-normal and scripted-easy Red (random Red is skipped: S4 already showed it is uninformative for a
from-scratch-level policy, and S7/S9 found it noisy and inconclusive for these learners). This is paired against
**S5's own archived paired-eval rows for the same 100 worlds** (a subset of S5's full 400-world evaluation) — an
autonomous-vs-autonomous comparison, exactly analogous to S5's own comparison against the S4 teacher. No new
evaluation-world seeds are declared.

**Budget.** Bound: round-zero 25,600 + DAgger rounds 307,200 + development-normal eval and label-error 60,000 +
critic warm start 230,400 + paired eval (2 arms × 100 worlds × 200 × 3 seeds) 120,000 = **743,200**, hard cap
950,000 (aborts unsealed if exceeded). Estimated 1–1.5 hours of wall time by analogy with S5; the run is a single
background process, waited on by PID, that refuses to overwrite.

## 2. Pre-declared readings (fixed now)

Success and units-lost fraction `L` use S4/S5's definitions. The primary comparison is **success minus S5's archived
success, per seed, world-paired**, on scripted-normal.

| # | Prediction | Rule |
| --- | --- | --- |
| P1 | Seeds 97102 and 97103 (the two S9 identified as aim-related) each show a material gain on scripted-normal | success ≥ S5's archived success + 0.10, for **both** seeds |
| P2 | Scripted-easy does not regress materially for any seed | success ≥ S5's archived easy success − 0.15, for **all three** seeds |
| P3 | The critic warm-start gate still passes for all three seeds | `gatePassed` true for all three (re-checked at 3v3, since this is a fresh check, not assumed from S5) |
| P4 (descriptive, not pass/fail) | Seed 97101's scripted-normal gain is smaller than the smaller of 97102's and 97103's gains | consistent with S8's finding that 97101's deficit is not primarily aim; reported either way, not scored as a failure |

`label_error`'s `throwAimHeadingErrorDegrees` is reported per seed as a proximal check (did the target metric move)
but is not itself a pass/fail gate.

**Pre-declared contingency if P1 fails.** A null or mixed result cannot, by itself, distinguish "the loss
change does not help" from "this fresh training-data draw was worse than S5's." The cheap next check, named now
rather than chosen after seeing the result, is a second run with `aimWeight = 1` on the **same** fresh seed bands as
this run — holding the data draw fixed and removing the loss change, isolating it from the data-draw confound. That
run is not authorized here; it is a candidate for a following declaration.

## 3. What this does and does not decide

It decides whether this one bounded change measurably narrows the scripted-normal gap for the seeds the diagnosis
implicated, and whether it does so without breaking scripted-easy or the critic gate. It does not identify a
mechanism beyond what S6–S9 already located, does not test necessity, does not compare against other repair
candidates (a relative-to-enemy decoder, an explicit enemy-selection head, or other weight values), and a null or
mixed result does not rule those out. Because training data differs from S5's, any result is confounded with data
draw as stated in §1. This does not attempt R1 qualification.

## 4. Verification and archive

New module `options/roster_imitation_repair.py` and tests; no existing module is edited. The declaration pins the
S4, S5, S7, S8, S9 manifests, `full_authority_imitation.py`'s, `mixture_imitation.py`'s, and
`full_authority_train_v1.py`'s digests (matching R1n-i's precedent of cross-checking against a prior run's own
recorded digests), the declaration digest, and the module digest. Gates before training: client and training
`pytest`, `npm run build`, `npm test` (366/367, the one documented exception), plus a unit test that the new weighted
loss is bit-identical to `fi.imitation_loss`'s total when `aimWeight = 1` (a regression guard on the reimplementation
itself). Sealed archive `runs/m8_s10_throw_aim_repair_v0/`; results in `reviews/m8_s10_results.md`. A failed gate or
unheld prediction is a finding, not patched around.
