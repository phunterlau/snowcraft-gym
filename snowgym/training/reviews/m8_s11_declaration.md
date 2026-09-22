# M8-S11 — the pre-declared contingency: isolate the loss weight from the data draw

Declared 2026-09-22, before any training. Ordinary autonomous training, not teacher-assisted.
`autonomousQualificationEligible` stays false (a development experiment).

## 0. Why, and what this deliberately reuses

[S10](m8_s10_declaration.md) raised the imitation loss's throw-aim coefficient from 1 to 5, on fresh training data,
and found seed 97101's scripted-normal success jump to 0.51 — but stated this could not be attributed to the loss
change, because the training data was also a fresh draw relative to S5. S10's own declaration §2 pre-declared this
run as the contingency: rerun at `aimWeight = 1` on the **same** fresh bands, changing nothing else.

**Deliberately reused from S10, not re-audited as new bands:** the round-zero, DAgger-round, critic-fold, and
development-normal seed bands (4200000+). This is not a collision; it is the point of the run. Every other design
choice (optimizer seeds, architecture, fit/round/epoch counts, scope reduction to a normal-only development read,
paired-eval worlds and arms) is identical to S10.

**One consequence of the reuse worth stating precisely:** round-zero data (model-independent, teacher self-play)
will reproduce bit-identical rows to S10's archived round-zero. DAgger rounds 1–4 will **not** reproduce S10's,
because they roll out the model being trained, and this run's model diverges from S10's the moment its first fit
uses a different loss. Re-running the simulator across every round is therefore necessary, not a redundant repeat of
S10's decisions.

## 1. The comparisons this design enables

| Comparison | Holds fixed | Varies | Isolates |
| --- | --- | --- | --- |
| S10 vs S11 | data draw (both use bands 4200000+), optimizer seeds, architecture | `aimWeight` (5 vs 1) | **the loss weight's effect alone** — the primary question |
| S11 vs S5 | `aimWeight` (both 1), optimizer seeds, architecture | data draw (S11's 4200000+ vs S5's 3300000+) | the data draw's effect alone |
| S10 vs S5 | optimizer seeds, architecture | both loss weight and data draw | the combined effect (already reported in S10) |

If the three comparisons are roughly consistent — (S10 − S11) + (S11 − S5) ≈ (S10 − S5) on scripted-normal success —
that is a sanity check on the decomposition, reported but not required to hold exactly (three separate stochastic
training runs, not a deterministic identity).

## 2. Classification rule (fixed now, not a pass/fail prediction)

Unlike S10's P1, this step has no confident directional prior — that is the reason it exists. Rather than force a
single prediction, three mutually exclusive outcomes for **seed 97101's scripted-normal success** are defined now:

| Outcome | Rule | Reading |
| --- | --- | --- |
| **Loss-weight-driven** | S11 ≤ 0.20 (close to S5's 0.00, far from S10's 0.51) | The `aimWeight` change caused the gain; the data draw alone does not reproduce it |
| **Data-draw-driven** | S11 ≥ 0.35 (close to S10's 0.51) | This training-data draw alone reproduces most of the gain; the loss change was not necessary |
| **Mixed or inconsistent** | 0.20 < S11 < 0.35, or S11 exceeds S10's 0.51, or the other two seeds move materially | Neither clean story fits; reported descriptively |

0.20 and 0.35 split the 0.00–0.51 range roughly into thirds around the midpoint; they are not fitted to any result
observed before this declaration.

Also reported, not classified: 97102's and 97103's scripted-normal success and 97102's easy regression, at
`aimWeight = 1` on the fresh bands — checking whether those seeds' null results were about the data or the loss
too; and the critic warm-start gate (re-checked, not assumed to pass because S10's did).

**If S11's 97101 success exceeds S10's 0.51** the rule above routes it to "mixed-or-inconsistent," but that outcome
is not uninformative: it would mean reverting the loss weight helped *more* than keeping it raised, i.e. the weight
increase was actively harmful for this seed on this data draw. This is stated now so that result is not read as a
null finding if it occurs.

`s10_rows` reads S10's layout (`condition-M/seed-{seed}/paired-eval/{arm}/episodes.jsonl`); `rp.s5_rows` reads S5's
own, different layout (`paired-eval/seed-{seed}/{arm}/episodes.jsonl`). Both are read-only archive lookups, not
written to.

## 3. What this does and does not decide

It decides which of loss-weight or data-draw better explains S10's single large result, or whether neither does
cleanly. It does not identify why the fresh data draw would matter (if it does), does not test intermediate
`aimWeight` values, and does not by itself justify any further tuning — that is explicitly deferred to whatever this
run shows.

## 4. Verification and archive

New module `options/roster_imitation_repair_isolated.py`; no existing module is edited, including
`roster_imitation_repair.py` (S10's), whose `train_and_measure`, `weighted_imitation_loss`, and `fit_weighted` are
imported and reused unchanged — only the configuration passed to them differs. The declaration pins S10's manifest,
`roster_imitation_repair.py`'s own digest, this module's digest, and the declaration digest. Budget and gates are
identical to S10's (bound 743,200, cap 950,000; `npm test` 366/367, the one documented exception). Sealed archive
`runs/m8_s11_throw_aim_isolation_v0/`; results in `reviews/m8_s11_results.md`.
