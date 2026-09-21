# M8-S6 — archive-only failure breakdown of the S5 3v3 initializers

Declared 2026-09-20, before any number in this note was computed. Analysis of sealed
archives only: **no simulator, no training, no new collection, no seed band.**
`autonomousQualificationEligible` stays false.

## 0. Why, and why archive-only first

[S5](m8_s5_results.md) returned "not viable" and, by its declaration, calls for an
R1n-i-style diagnosis before any PPO. R1n-i's module cannot be reused as-is at 3v3 (its
deployed view guards against more than one live enemy), and a trace-level diagnosis is a
real collection step. The episode rows S4 and S5 already archived carry contact, damage
(0–300 across three red units), first-hit and final decision, and engagement distances,
enough to say *where* the failure sits before deciding whether traces are worth building.
The 2026-09-18 review's "archive-only breakdown" was the cheap step that reframed R1n-h;
this is that step for S5.

## 1. Questions and predictions (fixed before computing)

Cells: three S5 seeds × three Red arms (400 paired worlds each) and the S4 teacher on the
same worlds. Damage is the episode's `finalTargetDamage` (success needs at least 240).

1. **Contact.** Fraction of episodes with any target damage.
   *Prediction:* at least 0.9 in every cell except possibly random Red (the S5 development
   evaluations already showed contact 0.79–1.00).
2. **Where damage stalls.** Bin final damage into none, under 100, 100–199, 200–239, and 240
   or more, for non-successes.
   *Prediction:* against scripted-normal the learners die with under 200 damage; against
   random Red many time out in the 200–239 band, i.e. they kill two red units and fail to
   finish the third.
3. **Outcome composition against random Red**, the anomaly S5 flagged: success, team wipe,
   timeout with damage of at least 100, timeout with under 100.
   *Prediction:* timeouts carry substantial damage (a finishing failure), wipes carry less.
4. **Engagement geometry.** Median minimum distance and distance at first hit, learners
   versus teacher.
   *Prediction:* learners engage at similar or closer range than the teacher, so range
   keeping does not distinguish them.
5. **Exchange.** Mean damage dealt per blue unit lost, learners versus teacher, where units
   were lost.
   *Prediction:* learners trade far worse than the teacher against scripted-normal.

Each prediction is a hypothesis about the *pattern*, not a claim about mechanism. A
prediction that fails is reported as a failure of the prediction.

## 2. What this can and cannot show

The rows have no per-unit death times, movement, throw timing, or projectile data, so this
cannot separate aim, movement, cohesion, or target-selection failures, and cannot show why
random Red defeats the learners. It can show whether the failure is contact, finishing, or
attrition, and how it differs from the teacher. Its output is a *pointer*: if the pattern
supports a trace-level diagnosis, that is a separate declaration (with its own module,
seed reuse, and budget); if it does not, the next step is decided from what this shows.
Nothing here is an attribution, and outcome categories are descriptive, not causal.

## 3. Implementation and archive

New pure module `options/roster_failure_breakdown.py` plus tests; no existing module
is edited. It reads `runs/m8_s5_roster_imitation_v0/paired-eval/**/episodes.jsonl` and
`runs/m8_s4_roster_baseline_v0/teacher/**/episodes.jsonl` after verifying both manifests,
and writes a sealed `runs/m8_s6_breakdown_v0/` (declaration pinning both manifests, the
declaration digest and the module digest; `report.json`; manifest). The paired-world
alignment is checked by seed. Results in `reviews/m8_s6_results.md`. Gates before commit:
client and training `pytest`, `npm test` (366/367, the one documented exception).
