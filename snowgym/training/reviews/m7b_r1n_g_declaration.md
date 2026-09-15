# R1n-g — why does blue's offense fail against scripted red? A label-error diagnostic, no training

Declared 2026-09-14, before any collection. No policy is trained, updated,
or evaluated for success/death here. This is a diagnostic reusing R1n-c's
existing `label_error` measurement, pointed at states visited under
`ScriptedAiAgent` instead of `RandomAgent`.

## 0. Why this exists, and what it does not decide

R1n-f found that all six R1n-c/R1n-e policies score 0% against scripted red
at both tested difficulties, and that the failure is bilateral: blue's
offense collapses as completely as its defense. On the easier arm, the six
policies combined **threw 25,008 snowballs and landed zero hits.** That
number is unexplained. It could mean:

- blue rarely attempts a throw the teacher would take (a **type** failure —
  wrong action, or no throw opportunity is recognized at all);
- blue throws about as often as the teacher, but aims badly (an **aim**
  failure);
- blue throws and aims reasonably, but its power/range calibration is wrong
  for a target that moves and takes cover (a **power/range** failure).

Each of these motivates a different fix to the mixture-imitation curriculum
this diagnostic exists to inform (roadmap step: mixture-imitation retraining
is deferred to **R1n-h**, designed with this diagnostic's answer in hand,
matching how R1n-d shaped R1n-e's design before any PPO ran).

**R1n-g decides nothing about R1 qualification, trains nothing, and
authorizes nothing.** `autonomousQualificationEligible` stays false.

## 1. Method

Reuses `full_authority_imitation.py` unchanged:

- `Labeler(model=model, deterministic=True, record=True)` and `collect()`:
  the frozen policy **acts** (deterministically, matching R1n-f's evaluation
  mode) while the **teacher's** plan-controller labels every visited state
  (`plan_teacher_tensor_actions_indices`) — the same "model acts, teacher
  labels" collection R1n-c used for its own held-out label error, applied
  here to states the model reaches under a different opponent.
- `label_error(model, part, cfg)`: per-action-type recall and precision
  against the teacher's label. The move/aim/power error terms are each
  conditioned only on the **teacher's** label type (move-heading/endpoint
  error on states the teacher labels MOVE, aim/power error on states the
  teacher labels THROW) — read from the model's own move/throw/power heads
  regardless of which action type the model actually chose there. So
  `throwAimHeadingErrorDegrees` answers "if the model had thrown here, how
  well would it have aimed," not "how well did the model aim on the throws
  it actually attempted." This is `label_error`'s existing, unedited
  behaviour (confirmed by reading `full_authority_imitation.py:212-256`);
  R1n-g does not change it, only reports it correctly (see §2).

**Comparators:** the same six frozen checkpoints R1n-f evaluated — the three
R1n-c initializers (`fit-4.pt`) and the three R1n-e finals (`update-200.pt`),
loaded exactly as `opponent_transfer.py` loads them. No new checkpoint is
produced.

**Opponent arms:** the same three as R1n-f, varied by
`opponent_transfer.scenario_override` (reused unchanged, imported, not
reimplemented) — random (reference), scripted easy, scripted normal. Each
arm uses its own fresh 128-world seed band, shared across all six
comparators (paired by world within an arm, matching R1n-c/R1n-f's design).

Every comparator runs against every arm: 6 × 3 = 18 (comparator, arm)
label-error measurements, on 128 episodes each.

## 2. Measures

For each of the 18 (comparator, arm) cells, `label_error`'s full output:
`typeAccuracy`, `perType` recall/precision for noop/move/throw/hold,
`moveHeadingErrorDegrees`, `moveEndpointErrorWorld`,
`throwAimHeadingErrorDegrees`, `powerMeanAbsoluteError`.

**Primary comparison:** for each metric, the arm R (random) value against
the arm E and arm N values, for the same checkpoint. A metric that moves
sharply between arms localizes the failure; one that stays flat rules that
head out.

**Specifically decomposing the three candidate explanations (§0):**
- **Type failure:** `perType["throw"]["recall"]` per arm. R1n-c's own
  `throwRecallFlag` (0.5) is reused as the reference threshold, not a new
  invention.
- **Aim failure:** `throwAimHeadingErrorDegrees` per arm. This is conditioned
  on the **teacher's** label being THROW, not on the model having chosen
  THROW — it measures whether the model's throw head *would* aim well on
  the states where a throw was correct, independent of whether the model
  fired there at all. It therefore cannot by itself distinguish "blue
  doesn't throw" from "blue throws and aims well" — that distinction comes
  from reading it together with `perType["throw"]["recall"]` on the same
  cell. The row count behind each aim/power mean (the number of
  teacher-labelled-THROW states in that cell, i.e. `perType["throw"]["support"]`)
  is recorded alongside every error mean in `label-error.json` and reported
  in the results table, so an aim number computed on a handful of states is
  never read as equivalent to one computed on thousands.
- **Power/range failure:** `powerMeanAbsoluteError` per arm (same
  teacher-label conditioning as aim, same caveat), and
  `moveHeadingErrorDegrees`/`moveEndpointErrorWorld` as a secondary check on
  whether *positioning* (not just throwing) diverges from the teacher.

**Cross-checks, not primary:**
- Arm R's numbers against R1n-c's own archived split-B label error for the
  same checkpoints (`runs/m7b_engage_r1n_c_v0/seed-*/label-error.json`) —
  different worlds, same opponent, same checkpoint; a large divergence would
  flag a measurement problem in this diagnostic, not a new finding.
- Initializer versus final, per arm: does PPO change the label-error profile
  at all, even though R1n-f showed it does not change the win/loss floor.

## 3. Predictions, stated for falsification

- `perType["throw"]["recall"]` drops from its arm-R level on at least one
  scripted arm, below `throwRecallFlag` (0.5) — the type failure is at least
  part of the story, given zero hits landed. **Note:** R1n-f's own mechanism
  counters already argue against this — blue threw 8–12 times per episode
  against scripted-easy red (`redProjectilesPerEpisode`-scale volume, from
  the same policies), which is throw *volume* comparable to or higher than
  against random red, not a collapsed throw rate. So this prediction is
  disfavored by evidence already in hand; R1n-g is well powered to falsify
  it rather than confirm it.
- `throwAimHeadingErrorDegrees`, read on the states where the teacher labels
  THROW (see §2's conditioning note), is dramatically larger on the
  scripted arms than on arm R — given the volume argument above, if the
  type failure is disfavored, the zero-hit result should show up here or in
  the power/range measure instead.
- `moveEndpointErrorWorld`/`moveHeadingErrorDegrees` rise on the scripted
  arms, reflecting the same short engagement range and cover-seeking that
  R1n-f's mechanism counters found outside blue's training distribution.
- PPO changes label-error metrics by less than the R1n-c-to-R1n-f
  checkpoint-loading noise floor (established by the arm-R-vs-archive
  cross-check) — consistent with R1n-f's finding that PPO does not change
  the scripted-arm floor.

## 4. Recommendation format (informs R1n-h; authorizes nothing)

The result is reported as which of the three failures (§0) the evidence
localizes to, with the supporting numbers, and a recommendation for what
R1n-h's mixture-imitation curriculum should therefore prioritize:

| Localization | R1n-h implication |
| --- | --- |
| Type/detection failure dominates | The curriculum needs states where the teacher throws under a moving/covering opponent; volume of demonstration matters more than throw-aim tuning. |
| Aim failure dominates | The curriculum needs the aim/power regression losses re-weighted or re-scaled for the shorter engagement ranges scripted red creates. |
| Position/movement failure dominates | The curriculum needs positioning demonstrations (approach, spacing) under scripted red before throw-quality can be evaluated meaningfully at all. |
| No single factor dominates | Report the full profile; R1n-h's declaration states its curriculum design against all three. |

## 5. Artifacts retained

Per (comparator, arm): the collected episodes (`episodes.jsonl`, reusing
`full_authority_diagnostics.write_episodes`) and the `label_error` output
(`label-error.json`). A top-level `report.json` with all 18 cells, the
cross-checks, and the localization call. A sealed SHA-256 manifest over the
whole run, reusing `death_rate_ppo.seal`/`verify_sealed` (imported, not
reimplemented).

## 6. Budget and stopping

- 18 cells × 128 episodes × ~110 decisions/episode (scripted-red episodes
  run shorter than random-red ones per R1n-f, so this is a conservative
  overestimate) ≈ **253,440** decisions.
- Bound: 260,000. Cap: 400,000, enforced in code.
- No training, so there is no per-policy budget split; the guard is a
  single run-wide counter.

## 7. Seeds and generators

- Arm R (random): 994000–994127.
- Arm E (scripted easy): 995000–995127.
- Arm N (scripted normal): 996000–996127.
- Repo-wide scan of 994000–996999 found no existing use.
- No training RNG, no bootstrap sampling (the reported quantities are
  descriptive point estimates over 128 paired worlds per arm, matching
  `label_error`'s own precedent in R1n-c, which reported it without a
  confidence interval).

## 8. Verification before the implementation commit

- Python training tests, Python client tests, `npm run build`, `npm test`
  (accepting only the documented R1n-b preflight failure).
- A test that `label_error` on a tiny synthetic case reproduces a
  hand-computed recall/precision and aim-error value (guards against a
  reuse-site regression, not `full_authority_imitation.py` itself, which is
  untouched).
- `auditSeedDocuments` on the serialized configuration, plus the repo-wide
  numeric scan recorded in §7.
- The arm-R cross-check against R1n-c's archived split-B label error (§2)
  is run and reported before the scripted-arm numbers are interpreted, so a
  measurement problem is caught before it is mistaken for a finding.
