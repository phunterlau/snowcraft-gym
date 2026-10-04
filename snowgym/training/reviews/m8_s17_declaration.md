# M8-S17 — qualification replication of S16's left/right command control

Declared 2026-10-04, before any simulator call on either panel below. Frozen-checkpoint evaluation, no training.
`autonomousQualificationEligible` stays false: this replicates a development finding. It is not PLAN.md's
per-mission qualification.

## 0. Question, scope, and what a pass can mean

S16 (`m8_s16_results.md`) found all six learned policies `controllable` on the `leftmost`/`rightmost` axis on its
192-world development panel. Its declared recommendation was to replicate on the reserved qualification panel
(2730000–2730399) before any training. This step does that.

**Scope was chosen by me, not explicitly by the user.** Asked whether the panel should cover left/right only or also
a centre-requested condition, the user said "Push and proceed" without choosing. I chose **left/right only**:
- it is exactly S16's primary claim and its declared recommendation;
- a centre-requested condition has no mirror flank and needs a new metric, designed on development data first.

A later centre-axis qualification can use its own fresh reserved band.

**What a pass can mean.** Spawns are fixed per spread, and the spreads d = 10/20/30 were chosen on development data
(S16 §9). Worlds on the qualification panel are therefore fresh draws of Red's random numbers on three
**already-seen geometries**. A pass shows the result is robust to the opponent's randomness and to the panel draw. It
does **not** show geometric generalization. That gap is addressed only by the labelled secondary in §3, which is
excluded from the verdict.

## 1. Reuse, pinned

`options/command_control_audit.py` (S16) is reused **unchanged**: its runner, tracker, preview grounding, scoring,
`contrasts`, `classify` and `mirror_identity`. `declare()` refuses to run unless that file's digest equals the
`implementationDigest` in S16's sealed `declaration.json`. S16's archive manifest is pinned, as are the S14
training, S14 probe, S12 and S15 manifests and the six checkpoint digests.

New module `options/command_control_qualification.py` composes those functions with this step's panels, conditions
and verdict. No existing module is edited.

## 2. Policies, conditions, layout

- **Policies:** the same six as S16, all evaluated, with no cohort or checkpoint selection: S14 `update-200.pt`
  finals and S14's σ-scaled initializer references for cohorts 1–3. The plan-aware teacher is the positive control.
- **Conditions:** `correct` (execution plan = requested flank) and `other` (execution plan = mirror flank), both
  execution modes.
  - Deterministic `other` serves only the mirror-identity check (S16 §3).
  - Dropped: `canonical` (the centre axis is out of scope), `zero-plan` (S16 showed it collapses the fighters
    physically) and `shuffled`, which is derivable: correct − shuffled ≈ Δo/2. PLAN.md's "≥ 20 points over shuffled"
    therefore corresponds to Δo ≥ 0.40, the controllable threshold.
- **Layout, stopping rule, metrics, informativeness guard and classification:** exactly S16 §2–§6.

## 3. Panels and seeds

| Panel | Role | Seeds | Spreads | Worlds |
| --- | --- | --- | --- | --- |
| Qualification | primary, verdict | 2730000–2730399 (the whole reserved panel) | d = (10, 20, 30)[i mod 3] | 400 (134/133/133) |
| Held-out geometry | secondary, never in the verdict | 2740000–2740199 | d = (15, 25)[i mod 2] | 200 (100/100) |

- A repo-wide scan, including ignored run directories, found 2730000–2749999 and 987000–987999 unused. The
  qualification panel was reserved, untouched, in S16.
- **No simulator call before the declaration commit.** No smoke check, preview reset or test touched either panel
  before the declaration and implementation were committed. Every live test overrides the panels to off-band seeds
  (5985000–5985999), and an offline test asserts that no test configuration overlaps either panel.
- **Stochastic torch seeds:** 987000 + 100·cohort + 40·panel + 20·condition + 2·task + policy. Here panel 0 is the
  qualification panel and 1 the held-out panel; condition 0 is correct and 1 other; task 0 is leftmost; policy 0 is
  the initializer. Values fall in 987100–987363, all distinct.
- **Bootstrap seed:** 987001, a separate numpy generator.

## 4. Gates (in order; a failure stops the run before later cells)

1. **Runner regression gate.** S16's `regression_gate`, unchanged: an exact replay of S14's first evaluation block on
   the default layout, 12 cells.
2. **Teacher gate on the qualification panel.** The same thresholds as S16:
   - mean `o`(correct) ≥ 0.90;
   - Δo lower 95% ≥ 0.70;
   - mean `o`(correct) ≥ 0.85 within each spread.

The teacher also runs on the held-out panel. If it misses those thresholds there, the secondary is reported as
invalid, but nothing stops.

## 5. Verdict (primary: the three S14 finals, deterministic, qualification panel)

Each final is classified by S16's rule. Given the exact deterministic mirror identity, `controllable` here is
equivalent to ō(correct) ≥ 0.70 with a lower 95% bound above 0.5. An `uninformative` final counts as **not
replicated** for that cohort.

| Verdict | Condition |
| --- | --- |
| `replicated` | all three finals `controllable` |
| `partially-replicated` | one or two |
| `not-replicated` | none |

If the identity check exceeds S16's learned tolerance (0.10), that is investigated before interpreting. Reported
but outside the verdict:
- the initializers;
- stochastic classes;
- per-spread and per-side breakdowns;
- the held-out-geometry secondary.

Cohort 2 is always shown separately (S15).

The verdict authorizes nothing. Recommendation only:
- **`replicated`:** the left/right command axis is ready to be built on. The next candidates, each needing the
  user's call and its own declaration, are the centre axis, mid-fight command switches, and the PLAN.md mission
  forks.
- **Otherwise:** diagnose before building on it.

## 6. Predictions (from S16's development numbers; fixed now)

1. The teacher gate passes.
2. Verdict `replicated`. S16's final Δo values were +0.93 to +0.96.
3. The cohort-averaged final deterministic Δo lies within ±0.10 of S16's +0.951.
4. The cohort 1 and cohort 3 deterministic initializers are again insensitive at d = 10: their d = 10 Δo interval
   contains 0.
5. Secondary: all three finals are `controllable` (deterministic) on the held-out spreads d = 15 and 25.

S16's post-hoc centre-command and side-bias readings are not predicted here; canonical is not run.

## 7. Budget

Bound, at horizon 200 per episode:

| Part | Calculation | Decisions |
| --- | --- | ---: |
| Regression gate | (as S16) | 153,600 |
| Teacher | 600 worlds × 2 tasks × 2 conditions × 200 | 480,000 |
| Learned | 600 worlds × 2 tasks × 2 conditions × 6 policies × 2 modes × 200 | 5,760,000 |
| **Total bound** | | **6,393,600** |

Hard cap: **6,400,000** simulator decisions. The cap was checked against the 49 seed bands of the four repo-JSON
scanner tests (S4, S5, S10, S12).

## 8. Archive and gates

New sealed archive `runs/m8_s17_command_control_qualification_v0/`:
- `declaration.json` with the pins of §1;
- `gate/`;
- `teacher/{qualification,heldout}/…`;
- `learned/{qualification,heldout}/…`;
- `report.json`.

Before running: training `pytest`, client `pytest`, `npm run build`, and `npm test` (only the documented
`trainSeedBase: 630000` failure). After archiving and before committing: the full training suite and `npm test`
again, plus a check of `report.json` integers against the scanner bands. Results go in `reviews/m8_s17_results.md`.
