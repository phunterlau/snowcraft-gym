# M8-S9 — splitting the throw target: which enemy vs. angular offset (teacher-assisted, 3v3)

Declared 2026-09-21, before any S9 intervention run. Collect-and-archive step. **Every arm is teacher-assisted
diagnosis, never autonomous behavior:** `assistType` records it and `autonomousQualificationEligible` stays false.
Nothing is trained; no checkpoint is selected or promoted.

## 0. Why

[S8](m8_s8_results.md) located the S5 learners' failure in the throw channel, mainly the **throw target** (0.24 / 0.99 / 1.00
success when replaced alone; power alone inert). But its `aim` arm replaces the target point wholesale, so it conflates
*which enemy is aimed at* with *how precisely*. This step separates them.

## 1. What the physics allows (read from the simulator, then used to define the split)

`ThrowSystem.tryThrow` computes the throw direction as the unit vector from the thrower to the target point and takes
speed and arc from power alone; **the distance to the target point is discarded.** A pre-declaration probe (below) showed the
learners' target points are about 32 world units from the thrower against the teacher's 7.5, roughly four times too far, which
is therefore physically irrelevant. Only the **heading** (direction from the thrower to the target point) matters, so the
split is defined in angular space.

For one comparable throw (learner and teacher both THROW, at least one living enemy), with thrower position `p` and living
enemies at bearings `φ_k` from `p`, and the two throws' headings `θ_L`, `θ_T`:

- the **chosen enemy** is the one whose bearing is angularly nearest the heading: `k = argmin_k |wrap(θ − φ_k)|`;
- the **offset** is the signed wrapped angle `δ = wrap(θ − φ_k)`.

Chosen-enemy and offset are therefore defined per throw for both learner (`k_L, δ_L`) and teacher (`k_T, δ_T`). "Chosen"
is an angular-proximity proxy, not a claim about the policy's internal target selection.

## 2. Arms

Same cells as S8: S5's three final policies (`fit-4.pt`, 97101–97103) × scripted-normal Red × worlds **2100000–2100099**,
blocks of 50, the learner acting deterministically and the plan teacher queried in the same state. Rules apply only to living
units where both learner and teacher THROW; the learner's action type and power are unchanged. The new target point is placed on
the ray from the thrower at the teacher's target distance, shortened if needed to stay inside the arena, so the direction is
never altered by clipping.

| Arm | New heading |
| --- | --- |
| `none` | unchanged (teacher still queried) |
| `aim` | the teacher's target, copied exactly (S8's rule; a reproduction gate, see below) |
| `aim-heading` | `θ_T` on a shortened ray (a check that the discarded distance is irrelevant) |
| `enemy` | `φ_{k_T} + δ_L`: the **teacher's** choice of enemy, the **learner's** angular offset |
| `offset` | `φ_{k_L} + δ_T`: the **learner's** choice of enemy, the **teacher's** angular offset |

**Gates.** (1) `none` must reproduce the S5 archived rows exactly (required). (2) `aim` must reproduce S8's archived `aim`
traces for the same seed episode by episode (every recorded state and action, after the archive's rounding), which is stricter than
comparing summary statistics and immune to summation order (required); it confirms the new harness matches S8's.
(3) `aim-heading` versus `aim` is reported, not required: equal outcomes would confirm that target distance is irrelevant.

**Descriptive measures** (from the `none` arm, per seed, over comparable throws): share of throws where `k_L = k_T`, and the
quartiles of `|δ_L|`, `|δ_T|` and `|θ_L − θ_T|`.

**Budget.** Bound 300,000 decisions (3 seeds × 5 arms × 100 worlds × 200), hard cap 420,000 (aborts unsealed). About an hour at
S8's rate; one background process, waited on by PID, refusing to overwrite. Recovery `R` uses S8's definition and reference:
`(yield_arm − yield_none) / (yield_tensor-path teacher − yield_none)` with the tensor-path teacher's 13.8 from S8's archive.

## 3. Predictions (fixed now; scored mechanically; all three seeds must satisfy each)

A disclosed peek informed P3 and P4: on 20 throwaway worlds (seeds 9301–9320, outside every research band) the learners chose the
teacher's enemy in 30–39% of comparable throws, and their median offset was 6–7° against the teacher's 2–4°. Those two are
therefore not independent of the peek. P1 and P2 are not informed by any intervention result.

| # | Prediction | Reads as |
| --- | --- | --- |
| P1 | `enemy`: `R ≥ 0.5` | the teacher's *choice of enemy* carries most of the aim effect |
| P2 | `offset`: `R < 0.5` | fixing angular precision on the learner's own choice is not enough |
| P3 | `none` arm: `k_L = k_T` on fewer than half of comparable throws | the learner does not mostly agree with the teacher's enemy. **Chance agreement is about 1/3 with three living enemies**, so this tests that agreement is not high, not that it is low; the peek's 30–39% is indistinguishable from random choice and P3 will very likely hold without being informative |
| P4 | `none` arm: median `|δ_L|` > median `|δ_T|` | the learner's angular offset is larger than the teacher's |

## 4. What this can and cannot show

**Pre-interpretation (fixed now, not a decision rule).** If the learner's choice of enemy is close to random, then: `enemy`
recovering most of the yield gap while `offset` does not means **target selection** is the deficit; `offset` recovering most while
`enemy` does not means the choice hardly matters and **angular precision** is; both recovering a lot means either fix alone would
suffice, which would point at redundancy rather than a single cause; neither recovering means the effect needs both together, as
`aim` (their combination) does. These readings are written before the results.


It shows whether the teacher's choice of enemy or the teacher's angular precision is sufficient on its own, given the learner's
other channels. It does not show necessity; the choice is defined by angular proximity, so a throw between two enemies is
assigned to one; aiming at a different enemy is not itself an error (success counts total damage, not which unit); and
sustained substitution shifts the visited states, as S8's non-monotone `aim + power` result showed. Success under assistance must
never be quoted as an autonomous result. Random Red is not covered.

## 5. Implementation and archive

New module `options/roster_target_split.py` and tests; no existing module is edited (it builds on `roster_intervention` and
`roster_trace`). The declaration pins the S4, S5, S7 and S8 manifests, the declaration digest and source digests. Sealed archive
`runs/m8_s9_target_split_v0/`; results in `reviews/m8_s9_results.md`. Gates before collection: client and training `pytest`,
`npm run build`, `npm test` (366/367, the one documented exception). A failed gate or unheld prediction is a finding, not
patched around. Live tests must exercise **every** arm name (S8's first launch crashed on one).
