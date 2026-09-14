# R1n-b: critic gate repaired; untrained fighters never make contact; teacher-first training with global decoding recommended

Completed 2026-09-13. The [declaration](m7b_r1n_b_declaration.md) was
committed at `365f67f`, its implementation and amendments A1–A8 at `45a66c9`,
and collection ran afterwards from `45a66c9`.

- One attempt, no retries, no aborted runs.
- 504,771 of the 600,000-decision cap (bound by item: 521,000).
- 276 s wall time, 1.41 GB peak memory.
- Zero rejected actions in every collection.
- All 46 manifest SHA-256 entries were re-verified after the run, and the
  pinned E3 source digests still match their archive.
- Archive: `runs/m7b_engage_r1n_b_v0/`.

No actor was trained, no checkpoint was promoted, and nothing below
authorizes a Phase D branch.

## Headline

1. **The repaired critic gate passes in all six arm/RNG configurations**, with
   predictive R² 0.989–0.999 where E3 reported −61 to −511. E3's stop was the
   gate's measurement design (F3/F4), not the critic. But the passing target is
   degenerate: see headline 3.
2. **Untrained full-authority fighters make no contact at all.** In 2,304
   episodes there was not one hit, and the uniform floor was 0/100 as well.
   Under the frozen reward, PPO from scratch (D1) receives no damage-dealing
   signal. The only non-clock signal is being hit.
3. **Local and global arms explore very differently, even with matched
   commanded noise.**
   - Local: never came closer than 14.6 world units to red.
   - Global: came within throw range 6 times and died 9 times.
   - B7 shows the commanded jitter is matched (median 2.19 vs 2.28 world
     units), so the difference comes from where each decoder's mean
     destination points, not from σ.
4. **The plan teacher is a usable 1v1 ceiling and label source (89/100), but
   only for the global decoder.** 77% of its MOVE labels lie beyond the local
   arm's 8-unit reach.
5. **At roster 1 the plan teacher and `SimpleBlueAgent` are the same
   controller.** The plan input carries no information in 1v1 Engage, so a
   wrong-plan control cannot be measured there.

The predeclared rules give **D3 (BC/DAgger-first)** for both arms, with the
label-audit flag set. Read together with the audit, the recommendation is
**D3 with the global decoder** (§6).

## 1. C1: teacher precondition, ceiling, and scripted reference

C1a passed: the tracker constructed at roster 1, and 5 plan-teacher decisions
were stepped with 0 rejections.

| Seeds 620000–620099 | Plan teacher (C1b) | `SimpleBlueAgent` (C1c) | Uniform floor (C2) |
| --- | ---: | ---: | ---: |
| Success | **89/100** | 89/100 | 0/100 |
| Contact (≥ 1 hit) | 94% | 94% | 0% |
| First hit, decision p50 / p95 (range) | 100 / 121.4 (80–152) | 100 / 121.4 | — |
| Completion, decision p50 / p95 | 120 / 143.8 | 118 / 145 | — |
| Blue death | 11% | 11% | 0% |
| Rejected actions | 0 / 12,130 | 0 / 12,017 | 0 / 20,000 |

- **All 11 teacher failures are blue deaths against random red, not
  timeouts.** The 89% ceiling is bounded by survival, not by reaching the
  target.
- **Horizon.** The teacher's 95th-percentile completion is 143.8 decisions,
  so the reviewer's ≥ 1.5× rule gives 216. The frozen 200-decision horizon is
  1.39× p95. A D declaration should decide whether to adopt 216; that would
  be a declared protocol change.
- **C1c is degenerate at roster 1, not an independent reference.** Comparing
  the two per seed:
  - same success outcome on 100/100 seeds;
  - same first-hit decision on 98/100;
  - identical blue–target distance trajectories over the first 60 decisions
    on 100/100;
  - `finalDecision` matches on 46/100; the rest differ after the outcome was
    already decided.

  With one enemy, the Engage plan grounds to "engage the only enemy," which
  the rule policy already does. The 89/100 figure shows the task is
  achievable. It does **not** show plan following.

## 2. Teacher label audit (D3 research)

| Quantity | Value |
| --- | --- |
| Labels by type | MOVE 10,092 · THROW 403 · NOOP 1,141 · HOLD 494 |
| MOVE own-to-destination distance, median / p90 | 24.1 / 48.4 world units |
| **MOVE beyond 8-unit local reach** | **77.1%** (flag threshold 20%) |
| MOVE / THROW targets at the arena bound (`|target| ≥ 0.999`) | 0.0% / 0.0% |
| THROW aim distance, median / p90 | 7.32 / 8.74 world units |
| THROW power, p10 / median / p90 | 0.74 / 0.80 / 0.97 |
| Blue–target distance at THROW decisions (`d*`) | 7.33 world units |
| Labeling throughput (teacher calls only; hardware-specific) | about 34,000 labels/s |

- The global decoder can represent every label in one decision, with no
  saturation, so `atanh` inversion is finite. The local R=8 decoder
  structurally cannot represent 77% of MOVE labels in one decision.
- Throws use the global `tanh` decode in both arms and are representable in
  both.
- The predeclared flag fired: a D3 declaration must say which decoder it
  imitates with, or declare a radius change or a multi-decision imitation
  target.

## 3. C2/C3: contact and exploration under random initialization

| 384 episodes × 3 RNGs per arm | Local | Global |
| --- | ---: | ---: |
| Episodes with a hit (damage > 0) | **0 / 1,152** | **0 / 1,152** |
| Closest approach: minimum / p5 / median | 14.55 / 34.18 / 50.45 | 3.51 / 13.65 / 26.36 |
| Episodes within 8 units / within `d*` = 7.33 | 0 / 0 | 8 / 6 |
| Blue deaths | 0 | 9 (0.78%) |
| Success | 0 | 0 |

The floor's closest approach has a median of 38.4 and never goes below 10.6.

**Mechanism.** `decode_move` anchors the local destination to the fighter's own
position, so an untrained local fighter jitters about its spawn point. The
global decode maps the head output to an arena-normalized point that does not
depend on the fighter's position, so an untrained global fighter walks toward
the arena interior and sometimes into red's range.

The B7 calibration at initialization shows matched commanded noise:
- move displacement median 2.19 (local) vs 2.28 (global) world units, p90
  3.79 vs 4.16, no clamping or saturation;
- throw-ray angular deviation median 2.15° vs 2.22°.

Matching σ therefore did not match exploration reach. This confirms, with
data, the handoff's concern that E3's "2-world-unit calibration" was a
near-origin sensitivity statement, not a matched behavior.

## 4. C3: corrected critic warm start

Held-out fold: 128 complete episodes on disjoint seeds. Gate: `predictiveR2 ≥ 0.25` and
`≥ timeOnlyR2 − 0.05`.

| Arm | RNG | Predictive R² [95% CI] | EV | Mean residual | Time-only R² | Clock skill | Untrained R² | Gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| local | 98001 | 0.9937 [0.9931, 0.9942] | 0.9938 | +0.0014 | 0.9975 | −1.51 | −37.5 | pass |
| local | 98002 | 0.9991 [0.9990, 0.9991] | 0.9991 | +0.0010 | 0.9975 | +0.63 | −50.3 | pass |
| local | 98003 | 0.9977 [0.9975, 0.9979] | 0.9977 | −0.0006 | 0.9975 | +0.09 | −56.0 | pass |
| global | 98001 | 0.9926 [0.9922, 0.9930] | 0.9927 | −0.0007 | 0.9975 | −1.92 | −37.8 | pass |
| global | 98002 | 0.9914 [0.9786, 0.9983] | 0.9915 | +0.0010 | 0.9897 | +0.17 | −50.0 | pass |
| global | 98003 | 0.9893 [0.9770, 0.9959] | 0.9893 | +0.0006 | 0.9869 | +0.18 | −55.8 | pass |

**The gate passed on an almost fully time-determined target. It certifies
that the critic can read the clock, and little else.**
- In all three local held-out folds and in global 98001, every episode lasted
  200 decisions with no damage either way. Each decision index has exactly one
  return value, and these four folds share identical held-out targets.
- The global 98002 and 98003 held-out folds each contain one episode that ended
  early in a blue death, plus episodes where blue took damage (6 and 5 distinct
  returns at decision 0; 25,586 and 25,572 rows). They are the only folds where
  returns vary beyond the clock, and the only interpretable positive clock
  skill (+0.17/+0.18). The critic picked up a little non-clock value there:
  being in danger.
- **Clock skill score is unstable when time-only R² → 1**, because both MSEs
  are tiny. Local 98001 (−1.51) and 98002 (+0.63) share identical targets and
  differ only in critic fit; global 98001 (−1.92) is in the same regime.
  Negative values there mean "slightly worse than a 20-bin clock lookup," not
  a failed critic; every such seed passes the declared gate with room to
  spare.
- **A1 in practice:** mean residuals are below 0.0015 everywhere, so
  predictive R² and EV agree to four decimals, as expected once the bias is
  near zero.
- **Carry-forward constraint:** this gate says nothing about value learning on
  a distribution where contact happens. Any D branch whose behavior policy
  reaches contact (as a BC-initialized policy would) must re-verify its critic
  on that policy's own complete episodes.

## 5. D research (reported, not gated)

**D1: repaired frozen-reward PPO from scratch.** Not supported.
- Contact is 0/2,304 random-init episodes and 0/100 for the floor, far below
  the 25% rule.
- In the local arm, returns are identical across episodes at each decision,
  so advantages carry no information beyond critic noise.
- In the global arm, the only variation is blue being hit or killed after
  wandering close (9 deaths, 0 damage dealt). Under the frozen reward that
  signal *penalizes approach*: the direction opposite to the skill.

**D2: approach shaping, measured offline with `Φ_d(s) = −max(0, d − 7.33)/128.06`; the reward was not changed.**

| Trajectories | Pre-contact transitions | `Φ_d` changes | Mean \|ΔΦ_d\| per decision (≈ world units) |
| --- | ---: | ---: | --- |
| Floor | 19,900 | 98.4% | 0.0023 (0.29) |
| Random-init local | 229,248 | 98.6% | 0.0020 (0.26) |
| Random-init global | 229,067 | 98.5% | 0.0025 (0.32) |
| Teacher | 10,134 | 99.1% | 0.0042 (0.54) |

- A distance potential would be dense: nonzero on about 98% of pre-contact
  transitions.
- Caveats:
  - The change includes red's random movement, which blue does not control, so
    density overstates controllable signal.
  - The early-potential/contact point-biserial is undefined for random-init
    (no contact variance) and −0.09 for the teacher (94% contact, so
    uninformative).
  - D2 is a new reward version and breaks comparability with every R1m/R1n
    number.

**D3: BC/DAgger-first.** Supported for the global decoder.
- The teacher is usable (89/100, 0 rejections).
- Labels are fully representable with global decoding (0% saturation).
- Labeling costs almost nothing next to simulation (about 34k labels/s).

Two constraints:
- The local decoder cannot imitate 77% of MOVE labels in one decision.
- At roster 1 the plan input is uninformative (C1b ≡ C1c), so R1's wrong-plan
  control and any plan-following claim need a second contrasting mission or a
  larger roster.

## 6. Decision rules and recommendation

**Runner output:**
- `teacherUsable: true`.
- Both arms: `criticHealthy: true`, `contactAdequate: false`, recommendation
  **D3**.
- `labelAuditRequiresDecoderStatement: true`.

**Recommendation** (not an authorization; the next step is a separate
declaration): **D3 with the global decoder**, in this order.

1. **Supervised pipeline that keeps its artifacts (Phase E).**
   - BC, then DAgger, from plan-teacher labels on the 1v1 scenario, into a
     global `FullAuthorityPolicyV1`.
   - Losses: cross-entropy on action type; move and throw latents by `atanh`
     of the normalized targets; power by logit.
   - Retain weights, optimizer state, fit history, and data digests.
   - Measure floor and ceiling on both development splits, and replicate
     across optimizer seeds.
2. **KL-anchored PPO fine-tuning under the frozen reward.**
   - Use the decoupled critic, re-warm-started on the BC policy's own
     complete episodes, where contact should be common.
   - Gate on R1's "+20 points over the initializer" so that PPO has to show
     real reward-driven improvement.
   - The declaration must decide the horizon (200 vs 216).
3. **A second contrasting mission sampled 50/50** before any plan-following
   claim.
4. **Leave-one-out ablations** from the working configuration: BC
   initialization off first; add D2 shaping only if PPO fine-tuning stalls.

The local arm is not carried into D3 unless a separate declaration changes its
radius or imitation target. Its random-init exploration results (§3) also
argue against it as a from-scratch vehicle.

Background for this design (not evidence from this run):
- demonstration-augmented policy gradients (Rajeswaran et al., 2018);
- KL regularization toward a supervised policy during RL fine-tuning
  (Vinyals et al., 2019);
- potential-based shaping preserving optimal policies under an absorbing
  terminal (Ng, Harada & Russell, 1999).

## 7. What this does and does not decide

- **Does** show:
  - E3's critic-gate stop was a measurement flaw, which the repaired gate
    removes;
  - frozen-reward PPO from random initialization has no contact signal in
    this 1v1 task;
  - the local and global decoders differ materially in random-init
    exploration and in teacher-label representability;
  - a coded 1v1 ceiling of 89/100 exists.
- **Does not** show:
  - that a critic can learn state value where contact happens (untested);
  - that BC or PPO will reach the ceiling;
  - that the plan input matters (uninformative at 1v1);
  - anything about 2v2 or 5v5.
- `autonomousQualificationEligible` stays false. No actor was trained.

## Verification

- The implementation gate passed before collection at `45a66c9`: 367
  TypeScript tests, build, 51 Python client tests, and 358 Python training
  tests (23 new).
- After collection:
  - all 46 manifest digests were re-verified;
  - `declaration.json` records `gitCommit 45a66c9` and matching pinned E3
    digests;
  - the budget guard was not triggered (504,771 ≤ 600,000).
- All per-seed comparisons and distance statistics in §§1–3 were recomputed
  from the archived `episodes.jsonl`, `trajectory-distances.npz`, and
  `critic-warm-start-arrays.npz`.
