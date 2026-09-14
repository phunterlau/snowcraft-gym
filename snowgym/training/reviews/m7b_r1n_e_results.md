# R1n-e: PPO cuts deterministic death rate by 25 points and raises success by 24; survival improved, three of three policies, replication pending

Collected 2026-09-14 from `a41af1a`.
- The [declaration](m7b_r1n_e_declaration.md) was committed at `68d2cb6`.
- The implementation and amendments A1–A6 are at `8f7544f`, A7–A9 at
  `1070890`, and A10 at `a41af1a`.

Run facts:
- **Attempts:** two.
  - The first (declared at `1070890`) was killed by the host at low system
    memory during update 8 of policy 97101, before any evaluation.
  - It is kept under `aborted-attempt-1/`, and the run was re-declared after
    the A10 memory fix.
  - The rerun's logged lines for updates 1–7 are identical to the first
    attempt's: successes, deaths, actor steps, anchor KL, and decisions.
- **Decisions:** 5,540,370, against a 9,000,000 cap (declared bound
  8,870,400). Per policy: 1,826,994, 1,847,976, and 1,865,400.
- **Resources:** 2,241, 2,179, and 2,182 s wall time per policy (1.83 h in
  total), with peak resident memory 1.30–1.35 GB.
- **Rejected actions:** 0 of 5,397,552 across training and evaluation.
- **Manifests:** the top-level manifest (`sha256:ee284f7e885e…`, 67
  artifacts) and the three policy manifests were re-verified after
  aggregation.
- **Archive:** `runs/m7b_engage_r1n_e_v0/`.

At runtime the policies act alone. The initializers came from
teacher-imitation training, so `autonomousQualificationEligible` stays
false.

**R1n-e makes no R1 qualification claim (declaration §0), and nothing below
authorizes R1n-f.**

## Headline

1. **Outcome: survival improved.** The primary test passes.
   - Seed-averaged deterministic death-rate difference (final − initializer)
     on split E (870000–870399): **−24.8 points, 95% CI [−27.9, −21.8]**,
     against the −5 threshold.
   - Every policy improved:

     | Policy | Death-rate difference | 95% CI |
     | --- | ---: | --- |
     | 97101 | −36.0 | [−41.0, −31.0] |
     | 97102 | −15.0 | [−19.3, −11.0] |
     | 97103 | −23.5 | [−28.5, −18.8] |

2. **The fewer deaths are not bought by avoiding the fight.**
   - **Success** rose by the same amount: **+24.0 points [+21.1, +27.0]**.
     The non-inferiority guard needed the lower bound above −5.
   - **Timeouts** rose by +0.5 [+0.1, +1.0], under the +5 guard.
   - **Contact** rose for every policy: 86.8% → 100%, 98.0% → 99.0%, and
     93.0% → 99.3%.
   - The policies engage more often and survive the engagement. That is the
     failure R1n-c located.
3. **Deterministic finals are at the teacher's level on other splits.**
   - Final success is 94.3%, 92.3%, and 92.0%; final death rate is 5.0%,
     7.8%, and 7.5%.
   - The coded teacher scored 93 / 91 success with 7% / 9% deaths on R1n-c's
     splits A and B.
   - The teacher was not evaluated on split E, so this is a comparison
     across splits.
4. **Stochastic execution at σ×0.5 shows the same effect.**
   - Death rate: −25.3 [−28.5, −22.1]; success: +24.0 [+20.9, +27.2].
   - The final deterministic − stochastic success gap is 1.8, 4.3, and 3.8
     points.
5. **The effect is five times the declared threshold and far outside the
   predicted range.** Declared prediction: `D` between −10 and −3. Of the
   four declared predictions, two failed: `D` overshot, and the KL stop did
   not bind in most updates (§5).
   - **A1's expectation** of about 4 or about 2 actor steps per update was
     also wrong, in the direction of *more* learning: the median was 64
     steps.
   - **Why.** The lr rule was calibrated on first-step KL, which did not
     predict cumulative movement. Minibatch gradients are not consistent
     across steps, so KL grows far more slowly than n².
6. **The weak critic did not prevent learning.**
   - Warm-start held-out R² was 0.070, 0.085, and 0.127. All three fail the
     absolute 0.25 condition of the R1n-b critic gate that R1n-c applied
     (`gatePassed` false).
   - With complete-episode Monte Carlo advantages, PPO learned anyway.
   - The design is bundled: σ×0.5, MC advantages, lr 1e-5, the anchor, and
     complete episodes changed together. This run does not attribute the
     gain to any one of them.

## 1. Critic warm start (A7 sanity stop)

| Policy | Held-out R² | 95% interval | Stop |
| --- | ---: | --- | --- |
| 97101 | 0.070 | [−0.011, 0.110] | no |
| 97102 | 0.085 | [−0.035, 0.160] | no |
| 97103 | 0.127 | [0.054, 0.182] | no |

- **Rule outcome:** none of the three point estimates is below 0, so §2's
  original point rule would also not have stopped any policy. The A7
  amendment did not change the outcome.
- **Warm-start data:** declared seeds 880000+ (train) and 885000+ (held
  out).

## 2. Training (200 updates × 64 complete episodes, stochastic at σ×0.5)

Windowed rates from `training-history.json`:

| Policy | Updates | Success | Death | Undiscounted return | Mean actor steps | Clip fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 97101 | 1–25 | 72.3% | 27.9% | +0.435 | 62.1 | 0.026 |
| 97101 | 176–200 | 88.2% | 11.2% | +0.780 | 51.3 | 0.037 |
| 97102 | 1–25 | 77.5% | 22.9% | +0.546 | 60.0 | 0.034 |
| 97102 | 176–200 | 88.0% | 11.3% | +0.776 | 42.4 | 0.035 |
| 97103 | 1–25 | 73.4% | 26.4% | +0.459 | 60.0 | 0.031 |
| 97103 | 176–200 | 88.6% | 11.1% | +0.787 | 50.4 | 0.037 |

- **Learning curves.**
  - 97101 and 97103 gain most in the first 75 updates, then run roughly
    flat at 88–89% training success over updates 126–200.
  - 97102 is still rising at update 200 (84.6% → 88.0% over the last 75
    updates).
  - No longer run was tested.
- **KL stop.**
  - It fired in 52, 72, and 69 of 200 updates.
  - It fired before the last epoch in 37, 56, and 58.
  - The median was 64 actor steps (range 4–68; rollouts above 8,192 rows
    have 17 minibatches).
  - Mean approximate KL per minibatch was 0.003–0.004.
- **Anchor KL (exact hybrid KL to the initializer).**
  - Maximum 0.615 (97101, update 56), 0.277, and 0.348.
  - Final 0.375, 0.154, and 0.179.
  - It rose and fell rather than growing monotonically.
- **Actor movement:** L2 parameter distance from the initializer was 0.241,
  0.213, and 0.237.

## 3. Evaluation on split E (400 worlds, same worlds for both policies)

**Deterministic** (initializer → final):

| Policy | Success | Death | Timeout | Contact |
| --- | --- | --- | --- | --- |
| 97101 | 59.5% → **94.3%** | 41.0% → **5.0%** | 0.0% → 1.0% | 86.8% → 100% |
| 97102 | 78.0% → **92.3%** | 22.8% → **7.8%** | 0.3% → 0.0% | 98.0% → 99.0% |
| 97103 | 69.0% → **92.0%** | 31.0% → **7.5%** | 0.0% → 0.8% | 93.0% → 99.3% |
| Mean | 68.8% → 92.8% | 31.6% → 6.8% | 0.1% → 0.6% | 92.6% → 99.4% |

**Stochastic, σ×0.5** (initializer → final):

| Policy | Success | Death | Timeout |
| --- | --- | --- | --- |
| 97101 | 61.8% → 92.5% | 39.0% → 6.5% | 0.3% → 1.0% |
| 97102 | 69.5% → 88.0% | 31.5% → 11.0% | 0.3% → 1.0% |
| 97103 | 65.5% → 88.3% | 34.3% → 11.5% | 0.3% → 0.5% |

**Paired differences with world-bootstrap 95% intervals** (points):

| | Death | Success | Timeout |
| --- | --- | --- | --- |
| Seed-averaged, deterministic | **−24.8 [−27.9, −21.8]** | +24.0 [+21.1, +27.0] | +0.5 [+0.1, +1.0] |
| Seed-averaged, σ×0.5 | −25.3 [−28.5, −22.1] | +24.0 [+20.9, +27.2] | +0.6 [0.0, +1.2] |
| 97101, deterministic | −36.0 [−41.0, −31.0] | +34.8 [+29.8, +39.8] | +1.0 [+0.3, +2.0] |
| 97102, deterministic | −15.0 [−19.3, −11.0] | +14.3 [+10.3, +18.5] | −0.3 [−0.8, 0.0] |
| 97103, deterministic | −23.5 [−28.5, −18.8] | +23.0 [+18.0, +28.0] | +0.8 [0.0, +1.8] |

- **Initializers match earlier archives.** The initializer rows agree with
  R1n-c's deterministic evaluations of the same `fit-4.pt` checkpoints:
  - success: A 54 / 79 / 62, B 61 / 83 / 78;
  - deaths: A 45 / 21 / 37%, B 39 / 17 / 22%;
  - contact: A 86 / 98 / 95%, B 85 / 97 / 93%.

  They also agree with R1n-d's `det` mode (55 / 72 / 71; deaths 46 / 28 /
  29%). Deterministic mode uses mean actions, so the σ×0.5 change does not
  affect it.
- **Success and death are near-disjoint.** An episode can count as both,
  when the target is eliminated but no blue unit is alive at the end. That
  happened in 2, 3, and 0 initializer episodes and in 1, 0, and 0 final
  episodes.
- **Not a pass of the unchanged R1 gate.** Per policy, the success gains
  are +34.8, +14.3, and +23.0: 97102 is below +20. Replication is still
  required. §0 of the declaration stands: R1n-e makes no R1 claim.

## 4. Decision rules

Order 1 does not apply: non-inferiority holds (`S⁻` = +21.1 > −5,
`T⁺` = +1.0 < +5), and `D⁻` < 0. Order 2 applies: `D` = −24.8 ≤ −5 and
`D⁺` = −21.8 < 0. Outcome: **survival improved**.

- **Incomplete:** does not apply; no sanity stop fired.
- **A8 (no effective training):** not reached. Final anchor KLs of 0.375,
  0.154, and 0.179 are all above the 0.01 floor anyway.
- **Checks:**
  - parameter change: yes;
  - rejection rate < 0.001: yes (0);
  - policies with lower death: 3 of 3;
  - σ×0.5 death difference: −25.3 [−28.5, −22.1].
- **Declared recommendation:** R1n-f, a fresh replication with new training
  RNGs and a new untouched split, then a second contrasting mission. It
  needs its own declaration.

## 5. Declared predictions

| Prediction (§5) | Result |
| --- | --- |
| `D` between −10 and −3 points | **Not met.** −24.8: five times the threshold, past the predicted range. |
| Success difference ≥ 0 | Met. +24.0. |
| The approximate-KL stop binds before the last epoch in most updates | **Not met.** 37, 56, and 58 of 200. |
| Mean KL to the initializer stays below 1 nat per living unit | Met. Maximum 0.615. |

**A1's stated expectation** (about 4 actor steps per update for 97101 and
about 2 for the others) was not met: the median was 64 for all three. The
rule itself was applied as declared.

**The §6 power analysis** assumed a detectable effect of about 4–5 points.
The observed effect is far outside that range, so the test was not
marginal.

## 6. What this does and does not decide

**Does show:**
- On 1v1 Engage, PPO on the frozen reward, started from R1n-c's imitation
  policies, cuts deterministic death rate and raises success by about 24
  points on a fresh split. All three policies improve.
- The gain comes from surviving engagements, with contact rising, not from
  avoiding them.
- Deterministic finals reach teacher-level success and death rates, judged
  against the teacher's rates on other splits.
- A critic with held-out R² near 0.1 is enough for complete-episode Monte
  Carlo PPO to learn. The 0.25 critic gate was not a necessary condition
  for this design.

**Does not show:**
- **Replicability.** There is one training run per initializer, and the
  training RNGs and evaluation split have not been replicated.
- **Which ingredient matters:** σ×0.5, MC advantages, lr 1e-5, the anchor,
  or complete episodes. They changed together.
- **Performance against the teacher on split E**, or whether the policies
  exceed it.
- **Transfer** to other missions, other scenarios, or teams larger than 1v1.
- **Training longer.** Whether more updates help or eventually drift was
  not tested.
- **The R1 gate:** R1n-e was not designed to satisfy it.

## 7. Exploratory behaviour analysis (post hoc, not pre-registered)

This section was produced after the outcome was known, at the user's
request. It declares nothing and gates nothing.

**Method.** The split-E deterministic evaluations were replayed for all three
policies, initializer and final (2,400 episodes), logging per-decision health,
positions, velocities, actions, and projectile ids. Every replayed episode
reproduced the archived outcome (success, survival, final decision). Hits were
attributed to individual snowballs; 100% of blue health drops matched a red
projectile. Scripts live outside the archive and nothing here was written into
the run directory.

**The opponent.** Red is `RandomAgent` at the Engage scenario's
`redController: "random"`. Every decision (10 Hz) it picks noop (25%), a
wander of up to 4 units (30%), or a throw at blue's *current* position with
power U(0.2, 1) (45%), subject to the 0.6 s cooldown. It does not lead its
target. Blue dies at 5 hits (100 health, 20 damage per hit); the Engage
mission completes at 80 damage, i.e. 4 hits.

**What changed, pooled over the three policies:**

| Within 12 units | Initializer | Final |
| --- | ---: | ---: |
| Red shots per episode | 4.02 | 3.40 |
| Red hit rate on those shots | 69.6% | 24.3% |
| Blue displacement during a snowball's flight, median | 0.50 | 1.37 |
| Share of red shots with blue displacement < 0.5 units | 50% | 22% |
| Share with displacement > 1 unit | 31% | 63% |
| Flight time, median decisions | 3 | 5 |
| Blue stationary (speed ≤ 1) | 35% | 17% |
| Hits taken per episode (all ranges) | 3.13 | 0.96 |
| Chance of another hit within 1.5 s of a hit | 70% | 45% |

- **Hit rate falls with displacement.** Red shots where blue moved less than
  0.5 units hit 94% (initializer) and 81% (final); shots where blue moved
  more than 1 unit hit 2% and 1%. The policies did not become harder to hit
  by standing elsewhere; they became harder to hit by not being where they
  were when the snowball was released.
- **Decomposition.** Splitting red's shots into cells of distance bucket ×
  target moving/still, about 27% of the reduction in hits taken comes from
  the change in exposure mix and about 73% from lower hit rates within cells.
  The within-cell change is accounted for by displacement during flight.
- **Damage cascades shrank.** Each hit stuns blue, which makes the next hit
  easier. Blue spent 8.6 → 2.4 decisions per episode stunned.
- **Offense.** Blue throws per episode 4.46 → 5.60, projectile hit rate
  72.9% → 68.7%, median throw distance 7.3 → 8.0 units, and throws from 8–12
  units 0.84 → 2.62 per episode. Red is stunned 14.5 → 16.7 decisions per
  episode.
- **Outcome accounting.** Across the 1,200 world–policy pairs: 319 death →
  win, 28 win → death, 51 deaths unchanged. Of the 94 initializer deaths
  that never landed a hit, the finals win 71.

**What this implies for replication.** The learned behaviour is evasion
against a thrower that aims where the target currently is. Against a
leading-aim opponent (`ScriptedAiAgent` uses a 0.18 s aim lead) the same
behaviour need not help. A stronger or different opponent is a separate
declared experiment, not a change to this one.

## Verification

- **Gate at `8f7544f` / `1070890` / `a41af1a`, before collection:**
  - training tests all passed (393 at `a41af1a`);
  - Python client and build passed;
  - `npm test` 366/367, with only the accepted R1n-b preflight failure.
- **Before collection, outcome-blind probes** measured lr (first-step KL),
  critic R² on off-band seeds, and memory. None evaluated death rate. Their
  numbers are recorded in A1, A7, and A10.
- **After collection:**
  - top-level and per-policy manifests re-verified: every artifact digest
    and the inventory;
  - `declaration.json` records `gitCommit a41af1a`;
  - the implementation, train, diagnostics, policy, and declaration digests
    equal the committed files, the E3 pins match, and the R1n-c source
    archive verified (95 artifacts);
  - `aborted-attempt-1/` (first attempt: declaration plus 97101 warm-start
    artifacts) is inside the sealed top-level manifest by design (A10). No
    analysis uses it.
  - `auditSeedDocuments` found no collision in the new `declaration.json`,
    `report.json`, or aborted declaration;
  - `npm test` still reports only the accepted failure.
- **Recomputed from the archive:**
  - tables come from `report.json` and `policy-report.json`;
  - training windows come from `training-history.json`;
  - success/death overlap counts come from the evaluation `episodes.jsonl`
    rows.
