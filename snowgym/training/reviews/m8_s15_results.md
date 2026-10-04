# M8-S15 results: S14's normal-only PPO retains random/easy competence on average, with one cohort regressing on random

Collected 2026-10-04 from `9ff1487` (declaration and implementation committed before running). Declaration:
[m8_s15_declaration.md](m8_s15_declaration.md). Archive: `runs/m8_s15_ppo_retention_eval_v0/` (sealed and verified,
`sha256:b90429f8…`). The run's declaration pins S14's training root (`sha256:39451d55…`), S14's probe root
(`sha256:28f7e4d4…`), S12's root (`sha256:ec7248ff…`), and the SHA-256 of all six loaded `.pt` files. 1,135,478
simulator decisions (71,349 fidelity + 1,064,129 evaluation) against the 2,073,600 bound and 2,100,000 cap. No
training. `autonomousQualificationEligible` stays false.

Both arms were in the initializer's imitation mixture. This is a **retention** check within that opponent family. It
is not an unseen-opponent test.

## Fidelity gate: exact

All 12 fidelity cells (3 cohorts × 2 policies × 2 modes) reproduced S14's archived first evaluation block exactly:
768 episodes on worlds 2600000–2600063, every row equal, deterministic and stochastic. This was run at S14's 64-world
block and the process-default 4 torch threads; the report records both values. The stochastic match is the
end-to-end proof that the initializer used here samples with S14's σ-scaled noise model, and not S13's unscaled
one.

## Headline

**Decision rule outcome: `retained-with-cohort-regression`.** All four cohort-averaged (arm, mode) cells are
non-inferior at δ = 0.05, and the rejection gate passes (zero rejected actions in every final-policy cell). However,
**cohort 2 is flagged on random Red**: its own intervals show harm beyond the margin on deterministic timeouts and
on stochastic success and timeouts.

Cohort-averaged paired differences, final − initializer, world-paired 95% bootstrap over 400 shared worlds:

| Arm / mode | Δ success | Δ `L` | Δ timeout | Δ team wipe | Cell |
| --- | --- | --- | --- | --- | --- |
| easy / det. | +0.000 [0, 0] | −0.036 [−0.043, −0.031] | +0.000 [0, 0] | +0.000 [0, 0] | non-inferior |
| easy / sto. | +0.000 [0, 0] | −0.021 [−0.027, −0.015] | +0.000 [0, 0] | +0.000 [0, 0] | non-inferior |
| random / det. | **+0.154 [+0.124, +0.183]** | −0.163 [−0.183, −0.143] | −0.087 [−0.113, −0.061] | −0.070 [−0.086, −0.055] | non-inferior |
| random / sto. | −0.010 [−0.027, +0.006] | −0.013 [−0.019, −0.007] | +0.011 [−0.005, +0.028] | −0.001 [−0.003, +0.000] | non-inferior |

Per cohort, random arm (the only arm where cohorts differ in direction):

| Cohort | Δ success det. | Δ timeout det. | Δ success sto. | Δ timeout sto. |
| --- | --- | --- | --- | --- |
| 1 | +0.270 [+0.225, +0.315] | −0.207 [−0.250, −0.168] | +0.022 [+0.000, +0.045] | −0.020 [−0.043, +0.003] |
| 2 | −0.092 [−0.138, −0.048]† | **+0.125 [+0.080, +0.170]** | **−0.090 [−0.125, −0.055]** | **+0.090 [+0.055, +0.125]** |
| 3 | +0.285 [+0.240, +0.330] | −0.177 [−0.220, −0.138] | +0.037 [+0.013, +0.062] | −0.037 [−0.062, −0.013] |

Bold marks the three cells the declared per-cohort rule flagged (harm beyond δ = 0.05 with confidence). † Cohort 2's
deterministic success loss is confidently harmful but not flagged: its upper bound, −0.048, sits just inside −δ.
This is a near-miss, not a pass.

`L` improves in every cohort and both modes on random, including cohort 2 (−0.101 deterministic, −0.012 stochastic).
Team wipe improves deterministically in every cohort. Under stochastic execution, wipes were already essentially zero
for both policies (cohorts 2 and 3 at exactly 0, cohort 1's interval reaching 0). Cohort 2's regression is confined
to finishing the option within the horizon.

## Absolute numbers

| Cohort | Arm | Init det. success / `L` / timeout | Final det. | Init sto. | Final sto. |
| --- | --- | --- | --- | --- | --- |
| 1 | random | 0.708 / 0.188 / 0.230 | 0.978 / 0.005 / 0.022 | 0.960 / 0.021 / 0.037 | 0.983 / 0.010 / 0.018 |
| 2 | random | 0.792 / 0.129 / 0.175 | 0.700 / 0.028 / 0.300 | 0.958 / 0.025 / 0.043 | 0.868 / 0.013 / 0.133 |
| 3 | random | 0.685 / 0.211 / 0.207 | 0.970 / 0.006 / 0.030 | 0.938 / 0.020 / 0.062 | 0.975 / 0.003 / 0.025 |
| 1–3 | easy | 1.000 / 0.021–0.052 / 0 | 1.000 / 0.000–0.003 / 0 | 1.000 / 0.021–0.033 / 0 | 1.000 / 0.002–0.009 / 0 |

Easy is saturated on success for both policies in both modes. PPO reduced the units lost there slightly.

## Predictions (declaration §6): 3 of 4 held

1. **Overall outcome `retained`: FAILED.** The outcome was `retained-with-cohort-regression` (cohort 2 on random).
2. **Random, deterministic, cohort-averaged `Δsuccess` > 0 with an interval excluding zero: held** (+0.154 [+0.124,
   +0.183]).
3. **Random stochastic `Δsuccess` smaller than the deterministic one: held** (−0.010 vs +0.154).
4. **Easy, deterministic, cohort-averaged `ΔL` ≤ 0: held** (−0.036).

**Cross-check (not a gate):** the initializer's deterministic random-arm success, 0.708/0.792/0.685, is within the
±0.12 tolerance of S13's 0.77/0.87/0.79 in every cohort, but lower in all three (−0.06, −0.08, −0.105). Two pieces
of evidence point to the world sample rather than loading:
- **Correlated worlds.** S13's three cohorts shared the same 100 worlds, so a favorable sample shifts all three in
  the same direction.
- **Identical actions.** A development check outside the archive ran S12's `act()` (S13's loader) and S14's PPO
  subclass `act()` side by side on 8 off-band random-Red worlds per cohort. Their deterministic actions were
  bit-identical over 39,540 unit-decisions: same action types, and a maximum target and power difference of 0.0.

## Interpretation

**On random Red, the deterministic gain is again mostly argmax repair.** The initializer's *stochastic* success on
random is already 0.94–0.96, far above its deterministic 0.69–0.79. S14 found the same deterministic/stochastic gap
on scripted-normal. Under stochastic execution the cohort-averaged change is −0.010 [−0.027, +0.006]: no detectable
gain, and no loss beyond the margin. S14's normal-only PPO therefore did not raise competence against random Red in
the sampled policy. It made the argmax policy behave more like the sampled one, and lowered casualties in both modes.

**Cohort 2's random-arm regression is a finishing failure, not a combat failure.** The following is a post-hoc
description, not a tested mechanism. Cohort 2's final deterministic policy times out in 120 of 400 worlds (the
initializer: 70; both: 50). Every one of those episodes has a hit, and the median target damage is 200 of the 240
needed for success. The policy is engaged, nearly done, and not losing units (zero wipes), yet it fails to close out
the last target before decision 200. The data here do not explain why cohort 2 alone does this.
- **Distance does not separate cohort 2.** Final policies' timed-out episodes keep a larger minimum distance than
  the initializer's in *every* cohort (4.9 vs 3.2, 5.2 vs 3.8, 5.7 vs 2.8).
- **Different subsets.** That comparison conditions on timeout episodes, which are differently selected for each
  policy.

## What this establishes and what it does not

**Establishes:**
- The S14 checkpoints can be loaded and replayed exactly. The replay identity holds only at a fixed block size and
  thread count.
- Normal-only PPO did not degrade the cohort-averaged random/easy competence beyond δ = 0.05 in either execution
  mode.
- Casualties fell on both arms.
- One of the three policies, cohort 2, loses finishing reliability against random Red in both modes.

**Does not establish:**
- Unseen-opponent generalization. Both arms were in the imitation mixture.
- That PPO improved the sampled policy against random Red. It did not detectably do so.
- A mechanism for cohort 2's regression.
- Uncertainty over future training runs. Intervals are over worlds, conditional on these three policies.

## Consequence

Per the declared recommendation for `retained-with-cohort-regression`, the S14 final/initializer pairs can serve as
the frozen executors for the command-controllability audit, with **cohort 2 reported separately** in every later
result. The audit is step B of the 2026-10-04 review. It needs its own declaration. Feasibility notes collected
during this step (read-only):
- **One cluster at reset.** The default 3v3 spawns place the three Reds 5 units apart, inside the grounder's
  6-unit cluster radius. Every S14 Engage target set was therefore all three Reds, so the fighters have never had a
  choice between clusters.
- **Mirrored axis.** `leftmost`/`rightmost` selectors and explicit `redSpawns` permit a mirrored target axis at 3v3.
- **Separate scoring.** `SnowGymBatchEnv.preview_plans` grounds a plan without activating it. The requested
  target's identity can therefore be scored independently of the execution plan.

None of this is authorized here.
