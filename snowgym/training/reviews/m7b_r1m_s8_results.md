# R1m-S8: sustained movement recovers nine same-state failures

Completed 2026-09-05. The [protocol](m7b_r1m_s8_declaration.md) and tested
implementation were committed as `ce9ea8d` before collection. The diagnostic
continuation gate passed. No PPO training, reward changes, provider calls,
browser input, or autonomous promotion occurred.

## Design and validity

Reuse every S6 squad-30 trajectory with a nonterminal state after its 30th
correction decision: 23 previously exposed training seeds. Seed 100011 timed
out after eight decisions from first hit and is the sole exclusion. Selection
does not depend on the later outcome. The excluded state remains a failure in
the earlier S6/S7 records; this experiment estimates a handoff-conditional effect.

Both arms replay the same original prefix and the same 30 teacher-correction
actions. Physical, plan, option and observation identities match at the fork.
The source arm generates the archived frozen-source suffix exactly. Move-rest
changes only living source-selected MOVE destinations, recomputing conditional
teacher recommendations on its current state until the original option ends.
Classifier weights and corrected-shot rules stay frozen in both arms. Their
later action choices and opponent actions can differ because their states diverge.
Recommendations are recomputed at the existing 10 Hz decision rate. This tests
how long the corrective controller remains active, not holding one action for
longer or sampling a temporally extended action.

All 46 repeated branch pairs matched exactly: 92 executions, not 92 independent
environments. All 23 generated source suffixes reproduced archived actions,
hashes, rewards, option records, action results and final outcomes. Source
weights and S3/S5/S6/S7 archives remained unchanged. The experiment consumed
14,624 prefix-plus-continuation decisions, below 19,200. Zero continuation
actions were rejected among 6,650 retained action results (13,300 with repeats);
prefix actions are excluded from those rejection counts.

The [archive](../runs/m7b_engage_r1m_s8_v0/manifest.json) binds 50 artifacts
(approximately 2.27 MB including its manifest): 46 complete compressed branches,
declaration, selection, duplicate digests and report. Inventory, source lineage,
all artifact and repeat digests, window/report recomputation, action-channel
isolation, prefix identity and simulator accounting verified. The implementation
gate passed 367 TypeScript, 51 client and 292 training tests plus build.
The results gate passed 367 TypeScript, 51 client and 293 training tests (711
total) plus build, including the new full-archive regression. Existing bundle-size
and legacy Gym-version warnings remain. No environment observation contract changed.

## Primary result

Intervals are paired seed-bootstrap intervals, 10,000 samples, RNG 992001.
They describe variability within this selected, previously exposed training
cohort; they are not qualification or generalization evidence.

| Endpoint | Source continuation | Sustained MOVE correction | Paired difference [95% interval] |
| --- | ---: | ---: | --- |
| Mission successes | 12/23 (52.2%) | 21/23 (91.3%) | +39.1 points [21.7, 60.9] |
| Discounted executor return from handoff | -0.3954 | +0.3097 | +0.7051 [0.3847, 1.0936] |
| Damage dealt after handoff | 113.04 | 139.13 | +26.09 [9.57, 44.35] |
| Damage received after handoff | 20.00 | 20.87 | +0.87 [-7.83, 11.30] |

The >=10-point gain, positive success/return interval lower bounds, and <0.1%
rejection criteria all pass. Nine failures recover, with no source success lost:
100003, 100004, 100009, 100012, 100013, 100017, 100019, 100020 and 100022.
This does not retroactively pass S6's different primary gate.

Mission success is the existing Engage threshold (activated target health at
most 20% of its initial denominator). It is not a full battle win: canonical
discounted reward is zero in both arms, and these options can terminate while
enemies remain. Keep these distinctions in replay labels and future comparisons.

## Physical interpretation

These are means of per-seed whole-tail statistics; trajectories terminate at
different times, so they are not common-duration measurements.

| Statistic | Source | Move-rest |
| --- | ---: | ---: |
| Occupancy within reference range 9 | 23.6% | 42.5% |
| Mean absolute range error from 6.5 | 7.717 | 4.401 |
| Outward fraction of MOVE orders beyond 9 | 48.8% | 1.1% |
| Relative range-opening fraction beyond 9 | 51.8% | 17.1% |
| Fraction of throw orders beyond 9 | 75.1% | 47.4% |
| Mean corrected MOVE destinations | 0 | 79.35 |

Final progress increases by 0.0522 [0.0191, 0.0887]. Continuation exposure is
15.04 decisions shorter on average [-24.04, -6.83], reflecting earlier mission
termination. Mean living fraction changes by -0.0087 [-0.0261, 0]: the correction
does not uniformly improve every secondary outcome. Readiness/range/action
metrics and their denominators remain in the machine-readable report.

The controlled intervention recovers nine failures without changing the horizon
or classifier weights. This establishes that continued MOVE destinations matter
for completion in these states under this opponent and shot assistance. It does
not isolate whether pursuit direction, formation, cohesion or dodging inside
the movement heuristic supplies the benefit. Range and throw-order statistics
help describe the behavior; they do not prove hit probability or projectile
launch counts.

### Seed 100019: the stalled S7 example

Both arms start at progress 0.36 with 79 decisions remaining. The source ends
at 0.76 when its budget expires. Sustained correction reaches 0.80 after only
25 decisions, leaving 54 decisions unused, with all five Blue fighters alive.
It deals 220 rather than 200 additional health damage and receives zero damage
in both branches. Range occupancy changes from 3.5% to 64.0%; 93 MOVE targets
are replaced. This is recovery from the same handoff, not a longer timeout.

### Remaining failures

- Seed 100000 has only 15 decisions left at handoff. Both arms end at progress
  0.40. Extra movement correction does not rescue this late state.
- Seed 100015 has 51 decisions left. Source reaches 0.64; move-rest reaches
  0.60 despite 197 replacements. Sustained teacher movement is not uniformly
  beneficial; this case remains available for later targeting/timing inspection.

Both fail by timeout with all assigned Blue alive. Do not alter their scoring,
remove them or extend the budget after seeing these results.

## What this changes for PPO

The evidence now supports treating the frozen continuation as an important
restriction on the earlier short-window movement experiment. That objective is

$$
J_K(\theta)=\mathbb{E}\left[
\sum_{t=0}^{K-1}\gamma^t r_t+\gamma^K V^{\pi_0}(s_K)
\right],
$$

where only the first $K$ decisions are learner-controlled. A sustained executor
would instead remain responsible for movement through option termination. S8
shows that a better movement continuation exists for many teacher-corrected
states. It does not show that PPO can discover it, that S3's learned prefixes
reach the same states, or that critic/exploration limitations disappear.

The return gain is not an artifact of accumulating more potential reward.
For each matched starting state, zero terminal potential makes the discounted
shaping sum equal to minus its starting progress. Maximum paired shaping
difference is below $2\times10^{-15}$. The mean executor gain decomposes into
approximately +0.7003 discounted mission reward and +0.0048 weighted combat
reward, with effectively zero shaping difference. The existing terminal mission
objective already values these recoveries; changing it is not required to
explain this result.

Keep stored-latent likelihood ratios, unused-head masking, and per-living-unit
normalization. Prefer a controlled test of sustained learned movement before
increasing KL limits, adding a terminal partial-progress reward, widening noise,
or introducing a new decoder. Any temporal action abstraction would still need
its own sampling/likelihood and discount contract; S8 does not validate one.

## Next predeclaration, not yet executed

Specify a bounded sustained-movement PPO comparison with a matched initialization,
explicit starting-state distribution, original option limits, and fixed shot
assistance. Make the learned control duration the intended change. If using these
post-correction states, record the teacher-controlled prefix as assistance and
never describe the run as autonomous. If returning to S3's first-hit reservoir,
state that S8 did not test the same distribution and reproduce its baseline.

Predeclare simulator and optimizer budgets, final-only selection, development
evaluation and replication gates. Longer controlled rollouts yield more actor
samples, so matching update count alone does not match data or optimization dose.
Use zero-residual deterministic parity and exact resume checks before fitting.
Keep runtime teacher MOVE overrides off in the learned segment, preserve the
current reward, and leave classifier/aiming changes for separate experiments.

Only a separately qualified autonomous executor with all runtime assistance
removed can close R1. This successful diagnostic does not advance M7b/M7c or
authorize online commander experiments.
