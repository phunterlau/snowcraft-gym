# R1m-S7: handoff changes movement; completion remains unresolved

Completed 2026-09-05. The tested [inspection protocol](m7b_r1m_s7_declaration.md)
was committed as `c8069c4` before collection. This report interprets S6 and
inspects its recorded actions. No policies, rewards, horizons, exploration
settings, qualification gates, or provider settings changed.

## What S6 established

The [S6 duration/scope experiment](m7b_r1m_s6_results.md) improved final success
from 9/24 to 12/24 when teacher destinations replaced all living source-selected
MOVE targets for 30 decisions. The paired gain was +12.5 percentage points,
95% interval [-4.2, +29.2]; return gain was +0.2399 [-0.0449, +0.5283].
Its primary gate failed. Local damage and progress improved, but those changes
did not establish reliable completion improvement. One-decision perturbations
from S5 and single-fighter corrections from S6 were weaker.

This favors investigating sustained engagement and the continuation policy.
It does not establish that PPO likelihoods are wrong, that a different coordinate
system would solve learning, or that larger action noise is sufficient.

## Exact replay and measurement boundaries

S7 replayed the saved Keep and squad-30 actions on all 24 S6 training seeds,
48 trajectories. Every prefix identity, transition hash, reward, complete option
record, action result, terminal boundary, and final outcome matched. Read-only
recommendations agreed with production and preserved state. No new policy
decisions were generated. Source checkpoints and S3/S5/S6 archives remained
unchanged.

The run used 8,405 prefix-plus-continuation decisions, below 9,600. Its
[manifest](../runs/m7b_engage_r1m_s7_v0/manifest.json) binds 50 artifacts: 48
compressed inspections, declaration, and report (approximately 2.04 MB including
the manifest). Inventory, file and manifest digests, all window summaries, report
recomputation, and decision accounting verified. The implementation commit passed
367 TypeScript, 51 client, and 287 training tests plus build. The results gate
passed 367 TypeScript, 51 client, and 288 training tests (706 total) plus build,
including the new archive-recomputation regression. Existing bundle-size and
legacy Gym-version warnings remain; no observation contract changed.

All measurements use detached server state. Range 9 and desired distance 6.5
are the teacher's reference values, not guarantees of a hit. Throw orders are
not projectile-launch counts. An outward move may be a dodge; these counts alone
do not label every action as incorrect. Ratios are computed within each seed's
window, then averaged over seeds with defined denominators.

## Around the 30-decision handoff

Seventeen paired seeds have complete [20,30) and [30,40) windows in both arms.
Seven others terminate too early in at least one arm and are excluded from these
paired temporal comparisons, while remaining in the full descriptive report.
The columns below are means on that complete cohort; the outward-MOVE ratio is
defined for all 17 squad-30 seeds but only 15 Keep seeds.

| Metric | Keep, offsets 20–29 | Keep, offsets 30–39 | Squad-30, offsets 20–29 | Squad-30, offsets 30–39 |
| --- | ---: | ---: | ---: | ---: |
| Damage dealt | 30.59 | 25.88 | 57.65 | 50.59 |
| Progress gain | 0.0612 | 0.0518 | 0.1153 | 0.1012 |
| Occupancy within 9 | 22.2% | 16.4% | 29.4% | 24.0% |
| Range error from 6.5 | 8.408 | 9.154 | 4.780 | 5.433 |
| Outward fraction of MOVE orders beyond 9 | 35.4% | 45.5% | 4.6% | 45.4% |

For squad-30, the within-seed outward-MOVE increase is +40.8 points
[23.3, 58.4]. Relative range-opening velocity beyond 9 increases from 13.2% to
48.1% of opportunities. The difference between squad-30's temporal change and
Keep's temporal change is +36.4 points [4.5, 69.5] for outward MOVE and +24.9
[2.1, 49.9] for range opening; each comparison has 15 defined seed pairs.

The corresponding damage contrast is -2.35 health [-36.47, 30.59], and progress
contrast is -0.0047 [-0.0729, 0.0612]. Immediate progress collapse is therefore
unsupported. Correction can leave useful positioning or in-flight projectiles,
so immediate damage need not track an order change instantly; that explanation
is a hypothesis, not an identified mechanism from these measurements.

These are descriptive, unadjusted paired-bootstrap intervals (10,000 resamples,
RNG 991001). Neither before/after nor the difference of temporal changes isolates
the causal effect of handoff: states, enemies, time, and intervention history
also change. Keep is not the same state at offset 30. No seed has defined
ready-and-in-range opportunities in all four windows, so that four-way paired
ratio contrast is unavailable; marginal percentages must not substitute for it.

## Where unsuccessful continuations end

All 12 squad-30 failures time out with every assigned Blue fighter alive.
Eleven have a post-handoff tail; seed 100011 has only eight decisions left at
the initial branch and never reaches handoff.

| Post-handoff statistic | Successful trajectories (12) | Failed trajectories with tail (11) |
| --- | ---: | ---: |
| Progress gain per decision | 0.01829 | 0.00449 |
| Occupancy within 9 | 34.3% | 11.8% |
| Range error from 6.5 | 4.360 | 11.379 |
| Outward fraction of MOVE orders beyond 9 | 36.4% | 62.4% |
| Fraction of throw orders beyond 9 | 67.3% | 83.6% |

The 12 failures average 32.7 final decisions without target-health progress;
eight have at least 38 such decisions. Successful episodes terminate when they
cross the progress threshold, so their zero trailing-stall count follows the
termination rule and is not independent evidence of better control. Success/fail
strata are outcome-conditioned, with unequal exposure, and cannot establish
causality. A legal throw with an available recommendation may still be poorly
timed or geometrically ineffective.

Budget also matters descriptively: squad-30 succeeds on 0/2 states with <=60
decisions remaining, 1/8 with 61–100, and 11/14 with >100. First-contact timing,
state difficulty, and budget are confounded. Do not extend the horizon or remove
late-contact cases after seeing these results.

### Concrete case: training seed 100019

The branch begins with 109 decisions remaining and progress 0.04. Teacher
movement reaches progress 0.36 by handoff; the frozen continuation reaches 0.76,
then makes no further progress for its final 45 decisions. Enemy 9 has 20 health
and enemy 10 has 100 at termination. Their 120 remaining health exceeds the
100-health success threshold by 20; the score correctly reports failure.

Before the final action, all five Blue fighters are 20.86–23.45 units from
their nearest living enemy, ID 9. All choose MOVE destinations directed away
from it, and none has a detected incoming projectile threat. All are alive and
movement is legal. This is a concrete late pursuit failure in the archived
trajectory, without a claim that changing only its last action would recover
the episode. Seed 100020 also stops at 0.76 but its last moves point toward the
enemy: final-action direction alone does not explain every failure.

## Implications for PPO and the control design

### Keep likelihood accounting; examine the controlled horizon

The existing latent Normal policy and its deterministic tanh-to-arena mapping
support valid stored-latent likelihood ratios. S4 reproduced those likelihoods;
S5 showed that destination perturbations affect physical motion. The evidence
does not justify replacing PPO merely to repair probability accounting.

The short recovery experiment instead optimizes a restricted objective. With
$K$ learned decisions followed by frozen source policy $\pi_0$, its expected
return is

$$
J_K(\theta)=\mathbb{E}_{\pi_\theta,\pi_0}
\left[\sum_{t=0}^{K-1}\gamma^t r_t+
\gamma^K V^{\pi_0}(s_K)\right].
$$

For trajectories terminating earlier, $K$ is truncated at termination. The
collector folds the actual frozen-tail rewards into the final learned reward;
it does not drop the tail. Its objective asks the learned prefix to produce
states that the frozen source can finish. A policy that would succeed with
continued corrective movement can still fail under this objective. S7 supplies
examples of weak later pursuit, but has not measured its causal contribution.

### Preserve the reward distinction

For the implemented potential term with terminal potential zero,

$$
\sum_{t=0}^{T-1}\gamma^t
\left(\gamma\Phi(s_{t+1})-\Phi(s_t)\right)=-\Phi(s_0).
$$

On a fixed starting state this term adds no separate terminal value for partial
progress. It can affect finite-sample learning with an approximate critic and
GAE. Combat reward also distinguishes damage outcomes. Nevertheless, reporting
more intermediate damage is not sufficient to demonstrate success on the
terminal mission objective. Adding a terminal damage reward would change the
objective and needs a separate declaration; no such change was made here.

### Exploration and credit still need physical meaning

The actor updates movement only when the frozen classifier selects MOVE, using
the shared team advantage and per-living-unit normalization. Corrected shots and
categorical decisions receive no actor likelihood. A trajectory may therefore
combine poor action timing with weak pursuit, while movement PPO can change
only one of those channels. A shared return also provides limited attribution
among simultaneous fighters. These restrictions are explicit experimental
boundaries, not evidence of an arithmetic error.

S5/S6 favor testing sustained useful motion before increasing independent
destination noise, KL limits, or optimizer budget. Relative coordinates alone
do not alter an equivalent action distribution. A future temporally persistent
movement policy would change the physical exploration process; it would need
explicit commitment state and likelihoods only at actual stochastic decisions,
with discounted rewards and bootstrapping matched to their duration.

## Recommended next experiment — not executed

Predeclare a same-state continuation fork at the end of S6's squad-30 correction.
For every archived state that reaches offset 30 without termination, reconstruct
that exact state and compare source continuation with teacher MOVE destinations
for the remaining original option budget. Keep classifier, corrected shots,
scoring, and seeds fixed; reproduce the archived source branch exactly. Report
all eligibility exclusions and paired completion/return intervals. Set its
decision budget and success gate before running it.

If sustained teacher movement reliably recovers these states, the next learned
comparison should keep movement control through the remaining option against
a matched full-duration initialization, rather than inherit the 30-decision
handoff. Keep the terminal success gate. If it does not recover them, inspect
shot timing and target pursuit before spending more PPO updates. Do not silently
add both a new action parameterization and a new reward in the same comparison.

The autonomous fighter remains unqualified. All S6/S7 trajectories retain their
teacher-assistance labels; no PPO run, commander experiment, or promotion was
authorized by this inspection's outcome.
