# R1m-S10: late-state learning audit

The full-horizon policy collected and selected late states, but this audit finds
weak local credit and little consistent movement toward the teacher's geometry.
The results narrow the diagnosis; they do not identify a single cause or change
S9's failed gate.

Protocol and implementation were committed as `fa21f41` before the audit.
Evidence: [immutable archive](../runs/m7b_engage_r1m_s10_v0/manifest.json),
[report](../runs/m7b_engage_r1m_s10_v0/report.json), and
[selection accounting](../runs/m7b_engage_r1m_s10_v0/selection.json).
The archive binds the S9 inventory, implementation, protocol and source lineage.
No optimizer steps, new trajectories, providers or policy changes were introduced:
the simulator only reconstructed archived trajectories.

## Reconstruction and selection

All 48 episodes matched completely, including actions, state hashes, option
records, latent samples, densities, values, rewards and GAE. All six first
minibatch losses matched with maximum error **0**. The audit used 8,399 simulator
decisions including prefix restoration, below the 9,600 cap. Source and archived
checkpoints remained unchanged. The archive contains 15 manifest-listed files
and occupies 1,411,431 bytes including its manifest.

Across all 30 S9 updates:

| Arm / time after first hit | Collected decisions | Selected occurrences | Unique selected decisions within updates |
| --- | ---: | ---: | ---: |
| Short, offsets 0–29 | 7,154 | 7,200 | 7,154 |
| Full, offsets 0–29 | 7,156 | 2,916 | 2,916 |
| Full, offsets >=30 | 10,963 | 4,284 | 4,284 |

Late decisions are 60.5% of full collection and 59.5% of its selected rows.
Unique selection rates are 40.7% early and 39.1% late. There is no late-row
exclusion in this sampling path. Equal optimizer-row budgets do reduce full's
early presentations from short's 7,200 to 2,916. This is an allocation tradeoff,
not a measured causal explanation of the performance difference. These counts
are within-update uniqueness, not globally distinct physical states.

## Credit quality

The following are collected late-decision statistics for the full arm. Updates
refer to their frozen behavior policies, before the numbered PPO update.

| Update | Late decisions | Critic explained variance | Between-episode fraction of advantage variance | Selected sample-direction / normalized-advantage correlation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 379 | -0.0045 | 87.0% | -0.015 |
| 11 | 264 | 0.0118 | 89.8% | -0.105 |
| 21 | 218 | 0.0142 | 91.1% | -0.007 |

The correlations concern displacement toward the available recommendation.
They are pooled descriptive correlations with repeated units/episodes, without
independent-opportunity confidence intervals. They do not measure the true
action advantage or prove that the recommendation is optimal. Early-state
advantage variance is even more episode-dominated: 95–98% for full and 96–99%
for short at these checkpoints.

The critic explains little variation in the archived GAE return targets. This
does not establish that critic approximation alone caused S9's failure: the
targets include bootstrap estimates, outcomes vary across episodes, and only
three behavior checkpoints are inspected.

## Exploration and cumulative output changes

Pooled living MOVE opportunities across the three audited updates:

| Arm / time | Opportunities | Mean sampled target shift | Mean recommendation gap | Within 3 latent sigma | Mean final-vs-behavior target shift | Mean gap reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short early | 2,393 | 0.973 | 7.943 | 7.40% | 1.050 | -0.0154 |
| Full early | 2,420 | 0.965 | 7.676 | 9.59% | 0.620 | -0.0400 |
| Full late | 2,661 | 0.902 | 10.958 | 3.04% | 0.720 | -0.0017 |

Distances are world-space target distances, not physical distance traveled.
The 3-sigma column uses Euclidean distance in standardized latent coordinates.
No sampled coordinate was saturated at absolute tanh output >=0.99. Every
recorded living MOVE had an available recommendation and legal MOVE mask;
helper readiness was true on only 2,535 of 7,474 opportunities. Readiness,
recommendation availability and legality are distinct fields.

Full's late recommendation gaps at updates 1/11/21 are 11.69/12.22/7.96; sampled
target shifts are 0.793/0.986/0.967. Final-policy gap reductions are
-0.0462/-0.0039/+0.0770. Among 22 full-late update/episode blocks with MOVE
opportunities, 12 have positive mean gap reduction; their median is +0.0341.
The pooled near-zero result is not evidence of uniform worsening. Outside-range
late opportunities also have mixed reductions: +0.0855/-0.0041/+0.1086.

Frozen ratio-one first-epoch mean-output scores project negatively toward the
recommendation in each full-late audit: -0.115/-0.412/-0.012. The first-minibatch
projections are -0.137/-0.584/+0.090. These are different summaries: first-epoch
scores hold behavior weights fixed even for later minibatches. They are neither
sequential Adam updates nor world-space velocity. Duplicate selected rows are
kept as occurrences, normalized in their actual minibatches, then summed.

## PPO interpretation

For the first minibatch, at behavior ratio one, the independent mean-output
ascent contribution is

$$
g_{ti} = \frac{m_{ti}}{B N_t}\widetilde A_t
\frac{z_{ti}-\mu_{ti}}{\sigma^2}.
$$

Here $N_t$ is the living roster, $B$ the minibatch size, $m_{ti}$ the living MOVE
mask, and $\widetilde A_t$ the minibatch-normalized squad advantage. The target
mapping $x=c+s\tanh(\mu)$ has Jacobian $s(1-\tanh^2\mu)$. The audit projects
that output-space direction toward the recommendation. Network parameter sharing,
gradient clipping, Adam and subsequent PPO ratios change the realized update.

The likelihood implementation can be correct while the reward-weighted local
search is uninformative. Most late recommendations lie far outside the current
small-noise neighborhood. At the same time, a shared squad advantage varies
mostly with episode outcome, and its local directional association is weak.
The observed final target changes show that parameters act on late states, but
do not show consistent progress toward those recommendations.

S8 already established that sustained teacher MOVE corrections can rescue many
specific training handoffs. S9 and S10 do not establish that independent small
endpoint perturbations can discover that sustained behavior within this budget.
Neither a blanket PPO bug claim nor an automatic increase in noise is supported.

## Next bounded decision

Before another actor run, predeclare a critic-only learnability diagnostic on
frozen on-policy states/returns. Separate fitting and validation by episode/seed,
handle behavior-policy versions explicitly, and compare against simple baselines.
Validation episodes would be disjoint from this diagnostic's fit, not unseen by
the original source policy. Discard the fitted diagnostic weights.

If the value signal can be learned, the next actor-space question is whether
temporally coherent movement exploration yields measurable action-to-return
credit under matched budgets. That needs its own protocol and likelihood/control
contract; S10 does not authorize a new decoder, reward sweep, larger noise or
teacher MOVE training. R1 and autonomous qualification remain open.

## Verification

Implementation gate: 367 TypeScript tests, 51 Python client tests and 306 Python
training tests passed; build passed. Targeted checks cover duplicate-safe scores
against autograd, zero unused gradients, offset boundaries, complete episode
parity, exact first-minibatch loss, RNG/source preservation, immutable output and
tamper rejection. No observation contract changed. Existing build chunk-size
and Gym legacy-version warnings remain.

Results gate also passed: 367 TypeScript, 51 client and 307 training tests
(725 total), plus build. The added archive regression rechecks inventory,
lineage, selection accounting, geometry summaries and reconstruction totals.
