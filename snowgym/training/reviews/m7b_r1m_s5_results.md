# R1m-S5: destination-to-motion boundary results

Completed 2026-09-05. The [declaration](m7b_r1m_s5_declaration.md) and tested
implementation were committed as `833cd68` before measurement. Evidence is in
[the immutable archive](../runs/m7b_engage_r1m_s5_v0/manifest.json).

## Finding

The action-to-motion map is direction-dependent: lateral target perturbations
produce much more immediate movement than outward radial perturbations. Larger
perturbations cause larger physical differences, but neither scale nor any
individual arm shows a reliable return improvement. There is no support here
for promoting a policy, widening PPO noise by default, or declaring destination
control invalid. No optimization or decoder change occurred.

## Protocol and verification

The first 24 eligible S3 training snapshots were selected in ascending seed
order, with no additional eligibility exclusions. Seeds are 100000 and
100002–100024; 100001 is absent from the archived first-hit inventory. One living,
source-selected MOVE fighter per snapshot was chosen by smallest eligible ID.
This is conditional, training-only evidence, not an untouched qualification set.

Ten branches per state used Keep, signed radial/lateral target shifts at 1/5
world units, and a conditional teacher-target reference. Only one fighter's
first movement destination changed. All branches then recomputed the frozen
source policy with corrected shots to the original option horizon. No new
orders were forcibly held or canceled. Changed persistent targets retained
normal engine semantics. Shot assistance makes every trajectory autonomous-
qualification ineligible.

All 240 independently repeated branch pairs matched exactly. All 24 Keep
branches matched their archived action digests and complete suffix state hashes.
Prefix restoration verified physical, plan, option and observation identity.
Read-only helper agreement, frozen source weights, original S3 source files and
artifact manifests passed. There were no clipped candidate destinations.

Selection used 2,493 prefix decisions. Selection plus both copies of all branches
used 90,475 simulator decisions, within the declared 107,400 budget. There were
zero rejected actions among 95,305 retained continuation action results, or
190,610 counting duplicate executions. These action counts exclude prefix and
selection replay. The archive contains 244 manifest-listed artifacts, including
240 compressed full branch traces, both repeat digests, exact selected prefixes,
configuration and report; its manifest self-digest and inventory verified.

## Immediate physical effect and full return

Position differences are relative to Keep after one decision (0.1 seconds),
not distance traveled from the initial position. Return is the full remaining
discounted executor reward. Each row has 24 paired seeds. Return intervals are
95% paired bootstrap intervals, descriptive and unadjusted for multiple tests.

| First target change | Effective target shift | Mean position difference | Return difference [95% CI] | Success |
| --- | ---: | ---: | --- | ---: |
| Keep | 0 | 0 | 0 | 9/24 |
| Radial +1 | 1 | 0.00412 | +0.0013 [-0.0020, +0.0062] | 9/24 |
| Radial -1 | 1 | 0.01774 | -0.0691 [-0.2073, +0.0032] | 8/24 |
| Lateral +1 | 1 | 0.03845 | -0.0707 [-0.2874, +0.1407] | 8/24 |
| Lateral -1 | 1 | 0.04748 | -0.0043 [-0.2111, +0.1963] | 9/24 |
| Radial +5 | 5 | 0.00630 | -0.0644 [-0.2806, +0.1529] | 8/24 |
| Radial -5 | 5 | 0.15912 | +0.0009 [-0.2735, +0.2760] | 9/24 |
| Lateral +5 | 5 | 0.12319 | -0.0678 [-0.2823, +0.1483] | 8/24 |
| Lateral -5 | 5 | 0.20067 | -0.1384 [-0.4201, +0.1405] | 7/24 |
| Teacher target | 7.5014 | 0.24409 | -0.0651 [-0.2811, +0.1500] | 8/24 |

The inward five-unit shift frequently enters the slowing region or reverses
direction; its large effect should not be interpreted as a contradiction of
the far-target radial derivative. Only 12/24 of those cases remain in the
declared unclipped, same-ray, outside-slow-radius subset. In that subset, mean
position difference is 0.00451. The corresponding means are 0.00086 for radial
+1 (22 cases), 0.00041 for radial -1 (19), and 0.00204 for radial +5 (22).

No perturbed branch had position difference <=1e-6 after the first decision.
Thus the full engine does not exhibit an exact zero-motion direction under
these finite perturbations. The analytic derivative describes desired velocity
at the initial state. The simulator then advances six physics ticks with changing
positions, separation and acceleration; the local approximation is not an exact
multi-tick equivalence claim.

## Scale comparison

For each seed, average its four signed directions before aggregating seeds.
No best direction is selected after observing returns.

| Perturbation scale | Mean first-decision position difference [95% CI] | Mean signed return difference [95% CI] | Seeds with any return change >1e-6 |
| --- | --- | --- | ---: |
| 1 unit | 0.02694 [0.01983, 0.03638] | -0.03570 [-0.14062, +0.05153] | 20/24 |
| 5 units | 0.12232 [0.09903, 0.14711] | -0.06741 [-0.27666, +0.15783] | 22/24 |

The paired mean position-difference increase is 0.09538 [0.07581, 0.11741].
Larger actions clearly change physical behavior more in this sample. Both mean
return differences are negative with intervals spanning zero. Return variation
and physical sensitivity are measurable even at the small scale; this contradicts
a strong claim that the current neighborhood is completely behaviorally inert.

One pair per arm terminated before decision 30 and is explicitly censored from
that fixed-time comparison. The remaining 23 pairs retained the selected fighter
alive in both branches. The archive reports position, velocity, nearest-enemy
range error, health and mission outcomes separately. Later separation can be
substantial even when the first position difference is small, because the
closed-loop policy and opponent react to diverging states.

## Design implications

1. **Physical conditioning matters.** Equal Euclidean target changes have unequal
   steering effects. A global isotropic destination distribution is an imperfect
   match for short-horizon control. This supports measuring controller-aware
   exploration, not replacing PPO's valid latent likelihood calculation.
2. **More amplitude alone has no demonstrated benefit.** Five-unit perturbations
   change motion more but do not improve mean return in this probe. The finite
   signed probes are not sampled Gaussian PPO policies, so this is also not a
   definitive rejection of every larger-variance training setup.
3. **The useful teacher intervention has unresolved temporal and team structure.**
   S2 corrected movement for all selected movers over 30 decisions on different
   development states. Here one teacher destination for one fighter at one
   decision produces a large physical response without a reliable reward gain.
   This does not contradict S2 or prove teacher movement is harmful. Duration,
   coordination, state coverage and subsequent control differ.
4. **Credit assignment remains unresolved.** This probe measures deterministic
   finite differences in outcomes, not advantages under S3's stochastic policy.
   It cannot establish whether PPO ranked its own sampled actions correctly.
   The S4 critic/advantage evidence remains relevant.

Do not choose a new decoder or widen the PPO budget on these results alone.
The next proposed diagnostic is a separately declared, matched training-state
teacher-duration/coordination contrast: one selected fighter versus all selected
movers, and one decision versus a 30-decision closed-loop correction, retaining
the source classifier and corrected shots. Keep the selected single fighter's
identity stable through casualties, record actual MOVE opportunities, and retain
the original option horizon. That experiment would test whether the positive
teacher effect transfers to these training states and whether duration/team
coordination is necessary. It has not been declared or run.

If a coherent correction is useful while independent short perturbations are
unhelpful, a subsequent action-duration or local-control experiment becomes
better motivated. Its action likelihoods, physical adapter and initializer need
their own contract. Autonomous Engage, the remaining missions and composition
remain gated.

## Verification gate

Targeted tests cover radial/lateral and arrival geometry, clipping, finite and
degenerate inputs, eligibility, action-head separation, one-fighter substitution,
unchanged source, exact baseline and duplicate trajectories, prefix/seed tamper
rejection, censoring, immutable output and manifest self-digest/inventory checks.
Full implementation and results gates: 367 TypeScript tests, 51 client tests,
278 training tests, and production build. Existing bundle-size and legacy-Gym
warnings remain. No environment tensor contract changed; no browser or provider
call was needed.
