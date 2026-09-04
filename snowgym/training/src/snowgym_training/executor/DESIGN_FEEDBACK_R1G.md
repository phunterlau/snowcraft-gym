# R1g: throw-channel results and executor design feedback

Date: 2026-09-04. Status: development diagnosis; R1 remains open.

Follow-up: [R1h completed the action-choice/movement diagnostic](../../../reviews/m7b_r1h_results.md).
Its results prioritize relative movement alongside shot geometry. The proposals
below retain the R1g reasoning at the time of that experiment.

## Question and experiment

The production plan-aware teacher can complete Engage, while its learned
imitation cannot. R1f reduced teacher-state throw-coordinate error from 18.71
to 9.74 world units, but increased mean throw-ray error from 31.81 to 45.65
degrees. This motivated a physical-channel experiment before further training.

R1g froze the R1f epoch-20 checkpoint and ran five arms on the same 40
development seeds, 200000–200039, in the open 5v5 Engage option under
`snowgym.sim.v2` and `snowgym.state.v2`. The option horizon is 200 decisions.
No optimizer steps, provider calls, browser input, or qualification seeds were
used. Model state was checked unchanged after collection.

On learner-selected throws, the direction arm replaces the aim point with the
nearest living enemy's position plus 0.18 seconds of velocity lead. The power
arm substitutes the production medium-range power rule. The combined arm
substitutes both. Action type and movement targets remain unchanged at the
intervened decision. Later decisions can differ as trajectories diverge.
The teacher arm executes the full production teacher through the same tensor
action bridge.

Recommendations are checked against actual teacher throw labels on visited
states. When the teacher chooses to move or dodge, the shot recommendation
is still defined independently; its movement target is never used as an aim
label. Some learner trajectories contain no teacher-selected throws, so this
agreement check has no coverage in those episodes.

## Results

| Executed controller | Engage success | At least one hit | Mean mission progress |
| --- | ---: | ---: | ---: |
| Frozen learner | 0/40 (0%) | 52.5% | 6.6% |
| Learner + direction recommendation | 7/40 (17.5%) | 95% | 46.7% |
| Learner + power recommendation | 0/40 (0%) | 42.5% | 6.2% |
| Learner + direction and power | 10/40 (25%) | 95% | 52.7% |
| Full production teacher | 40/40 (100%) | 100% | 84.9% |

All arms had zero rejected actions. Success means completing the Engage option,
not winning a complete battle. The learner baseline reproduces R1f development
success, progress, contact, and hit records seed by seed.

Relative to the learner, direction replacement increased success by 17.5
percentage points (paired-bootstrap 95% interval: 7.5–30 points) and progress
by 40.1 points (32.0–48.4). Replacing both increased success by 25 points
(12.5–40). Power alone changed progress by −0.4 points (−3.0–2.3).

A secondary paired comparison of both replacements versus direction alone
gives +7.5 success points (−5–20) and +6 progress points (−0.3–12.3). The
extra three successes do not establish an independent power benefit. These
intervals are exploratory development diagnostics without multiple-comparison
adjustment; the same development seeds have informed earlier recovery work.

## What this says about the design

1. **Enemy selection and shot direction deserve priority.** Correcting that
   channel substantially improves hits and progress without retraining. The
   intervention includes both selecting an enemy and aiming at it, so it does
   not identify which of those two contributes most.
2. **Coordinate regression is a weak physical objective.** The simulator turns
   the target point into a direction from the shooter; power separately controls
   projectile dynamics. For shooter position $x$ and target point $y$, the
   relevant ray is $u=(y-x)/\lVert y-x\rVert$. Endpoint error can fall while
   angular error rises, as R1f demonstrated. Handle the zero-length ray explicitly.
3. **The current target representation needs a gradient audit.** In the present
   model, `target_inputs = [actor_input.detach()]` prevents target and power losses
   from training the v3 actor entity adapters through that path. A new relational
   shot module needs its own trainable inputs or an explicitly tested gradient
   connection while inherited parameters remain frozen as intended.
4. **Power is not the first isolated change.** Power-only replacement gives no
   supported gain here. It uses the recommended enemy's distance even when the
   learner aims elsewhere, so this is not a general test of optimal power.
5. **Aim correction is insufficient.** Both recommendations achieve only 25%
   success versus the teacher's 100%. The remaining action-selection and movement
   policy must be investigated. In the unmodified learner, 1,105 of 1,468 proposed
   throws occur where the teacher would not throw. This disagreement does not
   establish that each shot is wrong or separate dodging, range, and positioning.

A controller that randomly chooses a living enemy already has a useful
targeting prior. Its action distribution is very different from sampling an
arbitrary arena coordinate. Comparisons should report this structural advantage
and use matched execution priors when assessing learned strategy.

## Proposed next bounded steps

### 1. Separate firing choice from movement

Predeclare a no-training conditional-action matrix using the same frozen actor
and development protocol. With shot recommendations held fixed, intervene on
movement versus action choice separately. If action type changes, obtain the
target from the newly selected action's head; never reuse a throw target as a
move target or the reverse. Test intervention isolation before collection.
Record disagreement reasons where available, shot distance, readiness, and
teacher coverage. This establishes which remaining channel merits repair.

### 2. Test a target-relative shot interface

Use a separately versioned architecture ablation with per-ally, masked
per-enemy selection scores and target-relative position/velocity features.
Enemy indices are lookup slots, not numeric ID features. Decode the selected
enemy through a shared deterministic aim/lead/power prior initially. Preserve
the existing movement path to limit the first ablation's scope.

Compare learned selection against the same nearest-enemy prior. Introduce a
bounded angular or power residual only in a later isolated ablation. Evaluate
direction error and physical shot outcomes alongside any coordinate loss.
The public move/throw target action contract can remain compatible, but the
checkpoint and PPO likelihood contracts must identify the new selection
distribution. Learned enemy selection must contribute to stored and reevaluated
log probabilities; deterministic decoding adds no independent sampled action.

Required tests include living-target masks, casualty replacement, permutation
equivariance, finite zero-distance handling, coordinate/arena transforms,
teacher-shot parity, gradient reachability, unused-head zero gradients, and
stored-versus-reevaluated likelihood equality. Keep legacy checkpoints loadable.

### 3. Restore physical and plan gates before PPO

Run a bounded supervised fit and closed-loop evaluation with the same-state
HOLD counterfactual. R1f weakened plan selectivity, so successful shooting alone
cannot justify promotion. Check sampled as well as deterministic execution
before restarting PPO. Predeclare the configuration and learning thresholds
before fitting; do not tune on qualification seeds.

R1's recovery threshold and M7b/M7c qualification criteria remain unchanged.
R1g is an oracle-assisted diagnostic, not a learned-executor result. No new
architecture or automatic aiming override was installed in the production actor
by this experiment.

## Evidence and reproduction

- [Frozen configuration](../configs/m7b_engage_r1g_throw_channels_v0.json)
- [Runner and recommendation implementation](../options/throw_channels.py)
- [Result report](../../../runs/m7b_engage_r1g_throw_channels_v0/report.json)
- [Manifest and provenance](../../../runs/m7b_engage_r1g_throw_channels_v0/manifest.json)
- [Preceding R1f review](../../../reviews/m7b_r1f_results.md)

Each arm retains all 40 episode records, state-hash trajectories, and executed
action-sequence digests. These are audit records, not visual replay JSON files.
The manifest binds checkpoint, protocol, simulator capabilities, source hashes,
and artifact hashes. Reproduce from `snowgym/training` into a new output path:

```bash
.venv/bin/python -m snowgym_training.options.throw_channels \
  --checkpoint runs/m7b_engage_r1f_supervised_probe_v0/epoch-020 \
  --output /tmp/snowgym-r1g-reproduction
```

The command starts a detached batch host; no HTTP server or browser is needed.
