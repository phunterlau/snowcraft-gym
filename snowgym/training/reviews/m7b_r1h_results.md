# R1h: action choice versus movement destinations

Date: 2026-09-04. Status: completed development diagnostic; R1 remains open.

## Result

With teacher-style shots held fixed, replacing movement destinations recovers
40/40 Engage successes while retaining the frozen learner's action-choice policy.
Replacing action choice while retaining learned movement gives 11/40, compared
with the shot-corrected baseline's 10/40. Movement destinations are the larger
remaining bottleneck in this frozen open 5v5 scenario.

| Action choice | Movement destination | Engage success | Hit rate | Mean progress |
| --- | --- | ---: | ---: | ---: |
| Learned | Learned | 10/40 (25%) | 95% | 52.7% |
| Learned | Teacher-style | 40/40 (100%) | 100% | 84.7% |
| Teacher | Learned | 11/40 (27.5%) | 90% | 53.7% |
| Teacher | Teacher-style | 40/40 (100%) | 100% | 84.9% |
| Full production teacher | Full production teacher | 40/40 (100%) | 100% | 84.9% |

The first four arms use the same nearest-enemy aim/lead and power recommendation.
All 200 episodes had zero rejected actions. No model parameters changed and no
training or provider calls occurred. Success is the fixed Engage option's
criterion within 200 decisions, not a full-battle win or learned-policy
qualification.

## Experimental controls

- Checkpoint: R1f epoch 20, digest
  `sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697`.
- Seeds: the same 40 paired development seeds, 200000–200039; no qualification
  seeds. Simulator and state hash versions are both v2.
- The shot-only baseline exactly reproduces R1g's direction-plus-power action
  digests and complete state-hash trajectories on every seed.
- Combined teacher choice and movement exactly reproduce the full teacher's
  state-hash trajectories on every seed. Unused tensor fields can differ, so
  action digests are not required to match for this comparison.
- Every actual production teacher move and throw on visited states is checked
  against its recommendation. Recommendations are independently defined even
  when the teacher chooses another action type.
- Conditional learned target heads are selected after any action-type change.
  The older R0 helper could retain the wrong conditional target; its historical
  action-only evidence should not be treated as this controlled comparison.

## Paired development comparisons

| Intervention contrast | Success difference | Paired-bootstrap 95% interval |
| --- | ---: | ---: |
| Movement replacement with learned choice | +75 points | +60 to +87.5 |
| Choice replacement with learned movement | +2.5 points | −12.5 to +17.5 |
| Movement replacement with teacher choice | +72.5 points | +57.5 to +85 |
| Choice replacement with teacher movement | 0 points | 0 to 0 |
| Choice × movement interaction | −2.5 points | −17.5 to +12.5 |

Movement replacement with learned choice also increases progress by 32 points
(23.7–40.4). Choice replacement with learned movement increases progress by
1 point (−6.4–8.4). Intervals use 10,000 paired bootstrap draws, seed 750001.
They are exploratory, without multiple-comparison adjustment. These development
seeds have informed earlier recovery work. A zero bootstrap interval at the
40/40 ceiling does not prove that the two policies are equivalent generally.

## Why the movement result matters

In the shot-corrected baseline, 1,739/2,010 executed throws (86.5%) occur beyond
the teacher's nine-unit firing threshold. Mean shot distance is 15.28 units.
After replacing movement destinations, the corresponding counts are 695/1,457
(47.7%), and mean distance falls to 10.00 units. Full-teacher mean shot distance
is 7.99 units, with no shots outside its threshold. The threshold is the
teacher's policy rule, not a claim that every longer shot is physically unable
to hit.

All executed throws pass the recorded readiness predicate. The baseline has
132 shots with a nearby incoming threat and the movement-replaced arm has 134.
Readiness, threat, and range flags describe overlapping state conditions; they
are not exclusive explanations for teacher choices.

Teacher choice with learned movement eliminates out-of-threshold shots but
completes only 11 missions. Better firing decisions alone do not repair where
the actor moves. Conversely, teacher-style destinations allow the learned
action-choice policy to complete this task despite substantial firing
disagreement. The restored movement recommendation includes range keeping,
formation, cohesion, and immediate dodging; this matrix does not isolate those
components. Actions later in a trajectory can change because the visited state
changes, even when the action-choice parameters remain frozen.

## Feedback on architecture and training priorities

1. Retain the existing action classifier for the first representation ablation.
   R1h gives little evidence that replacing it is the highest-value first step
   under corrected shot geometry. It still needs evaluation on other missions,
   sampled execution, and harder conditions.
2. Expand the proposed target-relative redesign to include movement. R1g shows
   a shot-direction/selection bottleneck; R1h shows that a shot-only repair
   leaves an important movement bottleneck. Purely extending action-logit
   training or PPO duration is not the next justified intervention.
3. Use per-unit relative geometry and a mission-aware movement reference, with
   explicitly trainable adapters. Keep the action-type contract and separate
   move/throw heads. Audit target-loss gradients into the new representation:
   the existing detached target path cannot train the actor's v3 entity adapters
   through target loss.
4. Keep hand-coded priors explicit. A teacher-derived movement reference can
   provide a controlled scaffold, but it already contains useful tactical
   behavior. Learned residuals must improve over the same fixed-prior baseline;
   retaining that prior is not evidence of learning. A nearest-enemy Engage
   reference alone cannot represent HOLD, WITHDRAW, or SUPPORT.

### Next bounded milestone

Predeclare a target-relative representation/gradient probe before additional
PPO. Freeze the inherited action classifier and compare the existing absolute
target path with trainable relative shot and movement paths on the same audited
teacher corpus. Separate representation changes from any added execution prior.
Measure teacher-ray error, movement destination error, gradient reachability,
and closed-loop Engage progress, with a same-state HOLD counterfactual to catch
loss of plan selectivity.

First require small-batch fitting and conditional-head correctness, then a
bounded supervised run and paired development evaluation. For any sampled
enemy-selection or residual distribution, version the checkpoint/likelihood
contract and test stored-versus-reevaluated log probabilities before PPO.
Success must be attributable to changed learned parameters relative to its
initializer and matched prior. Existing R1 and M7b/M7c thresholds remain fixed.

This document selects a design direction; the relative architecture and its
training configuration are not implemented or qualified by R1h.

## Artifacts and reproduction

- [Report](../runs/m7b_engage_r1h_control_channels_v0/report.json)
- [Manifest](../runs/m7b_engage_r1h_control_channels_v0/manifest.json)
- [Frozen configuration](../src/snowgym_training/configs/m7b_engage_r1h_control_channels_v0.json)
- [Runner](../src/snowgym_training/options/control_channels.py)
- [R1g design feedback](../src/snowgym_training/executor/DESIGN_FEEDBACK_R1G.md)

The manifest binds checkpoint, source, protocol, capability, and artifact
digests. Each arm retains all 40 episode records, per-decision state hashes,
action-sequence digests, and control telemetry. These are audit artifacts,
not visual replay JSON files. From `snowgym/training`, use a fresh output path:

```bash
.venv/bin/python -m snowgym_training.options.control_channels \
  --checkpoint runs/m7b_engage_r1f_supervised_probe_v0/epoch-020 \
  --output /tmp/snowgym-r1h-reproduction
```

The command starts its own detached batch host; no HTTP server or browser is
needed. Output directories cannot be overwritten.
