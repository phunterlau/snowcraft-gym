# Fixed-plan mission options

This package owns the M7b executor objective. Each episode activates one
validated `CommandPlan`; commander scheduling, lifecycle replacement, and
provider calls remain disabled. The TypeScript server continues to own physics,
assignments, target replacement, plan projection, and action validation.

`FixedPlanOptionBatchEnv` combines the v3 physical observation with fresh
`[3,38]` plan tensors, `[3,20]` role state, and mission progress. Its reward
record keeps five values separate:

```text
mission
combat
potential shaping
canonical battle reward
executor = mission + 0.1 * combat + shaping
```

Combat is clipped to `[-1,1]` after damage dealt and received are normalized by
the corresponding team’s initial maximum health. Mission success produces
`+1`. Assigned-group elimination, battle failure before success, or the option
horizon produces `-1`.

## Frozen definitions

| Option | Success condition | Horizon |
| --- | --- | ---: |
| Engage | Selected objective health is at most 20% | 200 |
| Advance | At least 80% of living members are within 10% of arena diagonal from the frozen region anchor for 10 decisions | 150 |
| Hold | Every living member is within 8% of arena diagonal from the activation anchor for at least 90% of 150 decisions, with half the assigned force alive | 150 |
| Withdraw | At least 80% of living members remain within 10% of arena diagonal from the backfield anchor for 20 decisions, with half the assigned force alive | 200 |
| Flank | The group reaches the commanded signed 20% activation-frame lateral extent and damages an enemy within the following 50 decisions | 200 |
| Focus | After damage equal to 10% of one mean enemy maximum-health unit, target-damage HHI is at least 0.65 | 200 |
| Distributed | At least two enemies each receive meaningful aggregate damage and damage entropy normalized over the initial enemy roster is at least 0.65 | 200 |
| Support | Both groups retain half their assignments while their living centroids remain 8–18% of arena diagonal apart for 30 decisions | 300 |

“Settled” in Withdraw is operationalized as maintaining the arrival condition
for 20 consecutive decisions. No velocity threshold is added. Damage
concentration uses health loss per enemy ID as host-side evaluation metadata;
IDs do not enter the learned tensors.

The immutable protocol is
[`m7_option_protocol_v0.json`](../configs/m7_option_protocol_v0.json). It fixes
thresholds, 10 Hz timing, 40 paired development seeds and 100 paired
qualification seeds per mission, plus disjoint training, plan-generation, and
sealed-map ranges.

## Teacher achievability gate

`evaluate_teacher_option` runs the production `PlanAwareTeamController` through
the same persistent server and option tracker used by learning. The test suite
executes all eight options on predeclared seeds. The support construction uses
a 4+1 grouping on a 30×20 arena because the teacher’s bounded lateral support
offset cannot reach the unchanged 8% minimum on the original 100×80 proof
arena. This scenario adjustment was made before PPO thresholds were frozen.

Passing the teacher gate establishes achievability. It does not establish a
learned-policy result or M7b qualification.
