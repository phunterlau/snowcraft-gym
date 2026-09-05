# Selective tactical repair: review and mechanism audit

Declared 2026-09-05 against source commit `c5b3099`, before new collection.

## Case evidence

Terrain 5v5, seed 620002, tick 522 (8.7 seconds): four blue remain against five
red. The original nearest-cluster binding contains enemy 8. All surviving blue
fighters are throw-ready, but none is within executor range of that target.
The preceding two-second window records 100 own health lost and 20 enemy health
lost. Keeping the activation loses after another 4.9 seconds, dealing 40 damage
and receiving 380.

Reactivating identical symbols selects enemies 6 and 7. The first action changes
from four moves to two throws and two moves. Blue wins after another 9.8 seconds
with two survivors, dealing 480 damage and receiving 220. A one-second delay
loses. Both branches reproduce exactly. No LLM was involved.

Full reactivation also rebuilds assignments and anchors; isolate those effects
before attributing all improvement to retargeting. In the terrain 10v10 control,
keeping wins with six blue alive, while reactivation is censored after 300
decisions with one blue and three red alive. Refresh can help or harm.

Source: [sealed report](../snowgym/orchestration/recovery/examples/preflight-20260905-v0/report.json)
and [fixtures](../snowgym/orchestration/recovery/examples/preflight-20260905-v0/fixtures.json).

## Emerging hypotheses

- Explicit commitment rules could separate persistent strategic objectives from
  refreshable tactical bindings.
- Bounded local firing could exploit opportunities without replacing movement
  objectives or role assignments.
- A retain/repair/escalate policy could preserve useful commitments and request
  commander intervention only when local repair is insufficient.
- Precomputed, host-validated contingencies could reduce provider-latency costs.

These remain hypotheses. Learning termination has precedent in
[Option-Critic](https://arxiv.org/abs/1609.05140); grounding strategic choices in
execution capability has precedent in [SayCan](https://arxiv.org/abs/2204.01691).
Novelty and learned-policy benefits are unassessed.

## Approved experiment

| Arm | Intervention |
| --- | --- |
| `keep` | Preserve activation and executor behavior. |
| `refresh_binding` | Resolve enemy-cluster selectors once, replacing only tactical bindings with a separate revision. Preserve assignments, commands, activation anchors/time and plan version. |
| `local_fire` | After dodge and normal assigned-target firing, permit a ready fighter to shoot the nearest living non-assigned enemy in executor range only when no assigned candidate is in range. Break ties by ID; reuse production lead/power; preserve movement logic and bindings. |
| `reactivate` | Use existing full same-symbol activation. |

Binding refresh retains surviving-member tracking and elimination fallback;
other objective kinds remain unchanged. Local firing stays enabled after
intervention until termination or horizon. Add no visibility or accuracy
assumption; proximity does not establish threat. Do not add a combined arm.

Use the three archived casualty cases as known examples. Reserve 40 fresh seeds
per roster: 630000–630039 for 5v5, 630040–630079 for 10v10, and 630080–630119
for 6v10, on arena6 with normal scripted Red at 10 Hz. Audit seed fields and
allocation/schedule declarations before collection; collisions stop preflight
without substitution. These are diagnostic, not qualification seeds.

Scan at most 300 decisions, retain the first unchanged recent-casualty predicate,
and report absent events without resampling. For each fixture run four arms at
0/1/2/4/8-second delays and a shared 300-decision horizon including delay.
Skip intervention if the battle ends first. Restore exact action prefixes and
independently rerun every continuation, comparing actions and hashes.

Record original/intervention target memberships, assignments, anchors, binding
revision, first changed action, opportunistic shots, original/active range
occupancy, damage, survivors, outcomes, rejections and frozen-target completion.
Keep original scoring membership and health denominator immutable.

Report paired differences against keep, and binding refresh against full
reactivation, separately by roster and delay. Use 10,000 paired bootstrap
resamples with analysis RNG 730001. Intervals are exploratory, unadjusted for
multiple comparisons, and exclude historical cases. Fewer than 20 qualifying
seeds means under-coverage; do not collect more. No automatic promotion.

## Delivery and boundaries

Commit declaration, tested harness, and sealed results as separate milestones.
Before each commit run TypeScript tests, build, Python client tests and Python
training tests. Test intervention isolation, readiness/range/ties/dodge,
unchanged default behavior, historical parity, exact delayed prefixes,
terminal handling, seed collision detection, tamper rejection and fixed scoring.

Keep CommandPlan, Gym contracts, production defaults, historical artifacts and
unrelated edits unchanged. No provider calls, learned repair policy, new scenario
design, R1n/M7c gate changes or automatic promotion. Pushing is a later request.
