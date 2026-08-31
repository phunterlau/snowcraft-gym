# SnowGym hierarchical command contract

This directory is the renderer-free boundary between a slow strategic
commander and SnowGym's synchronous unit controllers. The initial C0 milestone
contains no model SDK, network request, wall-clock scheduler, or browser code.

## Command levels

The commander emits one `snowgym.command-plan.v0` object for one to three
fixed roles: `main`, `maneuver`, and `reserve`. It chooses only relative force
weights, deterministic unit-selection strategies, symbolic objectives, and
bounded engagement doctrine.

The commander cannot choose plan IDs, source ticks, state hashes, unit IDs,
enemy IDs, coordinates, movement targets, throws, dodges, activation timing,
or fallback behavior. Those values are owned by the host, grounder, and fast
unit policies.

Example model output:

```json
{
  "schemaVersion": "snowgym.command-plan.v0",
  "intentSummary": "Pin the largest cluster while preserving a reserve.",
  "groups": [
    {
      "role": "main",
      "allocationWeight": 7,
      "selection": "balanced",
      "order": {
        "mission": "engage",
        "objective": { "kind": "enemy_cluster", "select": "largest" },
        "approach": "direct",
        "engagement": {
          "posture": "balanced",
          "fire": "focus",
          "preferredRange": "medium",
          "cohesion": "normal"
        }
      }
    },
    {
      "role": "reserve",
      "allocationWeight": 3,
      "selection": "rearline",
      "order": {
        "mission": "support",
        "objective": { "kind": "ally_group", "role": "main" },
        "approach": "direct",
        "engagement": {
          "posture": "conservative",
          "fire": "opportunistic",
          "preferredRange": "long",
          "cohesion": "normal"
        }
      }
    }
  ]
}
```

`intentSummary` is required by the strict output schema but may be `null`. It
is trace-only and never changes execution.

## Deterministic grounding

`PlanValidator` rejects unknown fields and invalid mission/objective pairs.
`GroupAllocator` uses Hamilton apportionment with deterministic tie-breaking,
then guarantees one unit per declared group when the roster permits it.
Membership is stable for a plan and each living ally is assigned exactly once.

`TargetResolver` evaluates symbolic enemy clusters and team-relative regions
from the current detached observation. Enemy and ally IDs may appear in the
grounded host plan, but never in model output.

`PlanStore` clones and freezes a fully grounded, host-enveloped plan before an
atomic synchronous swap. Later asynchronous commander code must continue to
execute the current snapshot while a replacement request is in flight.

## Next implementation boundary

C1 may add `PlanAwareTeamController` and reactive per-unit execution over a
hard-coded command plan. It must not introduce OpenAI calls yet. Mock latency,
the asynchronous scheduler, and the `gpt-5.6-luna` adapter remain separate
later milestones.
