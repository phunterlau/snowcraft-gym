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

## C1 headless executor

`PlanGrounder` combines validation output, stable assignments, and initially
resolved objectives. `PlanAwareTeamController` rereads symbolic objectives from
the current observation at each decision while preserving membership for the
life of the active plan. `ReactiveUnitPolicy` applies this priority order:

```text
immediate survival -> action legality -> fire doctrine -> mission movement -> cohesion
```

Run the deterministic renderer-free C1 demonstration with:

```bash
npx tsx snowgym/orchestration/examples/commanded-10v10.ts --json
```

It runs a 6:3:1 split on `arena6.json` and reports assignments, missions,
per-role physical-action counts, terminal status, and survivors. Add
`--output PATH` to write a normal visual-replay JSON later; the headless run
does not start a server or browser.

## C2 lifecycle and delayed-plan reconciliation

`PlanLifecycle` checks the active snapshot without blocking the physical
controller. It detects plan expiry, major own-force loss, assigned-group
elimination, and completed objectives. A trigger atomically installs a
deterministic one-group fallback, so execution never depends on commander
availability.

When a candidate plan arrives, `PlanReconciler` validates its strict command
schema and provenance against the newest detached observation. It can perform
only bounded repairs: drop optional groups that no longer fit the living
roster, replace support for a removed group, or replace an enemy objective
after elimination. Any other invalid plan is rejected while the previous
snapshot remains active. Lifecycle events are returned as detached trace data.

The monitoring ownership and escalation rules are specified in
`snowgym/PLAN.md`. Mock latency and the asynchronous request scheduler are C3;
the server-only `gpt-5.6-luna` adapter remains a later milestone after C3 proves
that commander latency cannot block the 10 Hz controller.
