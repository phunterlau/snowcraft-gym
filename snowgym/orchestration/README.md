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
`snowgym/PLAN.md`.

## C3 asynchronous mock commander

`summarizeStrategy` converts the current observation and plan snapshot into a
compact, versioned summary without unit IDs, enemy IDs, or raw projectile
trajectories. `CommanderClient` is provider-neutral and returns untrusted plan
data. `CommanderScheduler.tick(observation)` is synchronous: it polls completed
provider work, evaluates lifecycle triggers, and starts requests without ever
awaiting them.

The scheduler permits one request in flight, coalesces newer triggers, enforces
a minimum request interval, aborts on a simulation-tick deadline, and ignores a
response that arrives after timeout. A response can be held until a configured
simulation tick even if the mock finished earlier, which makes headless latency
experiments exactly replayable. Real provider latency is still asynchronous
wall-clock behavior and is recorded separately as response metadata.

Use this order in a runner:

```ts
const action = controller.act(observation, dt); // always synchronous
const result = environment.step(action);
scheduler.tick(result.observation); // never await this
```

Run the renderer-free delayed-commander demonstration with:

```bash
npx tsx snowgym/orchestration/examples/delayed-mock-10v10.ts --json
```

The default request is captured at tick 0, becomes eligible after 90 simulation
ticks (1.5 simulated seconds), and is reconciled/activated against the then
current battle. `--latency-ticks` changes this deterministic delay.

The next boundary is the server-only `gpt-5.6-luna` adapter using
`OPENAI_API_KEY` and strict structured output. The provider must plug into
`CommanderClient`; it must not change the synchronous controller or scheduler
contract.

## C4 provider adapter and live-test boundary

`providers/OpenAICommanderClient.ts` is the server-only adapter. It
uses the exact `gpt-5.6-luna` model, the Responses API, configurable reasoning
(`medium` by default), `store: false`, and the existing command-plan JSON Schema
as strict `text.format`. It reads `OPENAI_API_KEY` only at runtime and performs
defensive response/refusal/incomplete/usage parsing before the normal host-side
plan validation and reconciliation.

The live request contains only trigger names, state-hash provenance, arena
dimensions and obstacle count, aggregate force health/centroid/spread, hostile
projectile count, aggregate group status, and the current symbolic plan. It
does not contain unit IDs, enemy IDs, raw projectile trajectories, repository
files, or the API key in the JSON payload.

After explicitly approving that outbound summary, run the headless smoke with:

```bash
npx tsx snowgym/orchestration/examples/openai-commander-smoke.ts --json
```

This makes one API request and locally validates the returned plan. It does not
start a server or browser. Live API access is deliberately excluded from
`npm test`. The adapter smoke passed against `gpt-5.6-luna` on 2026-08-31;
availability and latency remain account/runtime-dependent.

## Single-request live battle

The bounded live runner disables automatic lifecycle monitoring, manually
starts one request at tick 0, and fails if its trace contains anything other
than exactly one `request_started` event. The simulator remains renderer-free
and is paced at 100 ms per 10 Hz decision so real inference overlaps play:

```bash
npx tsx snowgym/orchestration/examples/openai-commanded-10v10.ts --json
```

The authorized seed-42 acceptance run made one Luna request, continued control
during 3.51 seconds of inference, activated the reconciled response at tick
204, and reached a 9–0 blue victory with zero rejected physical actions. The
single-request mode is for bounded provider demos only; normal orchestration
keeps automatic lifecycle monitoring enabled.

## C5 trajectory-aware closed loop

`trajectory/TrajectoryMonitor.ts` is the passive first slice of continuous
orchestration. A runner records the observation before a decision, the immutable
plan snapshot used by the controller, the adapter's physical action results,
and the observation after the step. The monitor keeps a bounded rolling window
for one plan version and emits only group aggregates. Unit IDs remain internal
to aggregation and never appear in `snowgym.trajectory-digest.v0`.

The digest classifies mission-aware progress and records health, cohesion,
issued-action, rejected-action, and stuck-fraction evidence. It is initially
telemetry-only: it cannot change the controller, environment, scheduler, or
plan store. Later C5 slices debounce trajectory signals, distinguish soft
replanning from hard fallback, and pass the same bounded digest to the existing
stateless commander request.

`TrajectorySignalDetector` now emits latched `plan_stalled` and
`action_rejection_repeated` signals only after an activation grace period. Soft
signals keep the current plan active while `CommanderScheduler` makes an
asynchronous request. Existing loss, elimination, completion, and expiry
conditions remain hard fallback triggers. The scheduler counts every provider
attempt and can enforce a per-episode maximum even when calls fail or time out.

Run the exact-replay multi-request mock demonstration with:

```bash
npx tsx snowgym/orchestration/examples/trajectory-mock-10v10.ts --json
```

The legacy filename retains the original 10v10 default, but both the mock and
Luna trajectory runners now accept validated bundled-map configurations:

```bash
npx tsx snowgym/orchestration/examples/trajectory-mock-10v10.ts \
  --blue-units 6 --red-units 10 \
  --map arena6.json --red-difficulty easy \
  --seed 13 --json
```

`--blue-units`, `--red-units`, `--map`, and `--red-difficulty` reach the same
`createMapScenario` and `SnowEnvironment` contracts used by deterministic
examples. A map-capacity or configuration error is raised before the commander
loop starts. Configurability does not imply that the default doctrine is
effective for every roster imbalance; qualifying strategy claims require
paired baseline results over frozen seeds.

When blue starts outnumbered, the host now keeps one aggregate group at medium
range with distributed fire and loose cohesion while commander advice is
pending. This deterministic economy-of-force opening replaces the parity
default that advances into the enemy backfield; it does not constrain later
valid mock or Luna plans. A committed seed-14 6v10 example survives two
major-loss fallbacks and delayed mock responses before blue wins 1–0:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/trajectory-6v10-blue-win-seed-14.json&trace=/replays/trajectory-6v10-blue-win-seed-14.commander.json
```

This is a deterministic possibility proof, not an LLM result or balance
claim. In a 40-seed easy-red development sweep the opening won 11 episodes;
future evaluation should freeze a separate seed suite before comparing it with
online commanders.

Each later request can include the current `snowgym.trajectory-digest.v0` plus
one `snowgym.plan-outcome.v0` summary for the preceding plan. This explicit,
host-owned history keeps `store: false`, deterministic capture, provider
replacement, and stale-plan reconciliation intact. The OpenAI adapter forwards
those aggregates but continues to return only `snowgym.command-plan.v0`.

After the deterministic gates pass, run the capped, wall-clock-paced Luna
battle headlessly with:

```bash
OPENAI_API_KEY=... npx tsx \
  snowgym/orchestration/examples/openai-trajectory-10v10.ts \
  --blue-units 10 --red-units 10 \
  --map arena6.json --red-difficulty easy \
  --reasoning medium --max-requests 3 --json
```

The hard request limit counts failures and timeouts as provider attempts. The
runner aborts any outstanding request when the episode ends and never requires
a server, renderer, or browser.

The explicitly authorized seed-42 C5 acceptance run passed on 2026-09-01. The
host detected `plan_stalled` at ticks 498, 966, and 1125 while continuing the
10 Hz executor. It made two of the three permitted Luna requests: the first
returned an accepted split-force plan after 7.13 seconds (1,392 input tokens,
430 output tokens, including 241 reasoning tokens), and the outstanding second
request was aborted when the episode ended. The battle completed at tick 1125
with a 4-0 blue victory and zero rejected physical actions.

## Commander trace and replay overlay

Both trajectory runners can write the visual replay and a separate
`snowgym.commander-trace.v0` sidecar. The deterministic path requires no API
key:

```bash
node --import tsx snowgym/orchestration/examples/trajectory-mock-10v10.ts \
  --output public/replays/trajectory-10v10.json \
  --trace-output public/replays/trajectory-10v10.commander.json \
  --json
```

Use the same `--output`, `--trace-output`, and optional `--force` arguments with
`openai-trajectory-10v10.ts` to record an explicitly authorized provider run.
The sidecar is bound to the replay scenario, seed, final tick, and final public
state hash. Its parser rejects mismatched replays, malformed timelines, unknown
event kinds, and engine entity-ID fields.

Start the normal Vite UI and load both artifacts through the existing replay
viewer:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/trajectory-10v10.json&trace=/replays/trajectory-10v10.commander.json
```

The optional overlay follows the scrubber and shows the active symbolic plan,
aggregate group progress and stuck fractions, and recent scheduler/lifecycle
events. A replay without `trace=` behaves exactly as before. Both files can
also be selected independently with the viewer's local-file controls.

## Deterministic commander latency benchmark

```bash
npm run snowgym:commander:benchmark -- \
  --seeds 11,12,13,14,15 \
  --latency-ticks 0,6,15,30,60,120,240,480 \
  --blue-units 10 --red-units 10 --map arena6.json \
  --output snowgym/orchestration/artifacts/latency-v0.json --json
```

At 60 simulation Hz these ticks represent 0, 100, 250, 500 ms and 1, 2, 4,
8 seconds. Output includes every episode, aggregate outcomes by latency, and a
semantic digest. Existing files are not overwritten without `--force`. This
mock benchmark is the deterministic gate before separately authorized live
provider experiments; it does not claim LLM quality.
