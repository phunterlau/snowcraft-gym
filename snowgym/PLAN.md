# SnowGym implementation plan

## Validated repository state

This plan was reconciled against `refs/snowgym_implementation_note.md`, the
current systems, browser wiring, tests, and build configuration on 2026-08-31.

| Capability             | Current engine state                                                                                                     | SnowGym decision                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Multiple blue units    | `Game.spawnSquads` already honors `maxPlayers`; bundled maps have three blue spawns, while normal `main.ts` requests one | Reuse it; do not patch spawning                                                                             |
| Projectile attribution | `Snowball` and `SnowballThrown` already carry owner and team; collision rejects same-team hits                           | Reuse it; friendly fire stays disabled                                                                      |
| Blue action submission | Throwing has generic `tryThrow`; movement only accepted selected-unit UI commands                                        | Add one generic per-unit `tryMove` seam                                                                     |
| Red control            | The classic `AISystem` behavior is available through `ScriptedAiAgent`; seeded random is a second opponent               | Select both through the common `TeamController` boundary                                                    |
| Round termination      | Team counts are generic, but blue loss waits for single-hero lives                                                       | Configure the no-respawn scenario with zero reserve lives; redesign only if a later environment requires it |
| Determinism            | `World` owns a seeded RNG, physics uses fixed 60 Hz steps, and status exposes versioned public-state hashes              | Record provenance and exact actions; keep cross-language golden and replay assertions before benchmarking   |
| Headless use           | Most systems are DOM-free, but `Game` constructs renderer/input and owns the private step loop                           | Compose the systems directly in a DOM-free `SnowEnvironment`                                                |
| RL contract            | Canonical reset/step, masked fixed-shape Gym spaces, configurable rosters, and terrain observations are implemented      | Keep HTTP as the reference transport; add a direct batched transport before high-throughput training        |

## Milestones

### M0 — autonomous blue control and server status (current)

- [x] Canonical `UnitAction` / `TeamAction` types, free of UI inputs
- [x] Explicit `hold` action that cancels stale movement without changing `noop`
- [x] Detached entity observation with deterministic ID ordering
- [x] `TeamController` policy contract
- [x] Simple blue dodge / approach / throw policy
- [x] Validating SnowCraft action adapter
- [x] DOM-free `reset`, `observe`, and `step` lifecycle
- [x] 10 Hz decisions over the existing 60 Hz simulation
- [x] JSON status, reset, step, and autoplay endpoints
- [x] Verified deterministic autonomous completion in the Node integration test
- [x] Verified live HTTP reset/status/step/autoplay flow

Exit criterion: blue units independently move and throw without selection or
human input, a 3v3 match reaches a team-elimination result, and a client can
retrieve the result from the server without a renderer.

### M1 — reproducible environment contract

- [x] Extract a DOM-free simulation composition root with explicit system order.
- [x] Implement `reset(seed)`, `observe(team)`, and blue-team `step(action)`.
- [x] Advance a configurable integer number of physics ticks per policy decision.
- [x] Return terminal-only reward (`+1`, `-1`, `0`), `terminated`, `truncated`, and
      structured `info`; keep diagnostic event rewards separate.
- [x] Migrate the existing red behavior behind the same `TeamController` boundary.
      The scripted red squad now runs through `ScriptedAiAgent`, which re-runs
      the classic `AISystem` per-tick logic and reports its orders as semantic
      actions; full-episode traces are bit-identical to direct AI registration.
- [x] Record scenario, seed, action trace, simulation version, upstream base
      commit, and one public-state hash per replay frame.
- [x] Add exact same-seed/action-sequence state-hash tests and max-tick
      truncation tests.

Exit criterion: a Node test can run and exactly replay 3v3 without DOM, Canvas,
WebGL, Three.js rendering, browser timing, or input state.

### M2 — Gymnasium bridge

- [x] Publish the initial versioned `snowgym.v0` server schema.
- [x] Add fixed-shape numeric action/observation spaces and masks.
- [x] Register `gym.make("SnowGym/Squad-v0")`.
- [x] Add a project-local Python environment and locked dependencies.
- [x] Pass Gymnasium's environment checker against the live server.
- [x] Add a terminal-only scripted-blue demo command.
- [x] Strictly validate mutating request fields and isolate scripted stepping
      from explicit external actions.
- [x] Add optimistic state-hash guards, step idempotency, and machine-readable
      server capability discovery.
- [x] Check all three registered Gym environments, including map-backed v2,
      and support JSON CLI summaries.
- [x] Record versioned visual-replay JSON from detached server state.
- [x] Add a renderer-free CLI/function that builds reproducible M-vs-N examples
      for open arenas or bounded native map spawn pools.
- [x] Replay recordings through the existing Three.js rendering engine without
      coupling the Gym environment to rendering.
- Let the initial Python adapter consume the JSON server for correctness, then
  add a long-lived batch host/direct transport for training throughput.
- Add vectorized environment support over the future batch transport.
- [x] Add a shared TypeScript/Python golden fixture for the versioned public-state
      canonicalization and hash contract.
- Benchmark throughput at 1, 2, 5, 10, 20, and 60 Hz decision rates.

Exit criterion: `gym.make("SnowGym/Squad-v0")` passes Gymnasium's environment
checker and deterministic cross-language fixtures.

### M3 — configurable N-blue versus M-red scenarios (core complete)

- [x] Replace the fixed `THREE_VS_THREE_OPEN` assumption with validated scenario
      configuration: `blueUnits`, `redUnits`, spawn layout, arena dimensions, red
      difficulty, decision rate, seed, and max ticks.
- [x] Generate deterministic non-overlapping spawn layouts when explicit spawns are
      omitted; reject counts that cannot fit the arena.
- [x] Publish fixed roster maxima for `SnowGym/Squad-v1` (eight slots) and
      `SnowGym/Squad-v2` (ten slots); represent smaller N/M configurations with
      unit-presence and legal-action masks so a registered Gym environment's
      spaces never change after construction.
- [x] Extend reset/server configuration, replay metadata, reward/termination logic,
      and observations without breaking `SnowGym/Squad-v0` 3v3 recordings.
- [x] Load the bundled SnowCraft maps as scenario terrain: obstacles affect
      line-of-sight, cover, and collision, spawn points come from the map, and
      obstacles are exposed to policies as a fixed-capacity masked tensor.
- [x] Add a native 10v10 map whose browser JSON and headless registry definitions
      are contract-tested for exact parity.
- [x] Add a deterministic matrix covering 1v1, 1v3, 3v1, 3v3, and maximum-size
      fights, plus invalid counts/spawns and same-seed replay checks.
- [x] Migrate red behavior to the common `TeamController` boundary, then add
      independently selectable scripted, random, learned, or external opponents
      (scripted and random shipped; learned/external remain future).
- Benchmark episode throughput and balance by N/M configuration before training.

Exit criterion: one versioned environment can reset into multiple validated
N-vs-M configurations while retaining fixed Gym spaces, deterministic replay,
team elimination, and renderer-free server status.

### M4 — hierarchical commander (C4 complete)

- [x] Define the bounded `snowgym.command-plan.v0` group action space and strict
      JSON schema without unit IDs, enemy IDs, coordinates, or physical actions.
- [x] Add strict runtime validation for mission/objective compatibility, unique
      fixed roles, allocation bounds, and acyclic support relationships.
- [x] Add deterministic weighted group allocation for 3v3 through 10v10.
- [x] Add team-relative region and symbolic enemy-cluster target resolution.
- [x] Add a trusted host envelope and immutable atomic `PlanStore`.
- [x] Add a synchronous plan-aware controller and reactive per-unit executor.
- [x] Add a deterministic headless 10v10 split-force demonstration whose
      per-role action traces prove distinct group execution and exact replay.
- [x] Add synchronous plan lifecycle triggers for plan expiry, major own-force
      loss, assigned-group elimination, and objective completion.
- [x] Reconcile delayed candidates against the current living roster, repair
      bounded support/target drift, reject invalid candidates without replacing
      the active plan, and atomically activate accepted plans.
- [x] Keep a deterministic one-group fallback and detached lifecycle trace for
      every accepted, repaired, rejected, and fallback activation.
- [x] Add an ID-free, versioned strategic summary and a provider-neutral async
      `CommanderClient` boundary.
- [x] Prove non-blocking operation with a delayed mock commander: one request in
      flight, cooldown-governed trigger coalescing, simulation-tick timeout,
      stale-response reconciliation, and ignored late responses.
- [x] Add deterministic simulated-latency scheduling and a headless 10v10 C3
      demonstration with exact action/state/trace replay coverage.
- [x] Add a server-only `gpt-5.6-luna` Responses API adapter using strict
      structured output, reasoning, `store: false`, and environment-only
      `OPENAI_API_KEY`.
- [x] Gate the provider adapter on mocked error/refusal/timeout tests plus one
      explicitly authorized live headless schema smoke; never include it in the
      deterministic default test suite.
- [x] Run a separately authorized, wall-clock-paced live battle with automatic
      lifecycle requests disabled and a code-enforced limit of exactly one
      external request. The battle continued at 10 Hz, activated Luna's stale
      symbolic plan mid-battle, and completed without rejected physical actions.

Exit criterion: a slow commander can replace validated symbolic group plans
without blocking the 10 Hz physical controller or exposing transient unit
control to the model.

#### Agent monitoring and correction contract

Monitoring is host-owned and hierarchical. The LLM never polls individual
units, waits inside `step`, or corrects physical actions directly.

| Layer                         | Cadence                                          | Status consumed                                                                     | Corrections it may make                                                                                        | Escalation                                                                                    |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Simulation and action adapter | 60 Hz                                            | collision, legality, cooldown, alive state                                          | reject illegal physical actions and advance authoritative state                                                | report action results to the controller trace                                                 |
| Reactive unit policy          | each team decision, normally 10 Hz               | assigned group, live position/health/state, projectiles, current symbolic objective | dodge, choose a legal throw, reacquire targets, restore range/cohesion, or continue mission movement           | never waits for the LLM; future stuck/repeated-rejection summaries go to lifecycle monitoring |
| Plan-aware team controller    | each team decision                               | immutable active-plan snapshot plus current detached observation                    | late-bind enemy clusters, ally-group positions, and map-region objectives while keeping plan membership stable | continue the last valid snapshot during any commander request                                 |
| Plan lifecycle monitor        | each team decision                               | plan age, living assigned units, whole-team loss fraction, and objective progress   | activate deterministic fallback on expiry, major loss, group elimination, or completion                        | schedule a commander request; coalesce duplicate triggers while one is in flight              |
| Plan reconciler               | only when a candidate arrives                    | candidate provenance plus the newest detached observation                           | validate, shrink infeasible groups, repair missing support/enemy objectives, ground, then atomically activate  | reject unsafe drift and retain the old plan or fallback                                       |
| LLM commander                 | event-driven and rate-limited, target 0.1–0.5 Hz | a compact strategic summary captured with source tick/hash, never engine entities   | propose only a strict symbolic `CommandPlan`                                                                   | timeout/error cannot block control; host fallback and later retry policy remain authoritative |

The C3 delayed-commander test must prove this sequence:

```text
observe -> execute current plan -> detect/coalesce trigger -> request asynchronously
        -> keep executing current plan -> receive stale candidate
        -> reconcile against newest observation -> atomic activate or reject
```

C3 tests cover request timeout, duplicate-trigger coalescing, late response
after fallback, invalid response, provider failure, uninterrupted synchronous
control, and deterministic replay of the same mock-latency schedule. Stuck
detection and repeated action-rejection thresholds remain planned explicit,
debounced lifecycle signals; they must be host-computed rather than inferred by
the model.

### M4.1 — trajectory-aware closed-loop commander (C5)

- [x] Add a passive, bounded trajectory monitor over pre-step observation,
      active-plan snapshot, physical action results, and post-step observation.
- [x] Publish an ID-free, versioned group trajectory digest with mission-aware
      progress, health/cohesion trends, action counts, rejection counts, and a
      host-computed stuck fraction.
- [x] Prove that enabling telemetry cannot change physical actions, public-state
      hashes, plan activation, or replay results.
- [x] Add debounced `plan_stalled` and `action_rejection_repeated` signals with
      activation grace periods and recovery hysteresis.
- [x] Separate soft signals, which retain the current plan while replanning,
      from hard lifecycle failures, which activate deterministic fallback.
- [x] Pass the bounded trajectory digest and preceding plan outcome to each
      stateless commander request without exposing unit IDs or raw trajectories.
- [x] Run deterministic multi-request mock battles with exact state, plan, signal,
      latency, and scheduler-trace replay coverage.
- [x] Add an opt-in, wall-clock-paced Luna battle with one in-flight request, a
      code-enforced per-episode call limit, explicit token/latency accounting,
      and uninterrupted fallback on provider failure.
- [x] Pass an explicitly authorized live C5 acceptance battle: host-computed
      trajectory stalls triggered bounded Luna replanning while the executor
      continued at 10 Hz, used two of three permitted requests, activated a
      valid plan, and finished with zero rejected physical actions.
- [x] Record a versioned, ID-free commander trace sidecar bound to the replay's
      final public-state hash; add an optional scrubber-aware plan, aggregate
      trajectory, and lifecycle overlay to the existing replay UI.

Exit criterion: host-computed trajectory evidence can trigger bounded Luna
replanning during an episode while the 10 Hz executor continues synchronously,
and identical mock latency schedules reproduce identical actions, state hashes,
trajectory digests, plans, and lifecycle traces.

### M5 — multi-agent and research adapters

- [x] Add a two-team PettingZoo ParallelEnv over the same server and simulator,
      with mirrored fixed-capacity observations, simultaneous guarded joint
      actions, zero-sum terminal rewards, and the official Parallel API gate.
- Optional partial observations, semantic raster/pixels, latency injection, and
  scripted/random/learned/remote opponent adapters.
- Batched training benchmarks and versioned evaluation scenarios.

## Guardrails

- No RL, Python, transport, or SnowGym imports from `src/`.
- Policies receive observations and return actions; they never hold engine
  entities or mutate world state.
- Canonical benchmark reward stays terminal-only until experiments explicitly
  choose a shaped reward.
- Every change outside `snowgym/` is recorded in `UPSTREAM_PATCHES.md`.
- Human browser behavior must continue to pass the existing test/build/smoke
  suite after each milestone.
- Repository agents should follow the root `AGENTS.md` and the repo-local
  `.agents/skills/snowgym/SKILL.md` guarded workflows.
