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

### M4 — hierarchical commander (C0 complete)

- [x] Define the bounded `snowgym.command-plan.v0` group action space and strict
      JSON schema without unit IDs, enemy IDs, coordinates, or physical actions.
- [x] Add strict runtime validation for mission/objective compatibility, unique
      fixed roles, allocation bounds, and acyclic support relationships.
- [x] Add deterministic weighted group allocation for 3v3 through 10v10.
- [x] Add team-relative region and symbolic enemy-cluster target resolution.
- [x] Add a trusted host envelope and immutable atomic `PlanStore`.
- Add a synchronous plan-aware controller and reactive per-unit executor.
- Add plan lifecycle, reconciliation, deterministic fallback, and trace records.
- Prove non-blocking operation with a delayed mock commander before adding a
  live model provider.
- Add a server-only `gpt-5.6-luna` Responses API adapter using strict structured
  output and `OPENAI_API_KEY` after the latency architecture passes tests.

Exit criterion: a slow commander can replace validated symbolic group plans
without blocking the 10 Hz physical controller or exposing transient unit
control to the model.

### M5 — multi-agent and research adapters

- PettingZoo adapter over the same simulator, not a second implementation.
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
