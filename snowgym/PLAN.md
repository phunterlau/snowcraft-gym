# SnowGym implementation plan

## Validated repository state

This plan was reconciled against `refs/snowgym_implementation_note.md`, the
current systems, browser wiring, tests, and build configuration on 2026-08-30.

| Capability             | Current engine state                                                                                                     | SnowGym decision                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Multiple blue units    | `Game.spawnSquads` already honors `maxPlayers`; bundled maps have three blue spawns, while normal `main.ts` requests one | Reuse it; do not patch spawning                                                                             |
| Projectile attribution | `Snowball` and `SnowballThrown` already carry owner and team; collision rejects same-team hits                           | Reuse it; friendly fire stays disabled                                                                      |
| Blue action submission | Throwing has generic `tryThrow`; movement only accepted selected-unit UI commands                                        | Add one generic per-unit `tryMove` seam                                                                     |
| Red control            | `AISystem` is hard-coded to red units and blue targets                                                                   | Keep it unchanged for the first server milestone; migrate it behind `TeamController` later                  |
| Round termination      | Team counts are generic, but blue loss waits for single-hero lives                                                       | Configure the no-respawn scenario with zero reserve lives; redesign only if a later environment requires it |
| Determinism            | `World` owns a seeded RNG and physics uses a fixed 60 Hz step                                                            | Fix scenario seed and decision cadence; add replay assertions before publishing a benchmark                 |
| Headless use           | Most systems are DOM-free, but `Game` constructs renderer/input and owns the private step loop                           | Compose the systems directly in a DOM-free `SnowEnvironment`                                                |
| RL contract            | No canonical action, observation, reset, or step API existed                                                             | Provide reset/observe/step now and stabilize the schema before Python                                       |

## Milestones

### M0 — autonomous blue control and server status (current)

- [x] Canonical `UnitAction` / `TeamAction` types, free of UI inputs
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
- Migrate the existing red behavior behind the same `TeamController` boundary.
- Record scenario, seed, action trace, simulation version, and upstream commit.
- Add same-seed/action-sequence state-hash tests and max-tick truncation tests.

Exit criterion: a Node test can run and exactly replay 3v3 without DOM, Canvas,
WebGL, Three.js rendering, browser timing, or input state.

### M2 — Gymnasium bridge

- [x] Publish the initial versioned `snowgym.v0` server schema.
- [x] Add fixed-shape numeric action/observation spaces and masks.
- [x] Register `gym.make("SnowGym/Squad-v0")`.
- [x] Add a project-local Python environment and locked dependencies.
- [x] Pass Gymnasium's environment checker against the live server.
- [x] Add a terminal-only scripted-blue demo command.
- [x] Record versioned visual-replay JSON from detached server state.
- [x] Replay recordings through the existing Three.js rendering engine without
      coupling the Gym environment to rendering.
- Let the initial Python adapter consume the JSON server for correctness, then
  add a long-lived batch host/direct transport for training throughput.
- Add vectorized environment support over the future batch transport.
- Expand contract tests to shared TypeScript/Python golden fixtures.
- Benchmark throughput at 1, 2, 5, 10, 20, and 60 Hz decision rates.

Exit criterion: `gym.make("SnowGym/Squad-v0")` passes Gymnasium's environment
checker and deterministic cross-language fixtures.

### M3 — configurable N-blue versus M-red scenarios (core complete)

- [x] Replace the fixed `THREE_VS_THREE_OPEN` assumption with validated scenario
      configuration: `blueUnits`, `redUnits`, spawn layout, arena dimensions, red
      difficulty, decision rate, seed, and max ticks.
- [x] Generate deterministic non-overlapping spawn layouts when explicit spawns are
      omitted; reject counts that cannot fit the arena.
- [x] Choose and publish fixed roster maxima for `SnowGym/Squad-v1`; represent
      smaller N/M configurations with unit-presence and legal-action masks so a
      registered Gym environment's spaces never change after construction.
- [x] Extend reset/server configuration, replay metadata, reward/termination logic,
      and observations without breaking `SnowGym/Squad-v0` 3v3 recordings.
- [x] Add a deterministic matrix covering 1v1, 1v3, 3v1, 3v3, and maximum-size
      fights, plus invalid counts/spawns and same-seed replay checks.
- Migrate red behavior to the common `TeamController` boundary, then add
  independently selectable scripted, random, learned, or external opponents.
- Benchmark episode throughput and balance by N/M configuration before training.

Exit criterion: one versioned environment can reset into multiple validated
N-vs-M configurations while retaining fixed Gym spaces, deterministic replay,
team elimination, and renderer-free server status.

### M4 — multi-agent and research adapters

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
