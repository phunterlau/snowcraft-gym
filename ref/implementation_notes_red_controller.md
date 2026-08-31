# Implementation Notes — Red TeamController Migration + Selectable Opponents

> 2026-08-30, branch `agents/repo-design-documentation-update`.
> Continues from `ref/kimi_pickup.md`. Closes the biggest open seam in
> `snowgym/PLAN.md` M1/M3: red behavior now runs behind the same
> `TeamController` boundary as blue, with selectable opponents.

## What changed

### SnowGym layer (`snowgym/`)
- **`agents/TeamController.ts`** — interface evolved: `act(observation, dt)`.
  The fixed timestep is now passed so controllers with internal per-tick state
  can advance deterministically. Policies still never touch engine entities.
- **`agents/ScriptedAiAgent.ts`** (new) — wraps the classic `AISystem` as a
  `TeamController`. Re-runs the AI's per-tick reactive logic internally
  (decision timers, dodges, aim error, cover) and *reports* the orders it
  issued as semantic `move`/`throw`/`noop` actions. The report is pure — it is
  not re-applied to the world (re-issuing a throw would double-fire).
- **`agents/RandomAgent.ts`** (new) — seeded random baseline (noop / wander /
  throw at a random enemy). All randomness flows through `world.rng`, so
  episodes reproduce from the scenario seed.
- **`agents/opponents.ts`** (new) — `createRedController()` factory and
  `redController: "scripted" | "random"` selection + validation.
- **`core/SnowEnvironment.ts`** — red is now a `TeamController`
  (`redController` config field, default `scripted`). `physicsStep()` calls
  `redController.act(observeWorld(red), SIM.dt)` each tick in place of the old
  direct `redAI.update()`. Status `configuration` now reports `redController`.
- **`core/TeamControllerSystem.ts`** — passes `SIM.dt` through to `act`.
- **`server/SnowGymService.ts`** — `/reset` accepts `scenario.redController`;
  rejects unknown values with 400.
- **`replay/ReplayRecording.ts`** — `configuration` type allows optional
  `redController`.
- **`python/`** — `SnowGymEnv` gains `red_controller` (default `"scripted"`,
  validated), `default_scenario_config()` includes it, demo CLI gains
  `--red-controller`. Reset `options={"scenario": {...}}` can switch opponent
  on a Squad-v1 env without changing tensor shapes.

### Upstream patches (logged in `snowgym/UPSTREAM_PATCHES.md`)
- **`src/systems/AISystem.ts`** — optional `AiSquad { controlled, target }`
  constructor param, default `{ Enemy, Player }`; three hardcoded team checks
  now read from the squad; `events` param widened to accept `null` (it was
  already unused). Browser behavior identical.
- **`src/systems/MovementSystem.ts`** — added public `tryHold(player)` (cancel
  a stale move target without disturbing other states), for controllers that
  report orders on behalf of a unit.

## The hard part: provable behavioral equivalence

A naive "wrap the AI and re-apply its actions" bridge **diverged** from the old
direct-registration path. Two floating-point cadence bugs were the cause:

1. **Missing `dt`:** an early draft left the AI's internal advance at
   `remaining = 1` (one *second*) instead of `dt`, so the AI ran 60× per tick.
   Caught by instrumenting internal `ai.update` call counts (180 vs 3).
2. **FP timer drift:** the engine's AI decrements decision timers by
   `SIM.dt = 1/60` per tick; because `60 × (1/60) = 0.9999999999999994 < 1` in
   IEEE-754, resetting a timer to an integer number of ticks lands the next
   decision **one tick apart per batch of 60**, compounding. Fixed by having
   the bridge advance in exact `SIM.dt` slices, preserving bit-identical
   floating-point timer cadence.

**Verification method:** captured full-episode traces (winner, final tick, all
unit healths, all unit positions) from the OLD path by `git stash`-ing the new
code, then from the NEW controller-bridged path, and diffed. Across 5 scenarios
(3v3 normal, 3v3 hard, 5v2 hard wide arena, 1v3 easy, 1v1) the traces are
**bit-identical**. Also confirmed end-to-end through HTTP+Python: the
regenerated `blue-seed-42.json` replay is 294 ticks / blue 3-0, matching the
old-path trace.

## Validation snapshot (all green)
- 163/163 TS tests (29 files) · `tsc --noEmit` clean · ESLint clean (1
  pre-existing warning in `PickupSystem.test.ts`) · `vite build` OK.
- 8/8 Python tests · Gymnasium `check_env` passes for `SnowGym/Squad-v0` and
  `SnowGym/Squad-v1`.
- Live HTTP: `redController:"random"` reset/autoplay works; bad value → 400.
- Browser replay smoke passes; demo recorded to `public/replays/demo-best.json`.

## Known limitations / next seams
- The scripted bridge's throw report uses a fixed mid power (0.5) — it's a
  *report* for observation only, not re-applied, so it doesn't affect behavior,
  but a future "report exact orders" improvement could carry the real power.
- `RandomAgent` is intentionally weak (a baseline floor), not a benchmark
  opponent.
- Learned/external opponents remain future (M3 leftover / M4).

## Pointers for the next task (maps)
- `MapLoader.build(data: MapData)` is headless-friendly (pure; no fetch needed
  if JSON is supplied). `SnowEnvironment.reset` currently uses
  `createEmptyArena` — swap to `MapLoader.build` + map spawns.
- Maps live in `public/maps/arena{1..5}.json` (obstacles + `spawns`).
- For fixed-shape Gym observations, decide whether to expose obstacles (they
  affect LoS/cover) — likely a fixed-capacity obstacle list per registered map.
