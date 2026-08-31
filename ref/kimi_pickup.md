# Kimi Pickup Notes — snowcraft-gym

> Written 2026-08-30 on branch `agents/repo-design-documentation-update`
> (worktree `repo-design-documentation-update`, HEAD `b2db095`).
> Note: a `ref/` directory did not exist in this worktree; this file creates it.
> The design docs referenced below live at the repo root and in `snowgym/`.
> `snowgym/PLAN.md` mentions a `refs/snowgym_implementation_note.md`, which is
> not present in the repo (it was an external reconciliation note).

## What this repository is

Two layers in one repo:

1. **SnowCraft** (`src/`, `public/`, root docs) — a faithful browser remake of
   the 1999 Flash game SnowCraft. Three.js + TypeScript + Vite, Preact for DOM
   UI, no external assets (procedural geometry, synthesized Web Audio).
   Deployed to GitHub Pages. Spec: [`design.md`](../design.md).
2. **SnowGym** (`snowgym/`) — a newer extension layer that turns the SnowCraft
   simulation into a headless, Gymnasium-ready RL environment with autonomous
   blue-team control, configurable N-blue vs M-red fights, and visual
   recording/replay. Spec/roadmap: [`snowgym/PLAN.md`](../snowgym/PLAN.md),
   usage: [`snowgym/README.md`](../snowgym/README.md).

A third doc, [`multiplayer-plan.md`](../multiplayer-plan.md), is a *planning
only* document for future 1v1 online multiplayer (server-authoritative
snapshots). **No multiplayer code exists yet.**

## SnowCraft engine — current design (as built, not just as specced)

- **Strict layering:** Simulation → Renderer → Three.js scene. Game logic
  (`core/`, `game/`, `systems/`, `physics/`, `ecs/`, `utils/`) has **zero
  Three.js/DOM imports**; only `render/` and `ui/` touch Three/Preact. This is
  what makes both SnowGym headless operation and the multiplayer plan feasible.
- **Fixed timestep:** 60 Hz simulation with accumulator (`core/GameLoop.ts`);
  rendering interpolates at display refresh.
- **Update order** per tick: Input → AI → Movement → Throw → Projectile →
  Collision → Damage → Round → Animation (`core/Game.ts`).
- **Seams:** typed `EventBus` (`core/EventBus.ts` + `core/events.ts`),
  command-based input (`core/commands.ts`, `engine/InputManager.ts`), seeded
  PRNG (`utils/Random.ts`, mulberry32), `IdAllocator` in `ecs/Entity.ts`.
- **Pooling** for snowballs/particles/vectors; no hot-path allocation.
- **Gameplay divergence from original design.md:** the shipped game controls a
  **single hero** with respawning lives (not a 3-child squad with click-select,
  though selection systems still exist), WASD + right-click move, hold-LMB
  charge throw. Configurable: lives, opponent count, enemy lives, AI
  difficulty, buffs (arena pickups). Score + local leaderboard in menus.
- **Systems present** (`src/systems/`): AI, Movement, Throw, Projectile,
  Collision, Damage, Round, Animation, Selection, AutoSelect, Respawn, Pickup.
  Physics: spatial hash broadphase + primitive narrowphase, line-of-sight,
  pathfinding. Maps are JSON in `public/maps/`.
- **Verification:** `npm test` (vitest, pure sim logic) — **151 tests / 28
  files, all passing** as of this writing. Plus puppeteer browser smoke tests
  (`smoke`, `smoke:interaction`, `smoke:combat`).

## SnowGym — current design and status

Extension-layer rule: `snowgym/` may import `src/`; `src/` must **never**
import snowgym/RL/Python/transport code. Guardrails in PLAN.md: policies only
see observations and return actions; terminal-only reward (+1/−1/0); every
upstream change logged in [`snowgym/UPSTREAM_PATCHES.md`](../snowgym/UPSTREAM_PATCHES.md).

### Architecture (all under `snowgym/`)

- `actions/UnitAction.ts` — canonical engine-independent actions:
  `noop | move(x,y) | throw(x,y,power)` per unit, wrapped in a `TeamAction`.
- `observations/Observation.ts` — detached, deterministic (id-sorted) snapshot:
  allies/enemies (pos, vel, health, FSM state, cooldown, charge), projectiles
  (incl. height), arena, match counts. Uses canonical `"blue"`/`"red"` names
  instead of SnowCraft's internal `Team.Player`/`Team.Enemy`.
- `agents/` — `TeamController` interface (`act(observation) → TeamAction`) and
  `SimpleBlueAgent`: minimal scripted baseline (dodge incoming projectile →
  throw at nearest enemy in range with lead → close distance). Deterministic.
- `adapters/SnowCraftActionAdapter.ts` — validates team ownership, applies
  actions via engine seams, reports per-action results.
- `core/SnowEnvironment.ts` — **the key piece**: DOM-free composition root that
  wires World + Throw/Movement/AI(red)/Projectile/Collision/Damage/Round
  systems directly (bypassing `Game`'s renderer/input). One `step()` = 1 blue
  decision + `60/decisionHz` physics ticks (default 10 Hz decisions = 6 ticks).
  Blue lives forced to 0 → elimination is terminal. Terminal-only reward;
  `terminated`/`truncated` (maxTicks)/structured info; stepping a finished
  episode throws `EpisodeCompleteError`.
- `core/TeamControllerSystem.ts`, `scenarios/Scenario.ts` — validated scenario
  factory `createOpenScenario()`: 1–8 units per side, arena 12–120, explicit or
  deterministically generated non-overlapping spawns, `maxTicks`, seed.
  Default: `THREE_VS_THREE_OPEN` (seed `0x5a17c0de`).
- `server/` — `SnowGymService` (transport-independent handler) + `main.ts`
  (Node HTTP via `tsx`, binds 127.0.0.1:8787). Endpoints: `GET /health`,
  `GET /status`, `POST /reset` (seed + optional scenario), `POST /step`
  (optional external action; default = scripted blue policy), `POST /autoplay`.
  API version `"snowgym.v0"`.
- `replay/` — versioned `snowgym.replay.v0` JSON (frames = observations at
  decision cadence + actions + outcome), validated by `parseReplayRecording`,
  played back through the **existing Three.js renderers** via `replay.html`
  (second Vite entry) with play/pause/scrub/0.5–4x. Not a video; no pixels fed
  back to agents.
- `python/` — `snowgym_client` package (uv-managed, locked): Gymnasium envs
  `SnowGym/Squad-v0` (fixed 3v3) and `SnowGym/Squad-v1` (configurable N vs M,
  fixed 8+8 slots with `ally_mask`/`enemy_mask`/`unit_action_mask` so shapes
  never change), Gymnasium checker (`snowgym-check`), demo CLI
  (`snowgym-demo`) that can record replays. Transport is HTTP (correctness
  reference); a batched direct transport is explicitly future work.

### Upstream patches (the only SnowCraft changes SnowGym made)

Per UPSTREAM_PATCHES.md: (1) `MovementSystem.tryMove(player,x,y)` public seam;
(2) `ThrowSystem.tryThrow` rejects invalid states before side effects;
(3) `vite.config.ts` — snowgym tests in vitest + `replay.html` entry;
(4) `tsconfig.json` — include `snowgym`; (5) `package.json` —
`snowgym:server` / `snowgym:replay:smoke` scripts, `tsx`, `@types/node`.

### Milestone status (from PLAN.md checkboxes)

- **M0 autonomous blue control + server: DONE.** 3v3 blue scripted policy vs
  existing red AI, deterministic completion verified in Node integration test
  and live HTTP flow.
- **M1 reproducible env contract: core DONE** (DOM-free composition, reset/
  observe/step, ticks-per-decision, terminal reward). Open items: migrate red
  AI behind `TeamController`; record sim version/upstream commit in replays;
  state-hash and truncation tests.
- **M2 Gymnasium bridge: core DONE** (v0 schema, fixed-shape spaces, Squad-v0
  registered, checker passing, demo, replay record+playback). Open: batch
  transport, vectorized envs, cross-language golden fixtures, throughput
  benchmarks at various decision Hz.
- **M3 configurable N vs M: core DONE** (validated scenarios, Squad-v1 masks,
  deterministic matrix 1v1/1v3/3v1/3v3/max-size). Open: red behind
  `TeamController` with selectable scripted/random/learned/external opponents;
  throughput/balance benchmarks.
- **M4 multi-agent/research adapters: NOT STARTED** (PettingZoo, partial
  observability, raster obs, latency injection).

### Git history for SnowGym

Two commits: `db920e9 feat(snowgym): add autonomous Gym and configurable
replays`, `411dd8f docs(snowgym): document milestones and demo workflows`, plus
root README announcement `b2db095` (branch HEAD).

## How to run / verify

```bash
npm ci
npm test                    # 151 tests pass (28 files)
npm run typecheck && npm run lint
npm run snowgym:server      # headless env server on 127.0.0.1:8787
cd snowgym/python && uv sync --extra dev
.venv/bin/snowgym-check     # Gymnasium checker against live server
.venv/bin/snowgym-demo --seed 42 --record ../../public/replays/blue-seed-42.json
npm run dev                 # then open /replay.html?recording=/replays/blue-seed-42.json
npm run snowgym:replay:smoke  # self-contained browser replay acceptance test
```

## Design tensions / things to know before changing code

- **Red control is the main unfinished seam.** `AISystem` is still hard-coded
  to red-units-vs-blue-targets; PLAN defers migrating it behind
  `TeamController`. SnowEnvironment instantiates `AISystem` directly.
- **Team naming asymmetry:** sim says `Team.Player`/`Team.Enemy`; SnowGym's
  public contract says `blue`/`red` (mapped in `Observation.ts`). The
  multiplayer plan wants POV-relative colors (you're always blue) — a third
  mapping to keep in mind.
- **SnowCraft's human game is single-hero + respawns; SnowGym is squad +
  elimination, zero reserve lives.** The `RespawnSystem` is simply not
  registered in the headless environment.
- **Determinism** holds within a fixed seed + action sequence in Node
  (integration-tested), but cross-platform FP determinism is explicitly
  distrusted — that's why the multiplayer plan chose server-authoritative
  snapshots over lockstep.
- **Multiplayer is unimplemented**; its plan identifies the ownership refactor
  (ownership ≠ team; `ENEMY.*` handicaps becoming AI-only) as the deepest
  future change, and it would also serve SnowGym's "external opponent" goal.
- Toolchain is bleeding edge (Vite 8, TS 6, vitest 5 beta, prettier 4 alpha) —
  flagged as a risk in the multiplayer plan.
