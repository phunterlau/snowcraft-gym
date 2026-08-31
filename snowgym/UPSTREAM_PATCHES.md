# Upstream patch ledger

## `src/systems/AISystem.ts`

Reason: the red squad had to run behind SnowGym's `TeamController` boundary
without changing its behavior in the browser game.

Change: added an optional `AiSquad` parameter (`{ controlled, target }`) to the
constructor, defaulting to the classic `{ Enemy, Player }` pairing, and widened
the `events` parameter to accept `null` (it was already unused). The three
hardcoded `Team.Enemy`-drives-`Team.Player` checks now read from the squad.

Upstream behavior: unchanged; the browser game constructs `AISystem` with the
default squad and identical difficulty tuning.

SnowGym dependency: `ScriptedAiAgent` re-runs the classic AI per-tick inside a
`TeamController` and reports its orders as semantic actions; full-episode
traces are bit-identical to direct registration.

## `src/systems/MovementSystem.ts`

Reason: semantic controllers need to order one known unit without selection or
UI commands.

Change: added public `tryMove(player, x, y)`, routed the existing group move
command through it, and made rejected state-incompatible orders atomic. Later
added public `tryHold(player)` so a controller that reports orders on behalf of
a unit can cancel a stale move target without disturbing other states.

Upstream behavior: unchanged; group formation, clamping, state transitions, and
movement integration use the same logic as before.

SnowGym dependency: `SnowCraftActionAdapter` uses this generic engine seam.

## `src/systems/ThrowSystem.ts`

Reason: arbitrary Gym actions can arrive while a unit is in a state that cannot
start a throw.

Change: `tryThrow` now rejects state-incompatible orders before acquiring or
launching a projectile.

Upstream behavior: valid human and AI throws are unchanged; invalid throws no
longer create a projectile before a failed state transition.

SnowGym dependency: keeps rejected semantic actions free of partial side
effects.

## `vite.config.ts`

Reason: SnowGym tests must participate in normal verification.

Change: added `snowgym/tests/**/*.test.ts` to Vitest discovery and added
`replay.html` as a second Vite build entry.

Upstream behavior: source test discovery remains enabled.

SnowGym dependency: keeps extension-layer tests in the standard test command
and builds the visual replay viewer without changing the normal game entry.

## `tsconfig.json`

Reason: the headless environment and server must be checked by the repository's
strict TypeScript command.

Change: added `snowgym` to the compiler include list.

Upstream behavior: existing `src` and configuration checking is unchanged.

SnowGym dependency: covers the new server and environment sources.

## `package.json` and `package-lock.json`

Reason: the TypeScript server needs a direct Node execution entry point and Node
type declarations.

Change: added `npm run snowgym:server`, `npm run snowgym:example`, the SnowGym
replay browser-smoke command, `tsx`, and `@types/node`.

Upstream behavior: existing browser scripts and runtime dependencies are
unchanged.

SnowGym dependency: runs `snowgym/server/main.ts` without a separate build
pipeline.

## `public/maps/arena6.json` and `src/main.ts`

Reason: the terrain-backed 10v10 SnowGym acceptance scenario needs a map whose
native schema contains ten valid spawn points for each team.

Change: added the 64x48 `Winter Front` arena with symmetric cover and 20 spawn
points, then registered it in the browser map menu.

Upstream behavior: existing maps and the single-player defaults are unchanged;
the browser continues to cap spawned units according to its match settings.

SnowGym dependency: the headless mirror in `snowgym/scenarios/maps.ts` feeds the
same JSON data to `MapLoader.build`, so server physics and browser rendering use
identical terrain and spawn definitions.

## `AGENTS.md` and `.agents/skills/snowgym/`

Reason: coding agents and LLM policy operators need one discoverable source for
the repository boundary, safe mutation workflow, environment versions, and
verification commands.

Change: added a concise root agent guide plus a repo-local SnowGym skill with
guarded HTTP contract, troubleshooting, and a state-hash/idempotency-aware step
wrapper.

Upstream behavior: unchanged; these files are operational documentation and a
local command wrapper only.

SnowGym dependency: directs agents to the renderer-free interfaces, capability
endpoint, safe artifact workflow, and required verification gates.
