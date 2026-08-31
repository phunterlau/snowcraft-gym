# Upstream patch ledger

## `src/systems/MovementSystem.ts`

Reason: semantic controllers need to order one known unit without selection or
UI commands.

Change: added public `tryMove(player, x, y)`, routed the existing group move
command through it, and made rejected state-incompatible orders atomic.

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

Change: added `npm run snowgym:server`, the SnowGym replay browser-smoke
command, `tsx`, and `@types/node`.

Upstream behavior: existing browser scripts and runtime dependencies are
unchanged.

SnowGym dependency: runs `snowgym/server/main.ts` without a separate build
pipeline.
