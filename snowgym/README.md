# SnowGym

SnowGym is an extension layer over SnowCraft for autonomous teams and future RL
environments. Code in this directory may import the upstream engine in `src/`;
the upstream engine must not import SnowGym.

## First runnable milestone

The headless server runs three policy-controlled blue units against the
existing three-unit red AI with a fixed seed, fixed spawns, no obstacles, no
buffs, no respawns, and no human input:

```bash
npm install
npm run snowgym:server
```

The server binds only to `127.0.0.1` and exposes JSON—there is no chart or
rendering dependency:

```bash
curl http://127.0.0.1:8787/status
curl http://127.0.0.1:8787/capabilities
curl -X POST http://127.0.0.1:8787/reset \
  -H 'Content-Type: application/json' -d '{"seed":42}'
curl -X POST http://127.0.0.1:8787/step-scripted \
  -H 'Content-Type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8787/autoplay \
  -H 'Content-Type: application/json' -d '{"maxDecisions":2000}'
```

`GET /status` returns environment metadata plus the current blue-team
observation. `GET /capabilities` publishes versions, endpoint and action
constraints, decision rates, map capacities, and Gym IDs. `POST /step` requires
an explicit canonical action under `action`; `POST /step-scripted` runs the
built-in blue policy for one decision. `POST /autoplay` runs that policy until
termination, truncation, or the supplied decision limit. `POST /reset` starts a
seeded episode. Unknown request and action fields are rejected.

`POST /step-joint` accepts explicit `actions.blue` and `actions.red` team
actions, applies both before advancing the same physics decision, and returns
mirrored detached observations plus zero-sum rewards. It is the transport used
by the PettingZoo adapter; the built-in red controller is not run during a
joint decision.

The reference server owns one shared episode. Mutating calls accept an optional
`expectedStateHash` from the latest `/status` and an `idempotencyKey`. A stale
hash returns HTTP 409 without advancing; an exact retry using the same key
returns the cached response, while reuse with different input returns HTTP 409.
The Python Gym client supplies both guards automatically.

`POST /reset` also accepts an optional configurable fight. Team sizes are in
`[1, 10]`; omitted spawn arrays are generated deterministically:

```bash
curl -X POST http://127.0.0.1:8787/reset \
  -H 'Content-Type: application/json' \
  -d '{
    "seed": 42,
    "scenario": {
      "blueUnits": 5,
      "redUnits": 2,
      "arenaWidth": 50,
      "arenaHeight": 24,
      "decisionHz": 20,
      "maxTicks": 3600,
      "redDifficulty": "hard",
      "redController": "random"
    }
  }'
```

Optional `blueSpawns` and `redSpawns` arrays accept explicit `{ "x", "y" }`
positions. The server rejects unknown fields, invalid counts, out-of-arena
positions, overlaps, and decision rates that do not divide 60 Hz.

The red team is controlled through the same `TeamController` boundary as blue.
`redController` selects the opponent: `"scripted"` (default) is the classic
utility-scored squad AI with per-tick dodges, aim error, and cover play —
behaviorally identical to the browser game's AI at the same `redDifficulty` —
while `"random"` is a seeded baseline that wanders and throws at random.

A scenario can instead target a bundled SnowCraft map by id
(`arena1.json`–`arena6.json`). A map fixes the terrain and native spawn pool;
optional `blueUnits` and `redUnits` select evenly distributed native spawns up
to that map's capacity, while arena dimensions and custom spawn arrays must be
omitted. Obstacles affect line-of-sight, cover, and collision, and are exposed
to the policy as a fixed-capacity masked `obstacles` tensor:

```bash
curl -X POST http://127.0.0.1:8787/reset \
  -H 'Content-Type: application/json' \
  -d '{ "seed": 42, "scenario": { "map": "arena4.json", "redDifficulty": "hard" } }'
```

Example externally supplied action:

```json
{
  "action": {
    "actions": [
      { "type": "move", "unitId": 1, "x": -8, "y": 0 },
      { "type": "throw", "unitId": 2, "x": 5, "y": 1, "power": 0.7 },
      { "type": "hold", "unitId": 3 }
    ]
  }
}
```

`noop` means "issue no new order": it does not cancel an existing movement
target. A live unit omitted from `actions` also retains its prior movement
order. `hold` explicitly cancels a pending movement order and leaves the unit
in place when its current state accepts orders.

The JSON contract reports `apiVersion: "snowgym.v0"` and uses canonical
`"blue"` / `"red"` team names rather than SnowCraft's internal player/enemy
labels. Status and step info also report `simulationVersion`,
`upstreamBaseCommit`, `stateHashVersion`, and `stateHash`. The hash is a
versioned, non-cryptographic checksum of the detached public observation. It is
intended for deterministic regression and exact-action replay checks; it does
not cover hidden controller or RNG internals.

The blue policy consumes detached entity observations and emits semantic
`noop`, `hold`, `move`, and `throw` actions. `SnowCraftActionAdapter` validates team
ownership and applies accepted actions through the movement and throwing system
APIs. One environment step advances six 60 Hz physics ticks, giving the policy
a 10 Hz decision rate.

## Blue-team demo

Terminal 1, from the repository root:

```bash
npm run snowgym:server
```

Terminal 2:

```bash
cd snowgym/python
uv sync --extra dev
.venv/bin/snowgym-check
.venv/bin/snowgym-demo --seed 42 --max-decisions 2000 \
  --record ../../public/replays/blue-seed-42.json
```

The seed-42 acceptance run should report a completed blue win and the surviving
team counts directly in the terminal, then write a portable visual recording.
Both `snowgym-check` and `snowgym-demo` accept `--json` for agent-friendly
machine-readable output; the checker validates Squad-v0, Squad-v1, and
Squad-v2.

## PettingZoo two-team environment

`SnowGymParallelEnv` exposes `blue` and `red` as simultaneous team-level agents
through PettingZoo's Parallel API. Both sides receive the same fixed-capacity
numeric schema from their own ally/enemy perspective, and both actions are
translated back to the server's semantic action contract:

```bash
# With npm run snowgym:server running in another terminal:
cd snowgym/python
.venv/bin/snowgym-parallel-check --cycles 100 --json
```

The checker runs PettingZoo's official `parallel_api_test` against the live
server. Training and correctness remain renderer-free.

For partial-information and latency experiments, `SnowGymResearchParallelEnv`
wraps the same parallel environment without changing server state or physics.
Local visibility is measured in world units; action and observation delays are
integer team decisions. The info dictionary reports the current authoritative
tick separately from `research.observationSourceTick` and
`research.appliedActionSourceTick`:

```bash
.venv/bin/snowgym-parallel-check --cycles 100 \
  --visibility-radius 8 \
  --action-delay-steps 2 \
  --observation-delay-steps 2 \
  --semantic-raster-size 32 \
  --json
```

Initial action-delay slots issue semantic no-ops, and initial delayed
observations repeat the reset observation. Profiles are deterministic and keep
the standard fixed-capacity spaces, so the official Parallel API checker still
applies.

When `semantic_raster_size` is enabled, each observation retains all existing
entity tensors and adds a float32 `semantic_raster` with five channels in this
order: allies, enemies, friendly projectiles, hostile projectiles, and
obstacles. The square grid is derived directly from detached world-space
semantics and never starts WebGL or feeds rendered pixels into training. Enemy
and hostile-projectile channels respect the configured local visibility.

To replay it through SnowCraft's existing Three.js arena, character, snowball,
particle, camera, lighting, and asset renderers, start the normal Vite server
from the repository root:

```bash
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/blue-seed-42.json
```

The replay viewer has play/pause, scrubbing, 0.5x-4x speed, and a local JSON
file picker. Recording is based only on detached server observations and
semantic actions; training remains headless and does not depend on WebGL. The
versioned `snowgym.replay.v0` JSON is a visual record at the 10 Hz policy
decision cadence, with interpolation for smooth playback. It is not a video and
does not feed rendered pixels back into the agent. New recordings include
simulation provenance and one public-state hash per frame. The replay parser
remains backward compatible with earlier `snowgym.replay.v0` artifacts that do
not contain these optional fields.

Trajectory-aware commander runs can additionally write a separate, versioned
commander-trace sidecar. Add `&trace=/replays/<trace-file>` to the replay URL,
or use **Open commander trace** in the viewer. The optional overlay shows the
symbolic plan and aggregate orchestration evidence at the current scrubber
position; it is never used as agent input. See
[`orchestration/README.md`](orchestration/README.md#commander-trace-and-replay-overlay)
for generation commands and the replay-binding contract.

### Example replays

Ready-made recordings live in [`public/replays/`](../public/replays/). With the
Vite server running (`npm run dev -- --host 127.0.0.1`), open the viewer and
either pick a file with **Open recording**, or load one directly:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/<file>
```

| File                              | Scenario                                       | Result   |
| --------------------------------- | ---------------------------------------------- | -------- |
| `blue-seed-42.json`               | Open 3v3, normal scripted red (acceptance run) | blue 3–0 |
| `blue-5v2-hard.json`              | Open 5v2, hard scripted red                    | blue win |
| `example-open-3v3.json`           | Open 3v3, scripted red                         | blue 3–0 |
| `example-open-1v3-hard.json`      | Open 1 blue vs 3 hard red                      | red win  |
| `example-open-2v5-normal.json`    | Open 2 blue vs 5 red                           | red win  |
| `example-open-8v8.json`           | Open 8v8 on a large arena                      | blue 7–0 |
| `example-winter-front-10v10.json` | Winter Front (`arena6`) 10v10                  | blue 9–0 |
| `example-forest-3v3.json`         | Pine Forest (`arena4`) 3v3 — dense tree cover  | blue 3–0 |
| `example-pond-5v2-hard.json`      | Frozen Pond (`arena2`), hard red               | blue win |
| `example-village-random.json`     | Village Skirmish (`arena3`) vs `random` red    | blue win |

Map recordings render the terrain (trees, rocks, forts) and show units using
cover. Record your own with `--record PATH` on `snowgym-demo` (see below).

The browser acceptance command is self-contained:

```bash
npm run snowgym:replay:smoke
```

It reuses a reachable replay server or starts a temporary Vite server on port
5173, verifies WebGL rendering, seeking, the terminal winner, and rewind/play,
writes `/tmp/snowgym-replay.png`, and then stops the server it started.

## Build configurable examples

`snowgym:example` builds a complete replay directly from the headless simulator;
it does not require the HTTP server, Python, a browser, or WebGL:

```bash
# Open arena: arbitrary rosters through 10v10
npm run snowgym:example -- \
  --blue-units 4 --red-units 7 --map open \
  --arena-width 54 --arena-height 40 --seed 11

# Terrain map: roster must fit the map's native spawn capacity
npm run snowgym:example -- \
  --blue-units 5 --red-units 2 --map arena6.json \
  --red-difficulty hard --seed 17 \
  --output public/replays/example-arena6-5v2.json
```

Run `npm run snowgym:example -- --help` to list map capacities and all inputs,
including decision rate, tick/decision limits, red controller, and output path.
The builder records the scenario, semantic actions, provenance, observations,
and verified frame hashes. It refuses to replace an existing output unless
`--force` is supplied.

## Gymnasium client

Importing `snowgym_client` registers legacy fixed-3v3 `SnowGym/Squad-v0`,
eight-slot configurable `SnowGym/Squad-v1`, and ten-slot configurable
`SnowGym/Squad-v2`:

```python
import gymnasium as gym
import snowgym_client

env = gym.make(
    "SnowGym/Squad-v2",
    server_url="http://127.0.0.1:8787",
    blue_units=5,
    red_units=2,
    arena_width=50,
    arena_height=24,
    decision_hz=20,
    red_difficulty="hard",
)
observation, info = env.reset(seed=42)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

`Squad-v1` retains its original eight ally/enemy slots. `Squad-v2` exposes ten
slots per team and supports fights through 10v10. `ally_mask`, `enemy_mask`, and
`unit_action_mask` identify present and actionable slots, so changing N/M at
construction or through `reset(options={"scenario": ...})` never changes an
environment version's tensor shapes. The action space contains one action type,
normalized target, and throw power per blue slot. The HTTP adapter is the
correctness/reference transport; a direct batched transport remains planned for
high-throughput training.

Configurable command-line examples:

```bash
# 1 blue vs 3 easy red
.venv/bin/snowgym-demo --blue-units 1 --red-units 3 --red-difficulty easy

# 5 blue vs 2 hard red, with a replay artifact
.venv/bin/snowgym-demo --blue-units 5 --red-units 2 --red-difficulty hard \
  --record ../../public/replays/blue-5v2-hard.json

# scripted blue vs the seeded random red baseline
.venv/bin/snowgym-demo --red-controller random

# 10v10 on the terrain-backed Winter Front map
.venv/bin/snowgym-demo --map arena6.json \
  --record ../../public/replays/example-winter-front-10v10.json
```

## Layout

```text
actions/        canonical semantic action types
observations/   detached entity-state observations
agents/         policy interface and scripted blue baseline
adapters/       SnowCraft action application boundary
core/           decision controller and DOM-free environment lifecycle
scenarios/      deterministic scenario metadata
server/         local JSON status/reset/step/autoplay API
python/         Gymnasium package, checker, tests, and demo CLI
replay/         versioned replay validation and existing-engine visual playback
examples/       renderer-free configurable replay builder and CLI
reproducibility/ versioned public-observation canonicalization and hashing
fixtures/       cross-language TypeScript/Python contract fixtures
tests/          SnowGym-owned unit and integration tests
```

See [PLAN.md](./PLAN.md) for the repository audit and staged RL roadmap, and
[UPSTREAM_PATCHES.md](./UPSTREAM_PATCHES.md) for the small upstream change ledger.
