# SnowGym Best Milestone — 2026-08-30

## Milestone status

SnowGym has reached a verified autonomous, Gymnasium-ready, configurable squad
milestone:

- Blue is controlled as a team through a code/policy boundary with semantic
  `noop`, `move`, and `throw` actions.
- Simulation, reset, step, reward, termination, and status are server-side and
  headless. The agent does not need a chart, graph, browser, Canvas, or WebGL.
- `SnowGym/Squad-v0` preserves the initial fixed 3v3 Gym contract.
- `SnowGym/Squad-v1` supports deterministic 1–8 blue versus 1–8 red fights with
  fixed eight-slot tensors and ally, enemy, projectile, and legal-action masks.
- Visual recordings are detached JSON artifacts and replay through the existing
  SnowCraft Three.js renderers without coupling rendering to training.

This milestone is published on GitHub `main` at:

```text
b2db0958a6b60599fe988fb28a16d41920610449
```

Relevant commits:

```text
db920e9 feat(snowgym): add autonomous Gym and configurable replays
411dd8f docs(snowgym): document milestones and demo workflows
b2db095 docs: announce SnowGym in root README
```

## Implemented architecture

```text
Gymnasium policy
    ↓ fixed numeric action space
Python SnowGym client
    ↓ versioned JSON reset/step
SnowGym server
    ↓ semantic TeamAction
SnowCraft action adapter
    ↓
DOM-free SnowEnvironment at 60 Hz
    ├── scripted blue policy at configurable decision Hz
    ├── existing red AISystem at easy/normal/hard
    └── terminal-only reward and team-elimination outcome

Detached observations + semantic actions
    ↓
snowgym.replay.v0 JSON
    ↓
existing Three.js arena/player/snowball/particle renderer
```

The transport schema remains `snowgym.v0`; `Squad-v0` and `Squad-v1` are
Gymnasium environment versions over that backward-compatible additive schema.

## Configurable fight contract

`POST /reset` and `SnowGym/Squad-v1` support:

- `blueUnits`: integer in `[1, 8]`
- `redUnits`: integer in `[1, 8]`
- `arenaWidth` and `arenaHeight`: each in `[12, 120]`
- deterministic generated spawns or explicit `blueSpawns` / `redSpawns`
- `decisionHz`: a positive divisor of the 60 Hz simulation rate
- `maxTicks`
- `redDifficulty`: `easy`, `normal`, or `hard`
- explicit episode seed

Unknown fields, invalid counts, overlapping/out-of-arena spawns, and invalid
decision rates are rejected at the server boundary.

## Verified results

Automated verification:

- TypeScript: 28 files and 151 tests passed.
- Python: 7 tests passed.
- Gymnasium checker passed against the live server for both
  `SnowGym/Squad-v0` and `SnowGym/Squad-v1`.
- Strict TypeScript checking and the two-entry Vite production build passed.
- ESLint reported zero errors; one unrelated pre-existing warning remains in
  `src/systems/PickupSystem.test.ts`.
- Prettier checks and `git diff --check` passed for SnowGym-owned changes.
- Browser replay verification passed with a 1280×800 WebGL canvas, seek,
  terminal winner, rewind, replay, and no console/runtime/request errors.

Live seed-42 acceptance fights:

| Configuration           | Result   | Decisions | Final tick | Survivors     |
| ----------------------- | -------- | --------: | ---------: | ------------- |
| 3 blue vs 3 red, normal | Blue win |        49 |        294 | blue 3, red 0 |
| 1 blue vs 3 red, easy   | Red win  |        54 |        324 | blue 0, red 3 |
| 5 blue vs 2 red, hard   | Blue win |        34 |        201 | blue 5, red 0 |
| 8 blue vs 8 red, normal | Blue win |        72 |        432 | blue 7, red 0 |

Checked-in replay artifacts:

```text
public/replays/blue-seed-42.json
public/replays/blue-5v2-hard.json
```

## Reproduce the milestone

Start the authoritative simulation server:

```bash
npm run snowgym:server
```

In another terminal:

```bash
cd snowgym/python
uv sync --extra dev
.venv/bin/snowgym-check
.venv/bin/snowgym-demo --seed 42 --blue-units 5 --red-units 2 \
  --red-difficulty hard \
  --record ../../public/replays/blue-5v2-hard.json
```

Run the self-contained visual acceptance check from the repository root:

```bash
npm run snowgym:replay:smoke
```

To inspect the 5v2 recording interactively:

```bash
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/blue-5v2-hard.json
```

## Current limitations

- The blue controller is a deterministic scripted reference policy, not a
  trained neural policy.
- The JSON server is a singleton correctness/reference transport, not a
  high-throughput vectorized training host.
- Red still runs through the existing direct `AISystem`; it has not yet been
  migrated behind the same `TeamController` action boundary.
- The benchmark reward is terminal-only (`+1`, `-1`, or `0`).
- Replay frames are recorded at policy decision cadence and interpolated for
  viewing; they are not per-physics-tick traces or videos.
- PettingZoo, learned opponents, partial observations, and pixel observations
  are not implemented.

## Recommended next milestone

Build the training-throughput and reproducibility layer before training a
learned blue policy:

1. Add shared TypeScript/Python golden fixtures and same-seed action-trace state
   hashes for `Squad-v1` configurations.
2. Add a long-lived batch/direct transport and Gymnasium vector environment.
3. Benchmark reset/step throughput across roster sizes and decision rates.
4. Train and evaluate a PPO blue baseline against the scripted blue policy and
   fixed red difficulty matrix.
5. Then migrate red behavior behind `TeamController` for configurable learned
   or external opponents and future PettingZoo support.

The scripted blue policy and the seed/configuration matrix should remain frozen
as the regression baseline while the learned-policy path is added.
