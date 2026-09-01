# SnowGym Gymnasium client

This package maps the versioned SnowGym JSON server contract to a fixed-shape
Gymnasium environments registered as fixed-3v3 `SnowGym/Squad-v0` and
configurable `SnowGym/Squad-v1` with fixed eight-slot roster tensors,
`SnowGym/Squad-v2` with fixed ten-slot roster tensors, and
presence masks.

The package also exports `SnowGymParallelEnv`, a PettingZoo Parallel API
environment with simultaneous team-level `blue` and `red` agents. It uses the
same server episode and physics as the Gymnasium environments; there is no
second simulator implementation.

`SnowGymResearchParallelEnv` adds optional local visibility plus deterministic
action and observation delays measured in team decisions. These transforms are
agent-facing only: the wrapped server continues to own current state, hashes,
reward, termination, and physics. Each info records the visible observation
tick and the source tick of the action currently applied by the delay queue.

The TypeScript server remains authoritative for physics, reward, termination,
and seeded state. Python only translates numeric Gymnasium arrays to semantic
team actions and encodes returned entity observations.

See the parent [SnowGym README](../README.md) for setup, recording, and replay
commands. Pass `--record PATH` to `snowgym-demo` to write a portable
`snowgym.replay.v0` JSON artifact from detached server observations.

For example:

```bash
.venv/bin/snowgym-demo --blue-units 5 --red-units 2 --red-difficulty hard

# Winter Front defines ten spawn points per team, producing a map-backed 10v10.
.venv/bin/snowgym-demo --map arena6.json \
  --record ../../public/replays/example-winter-front-10v10.json

# Select smaller M/N rosters from evenly distributed native map spawns.
.venv/bin/snowgym-demo --map arena6.json --blue-units 5 --red-units 2

# Official PettingZoo Parallel API check against the live server.
.venv/bin/snowgym-parallel-check --cycles 100 --json

# Validate a partial-information, two-decision-latency profile.
.venv/bin/snowgym-parallel-check --cycles 100 --visibility-radius 8 \
  --action-delay-steps 2 --observation-delay-steps 2 --json
```
