# SnowGym Gymnasium client

This package maps the versioned SnowGym JSON server contract to a fixed-shape
Gymnasium environments registered as fixed-3v3 `SnowGym/Squad-v0` and
configurable `SnowGym/Squad-v1` with fixed eight-slot roster tensors,
`SnowGym/Squad-v2` with fixed ten-slot roster tensors, and
presence masks.

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
```
