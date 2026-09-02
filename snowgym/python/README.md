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
An optional fixed-size semantic raster adds ally, enemy, friendly-projectile,
hostile-projectile, and obstacle channels without invoking the renderer.

The TypeScript server remains authoritative for physics, reward, termination,
and seeded state. Python only translates numeric Gymnasium arrays to semantic
team actions and encodes returned entity observations.

`SnowGymBatchEnv` removes per-decision HTTP overhead without adding another
simulator. Its persistent newline-delimited subprocess owns independent
`SnowEnvironment` instances and returns fixed tensors with a leading batch
dimension. It supports per-slot seeds/scenarios, selective reset, guarded
single-team steps, raw joint-step requests, and explicit per-world failures.
`activate_plans()` sends validated symbolic plans through guarded server-side
grounding; `plan_observations()` returns current `plan_groups [B,3,38]` and
`plan_group_mask [B,3]` plus plan metadata without duplicating target resolution
or tactical geometry in Python.
`plan_teacher_actions()` reads the matching production executor labels at those
same state hashes without stepping, providing the safe oracle boundary for
plan-conditioned DAgger collection.
Run the live golden transport check while the HTTP server is available:

```bash
.venv/bin/snowgym-batch-check --worlds 8 --json
```

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
  --action-delay-steps 2 --observation-delay-steps 2 \
  --semantic-raster-size 32 --json

# Run the bundled versioned 3v3, map-backed 10v10, and research-profile suite.
.venv/bin/snowgym-benchmark --repeat 1 --output benchmark.json
```

`snowgym-benchmark` runs episodes sequentially because the reference HTTP
server owns one shared episode. The `results` and `summary` sections are
deterministic for a fixed suite, while the separate `performance` section is a
wall-clock measurement. The bundled `baseline-v0` suite uses independently
seeded, action-mask-aware random policies for both teams and includes the
scenario, policy, profile, final state hash, outcome, and rejected-action count
in every result. Pass `--suite PATH` for another
`snowgym.evaluation-suite.v0` document. Existing output is never overwritten
unless `--force` is supplied. Repeats replay the same declared seed and policy
stream; add explicit suite episodes with different seeds for varied trials.

For single-team learning against a joint-step opponent, wrap
`SnowGymParallelEnv` (or its research-profile wrapper) in
`SnowGymSingleTeamEnv`. `MaskedRandomOpponent` is a seeded baseline,
`LearnedOpponent` adapts an in-process policy callable, and `RemoteOpponent`
uses the versioned, ID-free `snowgym.opponent-observation.v0` /
`snowgym.opponent-action.v0` tensor contract behind an injected client. Invalid
or failed opponent responses raise before the authoritative server advances.
The original `SnowGymEnv` remains the direct route for the native TypeScript
scripted and seeded-random red controllers.
