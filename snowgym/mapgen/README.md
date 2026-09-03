# SnowGym GPT-5.6 Luna map generator

This server-only side tool asks `gpt-5.6-luna` to author exact SnowCraft map
geometry, then validates and evaluates it with the renderer-free simulator. The
model proposes geometry; deterministic code decides whether the result is a
usable research artifact.

Generated maps stay under this directory and do not become browser/server maps
unless `promote` is explicitly invoked. `OPENAI_API_KEY` is read only at
runtime. Normal tests and offline commands never call OpenAI.

## Quick start

From the repository root:

```bash
npm run snowgym:mapgen -- generate \
  --prompt "A balanced 10v10 battlefield with two lanes and a contested center" \
  --blue-capacity 10 --red-capacity 10 \
  --width 64 --height 48 \
  --topology lanes --symmetry mirror --density medium \
  --output snowgym/mapgen/artifacts/two-lane-10v10 \
  --evaluate --replay --json
```

Generation uses the Responses API with strict structured output, `store:
false`, medium reasoning by default, and at most two requests: a draft and one
repair containing deterministic validation errors. There are no hidden
transport retries. Use `--max-requests 1` to measure raw first-pass validity.

Offline operations:

```bash
npm run snowgym:mapgen -- control \
  --output snowgym/mapgen/artifacts/mirrored-control-10v10 \
  --blue-capacity 10 --red-capacity 10 --evaluate --replay
npm run snowgym:mapgen -- validate snowgym/mapgen/artifacts/two-lane-10v10
npm run snowgym:mapgen -- evaluate \
  snowgym/mapgen/artifacts/two-lane-10v10 \
  --seeds 41,42,43 --replay --force --json
```

To view `replay.json`, copy it to `public/replays/` under a distinct name, run
`npm run dev`, and open:

```text
http://127.0.0.1:5173/replay.html?recording=/replays/NAME.json
```

Promotion is deliberate and edits both static catalogs:

```bash
npm run snowgym:mapgen -- promote \
  snowgym/mapgen/artifacts/two-lane-10v10 --id arena7.json
```

The command refuses an existing map/catalog entry unless `--force` is present.
After promotion, run the full repository test and build gate before committing.

## Native map contract

Coordinates are centered at `(0, 0)`. Generated arenas are 12–120 units per
axis, have 1–10 explicit spawns per team, and contain at most 64 objects so the
fixed Gym obstacle tensor is not silently truncated.

| Type    | Per-instance attributes              | Simulation behavior                                            |
| ------- | ------------------------------------ | -------------------------------------------------------------- |
| `tree`  | `x`, `y`, optional `radius`          | blocks movement, projectiles, and sight; enlarged cover circle |
| `rock`  | `x`, `y`, optional `radius`          | blocks movement, projectiles, and sight                        |
| `fort`  | `x`, `y`, optional `width`, `height` | axis-aligned blocker for movement, projectiles, and sight      |
| `fence` | `x`, `y`, optional `width`, `height` | blocks movement/projectiles but not sight                      |
| `prop`  | `x`, `y`                             | decorative and non-blocking                                    |

`rotation` is intentionally rejected. It exists in the historical TypeScript
type but is ignored by current collision and rendering, so accepting it would
create misleading maps. Behavior flags are type-derived and cannot be authored
by the model.

Hard validation covers schema, dimensions, object budget, complete footprints,
spawn counts/spacing/clearance, and path connectivity from every spawn to an
opposing spawn. Overlap and balance descriptors are retained as warnings and
metrics rather than being mistaken for universal playability proofs.

## Artifact and suite formats

An accepted artifact directory contains:

- `map.json` — canonical exact geometry.
- `request.json` — normalized research condition.
- `manifest.json` — model, reasoning, revision, hashes, request IDs, latency,
  tokens, and every provider attempt.
- `validation.json` — first-pass and repair outcomes.
- `evaluation.json` — deterministic paired normal/swapped-spawn episodes.
- `replay.json` — optional replay understood by the existing Three.js viewer.

The artifact is the reproducible unit; an LLM response is not assumed to be
regenerable. Files are never overwritten without `--force`.

A ready-to-edit development matrix lives at
[`examples/development-suite.json`](./examples/development-suite.json). A batch
config uses this form:

```json
{
  "schemaVersion": "snowgym.map-suite.v0",
  "maps": [
    {
      "id": "dev-lanes-01",
      "brief": "Two lanes with a contestable center",
      "blueCapacity": 10,
      "redCapacity": 10,
      "width": 64,
      "height": 48,
      "topology": "lanes",
      "symmetry": "mirror",
      "density": "medium",
      "objectBudget": 40,
      "desiredCover": "medium",
      "split": "development",
      "tags": ["two-lane"]
    }
  ]
}
```

Run it with an explicit cost bound:

```bash
npm run snowgym:mapgen -- suite suite.json \
  --output snowgym/mapgen/artifacts/generalization-v0 \
  --max-maps 20 --max-requests 2 --evaluate --json
```

Keep evaluation configs and outputs sealed until the policy/executor is frozen.
Development and evaluation split labels are recorded, not inferred.

## Research use

The intended comparisons are native maps, valid first-pass Luna maps, repaired
Luna maps, and deterministic mirrored controls. Report first-pass validity,
repair yield, diversity/novelty descriptors, topology, path imbalance,
termination and win distributions, side swaps, provider latency, and token use.
These are observational environment and policy measurements; they do not by
themselves establish causal improvement, universal balance, or closed-loop
commander generalization.
