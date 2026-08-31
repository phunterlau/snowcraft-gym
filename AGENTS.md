# SnowCraft / SnowGym agent guide

## Source of truth

- `src/` is the upstream browser game. `snowgym/` may import it; `src/` must not import SnowGym.
- `snowgym/core/SnowEnvironment.ts` is the authoritative renderer-free simulation host.
- `snowgym/server/SnowGymService.ts` defines the JSON contract; query `GET /capabilities` before generating requests.
- `snowgym/PLAN.md` is the current roadmap. Files under `refs/` are historical design and milestone notes and may be stale.
- Record every required edit outside `snowgym/` in `snowgym/UPSTREAM_PATCHES.md`.

## Safe workflows

- Prefer `npm run snowgym:example -- ...` for deterministic, single-process replay generation.
- The HTTP server owns one shared episode. Before a mutation, read `/status`; send its `stateHash` as `expectedStateHash` and a unique `idempotencyKey`.
- `POST /step` requires an explicit action. Use `POST /step-scripted` only when the built-in blue policy is intended.
- A `noop` or omitted unit issues no new order; it does not cancel a previous movement order.
- Do not overwrite replay artifacts unless the task explicitly authorizes replacement; the example builder requires `--force`.
- Training and correctness must use detached server state. Visual replay is optional verification, never agent input.

## Environment versions

- `SnowGym/Squad-v0`: fixed 3v3 compatibility environment.
- `SnowGym/Squad-v1`: configurable fixed eight-slot tensors.
- `SnowGym/Squad-v2`: configurable fixed ten-slot tensors and map-backed 10v10.

## Verification

Run checks proportional to the change, with this full milestone gate:

```bash
npm test
npm run build
cd snowgym/python && .venv/bin/python -m pytest -q
```

For the live Gym contract, start `npm run snowgym:server` in one terminal and run:

```bash
cd snowgym/python
.venv/bin/snowgym-check
```

Use `.agents/skills/snowgym/SKILL.md` for operational examples and guarded HTTP control.
