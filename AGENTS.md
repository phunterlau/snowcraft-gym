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
- `GET /teacher-action` labels the current state without advancing it; guard the learner's following `POST /step` with the returned state hash.
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

Known failure, accepted on 2026-09-13, description corrected on 2026-09-23:
`npm test` has exactly one failing test. In `snowgym/tests/SelectiveRepair.
test.ts`, "runs the CLI preflight without credentials or new episodes and
verifies it" fails with `seed preflight collision`.

**As of 2026-09-23 it names five files**, all colliding on the single value
`trainSeedBase: 630000`: `m7b_engage_r1n_b_v0` (630000 is R1n-b's own real,
reserved seed band) plus `m8_s4_roster_baseline_v0`, `m8_s7_trace_diagnosis_v0`,
`m8_s8_channel_intervention_v0`, and `m8_s9_target_split_v0`. The last four
never actually use `trainSeedBase` — none of their own modules
(`roster_baseline.py`, `roster_trace.py`, `roster_intervention.py`,
`roster_target_split.py`) call `fold_seeds` or `warm_start_critic_mc`, the
only two functions that read that config key. It reaches their
`declaration.json` only because their `configuration()` functions spread
`full_authority_train_v1.configuration()` (whose default is
`trainSeedBase: 630000`) without overriding it, since they have no need to.
This is confirmed inert, not a real seed collision: none of these four steps
ever drew simulator seeds from 630000–630999. (`opponent_transfer.py` found
and worked around this same inherited-default problem in its own module
already; that fix was not propagated to `v1.configuration()`'s default or to
these four steps' configs.)

The gate passes when that is the only failure and every named file's
collision is on `trainSeedBase: 630000` alone (currently these five files;
the set may grow if a future step also fails to override the inherited
default, but should not add any *other* colliding value). Any other
failure, any other colliding *value*, or a file colliding on something
besides this one inert default blocks the gate. Fixing the root cause would
mean editing a shared base config default and four already-sealed archives'
serialized config — out of scope for a routine gate pass; see the erratum
in `snowgym/training/reviews/m7b_r1n_b_results.md` for R1n-b's own band, and
`snowgym/training/reviews/m8_s14_declaration.md` §14 for how this was found.

For the live Gym contract, start `npm run snowgym:server` in one terminal and run:

```bash
cd snowgym/python
.venv/bin/snowgym-check
```

Use `.agents/skills/snowgym/SKILL.md` for operational examples and guarded HTTP control.
