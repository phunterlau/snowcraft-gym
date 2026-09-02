---
name: snowgym
description: Operate and extend this repository's SnowGym headless simulator, Gymnasium environments, configurable M-vs-N scenarios, guarded JSON API, deterministic example builder, and visual replay workflow. Use for SnowGym server control, policy demos, replay generation, map selection, Gym checks, or SnowGym backend changes.
---

# SnowGym

Work from the repository root. Read `AGENTS.md`, then query the live server's
`GET /capabilities` when using HTTP. Treat `snowgym/PLAN.md` and implementation
as authoritative over historical notes in `refs/`.

## Choose the smallest workflow

- Build a deterministic replay without a server: use `npm run snowgym:example`.
- Exercise Gymnasium: start `npm run snowgym:server`, then use the Python package.
- Read live state: use `GET /status`; no browser or graph is required.
- Label a learner-visited state: use `GET /teacher-action`; it is read-only and returns the current state hash with the scripted blue action.
- Mutate live state: use `scripts/strict-step.mjs` or follow
  [the guarded contract](references/contract.md).
- Visualize only after producing a replay: use the existing `/replay.html` UI.

## Build a replay

```bash
npm run snowgym:example -- \
  --blue-units 10 --red-units 10 --map arena6.json --seed 42 \
  --output public/replays/example-winter-front-10v10.json --json
```

The builder refuses overwrite unless `--force` is explicitly passed. Inspect
`npm run snowgym:example -- --help` for native map capacities.

## Control one guarded step

Start the server, write a canonical team action to a JSON file, and run:

```bash
node .agents/skills/snowgym/scripts/strict-step.mjs --action /tmp/action.json
```

Use `--scripted` instead of `--action` only when the built-in blue policy is
intended. The script reads current state, supplies `expectedStateHash`, and
generates an idempotency key. It does not retry an uncertain mutation.

## Verify

Run targeted tests while editing. Before handing off a milestone, run `npm
test`, `npm run build`, and the Python tests. For live Gym verification, run
`.venv/bin/snowgym-check --json` from `snowgym/python` while the server is up.

Read [troubleshooting](references/troubleshooting.md) when the server is
unreachable, a mutation returns 409, a map rejects a roster, or replay smoke
fails.
