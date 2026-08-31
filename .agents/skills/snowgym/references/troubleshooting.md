# Troubleshooting

- Cannot connect: from the repo root run `npm run snowgym:server`; confirm
  `curl http://127.0.0.1:8787/health` returns `{"ok":true}`.
- `409 stale_state`: another request changed the singleton episode. Discard the
  proposed action, fetch `/status` again, and re-plan from the new observation.
- `409 idempotency_conflict`: generate a fresh key for the new request. Never
  reuse a key with altered input.
- Map roster rejected: query `/capabilities` and keep each team at or below that
  map's native capacity. `arena6.json` supports 10v10.
- Gym checker fails: run it from `snowgym/python` after `uv sync --extra dev` and
  while the TypeScript server is running.
- Replay smoke fails: first run `npm run build`, then run the replay command
  `npm run snowgym:replay:smoke`. It may reuse or start a local Vite server.
