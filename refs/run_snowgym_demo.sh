#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
port="${SNOWGYM_PORT:-8787}"
server_url=""
server_pid=""
ui_port="${SNOWGYM_UI_PORT:-5173}"
ui_url=""
ui_pid=""
server_log="$(mktemp "${TMPDIR:-/tmp}/snowgym-server.XXXXXX.log")"
ui_log="$(mktemp "${TMPDIR:-/tmp}/snowgym-ui.XXXXXX.log")"
replay_path=""

cleanup() {
  if [[ -n "$ui_pid" ]]; then
    kill "$ui_pid" 2>/dev/null || true
    wait "$ui_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  [[ -z "$replay_path" ]] || rm -f "$replay_path"
  rm -f "$server_log" "$ui_log"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$repo_root"
[[ -d node_modules ]] || npm install

if [[ -z "${SNOWGYM_PORT:-}" ]]; then
  while ((port <= 8797)); do
    server_url="http://127.0.0.1:${port}"
    if curl -fsS "$server_url/capabilities" 2>/dev/null |
      grep -q '"format":"snowgym.capabilities.v0"'; then
      break
    fi
    if ! curl -sS --max-time 0.2 "$server_url/" >/dev/null 2>&1; then
      break
    fi
    ((port += 1))
  done
fi
server_url="http://127.0.0.1:${port}"

if ! curl -fsS "$server_url/health" >/dev/null 2>&1; then
  npm run snowgym:server -- --port "$port" >"$server_log" 2>&1 &
  server_pid=$!
  for _ in {1..100}; do
    curl -fsS "$server_url/health" >/dev/null 2>&1 && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$server_log" >&2
      exit 1
    fi
    sleep 0.1
  done
  curl -fsS "$server_url/health" >/dev/null
fi

if ! curl -fsS "$server_url/capabilities" 2>/dev/null |
  grep -q '"format":"snowgym.capabilities.v0"'; then
  echo "SnowGym on port $port does not support the required API" >&2
  exit 1
fi

if [[ ! -x snowgym/python/.venv/bin/snowgym-demo ]]; then
  (cd snowgym/python && uv sync --extra dev)
fi

for argument in "$@"; do
  if [[ "$argument" == "--record" || "$argument" == --record=* ]]; then
    echo "This launcher manages its temporary recording; omit --record" >&2
    exit 2
  fi
done
replay_path="$(mktemp "$repo_root/public/replays/quick-demo.json.XXXXXX")"

snowgym/python/.venv/bin/snowgym-demo \
  --server "$server_url" \
  --seed "${SNOWGYM_SEED:-42}" \
  --max-decisions "${SNOWGYM_MAX_DECISIONS:-2000}" \
  --record "$replay_path" \
  "$@"

if [[ -z "${SNOWGYM_UI_PORT:-}" ]]; then
  while ((ui_port <= 5190)); do
    ui_url="http://127.0.0.1:${ui_port}"
    if ! curl -sS --max-time 0.2 "$ui_url/" >/dev/null 2>&1; then
      break
    fi
    ((ui_port += 1))
  done
  if ((ui_port > 5190)); then
    echo "No free UI port found from 5173 through 5190" >&2
    exit 1
  fi
fi
ui_url="http://127.0.0.1:${ui_port}"

if ! curl -fsS "$ui_url/replay.html" >/dev/null 2>&1; then
  npm run dev -- --host 127.0.0.1 --port "$ui_port" --strictPort >"$ui_log" 2>&1 &
  ui_pid=$!
  for _ in {1..100}; do
    curl -fsS "$ui_url/replay.html" >/dev/null 2>&1 && break
    if ! kill -0 "$ui_pid" 2>/dev/null; then
      cat "$ui_log" >&2
      exit 1
    fi
    sleep 0.1
  done
  curl -fsS "$ui_url/replay.html" >/dev/null
fi

replay_url="$ui_url/replay.html?recording=/replays/$(basename "$replay_path")"
if command -v open >/dev/null 2>&1; then
  open "$replay_url"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$replay_url"
else
  echo "No system browser launcher found; open $replay_url" >&2
  exit 1
fi

echo "Replay opened: $replay_url"
if [[ "${SNOWGYM_NO_WAIT:-0}" != "1" ]]; then
  echo "Press Ctrl-C to stop the temporary servers."
  if [[ -n "$ui_pid" ]]; then
    wait "$ui_pid"
  else
    while :; do sleep 1; done
  fi
fi
