# SnowGym backend review: LLM-agent and Codex readiness

Date: 2026-08-31
Reviewed repository state: `main` at `237ec10`
Scope: SnowGym simulation composition, JSON server, Python/Gymnasium bridge,
example builder, replay artifacts, repository guidance, and prospective
`AGENTS.md` / skill integration.

## Implementation follow-up

The first agent-readiness milestone was implemented later on 2026-08-31. The
server now strictly rejects unknown mutation fields, requires explicit actions
on `/step`, isolates built-in control at `/step-scripted`, exposes
`/capabilities`, and supports optimistic `expectedStateHash` plus idempotency
keys. The Python Gym client uses those guards automatically; the live checker
now covers Squad-v0, v1, and map-backed v2; relevant CLIs have JSON output; and
the repository now includes `AGENTS.md` plus a validated repo-local SnowGym
skill. Multi-session hosting, a true `hold` action, and batched direct transport
remain future work.

## Executive assessment

SnowGym is already a strong **code-agent development target** and a good
**deterministic artifact generator**. A Codex agent can inspect typed boundaries,
run the renderer-free example builder, execute tests, produce a replay, and
verify hashes without interacting with a browser or interpreting pixels.

SnowGym is not yet a safe **stateful tool API for an autonomous LLM policy**.
The raw HTTP service has a singleton mutable episode, no idempotency or
optimistic-concurrency guard, no machine-readable capability/schema endpoint,
and permissive top-level request parsing. A typo in a mutating request can
silently invoke the scripted policy or an entire autoplay run.

Current readiness, by access path:

| Access path | Readiness | Assessment |
| --- | ---: | --- |
| TypeScript `SnowEnvironment` | 8/10 | Typed, deterministic, renderer-free, and easy for a coding agent to extend |
| `buildReplayExample()` / `snowgym:example` | 9/10 | Best current agent entry point: one process, strong help, validated output, safe overwrite behavior |
| Python/Gymnasium | 7/10 | Conventional fixed spaces and masks, but requires a live server and remains synchronous/unbatched |
| Raw JSON HTTP control | 4/10 | Usable by a careful human client, unsafe as a direct LLM tool without a strict wrapper |
| Replay/UI workflow | 8/10 | Detached, reproducible, and visually verifiable; artifacts are large for LLM context |
| Repository `AGENTS.md` guidance | 0/10 | No repository-local `AGENTS.md` exists |
| Repository SnowGym skill | 0/10 | No SnowGym `SKILL.md` or agent wrapper scripts exist |

Overall: **6/10 agent-ready today; approximately 8.5/10 after the P1 safety and
guidance work below.**

## What was validated

The review reconciled the implementation with:

- `refs/snowgym_implementation_note.md`
- `refs/snowgym_best_milestone_2026-08-30.md`
- `snowgym/PLAN.md`
- `snowgym/README.md`
- server, simulator, Python, example, replay, map, and test sources

Current automated checks passed:

- TypeScript: 32 test files, 187 tests
- Python: 15 tests
- TypeScript/Vite production build
- `snowgym:example --help`

Known non-failures observed during validation:

- Gymnasium emits deprecation warnings when tests instantiate `Squad-v1`
  because `Squad-v2` exists.
- Vite reports the existing large-chunk warning.

No tracked backend files were changed by this review.

## Backend map

### 1. In-process simulation

`snowgym/core/SnowEnvironment.ts` is the authoritative headless composition
root. It owns reset, observation, semantic blue actions, red controller calls,
fixed 60 Hz physics ordering, terminal reward, termination/truncation, and
versioned public-state hashes.

This is the cleanest boundary for code-authored policies. Policies consume
detached `Observation` values and return `TeamAction`; they do not receive engine
entities or renderer objects.

### 2. Scenario and map construction

`snowgym/scenarios/Scenario.ts` supports open 1-10 vs 1-10 fights and bounded
subsets of native map spawn pools. `snowgym/scenarios/maps.ts` mirrors browser
JSON maps so headless execution does not require `fetch`. Contract tests compare
the mirror with `public/maps/*.json` exactly.

### 3. Transport-independent service

`snowgym/server/SnowGymService.ts` exposes five operations:

- `GET /health`
- `GET /status`
- `POST /reset`
- `POST /step`
- `POST /autoplay`

The service returns detached observations, status/provenance, action results,
and terminal state. It currently owns one mutable `SnowEnvironment` instance.

### 4. HTTP host

`snowgym/server/main.ts` binds to `127.0.0.1`, limits request bodies to 1 MB,
parses JSON, and supports graceful SIGINT/SIGTERM shutdown. These are good local
tool defaults.

### 5. Python/Gymnasium bridge

`snowgym/python/src/snowgym_client/` provides:

- a standard-library synchronous HTTP client;
- fixed-shape numeric observations/actions;
- presence and legal-action masks;
- `SnowGym/Squad-v0`, `Squad-v1`, and `Squad-v2` registration;
- checker, demo, recording, and cross-language state-hash support.

### 6. Example and replay path

`snowgym/examples/ExampleBuilder.ts` and `snowgym/examples/main.ts` are the most
agent-friendly interfaces currently present. They run the simulator directly,
accept M/N, map, seed, cadence, red policy, and limits, then emit a fully
validated replay. The CLI lists map capacities and refuses overwrite without
`--force`.

The replay path retains scenario/provenance/actions/frame hashes and reuses the
existing Three.js UI for optional visualization.

## Existing strengths for a Codex-style agent

1. **Renderer-free source of truth.** Status and control do not require chart,
   graph, Canvas, screenshots, or visual inference.
2. **Typed semantic actions.** `noop`, `move`, and `throw` are independent of
   mouse/keyboard/UI events.
3. **Deterministic identity and state.** Entity observations are ID-sorted;
   seeds, fixed physics cadence, provenance, and public-state hashes are exposed.
4. **Inspectable failures.** Applied actions return `accepted` plus reasons such
   as `duplicate_unit`, `missing_unit`, `wrong_team`, and `unavailable`.
5. **Safe artifact generation.** The example CLI is single-process, documents
   capacities, writes a portable artifact, and requires `--force` to overwrite.
6. **Stable RL tensors.** Gym environment versions have fixed roster capacity,
   masks, and bounded numeric spaces.
7. **Separation from upstream.** Most functionality stays under `snowgym/`, and
   changes outside it are tracked in `UPSTREAM_PATCHES.md`.
8. **Good test surface.** Determinism, map parity, configurable rosters,
   truncation, state hashes, server validation, Gym spaces, and replay parsing
   are covered.

## Findings, ordered by priority

### P1 — Mutating HTTP endpoints silently accept misspelled top-level fields

Scenario fields are allowlisted, but endpoint-level request objects are not.
This is dangerous for generated tool calls.

Confirmed behavior:

| Request | Result |
| --- | --- |
| `POST /reset {"sead":42}` | HTTP 200; typo ignored; requested seed not used |
| `POST /step {"aciton":{"actions":[]}}` | HTTP 200; world advanced six ticks using scripted blue |
| `POST /autoplay {"maxDecision":1}` | HTTP 200; full episode ran under the default 10,000-decision limit |

The most severe case is `/step`: absence of a recognized `action` means
"scripted step," so malformed external control silently changes policy.

Recommendation:

- reject unknown top-level keys on every endpoint;
- make `POST /step` require `action`;
- move built-in policy stepping to `POST /step-scripted`;
- reject unknown fields inside action objects;
- add negative tests proving rejected requests leave `stateHash` unchanged.

### P1 — The HTTP server is singleton and steps are non-idempotent

Every client shares one `SnowGymService.environment`. Any reset replaces the
episode for all clients. A network retry of `/step` advances twice. There is no
session ID, request ID, expected tick, or expected state hash.

This is unsuitable for parallel Codex tasks, tool retries, or uncertain network
completion.

Recommendation:

- introduce episode/session IDs;
- accept `expectedStateHash` or `expectedTick` on mutations and return `409
  stale_state` on mismatch;
- accept an idempotency key and cache the response for duplicate step requests;
- keep a single-session compatibility mode only if required.

### P1 — No repository agent guidance exists

There is no `AGENTS.md` in this repository and no SnowGym `SKILL.md`. An agent
must discover commands, version distinctions, map capacities, two-process Gym
setup, mutation semantics, and verification expectations from multiple files.

The root README links SnowGym, but it is a user guide rather than an operational
agent contract.

Recommendation: add a concise root `AGENTS.md` and a repository-local
`.agents/skills/snowgym/SKILL.md`. Proposed content is below.

### P1 — `noop` and omitted-unit semantics are easy to misinterpret

`SnowCraftActionAdapter` accepts `noop` but does not cancel an existing move
target. Units omitted from `TeamAction.actions` also retain earlier orders.

Confirmed example: after a move step, a unit advanced from `x=-11.7667` to
`x=-11.2` during a subsequent accepted `noop` step.

This can be a valid "issue no new order" definition, but it is not documented
prominently. An LLM is likely to interpret `noop` as "hold position."

Recommendation:

- explicitly document persistence in the protocol and skill;
- preferably add a distinct `hold` action that calls `MovementSystem.tryHold`;
- specify whether every live unit must appear once per team action;
- add action examples for continuing, stopping, and rejected commands.

### P2 — No machine-readable capability or schema discovery

There is no `/capabilities`, OpenAPI document, JSON Schema, or CLI `--json`
contract description. An agent cannot query action constraints, decision rates,
map capacities, environment versions, or endpoint schemas from the backend.

Recommendation: add `GET /capabilities` returning at least:

- API/simulation/hash/replay versions;
- action discriminators and required fields;
- coordinate and power conventions;
- allowed decision rates;
- roster maxima;
- map IDs and native capacities;
- red controllers/difficulties;
- Gym IDs;
- supported transports and singleton/session behavior.

Check this payload against a versioned JSON fixture.

### P2 — `snowgym-check` does not check the newest Gym environment

`snowgym_client.check` checks `Squad-v0` and `Squad-v1`, but not ten-slot
`Squad-v2`. The newest public Gym contract therefore is not covered by the
official live checker command.

Recommendation: check all three IDs, with a 10v10/map reset for `Squad-v2`, and
filter only expected version deprecation warnings.

### P2 — Raw observations and replay artifacts are expensive LLM context

The Winter Front 10v10 replay is 2,045,111 bytes. Its 120 frames are roughly
6.6-9.6 KB each (median 8.5 KB). Static obstacles cost about 3.7 KB per frame
and are repeated approximately 447 KB across the artifact.

A live 10v10 `/status` response is about 7.2 KB compact JSON; a one-decision
`/autoplay` response is about 17.6 KB because final state is repeated in
`result` and the snapshot.

Recommendation:

- add a compact observation/status projection for language agents;
- allow field selection or `includeObservation=false` where appropriate;
- publish a replay v1 that stores static arena/obstacles once;
- avoid placing full replay JSON in an LLM prompt; provide summaries and paths.

### P2 — CLI output is human-readable only

The example CLI is otherwise excellent, but it has no `--json` summary mode.
Errors are ordinary thrown exceptions/stack traces rather than stable error
objects. The server has no `--help`, readiness timeout, or managed background
workflow.

Recommendation:

- add `--json` to example/check/demo commands;
- make structured errors contain `code`, `field`, `message`, and allowed values;
- add a single command that starts a server on an available port, waits on
  health, runs a requested workflow, and always stops the owned process.

### P2 — Version names need an explicit matrix

An agent sees HTTP `snowgym.v0`, simulation `snowgym.sim.v1`, replay
`snowgym.replay.v0`, state hash `snowgym.state.v1`, and Gym `Squad-v0/v1/v2`.
These are legitimate independent versions, but their relationship is implicit.

Recommendation: put a small compatibility matrix in `AGENTS.md`, the skill, and
`GET /capabilities`.

### P3 — HTTP diagnostics and local-browser exposure can be improved

- Non-validation server failures are caught by the HTTP host and labeled
  `invalid_json`, which can hide backend defects.
- CORS allows `*`; while the server binds to loopback, any browser origin can
  attempt to mutate a running local episode.
- Requests have no structured audit log, request ID, or episode ID.
- Synchronous `/autoplay` has no cancellation/progress mechanism.

Recommendation: distinguish `invalid_json`, `invalid_request`, and
`internal_error`; restrict CORS unless cross-origin use is required; add compact
structured request logs; and favor the direct builder for long runs.

### P3 — One milestone note in `refs/` is stale

`refs/snowgym_best_milestone_2026-08-30.md` still says 1-8 squads, direct red
`AISystem`, and an older recommended next milestone. Current code supports
`Squad-v2`, 10v10 Winter Front, red `TeamController`, provenance hashes, and the
example builder.

An agent told to inspect `refs/` may select the stale note over current code.

Recommendation: retain it as historical evidence but add a prominent
"superseded" header linking to `snowgym/PLAN.md` and the newest milestone note.

## How Codex should use the backend today

Until the HTTP issues are fixed, the recommended routing is:

| Goal | Preferred path today |
| --- | --- |
| Build a deterministic example/replay | `npm run snowgym:example -- ...` |
| Add or test a code policy | Implement `TeamController`; use `SnowEnvironment` directly |
| Run baseline Gym behavior | Start local server, then use `SnowGym/Squad-v2` |
| Inspect current state manually | `GET /status`, read-only |
| Drive blue with generated actions | Use a strict wrapper; do not expose raw `/step` directly |
| Run a long scripted battle | Direct example builder, not HTTP `/autoplay` |
| Visual verification | Replay artifact through the existing viewer/smoke command |
| High-throughput RL | Not ready; direct batch/vector transport remains planned |

## Recommended root `AGENTS.md`

`AGENTS.md` should stay short and operational. It should not duplicate the full
README. Recommended sections:

1. **Scope and source of truth**
   - `snowgym/PLAN.md` and implementation/tests outrank historical `refs/` notes.
   - SnowGym may import `src/`; `src/` must not import SnowGym.
   - Record every upstream edit in `UPSTREAM_PATCHES.md`.
2. **Default workflow selection**
   - Use `snowgym:example` for generation and deterministic acceptance.
   - Start HTTP only for transport/Gym work.
   - Start Vite/browser only for replay/UI work.
3. **Mutation safety**
   - Raw server is singleton and steps are non-idempotent.
   - Never retry a step after an uncertain response.
   - Inspect every `actionResults` entry.
   - `noop` does not stop prior movement.
4. **Scenario constraints**
   - Open: 1-10 per team.
   - Maps: bounded by native capacity; arena dimensions/custom spawns forbidden.
   - `decisionHz` must divide 60.
5. **Artifact policy**
   - Write exploratory outputs under `/private/tmp`.
   - Check into `public/replays` only when requested.
   - Never use `--force` without resolving the exact target.
6. **Verification ladder**
   - TypeScript tests, Python tests, build, lint.
   - Live checker only for HTTP/Gym changes.
   - Replay smoke only for replay/render changes.
7. **Version matrix**
   - Explain API, simulation, replay, hash, and Gym versions separately.

## Recommended SnowGym skill

Suggested location:

```text
.agents/skills/snowgym/
├── SKILL.md
├── references/
│   ├── contract.md
│   └── troubleshooting.md
└── scripts/
    ├── capabilities.mjs
    ├── build-example.mjs
    ├── managed-server.mjs
    └── strict-step.mjs
```

Suggested trigger description:

> Build, run, inspect, verify, or replay SnowGym squad environments; generate
> configurable M-vs-N episodes; control blue through the local server; or work
> with SnowGym Gymnasium contracts.

The skill should route tasks rather than copy the README:

- **Generate** -> direct builder wrapper.
- **Verify** -> scoped tests, then full acceptance based on touched layer.
- **Gym** -> managed server lifecycle plus newest environment ID.
- **Replay** -> build artifact, verify hashes, optionally run UI smoke.
- **Manual policy control** -> strict step wrapper only.

The strict step wrapper should:

- validate the request locally against a bundled schema;
- require an explicit action (never fall back to scripted blue);
- fetch and require the expected state hash;
- reject duplicate unit IDs and out-of-range power before HTTP;
- print a compact JSON result, including rejected actions;
- refuse automatic retry after an uncertain mutation.

`AGENTS.md` is guidance, not a callable interface. The skill supplies repeatable
commands, but a stateful MCP server would ultimately be safer than shell/curl for
live LLM control. A future MCP surface should expose typed `capabilities`,
`create_session`, `observe`, `step`, `autoplay`, `build_example`, and
`verify_replay` tools.

## Recommended implementation order

### Phase A — Guidance and low-risk discoverability

1. Add root `AGENTS.md`.
2. Mark the old milestone note superseded.
3. Add a version matrix and action persistence note.
4. Add JSON output modes to existing CLIs.

### Phase B — Safe agent mutations

1. Strict top-level and per-action field allowlists.
2. Separate external `/step` from `/step-scripted`.
3. Add `expectedStateHash` and idempotency keys.
4. Add endpoint tests that assert rejected calls do not mutate state.

### Phase C — Skill and capability contract

1. Add `GET /capabilities` plus a golden fixture.
2. Add the repository SnowGym skill and guarded scripts.
3. Extend `snowgym-check` to `Squad-v2` and a map-backed 10v10 reset.
4. Add one skill-level smoke test from prompt-equivalent inputs to verified
   artifact output.

### Phase D — Multi-client and training architecture

1. Session-isolated server state.
2. Direct/batched transport and vector environment support.
3. Compact language-agent observation and replay v1 static-state deduplication.
4. Optional typed MCP facade for LLM control.

## Acceptance criteria for "Codex-friendly"

The backend should not be called agent-friendly until all of the following are
automated:

- misspelled fields return 400 and preserve state hash;
- `/step` without an explicit action cannot advance the environment;
- duplicated idempotency keys cannot advance twice;
- stale expected hashes return 409;
- two sessions can reset and step without cross-talk;
- `/capabilities` fully describes maps, actions, versions, and limits;
- `snowgym-check` covers `Squad-v2` and 10v10 map observations;
- all operational CLIs have parseable JSON summaries;
- the skill can build and verify open and mapped M-vs-N examples under `/tmp`;
- the skill documents persistent orders and checks all action results;
- full TypeScript/Python/build verification remains green.

## Bottom line

The simulation architecture is substantially more agent-ready than the current
agent interface. Codex can productively work on SnowGym today by calling the
direct example builder or typed TypeScript environment. The next milestone
should not be a broader agent feature: it should be the thin safety and
discoverability layer that turns those strong internals into a strict,
retry-aware, self-describing tool contract.
