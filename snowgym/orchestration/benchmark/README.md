# Commander backend comparison

For the next diagnostic, see the [progress-aware recovery preflight](../recovery/README.md):
observed recovery opportunities, enriched input, and deterministic delayed controls.

Compare `gpt-5.6-luna` and `gpt-6-astra` through the same Responses API adapter,
prompt, `snowgym.command-plan.v0` JSON schema, validator, grounder, and scripted
physical executor. This is a commander integration pilot, separate from learned
fighter qualification. No training checkpoint is involved.

The [2026-09-05 pilot results](examples/luna-astra-20260905-v0/README.md) contain
the completed 16-call comparison and frozen offline evidence. Luna-low and
Astra-low each won three of four cases at similar mean latency. One
Astra-medium plan failed the host's whitespace rule and used fallback.
The four-case pilot does not establish general model superiority.

## Choose a backend

Run from the repository root with `OPENAI_API_KEY` already set:

```bash
# Default: Astra / low
node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts --json

node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts \
  --backend astra --reasoning light --json

node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts \
  --backend astra --reasoning medium --json

node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts \
  --backend luna --reasoning medium --json
```

Each smoke invocation makes one paid request. `light` maps to the API's `low`
reasoning effort; it is not a separate model ID. Omitting `--backend` selects
Astra with `low` reasoning. Explicit `--backend luna` retains `medium` unless
`--reasoning` is supplied. The single-request battle and
trajectory-aware battle CLIs accept the same `--backend` and `--reasoning`
parameters. No SDK, browser, or web server is needed. Astra's supported efforts
and structured-output support are documented in the
[official model page](https://developers.openai.com/api/docs/models/gpt-6-astra).

The adapter sends `store: false` and uses environment-only credentials. Model
choice does not change the prompt, output schema, or aggregate evidence. Provider
response/request IDs, actual model, requested model, reasoning effort, latency,
input/output tokens, cached input tokens, and reasoning tokens are recorded.
Failed structured outputs retain usage when the API supplies it. Unknown usage
remains unknown. There are no automatic request retries or model substitutions.

## Reproduce the bounded pilot

First freeze and inspect the four snapshots and deterministic baseline outcomes.
This command makes **zero API calls**, even without credentials:

```bash
node --import tsx snowgym/orchestration/examples/commander-backend-benchmark.ts \
  --dry-run --output /tmp/snowgym-backends-preflight
```

After reviewing `requests.json`, run the paid comparison using those exact
fixtures and a different, new directory:

```bash
node --import tsx snowgym/orchestration/examples/commander-backend-benchmark.ts \
  --fixtures /tmp/snowgym-backends-preflight/fixtures.json \
  --backends luna,astra --reasoning low,medium --max-requests 16 \
  --output /tmp/snowgym-backends-live

node --import tsx snowgym/orchestration/examples/commander-backend-benchmark.ts \
  --verify /tmp/snowgym-backends-live
```

The default matrix is four snapshots × two models × two reasoning levels:
**16 sequential requests**, no retries, 4096 output tokens including reasoning,
60-second deadline per request. The request order rotates by snapshot so each
arm occupies each call position once. Cache state, load, and single-call
variability still affect timings. Use `--backends astra --reasoning low` for a
four-request subset. A request cap smaller than the requested matrix fails
before any API call. Existing output directories are never overwritten.

| Case | Seed | State at request |
| --- | --- | --- |
| Open 5v5 opening | 610001 | Reset, no trajectory history |
| Open 5v5 contact | 610002 | 30 decisions; recent engagement trajectory |
| Terrain 10v10 | 610003 | `arena6.json`, 40 decisions |
| Terrain 6v10 history | 610004 | `arena6.json`, 60 decisions; actual scheduled plan switch at decision 20 |

All cases use 10 Hz decisions and normal scripted Red. The initial plan is the
production single-main Engage fallback. The last case switches to the existing
direct-advance plan and includes the previous plan's measured outcome. Requests
have an empty trigger list because these are scheduled opportunities. No
failure or casualty trigger is fabricated.

The physical state, controller state, plan assignments/anchors, and trajectory
history are reconstructed from reset using the exact archived action prefix.
Each restored step hash and the final physical/plan/trajectory state are checked.
Only the existing aggregate commander request goes to OpenAI; raw observations,
action prefixes, physical unit IDs, and repository source remain local.

## Metrics and interpretation

For each returned plan the host validates, reconciles, grounds, then runs the
same reactive executor for up to **300 additional decisions (30 seconds)**.
The plan remains fixed. API wall-clock latency is measured separately and is
not injected into this continuation. This isolates tactical consequences of
plan choice from asynchronous response timing.

- Integration: schema-valid count, accepted/repaired count, errors, deadlines,
  fallback count, actual/requested model agreement, rejected physical actions.
- Inference: mean/median/p95 latency, input/output/reasoning/cache token means,
  and number of calls with known token usage. Output includes reasoning tokens;
  do not add reasoning tokens again to the output total.
- Combat: winner, surviving units, decisions to termination, censoring, damage
  dealt/received, and net normalized damage. Each team's denominator is its
  initial roster's total maximum health. Damage starts at the request snapshot.

`paired` in `report.json` contains per-case outcomes for every arm and the
deterministic fallback baseline. Accepted-plan summaries exclude fallback runs;
failures remain visible in integration statistics. A timeout has unknown usage
unless returned by the provider, and can still incur charges.

Four heterogeneous scenarios with one sampled plan per arm are exploratory.
Paired outcomes reveal disagreements on these states; they do not establish
general model superiority. The p95 interpolates only four observations.
These seeds are not fighter development/qualification seeds, and this pilot
does not qualify the learned executor or reopen M9 research gates. Online
replanning, latency-injected play, and repeated environment/provider samples
would be separate experiments.

## Artifacts

- `configuration.json`: predeclared settings, call order, source hashes, Git
  revision, Node version, and explicit scripted-executor research boundary.
- `fixtures.json`: physical snapshots, plans, exact prefixes, hashes, and
  aggregate requests, each with a digest.
- `requests.json`: credential-free wire bodies; only model and effort differ
  across the four arms for a given case.
- `baselines.json`: deterministic fallback continuations.
- `response-01.json` through `response-16.json`: raw symbolic output, provider
  metadata/errors, reconciliation outcome, continuation actions and state hashes.
- `report.json`: grouped metrics, paired outcomes, and limitations.
- `manifest.json`: file inventory and SHA-256 checksums for all artifacts.

Each completed call is saved before the next one begins. An interrupted run
retains partial evidence and is never automatically resumed or retried. The
sealed inventory detects changed, missing, or extra files; hashes are integrity
checks, not an external authenticity signature.

For deterministic simulated latency sweeps, see
[`CommanderLatencyBenchmark.ts`](CommanderLatencyBenchmark.ts). That separate
mock benchmark makes no provider calls.
