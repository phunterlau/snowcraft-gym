# Luna/Astra pilot — live results

Completed on 2026-09-05 after explicit approval of the 16-call budget. All
16 requests returned completed responses from the requested model. Fifteen
plans passed host validation; one Astra-medium plan used the existing fallback.
There were no retries, transport errors, or timeouts. The earlier blocked launch
made no API requests and is separate from this completed run.

The outbound payload was the existing strict command-plan schema and
instructions plus aggregate strategic state, the current symbolic plan, and
optional aggregate trajectory/previous-plan-outcome evidence. Raw unit state,
physical action prefixes, repository source, and the API key are not included
in the JSON payload. The API key is used only for authentication.

Actual allocation: eight `gpt-5.6-luna` and eight `gpt-6-astra` requests,
four each at `low` and `medium` reasoning per model. Each request was capped at
4096 output tokens including reasoning and 60 seconds, with no automatic retry.

## Comparison

Each arm receives the same four snapshots. Every returned plan runs through
the same grounder and scripted reactive executor. Latency is measured in wall
time; continuations start from the frozen state with zero simulated API delay
and hold one plan for up to 300 decisions at 10 Hz.

| Backend / reasoning | Mean latency | Median / p95 latency | Mean output tokens | Mean reasoning tokens | Host accepted | Battle outcomes |
| --- | --- | --- | --- | --- | --- | --- |
| Luna low | 3.576 s | 3.142 / 4.972 s | 254.50 | 129.0 | 4/4 | 3 wins, 1 censored |
| Luna medium | 4.207 s | 3.794 / 6.006 s | 302.00 | 192.5 | 4/4 | 2 wins, 2 losses |
| Astra low | 3.564 s | 3.480 / 3.874 s | 116.75 | 0.0 | 4/4 | 3 wins, 1 loss |
| Astra medium | 6.427 s | 6.041 / 7.766 s | 271.25 | 91.0 | 3/4 | 3 wins, 1 fallback loss |

Mean input tokens were 1359.25 for every arm. Every response reported usage;
all reported zero cached input tokens. Total usage was **21,748 input tokens
and 3,778 output tokens**, with **1,650 reasoning tokens already included in
the output total**. Dollar costs were not estimated. Astra-low was configured
with `reasoning.effort: low`; the API reported zero reasoning tokens on its
four responses. This is a usage observation, not an inference about internal
computation.

All **7,206 physical action results were accepted**, including the fallback
continuation. There were no reconciler repairs. `schemaValid` in `report.json`
means the existing host `parseCommandPlan` check, which includes constraints
beyond the provider JSON schema.

### Paired battle outcomes

Entries show winner and blue/red survivors. Damage is measured only after the
request snapshot; denominators are the teams' initial total maximum health.

| Case | Luna low | Luna medium | Astra low | Astra medium |
| --- | --- | --- | --- | --- |
| Open 5v5 opening | Blue, 5/0 | Blue, 5/0 | Blue, 2/0 | Blue, 2/0 |
| Open 5v5 contact | Blue, 5/0 | Blue, 3/0 | Blue, 3/0 | Blue, 5/0 |
| Terrain 10v10 | Blue, 5/0 | Red, 0/6 | Blue, 4/0 | Blue, 4/0 |
| Terrain 6v10 history | Censored, 1/10 | Red, 0/9 | Red, 0/8 | Fallback: Red, 0/9 |

Luna-low's final case was still running after the 30-second continuation; this
is neither a win nor a terminal draw. No backend won this 6v10 state.

Mean net normalized damage over accepted plans was 0.2033 for Luna-low,
-0.0633 for Luna-medium, 0.0267 for Astra-low, and 0.2867 for Astra-medium.
**The last number excludes the difficult rejected 6v10 case and is not a
matched four-case comparison.** On the three commonly accepted cases the means
were respectively 0.5067, 0.1933, 0.2600, and 0.2867. Per-case values remain
available in [`live/report.json`](live/report.json).

### Findings and limitations

- Astra-low exercised the existing command contract successfully. Its mean
  latency was within 12 ms of Luna-low in this sample and it used fewer output
  tokens. This sample cannot establish a reliable latency ranking.
- Medium reasoning did not improve the overall pilot outcomes. No default
  reasoning setting was changed after observing results.
- Tactical effects depended on the state. On terrain 10v10, both Astra efforts
  returned the same executable doctrine and produced identical trajectories.
  Luna-low and Luna-medium differed only in executable cohesion (`normal`
  versus `tight`); their matched continuations won and lost respectively. This
  isolates a doctrine effect on that state, not a general ranking of models.
- The 6v10 fixture exposed a contract mismatch: the provider schema allows a
  1–160-character `intentSummary`, while the host additionally rejects leading
  or trailing whitespace. Astra-medium returned a 160-character summary ending
  in a space. The host rejected the whole plan, including its proposed withdraw
  order. Its fallback loss is not evidence about that unexecuted strategy.
- Several summaries reached the 160-character limit and ended mid-sentence.
  This trace-only field deserves a separately versioned schema/host alignment
  review. Possible choices are encoding the whitespace constraint in the
  provider schema or explicitly permitting normalization of trace-only text.
  Neither change was applied to this pilot, and no response was repaired or
  regenerated to improve the result.
- Four heterogeneous snapshots with one provider sample per arm cannot support
  generalization, significance, or learned-fighter qualification claims. The
  p95 is descriptive over only four observations. The benchmark does not
  measure online replanning or the combat effect of real inference latency.

The practical result is a working secondary Astra backend and inspectable
paired evidence. Preserve both low-reasoning backends for future comparisons;
the pilot does not justify switching the production default or reopening M9.

## Live evidence and audit

[`live/`](live/) contains the unchanged raw plans and metadata, 16 unique
provider response IDs, exact physical action continuations, paired metrics,
source hashes, and a sealed file inventory. The manifest and report aggregation
were independently verified after collection. **All 16 continuations were
re-executed locally and matched their archived actions, state hashes, activation
outcomes, and metrics exactly**, without further provider calls.

The JSON artifacts are immutable. This README interprets them without altering
the predeclared settings or replacing negative evidence. Configuration captures
the source state at collection; later roadmap/documentation updates do not
rewrite that historical source digest.

## Frozen offline evidence

[`preflight/`](preflight/) contains the settings/source digests, four exact
state/action-prefix fixtures, all 16 credential-free request bodies, four
scripted fallback continuations, and an audited file manifest. Inputs and schema
are identical across arms apart from model ID and reasoning effort.

| Case | Scripted fallback result | Blue/red survivors | Net normalized damage after request |
| --- | --- | --- | --- |
| Open 5v5 opening | Blue wins | 5/0 | 0.880 |
| Open 5v5 contact | Blue wins | 5/0 | 0.600 |
| Terrain 10v10 | Red wins | 0/3 | -0.180 |
| Terrain 6v10 history | Red wins | 0/9 | -0.813 |

These are baseline outcomes, not LLM results. All four baseline continuations
terminated within the 300-decision horizon and issued zero rejected actions.

Verification passed: 338 TypeScript tests, production build, 51 Python client
tests, and 257 Python training tests. CLI tests cover credential-free preflight,
request caps, immutable output, and artifact tamper detection. Existing Gym
version-deprecation and browser bundle-size warnings remain non-blocking.

## Reproduce with a new, explicitly approved budget

From the repository root with `OPENAI_API_KEY` already set:

```bash
node --import tsx snowgym/orchestration/examples/commander-backend-benchmark.ts \
  --fixtures snowgym/orchestration/benchmark/examples/luna-astra-20260905-v0/preflight/fixtures.json \
  --backends luna,astra --reasoning low,medium --max-requests 16 \
  --output /tmp/snowgym-luna-astra-new-live-run
```

The output directory must be new. The run remains headless and uses the existing
scripted executor; learned-fighter qualification and M9 gates are unchanged.
See the [benchmark guide](../../README.md) for metrics and limitations.
