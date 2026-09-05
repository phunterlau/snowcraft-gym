# Astra commander: default change and planning review

Reviewed 2026-09-05. The operational default is now **`gpt-6-astra` with
`reasoning.effort: low`**, selected explicitly by the user. This changes the
commander client, request-body builder, and the smoke, single-request battle,
and trajectory-aware battle CLIs. `--reasoning light` remains an alias for low.
Explicit `--backend luna` selects `gpt-5.6-luna` with medium reasoning unless
overridden. The map generator keeps its own Luna configuration.

The output contract, prompt, grounding, physical executor, benchmark axes, and
archived pilot artifacts are unchanged. No new provider calls were made for
this default change or review. The earlier authorized 16-call pilot is complete.
Verification passed 344 TypeScript tests, 51 Python client tests, 257 Python
training tests, and the build. Tests confirm outgoing default requests,
explicit Luna overrides, all live CLI help, and byte-equivalent JSON request
objects for every explicitly configured pilot arm.

```bash
# Astra / low by default; running this command makes one API request
node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts --json

# Explicit comparison control
node --import tsx snowgym/orchestration/examples/openai-commander-smoke.ts \
  --backend luna --reasoning low --json
```

## What the evidence establishes

The [matched pilot](benchmark/examples/luna-astra-20260905-v0/README.md) used four
fixed server-state snapshots and the existing scripted fighter. Astra-low and
Luna-low each won three cases. Mean latency was 3.564 s versus 3.576 s; mean
output tokens were 116.75 versus 254.5. Astra-low passed all four host validations.
These measurements establish working integration. Four samples do not establish
a stable speed, cost, or tactical-quality advantage.

The open-arena and terrain cases favored different doctrines. Both Astra
efforts won terrain 10v10, while Luna-medium lost that case. On the opening
5v5, Luna preserved five fighters and Astra preserved two. No arm won 6v10.
All eight Astra responses used a single main role, so this pilot supplies no
evidence that Astra coordinates multiple roles better. More reasoning was
not consistently beneficial. Astra-low reported zero reasoning tokens despite
the requested low effort; usage accounting does not reveal internal computation.

Official documentation positions Astra for demanding multistep work and Luna
for cost-sensitive, high-volume work. That product positioning is a hypothesis
about suitability, not a measured SnowGym improvement. Both model pages list
1,050,000-token context, 128,000-token maximum output, text/image input,
structured outputs, and function calling. Vision, large context, ordinary
tool calling, and persisted reasoning should not be described as new Astra-only
features. [Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Luna model](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[Astra guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra).

## Verified capabilities beyond GPT-5.6

| Capability | Verified API behavior | Potential SnowGym use |
| --- | --- | --- |
| Asynchronous function/custom tools | Astra can continue working while application-owned tools run. The application tracks jobs and returns results on the original call IDs. | Start bounded route or candidate-plan evaluations concurrently while the commander considers other evidence. |
| Mid-turn steering | Astra accepts additional user-message input over Responses WebSockets during an active response. GPT-5.6 does not support it. | Supply a fresh casualty or target-change observation while a plan is being generated. |
| Cache-preserving effort updates | Astra supports `configuration_update` between responses in standard, single-agent mode, without changing the original request-level effort prefix. | Escalate from low to medium after a predeclared difficult-state trigger, then return to low. |

Sources: [async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling),
[mid-turn steering](https://developers.openai.com/api/docs/guides/steering), and
[reasoning configuration updates](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation).

These are available API capabilities; our current adapter uses none of them.
It sends a stateless HTTP Responses request and reads one final command object.
The existing `CommanderScheduler` already lets fighters continue while an
ordinary request is pending. API async tools would add overlap *inside the
commander's tool workflow* rather than create the existing control concurrency.

There are important implementation constraints:

- Async tools apply to application-run function/custom tools, not hosted tools,
  and should not be configured for programmatic tool calling. Tool results
  remain observations; only the host can activate a plan.
- Steering acceptance means queued input. It does not undo prior output or
  cancel running tools. A continuation can occur even after the initial response
  completes; its token/tool limits apply separately. The host needs an overall
  replan budget, source/version tracking, and a rule preventing superseded
  command activation. Disconnects require reconciliation of queued events.
- Effort updates work between responses. They are incompatible with automatic
  compaction/truncation, and `response.reasoning.effort` still reports the
  request-level value. Record effective effort in the host event ledger.

The linked feature guides specify these constraints. All need separate
integration tests before adoption. Ordinary per-request effort selection is
already available for both backends; it requires no new session mechanism.

## What currently limits planning

### Missing tactical evidence

[`StrategicSummary.ts`](commander/StrategicSummary.ts) exposes arena dimensions,
obstacle count, whole-team health/centroid/spread, projectile count, and group
mission/membership totals. It does not tell the model which lane is blocked,
which route offers cover, where separate enemy clusters sit, or whether a
supported role is in range. The final plan permits selectors such as `weakest`
and `leftmost`, but the input lacks separate cluster summaries to compare them.

Add a bounded, versioned server-derived summary containing:

- Per-lane passability, path cost, projectile cover, exposure, and local force
  balance, expressed relative to the existing tactical frame.
- Per-selector cluster counts, health, range, and movement direction, using
  the host's actual clustering and tie-breaking rules.
- Per-role readiness, health relative to initial assignment, objective range,
  supported-group separation, and measured mission progress.

The [role encoder](../training/plan/RoleStateEncoder.ts) already computes useful
physical summaries. Factor reusable pure calculations into a shared runtime
module rather than importing the training stack into the commander. Preserve
host-owned unit IDs, target resolution, and activation anchors. No graph,
screenshot, browser, or pixel input is needed.

Whole-force `healthFraction` currently averages living fighters. Pair it with
initial-force capacity and casualty totals so an apparent increase after a
casualty cannot be interpreted as recovery.

### Evidence age and causal attribution

The trajectory monitor defaults to 20 decisions, or two seconds at 10 Hz.
Mean Astra-low inference in the pilot was about 3.6 seconds. In real-time play,
the world can change substantially while the request is pending. Zero-delay
pilot continuations do not test this effect.

The scheduler's generic timeout defaults to 180 simulation ticks, about three
simulated seconds. Live examples explicitly use 600 ticks and pace decisions;
do not infer that the generic default suits provider latency. Unpaced simulation
can consume its tick deadline far faster than wall time. Keep simulation age
and provider wall-clock deadlines explicit in tests.

`previousPlanOutcome` in the scheduler is formed from the request-time trajectory
when a new candidate is activated. It therefore summarizes that bounded earlier
window, not necessarily the old plan's complete execution through activation.
Label those intervals precisely before using them as evidence of plan success.

Expose activation tick, source-state age, option budget where applicable, and
the time interval of every progress measurement. Distinguish damage exchange,
terrain blockage, cooldown/readiness, and insufficient dwell time before
attributing poor progress to doctrine. Avoid replacing a useful plan merely
because no damage occurred while closing distance.

### Execution knowledge and output reliability

The commander needs concise, measured semantics for `tight` cohesion, `focus`
fire, flank approaches, and role allocations. The terrain result showed that a
cohesion choice alone can change a matched battle outcome. Supply a small
versioned executor description and calibrated development examples; never use
qualification outcomes as prompt examples.

The pilot also exposed a schema mismatch. `intentSummary` has a length bound in
the provider schema; the host additionally rejects surrounding whitespace.
Astra-medium's proposed withdraw plan was rejected solely for that trace-only
text. Review a schema constraint or explicit trace normalization as a separate
contract change. Preserve the raw response and failure in the original pilot.

## Proposed sequence

These are review recommendations, not implementations or authorization for
additional provider spending.

1. **Align evidence and validation.** Resolve the summary constraint explicitly;
   add tests for raw-output preservation and executable-field equality. Specify
   source age and outcome intervals. Keep the final `CommandPlan` unchanged.
2. **Improve bounded server input.** Add lane/cluster/role evidence and executor
   semantics. Compare old versus enriched input for both Luna-low and Astra-low
   on identical snapshots. Hold prompt/output changes separate from input
   changes so results remain attributable.
3. **Test plan sensitivity and latency.** Use paired correct/zero/shuffled-plan
   controls, casualties, and fresh scenarios. Sweep deterministic delays before
   paid online tests. Measure acceptance, source age at activation, unnecessary
   plan changes, mission progress, survival, win rate, and token/latency budgets.
   Report per-case results and uncertainty; keep provider repetition separate
   from independent environment seeds.
4. **Add bounded candidate evaluation.** A read-only host tool can compare at
   most a few candidate plans from the same reconstructed state, returning
   damage, progress, exposure, and uncertainty. Preserve exact action-prefix
   restoration because scripted Red has hidden controller state. Give both
   models identical tools and compute budgets. Label simulator-assisted planning
   separately from observation-only planning and audit any privileged lookahead.
5. **Evaluate Astra-specific transport features.** First test async jobs, then
   steering, then effort updates independently with mocks. Enforce monotonic
   plan versions, stale-result rejection, bounded pending work, reconnect
   behavior, and total continuation budgets. Activate only a final validated
   command; no streamed partial command may reach the fighters.

For effort escalation, prefer a host rule based on persistent stalled progress
or contradictory outcomes after a minimum dwell period. Calibrate it on
development cases. A model's unsupported confidence statement should not
control its own spending. Keep low as the routine setting until paired
evaluation shows that extra inference improves outcomes enough to offset delay.

The most useful near-term hypothesis is that richer causal and spatial evidence
will improve plan choice. Astra's new transport features then offer ways to
reduce waiting and stale context. They do not remove the need for a capable
executor. M7c/M9 learned-fighter gates remain in force; scripted-executor side
experiments must retain their labels and separate seed allocations.
