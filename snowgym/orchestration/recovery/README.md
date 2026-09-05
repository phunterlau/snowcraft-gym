# Progress-aware commander recovery

Headless, provider-independent preflight for comparing Luna-low, Astra-low,
and Astra-medium with the existing scripted fighter. The final command remains
`snowgym.command-plan.v0`. This diagnostic does not qualify a learned fighter.

## Run

From the repository root:

```bash
node --import tsx snowgym/orchestration/examples/commander-recovery-benchmark.ts \
  --output /tmp/snowgym-recovery-preflight

node --import tsx snowgym/orchestration/examples/commander-recovery-benchmark.ts \
  --verify /tmp/snowgym-recovery-preflight
```

No API key, provider call, server, browser, or screenshot is used. Output must
not exist. The default scans four declared worlds for 300 decisions each, then
runs 300-decision continuations with 0/1/2/4/8-second delays. Each continuation
is rerun and compared exactly. Delays consume the shared continuation budget.

## Input contract

`RecoveryEvidence.ts` builds the optional `recoveryEvidence` request field from
detached server observations, the immutable activation observation/plan, and a
20-decision trajectory window. Existing requests omit it and remain unchanged.
The production scheduler does not enable this evidence automatically.

The evidence contains one row per assigned role: surviving force, health
relative to activation, movement/throw readiness, executor-range occupancy,
range excess, objective distance, conservative direct-path obstruction, frozen
target health/completion, and recent movement/throw/damage observations.
Source hash, source tick, plan version, activation time and window boundaries
are explicit. IDs, individual coordinates and physical actions stay local.

Readiness and throw range reuse the production reactive policy predicates.
They do not establish adapter acceptance or hit probability. Obstruction uses
straight segments and conservative public axis-aligned obstacle footprints;
it does not compute navigation paths, unit-radius clearance or projectile arcs.
Enemy health loss is whole-team evidence and cannot attribute damage to a role.
No calibrated skill-success model is available.

## Opportunity gates

The scanner selects the **first qualifying event per family per world**, before
any model outcome is known. It never mutates world state to manufacture an event.

| Family | Required evidence |
| --- | --- |
| Blocked advance | Full window; at least five accepted moves; stuck fraction at least 0.5; direct movement obstruction; objective farther than two world units |
| Target eliminated | Original activated enemy-cluster membership eliminated while another enemy remains alive |
| Recent casualties | Full window; at least one assigned member lost in the window |
| Throws without damage | Full window; at least five accepted throws and no enemy-force health loss |

These are diagnostic predicates, not mission-success labels. Blocked movement
can occur during Engage; the name does not require an Advance command. Throws
without damage can reflect flight time, range, obstruction or aim. A qualifying
predicate alone does not establish that a different command will help.

## Artifacts and scoring

- `configuration.json`: source-file digests, fixed arms, delays and budgets.
- `scan.json`: each world's scan length and observed families.
- `fixtures.json`: exact reset/action prefixes, every prefix hash, physical and
  plan snapshots, bounded history, activation observation and evidence digests.
- `requests.json`: old/enriched input crossed with three model/effort choices.
  These are frozen request bodies, **not provider responses**.
- `continuation-*.json`: complete actions, hashes, stage timeline and outcomes.
- `report.json`: missing families, control results and preflight limitations.
- `manifest.json`: immutable file inventory with SHA-256 digests. Verification
  detects changes; it does not authenticate a maliciously regenerated manifest.

The controls are `keep` (no activation, preserving all anchors) and
`reactivate_current` (same command symbols, grounded at the delayed activation
state). Their distinction matters for persistent regions and target replacement.
Candidate runs can use `continueRecoveryPlan`; rejected candidates use the host
fallback. Terminal episodes skip pending activation. Source age is simulated
time, independent of provider wall-clock latency.

Stage observations are 0: no measured contact; 1: at least one living fighter in
executor throw range; 2: damage observed after the request; 3: damage on at least
two decisions within a rolling 20-decision window; 4: an original frozen target
cluster eliminated. These stages can skip or regress. The trace records the
current stage and the report records its maximum and first occurrence. Range
stage ignores firing occlusion. Existing target elimination is explicitly
flagged so it cannot be credited to a newly generated plan. Stage 4 takes
precedence, so inspect damage metrics separately for target-replacement cases.

## First preflight and next gate

The initial four-world scan found recent-casualty opportunities on terrain 5v5,
10v10 and 6v10. It did not find blocked advance, partial target-cluster
elimination, or throws-without-damage opportunities. This is a coverage failure
of this small scenario set, not evidence about any LLM's recovery capability.

The [sealed preflight report](examples/preflight-20260905-v0/report.json) contains
30 continuations, each rerun exactly. Across the 30 recorded continuations there
were 10,024 action results and zero rejected actions. Four delayed activations
were skipped because the episode ended first. The 18 frozen provider request
bodies were not sent.

The zero-delay controls reveal why grounding must be controlled in the later
LLM experiment:

| Terrain case | Keep current activation | Reactivate identical symbols |
| --- | --- | --- |
| 5v5 | Red wins; blue 0, red 5 | Blue wins; blue 2, red 0 |
| 10v10 | Blue wins; blue 6, red 0 | Censored after 300 decisions; blue 1, red 3 |
| 6v10 | Red wins; blue 0, red 4 | Red wins; blue 0, red 4 |

These are three correlated-control examples with a scripted fighter, not model
results. Reactivation invokes the grounder on the casualty-altered state; even
unchanged symbols can select different targets. Future model comparisons must
include this reactivation control alongside keeping the existing plan.

Verification: 355 TypeScript tests, 51 Python client tests, 257 Python training
tests and build passed. All archived pilot request bodies remain unchanged.

Next: declare purpose-built scenarios with separated enemy clusters, measured
movement obstruction and firing failures; retain this negative scan unchanged.
Verify each predicate and a useful physical recovery before freezing the paired
provider protocol. Compare old/enriched inputs with identical instructions,
schema, executor, request opportunities and effort-specific budgets. Report
independent environment seeds separately from repeated provider samples.

Multi-request closed-loop recovery, recovery-time estimators, tactical lane
tables, calibrated mission feasibility and event-triggered scheduling remain
later work. Existing M7b/M7c gates remain unchanged.

## Benchmark motivation

[Robocurve's Astra report](https://openai.robocurve.org/gpt-6-astra/) motivates
stage-level scoring and separating easy manipulation from precision failures.
It used medium effort, unblinded grading and non-interleaved trials; the bowl
comparison used different rigs. Its results do not establish Astra-low versus
Luna performance.
[RoboCerebra](https://robocerebra.github.io/) motivates aligned subgoal/request
opportunities and disturbance recovery.
[SayCan](https://say-can.github.io/) motivates grounding commands in executor
capability; our initial geometric/readiness proxies are not its learned value
functions.
