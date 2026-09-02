# SnowGym implementation plan

## Validated repository state

This plan was reconciled against `refs/snowgym_implementation_note.md`, the
supplied next-step RL design, current systems, browser wiring, tests, and build
configuration on 2026-09-01.

| Capability             | Current engine state                                                                                                     | SnowGym decision                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Multiple blue units    | `Game.spawnSquads` already honors `maxPlayers`; bundled maps have three blue spawns, while normal `main.ts` requests one | Reuse it; do not patch spawning                                                                             |
| Projectile attribution | `Snowball` and `SnowballThrown` already carry owner and team; collision rejects same-team hits                           | Reuse it; friendly fire stays disabled                                                                      |
| Blue action submission | Throwing has generic `tryThrow`; movement only accepted selected-unit UI commands                                        | Add one generic per-unit `tryMove` seam                                                                     |
| Red control            | The classic `AISystem` behavior is available through `ScriptedAiAgent`; seeded random is a second opponent               | Select both through the common `TeamController` boundary                                                    |
| Round termination      | Team counts are generic, but blue loss waits for single-hero lives                                                       | Configure the no-respawn scenario with zero reserve lives; redesign only if a later environment requires it |
| Determinism            | `World` owns a seeded RNG, physics uses fixed 60 Hz steps, and status exposes versioned public-state hashes              | Record provenance and exact actions; keep cross-language golden and replay assertions before benchmarking   |
| Headless use           | Most systems are DOM-free, but `Game` constructs renderer/input and owns the private step loop                           | Compose the systems directly in a DOM-free `SnowEnvironment`                                                |
| RL contract            | Canonical reset/step, masked fixed-shape Gym spaces, configurable rosters, and terrain observations are implemented      | Keep HTTP as the reference transport; add a direct batched transport before high-throughput training        |

## Milestones

### M0 — autonomous blue control and server status (current)

- [x] Canonical `UnitAction` / `TeamAction` types, free of UI inputs
- [x] Explicit `hold` action that cancels stale movement without changing `noop`
- [x] Detached entity observation with deterministic ID ordering
- [x] `TeamController` policy contract
- [x] Simple blue dodge / approach / throw policy
- [x] Validating SnowCraft action adapter
- [x] DOM-free `reset`, `observe`, and `step` lifecycle
- [x] 10 Hz decisions over the existing 60 Hz simulation
- [x] JSON status, reset, step, and autoplay endpoints
- [x] Verified deterministic autonomous completion in the Node integration test
- [x] Verified live HTTP reset/status/step/autoplay flow

Exit criterion: blue units independently move and throw without selection or
human input, a 3v3 match reaches a team-elimination result, and a client can
retrieve the result from the server without a renderer.

### M1 — reproducible environment contract

- [x] Extract a DOM-free simulation composition root with explicit system order.
- [x] Implement `reset(seed)`, `observe(team)`, and blue-team `step(action)`.
- [x] Advance a configurable integer number of physics ticks per policy decision.
- [x] Return terminal-only reward (`+1`, `-1`, `0`), `terminated`, `truncated`, and
      structured `info`; keep diagnostic event rewards separate.
- [x] Migrate the existing red behavior behind the same `TeamController` boundary.
      The scripted red squad now runs through `ScriptedAiAgent`, which re-runs
      the classic `AISystem` per-tick logic and reports its orders as semantic
      actions; full-episode traces are bit-identical to direct AI registration.
- [x] Record scenario, seed, action trace, simulation version, upstream base
      commit, and one public-state hash per replay frame.
- [x] Add exact same-seed/action-sequence state-hash tests and max-tick
      truncation tests.

Exit criterion: a Node test can run and exactly replay 3v3 without DOM, Canvas,
WebGL, Three.js rendering, browser timing, or input state.

### M2 — Gymnasium bridge

- [x] Publish the initial versioned `snowgym.v0` server schema.
- [x] Add fixed-shape numeric action/observation spaces and masks.
- [x] Register `gym.make("SnowGym/Squad-v0")`.
- [x] Add a project-local Python environment and locked dependencies.
- [x] Pass Gymnasium's environment checker against the live server.
- [x] Add a terminal-only scripted-blue demo command.
- [x] Strictly validate mutating request fields and isolate scripted stepping
      from explicit external actions.
- [x] Add optimistic state-hash guards, step idempotency, and machine-readable
      server capability discovery.
- [x] Check all three registered Gym environments, including map-backed v2,
      and support JSON CLI summaries.
- [x] Record versioned visual-replay JSON from detached server state.
- [x] Add a renderer-free CLI/function that builds reproducible M-vs-N examples
      for open arenas or bounded native map spawn pools.
- [x] Replay recordings through the existing Three.js rendering engine without
      coupling the Gym environment to rendering.
- Let the initial Python adapter consume the JSON server for correctness, then
  add a long-lived batch host/direct transport for training throughput.
- Add vectorized environment support over the future batch transport.
- [x] Add a shared TypeScript/Python golden fixture for the versioned public-state
      canonicalization and hash contract.
- Benchmark throughput at 1, 2, 5, 10, 20, and 60 Hz decision rates.

Exit criterion: `gym.make("SnowGym/Squad-v0")` passes Gymnasium's environment
checker and deterministic cross-language fixtures.

### M3 — configurable N-blue versus M-red scenarios (core complete)

- [x] Replace the fixed `THREE_VS_THREE_OPEN` assumption with validated scenario
      configuration: `blueUnits`, `redUnits`, spawn layout, arena dimensions, red
      difficulty, decision rate, seed, and max ticks.
- [x] Generate deterministic non-overlapping spawn layouts when explicit spawns are
      omitted; reject counts that cannot fit the arena.
- [x] Publish fixed roster maxima for `SnowGym/Squad-v1` (eight slots) and
      `SnowGym/Squad-v2` (ten slots); represent smaller N/M configurations with
      unit-presence and legal-action masks so a registered Gym environment's
      spaces never change after construction.
- [x] Extend reset/server configuration, replay metadata, reward/termination logic,
      and observations without breaking `SnowGym/Squad-v0` 3v3 recordings.
- [x] Load the bundled SnowCraft maps as scenario terrain: obstacles affect
      line-of-sight, cover, and collision, spawn points come from the map, and
      obstacles are exposed to policies as a fixed-capacity masked tensor.
- [x] Add a native 10v10 map whose browser JSON and headless registry definitions
      are contract-tested for exact parity.
- [x] Add a deterministic matrix covering 1v1, 1v3, 3v1, 3v3, and maximum-size
      fights, plus invalid counts/spawns and same-seed replay checks.
- [x] Migrate red behavior to the common `TeamController` boundary, then add
      independently selectable scripted, random, learned, or external opponents
      (scripted and random shipped; learned/external remain future).
- Benchmark episode throughput and balance by N/M configuration before training.

Exit criterion: one versioned environment can reset into multiple validated
N-vs-M configurations while retaining fixed Gym spaces, deterministic replay,
team elimination, and renderer-free server status.

### M4 — hierarchical commander (C4 complete)

- [x] Define the bounded `snowgym.command-plan.v0` group action space and strict
      JSON schema without unit IDs, enemy IDs, coordinates, or physical actions.
- [x] Add strict runtime validation for mission/objective compatibility, unique
      fixed roles, allocation bounds, and acyclic support relationships.
- [x] Add deterministic weighted group allocation for 3v3 through 10v10.
- [x] Add team-relative region and symbolic enemy-cluster target resolution.
- [x] Add a trusted host envelope and immutable atomic `PlanStore`.
- [x] Add a synchronous plan-aware controller and reactive per-unit executor.
- [x] Add a deterministic headless 10v10 split-force demonstration whose
      per-role action traces prove distinct group execution and exact replay.
- [x] Add synchronous plan lifecycle triggers for plan expiry, major own-force
      loss, assigned-group elimination, and objective completion.
- [x] Reconcile delayed candidates against the current living roster, repair
      bounded support/target drift, reject invalid candidates without replacing
      the active plan, and atomically activate accepted plans.
- [x] Keep a deterministic one-group fallback and detached lifecycle trace for
      every accepted, repaired, rejected, and fallback activation.
- [x] Add an ID-free, versioned strategic summary and a provider-neutral async
      `CommanderClient` boundary.
- [x] Prove non-blocking operation with a delayed mock commander: one request in
      flight, cooldown-governed trigger coalescing, simulation-tick timeout,
      stale-response reconciliation, and ignored late responses.
- [x] Add deterministic simulated-latency scheduling and a headless 10v10 C3
      demonstration with exact action/state/trace replay coverage.
- [x] Add a server-only `gpt-5.6-luna` Responses API adapter using strict
      structured output, reasoning, `store: false`, and environment-only
      `OPENAI_API_KEY`.
- [x] Gate the provider adapter on mocked error/refusal/timeout tests plus one
      explicitly authorized live headless schema smoke; never include it in the
      deterministic default test suite.
- [x] Run a separately authorized, wall-clock-paced live battle with automatic
      lifecycle requests disabled and a code-enforced limit of exactly one
      external request. The battle continued at 10 Hz, activated Luna's stale
      symbolic plan mid-battle, and completed without rejected physical actions.

Exit criterion: a slow commander can replace validated symbolic group plans
without blocking the 10 Hz physical controller or exposing transient unit
control to the model.

#### Agent monitoring and correction contract

Monitoring is host-owned and hierarchical. The LLM never polls individual
units, waits inside `step`, or corrects physical actions directly.

| Layer                         | Cadence                                          | Status consumed                                                                     | Corrections it may make                                                                                        | Escalation                                                                                    |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Simulation and action adapter | 60 Hz                                            | collision, legality, cooldown, alive state                                          | reject illegal physical actions and advance authoritative state                                                | report action results to the controller trace                                                 |
| Reactive unit policy          | each team decision, normally 10 Hz               | assigned group, live position/health/state, projectiles, current symbolic objective | dodge, choose a legal throw, reacquire targets, restore range/cohesion, or continue mission movement           | never waits for the LLM; future stuck/repeated-rejection summaries go to lifecycle monitoring |
| Plan-aware team controller    | each team decision                               | immutable active-plan snapshot plus current detached observation                    | late-bind enemy clusters, ally-group positions, and map-region objectives while keeping plan membership stable | continue the last valid snapshot during any commander request                                 |
| Plan lifecycle monitor        | each team decision                               | plan age, living assigned units, whole-team loss fraction, and objective progress   | activate deterministic fallback on expiry, major loss, group elimination, or completion                        | schedule a commander request; coalesce duplicate triggers while one is in flight              |
| Plan reconciler               | only when a candidate arrives                    | candidate provenance plus the newest detached observation                           | validate, shrink infeasible groups, repair missing support/enemy objectives, ground, then atomically activate  | reject unsafe drift and retain the old plan or fallback                                       |
| LLM commander                 | event-driven and rate-limited, target 0.1–0.5 Hz | a compact strategic summary captured with source tick/hash, never engine entities   | propose only a strict symbolic `CommandPlan`                                                                   | timeout/error cannot block control; host fallback and later retry policy remain authoritative |

The C3 delayed-commander test must prove this sequence:

```text
observe -> execute current plan -> detect/coalesce trigger -> request asynchronously
        -> keep executing current plan -> receive stale candidate
        -> reconcile against newest observation -> atomic activate or reject
```

C3 tests cover request timeout, duplicate-trigger coalescing, late response
after fallback, invalid response, provider failure, uninterrupted synchronous
control, and deterministic replay of the same mock-latency schedule. Stuck
detection and repeated action-rejection thresholds remain planned explicit,
debounced lifecycle signals; they must be host-computed rather than inferred by
the model.

### M4.1 — trajectory-aware closed-loop commander (C5)

- [x] Add a passive, bounded trajectory monitor over pre-step observation,
      active-plan snapshot, physical action results, and post-step observation.
- [x] Publish an ID-free, versioned group trajectory digest with mission-aware
      progress, health/cohesion trends, action counts, rejection counts, and a
      host-computed stuck fraction.
- [x] Prove that enabling telemetry cannot change physical actions, public-state
      hashes, plan activation, or replay results.
- [x] Add debounced `plan_stalled` and `action_rejection_repeated` signals with
      activation grace periods and recovery hysteresis.
- [x] Separate soft signals, which retain the current plan while replanning,
      from hard lifecycle failures, which activate deterministic fallback.
- [x] Pass the bounded trajectory digest and preceding plan outcome to each
      stateless commander request without exposing unit IDs or raw trajectories.
- [x] Run deterministic multi-request mock battles with exact state, plan, signal,
      latency, and scheduler-trace replay coverage.
- [x] Add an opt-in, wall-clock-paced Luna battle with one in-flight request, a
      code-enforced per-episode call limit, explicit token/latency accounting,
      and uninterrupted fallback on provider failure.
- [x] Pass an explicitly authorized live C5 acceptance battle: host-computed
      trajectory stalls triggered bounded Luna replanning while the executor
      continued at 10 Hz, used two of three permitted requests, activated a
      valid plan, and finished with zero rejected physical actions.
- [x] Record a versioned, ID-free commander trace sidecar bound to the replay's
      final public-state hash; add an optional scrubber-aware plan, aggregate
      trajectory, and lifecycle overlay to the existing replay UI.
- [x] Generalize the deterministic and Luna trajectory runners across validated
      bundled-map blue/red rosters and red difficulties while preserving the
      original 10v10 defaults. Reject invalid map capacities before entering
      the commander loop.
- [x] Add a host-owned economy-of-force opening for outnumbered blue rosters so
      the synchronous executor does not blindly advance into the enemy
      backfield while commander advice is pending; retain the direct opening at
      parity.

Exit criterion: host-computed trajectory evidence can trigger bounded Luna
replanning during an episode while the 10 Hz executor continues synchronously,
and identical mock latency schedules reproduce identical actions, state hashes,
trajectory digests, plans, and lifecycle traces.

Configurable-runner acceptance (2026-09-02): targeted tests cover an
understrength 6v10 trace on `arena6`, a 3v3 provider-neutral run on `arena4`,
default 10v10 compatibility, deterministic replay, and pre-run rejection of a
10-unit roster on a three-spawn map. No provider request is part of this gate.

Understrength-opening acceptance (2026-09-02): deterministic seed 14 on
`arena6` with 6 blue versus 10 easy scripted red completes with a 1–0 blue win,
two simulated-latency mock responses, zero rejected physical actions, and a
replay-bound commander trace. The example demonstrates possibility, not a
held-out win-rate or online-LLM result.

### M5 — multi-agent and research adapters

- [x] Add a two-team PettingZoo ParallelEnv over the same server and simulator,
      with mirrored fixed-capacity observations, simultaneous guarded joint
      actions, zero-sum terminal rewards, and the official Parallel API gate.
- [x] Add deterministic local-visibility, action-delay, and observation-delay
      research profiles around the parallel environment, with authoritative and
      observation-source ticks reported separately.
- [x] Add an optional, fixed-shape, renderer-free semantic raster that respects
      local visibility and retains the entity-tensor observation contract.
- [x] Add a versioned baseline evaluation suite and sequential multi-episode
      benchmark over the reference HTTP transport, separating deterministic
      results from wall-clock performance.
- [x] Add a Gymnasium single-team wrapper with deterministic random/no-op,
      detached learned-callable, and versioned remote-client opponent adapters;
      retain the existing native scripted/random server-controller route.
- [ ] Deferred: add optional rendered pixels only after learned entity-policy
      baselines exist; pixels are never a correctness or default training input.
- [ ] Carried into M6.1: add a long-lived direct/vectorized batch host and
      benchmark it independently from the correctness/reference HTTP transport.

### Reconciled RL phase boundary

The environment and hierarchical commander are now sufficiently complete to
test the central open question: can a neural policy learn SnowGym's fast hybrid
physical control and later execute its slow symbolic plans? Until that is
answered, pixels, new commander verbs, prompt tuning, map expansion, UI polish,
large policies, and self-play leagues are explicitly deferred.

The next phase reuses rather than replaces the current contracts:

- `/step-scripted` already returns the exact teacher `TeamAction` under
  `info.action`; the exporter must convert it back to the fixed Gym tensor
  action with exact round-trip tests.
- `snowgym.replay.v0` already stores observations, semantic actions, state
  hashes, and outcomes; learned rollouts should continue to use it.
- `snowgym.evaluation-suite.v0` already fixes scenarios, seeds, policies, and
  research profiles; checkpoint evaluation should extend this runner rather
  than introduce another scenario format.
- `SnowGymEnv` and `SnowGymSingleTeamEnv` are the Python closed-loop seams;
  `LearnedOpponent` supports learned opposing teams. The simulator must remain
  unaware of Torch.
- HTTP remains the correctness transport. A batch host must compose the same
  `SnowEnvironment`, never reimplement physics or reward logic.

Scaled data collection deliberately waits for the batch host. A small HTTP
corpus is sufficient to validate the trajectory and model contracts first;
100k–1M transition runs are not an M6.0 prerequisite.

### M6.0 — teacher data and behavior-cloning proof

Goal: prove the observation/action/checkpoint loop can learn and execute the
existing scripted blue teacher before adding policy-gradient failure modes.

#### M6.0a — training and trajectory contracts

- [x] Add a separate `snowgym/training/` Python package with its own lockfile,
      tests, and `torch` dependency; keep `snowgym-client` lightweight.
- [x] Define `snowgym.trajectory.v0` as compressed, non-pickle NumPy shards
      plus a JSON manifest. Record observation/action tensors, masks, reward,
      termination/truncation, scenario, seed, tick, teacher name/version,
      simulation/API/hash versions, and pre/post public-state hashes.
      Compute content digests over canonical array names, dtypes, shapes, and
      raw bytes so archive timestamp/compression metadata cannot affect them.
- [x] Add semantic-`TeamAction` to Gym-action inversion. Prove
      semantic -> tensor -> semantic equality within explicit float tolerance,
      including noop/hold, target normalization, power, dead/absent slots, and
      arbitrary roster sizes.
- [x] Export a small renderer-free scripted corpus through the guarded HTTP
      client. Capture the pre-step observation and the exact action returned by
      `/step-scripted`; never infer labels from post-step motion. Persist action
      results and reject a qualifying corpus containing illegal teacher labels.
- [x] Make shard ordering, manifests, split assignment, and content hashes
      reproducible. Training, validation, and held-out evaluation seed sets must
      be disjoint and committed as versioned manifests.
- [x] Measure the scripted teacher and masked-random baseline on the first held-
      out 1v1 suite before using teacher win rate as a learned-policy ceiling.

M6.0a gate: two exports with the same specification have identical manifests,
canonical tensor digests, tensors, and state-hash trajectories; a dataset audit
rejects a changed schema, corrupt shard, illegal label, or seed overlap.

M6.0a acceptance (2026-09-01): two live 20-transition exports produced the
same manifest and dataset digest; a complete held-out episode produced 53
audited transitions. On evaluation seeds 201/202, scripted blue beat native
random red 2/2 with zero rejected actions, while masked-random blue won 0/2
and both episodes truncated as draws. The committed result is
`training/baselines/teacher_1v1_v0.json`.

#### M6.0b — first neural executor

- [x] Implement small shared MLP entity encoders, masked mean/max aggregation,
      global context, and a shared per-ally actor. Do not begin with a
      transformer.
- [x] Emit a masked categorical action type, bounded normalized target, and
      bounded throw power per present ally slot. Invalid action types receive no
      probability mass.
- [x] Train with masked categorical cross-entropy plus target MSE for move/throw
      actions and power MSE for throw actions only. Keep all loss weights in a
      versioned configuration.
- [x] Add deterministic CPU training acceptance, fixed data-loader ordering,
      a one-batch overfit test, finite-gradient checks, and exact checkpoint
      resume. Accelerators may be optional but are not the reproducibility gate.
- [x] Define `snowgym.checkpoint.v0` metadata: git commit, dataset manifest hash,
      SnowGym versions, architecture, optimizer/loss configuration, training
      seed, step, and evaluation suite.
- [x] Add a `TorchPolicy`/`LearnedOpponent` bridge with detached NumPy tensors,
      `eval()`/no-grad inference, action-space validation, and no Torch import in
      simulator code.
- [x] Extend the evaluation runner with checkpoint policies and deterministic
      metrics: win/draw rate, episode length, survivors, and health lost/dealt
      derived from observations. Preserve terminal-only benchmark reward.
- [x] Record learned closed-loop episodes as normal visual replays; browser
      viewing remains optional validation and never model input.

M6.0 exit: on an explicit held-out 1v1 seed suite, the behavior-cloned policy
is reproducible, respects every mask, executes closed-loop, and improves over
the masked-random baseline. Report teacher, random, and learned results together
rather than claiming teacher parity from training loss alone.

M6.0b acceptance (2026-09-01): a 212-transition scripted corpus from train
seeds 11–14 trained the versioned entity policy for 5,000 deterministic CPU
steps. Independent runs produced the same model/optimizer state and checkpoint
digests. On held-out seeds 201/202, learned blue won 2/2 in 54 decisions with
zero rejected actions and no health lost; scripted blue won 2/2 in 53, while
masked-random blue won 0/2 and timed out at 200. The committed checkpoint is
`training/checkpoints/bc_1v1_v0`, the joined report is
`training/evaluations/bc_1v1_v0.json`, and both learned episodes are normal
`snowgym.replay.v0` files under `public/replays/bc_1v1_v0/`. This is a narrow
1v1 behavior-cloning proof, not evidence of broader scenario generalization.

### M6.1 — persistent batch simulation

Goal: remove one-request-per-decision HTTP overhead without creating a second
simulator.

- [x] Add `snowgym/batch/` with a persistent subprocess host that owns multiple
      independent `SnowEnvironment` instances and exposes a versioned handshake,
      batched reset, batched joint/single-team step, close, and error messages.
- [x] Begin with compact framed or newline-delimited messages over stdin/stdout.
      Keep protocol logging on stderr so stdout remains machine-readable; do not
      require native bindings for the first implementation.
- [x] Add a Python `SnowGymBatchEnv`/client with fixed leading batch dimension,
      per-slot seeds/scenarios, independent terminal state, selective reset, and
      explicit failure semantics. Start with 8, then 32, then 64 worlds.
- [x] Preserve per-world deterministic RNG, ID allocation, controller state,
      masks, rewards, termination, and public-state hashes. One failed world
      must not silently advance any other world.
- [x] Add golden HTTP/batch parity over the same version, scenario, seed, and
      exact semantic action sequence. Compare every state hash, reward,
      termination flag, truncation flag, and action result.
- [x] Add `snowgym/training/benchmarks/throughput.py`. Report environment count,
      decision rate, ticks per decision, decisions/sec, simulation ticks/sec,
      wall-clock real-time factor, CPU utilization, payload bytes, and measured
      serialization share for 1, 8, 32, and 64 worlds.
- [x] Document the 64-world limitation if the target machine cannot sustain it;
      never weaken parity or silently omit failed slots to reach a throughput
      number.

M6.1 exit: the trainer directly consumes at least 32 persistent worlds; 8/32
parity is exact against HTTP, 64 works or has an evidence-backed limitation,
and benchmark results clearly separate simulation from serialization cost.

M6.1 acceptance (2026-09-01): the training benchmark consumed 1, 8, 32, and 64
persistent worlds through `SnowGymBatchEnv`; 8/32 direct-service golden tests
and an eight-world live HTTP check matched complete reset/step payloads exactly.
The 20-decision smoke measured approximately 218, 1,544, 4,707, and 6,964
decisions/s respectively. At 64 worlds it measured 41,783 simulation ticks/s,
696x aggregate real time, 1.34 CPU cores, 1.89 MB of protocol payload, and 4.3%
Python JSON serialization share. These are local short-run measurements, not a
cross-machine performance guarantee. The committed machine-readable report is
`training/benchmarks/batch_throughput_v0.json`.

After this gate, scale the versioned teacher corpus toward 100k–1M transitions
only if learning curves or scenario coverage require it.

### M6.2 — centralized hybrid-action PPO

Goal: demonstrate reward-driven improvement without commander, unit-level MARL,
partial observation, or self-play confounds.

- [x] Add a custom Torch PPO implementation for the squad-level actor. The
      joint log probability includes action type for present units, target only
      for move/throw, and power only for throw; masks also apply to entropy.
- [x] Add detached fixed-horizon rollout/GAE buffers with terminal versus
      time-limit truncation handled separately, strict fixed-world tensor
      validation, immutable snapshots, and deterministic time/world flattening.
- [x] Add the deterministic minibatch optimizer loop with global advantage
      normalization, clipped policy/value losses, finite-gradient enforcement,
      gradient clipping, and aggregated KL/entropy/clip diagnostics.
- [x] Add exact rollout-boundary PPO checkpoint/resume metadata, including
      model, optimizer, Torch random state, update/environment-step counters,
      architecture/configuration, training seed, curriculum provenance, and
      semantic state/metadata digests.
- [x] Collect directly from persistent `SnowGymBatchEnv` worlds with a bounded,
      non-reusing episode-seed schedule, selective terminal resets, explicit
      horizon truncation/value bootstrap, and restartable rollout boundaries.
- [x] Add an atomic headless PPO smoke runner that binds a frozen gate, rollout
      geometry, seed cursor, update metrics, and final checkpoint in a hashed
      machine-readable run manifest; prove resumed and uninterrupted updates
      reach the same semantic state.
- [x] Keep canonical evaluation reward at win `+1`, loss `-1`, draw `0`. If
      sparse learning blocks the smoke test, add an opt-in training wrapper with
      potential-based own-minus-enemy health shaping and test that it leaves
      terminal benchmark results unchanged.
- [ ] Use a gated curriculum: 1v1 random, 1v1 easy scripted, 3v3 random, 3v3
      scripted, 3v3 terrain, then 5v5 and 10v10. Do not advance a stage without
      fixed held-out evaluation evidence.
- [ ] Keep training and evaluation seeds disjoint. Evaluate checkpoint series,
      not only the best checkpoint, against masked-random, native random, and
      scripted baselines using the versioned suite.
- [x] Freeze a headless PPO checkpoint evaluator over each gate's disjoint
      held-out seeds, with deterministic PPO inference, masked-random and native
      scripted-blue comparisons, canonical returns, rejected-action accounting,
      threshold reporting, and a hashed result contract.
- [x] Add an atomic checkpoint-series runner that retains and evaluates every
      predeclared update, records its checkpoint/run/evaluation digests and the
      complete update curve, distinguishes development from qualifying runs,
      and never selects only the best checkpoint.
- [ ] Record learning curves, policy/value losses, entropy/KL, rejected actions,
      throughput, checkpoint provenance, and replay links in a machine-readable
      run manifest.

M6.2 exit: PPO reproducibly solves the defined 1v1-random gate, shows meaningful
improvement against easy scripted 1v1, and exceeds the random-policy baseline
in 3v3-random evaluation. Exact numerical thresholds and seed counts must be
frozen in the evaluation manifest before the qualifying training run.

M6.2 foundation (2026-09-01): the versioned curriculum freezes disjoint
training ranges and eight evaluation seeds for each 1v1-random,
1v1-easy-scripted, and 3v3-random gate before training. The actor-critic now
implements mask-aware categorical actions, tanh/sigmoid continuous heads,
conditional joint log probability and entropy, a centralized value head,
terminal-aware GAE, clipped PPO loss, KL/clip diagnostics, and opt-in health
potential shaping. Rollout storage, optimizer/resume, and qualifying runs remain
open; the M6.2 exit gate is not yet claimed. The fixed-horizon rollout buffer
now snapshots vector-world tensors, rejects inconsistent transitions, computes
terminal-aware GAE only when complete, and flattens time/world axes for the
optimizer without collapsing entity features. PPO updates use a reproducible
seed/update-index permutation and report sample-weighted diagnostics across
every epoch and minibatch. A restricted-load checkpoint test proves that an
update-save-restore-update sequence exactly matches uninterrupted optimizer
state and weights. The live collector now drives persistent batch worlds within
each rollout, selectively resets completed slots, and checkpoints its monotonic
episode-seed cursor. Collection boundaries deliberately truncate and bootstrap
unfinished worlds so exact resume does not depend on unserialized simulator
state. The headless smoke runner now exercises collection, optimization,
manifest writing, and exact resume end to end, while labeling its output as
infrastructure-only evidence. Qualifying training and held-out evaluation runs
remain open. The batch host now also exposes exact-parity `stepScripted` so the
held-out evaluator compares PPO, deterministic masked-random, and the native
scripted blue controller without an HTTP server or renderer. The evaluator
reports the frozen thresholds but does not turn a threshold miss into hidden
checkpoint selection. Training can now explicitly opt into potential-based
health shaping; checkpoints bind the reward mode, manifests retain shaped and
canonical sums separately, and held-out evaluation remains canonical. A
qualifying checkpoint series remains open. The curriculum now freezes all seven
planned gates through map-backed 10v10, with distinct 10,000-seed training
ranges and eight held-out seeds per gate. A live batch test proves every frozen
scenario and roster loads on the authoritative server; no gate is considered
advanced merely because it is defined. PPO now also supports an explicit,
digest-bound behavior-cloning initialization: the BC policy is loaded only at
the first update and its checkpoint/state/dataset provenance survives every
resume. A development series showed why retaining all checkpoints matters: its
BC-initialized update 1 passed 1v1-random at 8/8, while updates 5 and 10 regressed
to draws. A subsequently frozen conservative candidate (`ppo_1v1_bc_v0`) lowers
the learning rate to `3e-5`, uses one PPO epoch and full 200-decision rollouts,
and passed 8/8 at retained updates 1, 5, 10, and 25 in a reproduced development
run. This is tuning evidence, not yet the post-commit qualifying artifact.

M6.2 gate-1 acceptance (2026-09-02): the committed, BC-initialized
`ppo_1v1_bc_v0` qualifying series retains updates 1, 5, 10, and 25. Every
checkpoint won all eight held-out `1v1-random` episodes, while deterministic
masked-random won none; the final checkpoint averaged 60 decisions and all
policies recorded zero rejected actions. The series is bound to source commit
`60459b5`, the frozen series-config digest, curriculum digest, BC checkpoint and
dataset digests, and every child checkpoint/evaluation digest. The semantic
auditor restricted-loads every checkpoint and rejects modified child artifacts.
Only the first curriculum gate is advanced; 1v1 easy scripted and all later
gates remain open. A deterministic headless recorder produced the final
checkpoint's seed-3101 blue win in 60 decisions, and the existing browser/WebGL
replay smoke reached the terminal frame and restarted without UI errors. The
replay is companion evidence; the accepted v0 series manifest remains immutable.
The trainer now distinguishes exact same-gate resume from cross-gate PPO
transfer. Transfer imports model/value weights but resets the optimizer, update
counter, and episode-seed range while binding source checkpoint, state,
curriculum, gate, and update provenance. A gate-2 development transfer from the
accepted gate-1 final checkpoint remained at 0/8 versus easy scripted red
through update 25, while its teacher baseline won 8/8; gate 2 therefore remains
closed and requires a gate-specific training configuration.
The gate-specific `bc_1v1_easy_v0` initializer was regenerated after its source
specification commit and binds 184 audited teacher transitions. It won both
disjoint BC evaluation episodes against easy scripted red, versus 0/2
masked-random, with zero rejected actions. Conservative PPO development from
this initializer passed the frozen eight-seed gate at updates 1, 5, and 10 but
regressed by update 25, so the gate-2 candidate must stop at a predeclared
update 10. `ppo_1v1_easy_bc_v0` now freezes that checkpoint schedule and all
hyperparameters before qualification; no qualifying gate-2 result is claimed
yet.

M6.2 gate-2 acceptance (2026-09-02): the committed
`ppo_1v1_easy_bc_v0` qualifying series retains updates 1, 5, and 10; all three
pass the frozen held-out threshold. The final checkpoint won 5/8 against easy
scripted red versus 0/8 masked-random and 8/8 teacher, averaged 66.75 decisions,
and recorded zero rejected actions. The semantic auditor verifies all nested
checkpoint and evaluation digests. Gates 1 and 2 are advanced; 3v3-random and
all later gates remain closed.

### M7 — plan-conditioned learned executor

Goal: train the fast learned controller to follow the existing slow
`CommandPlan` language without online LLM calls.

- [ ] Add a deterministic synthetic plan curriculum that emits only schema-
      valid plans and records grounded assignments and source seeds.
- [ ] Add a fixed-size `PlanTensorEncoder` for mission, approach, posture, fire
      policy, preferred range, cohesion, relative objective/group geometry,
      group fractions, support relation, and plan age. JSON remains canonical;
      the tensor is only an RL adapter.
- [ ] Train identical executor architectures with and without plan input under
      the same data and optimization budget.
- [ ] Evaluate direct versus flank trajectories, focus versus distributed fire,
      hold/support/withdraw behavior, unseen valid directive combinations, and
      3v3/5v5 training to 10v10 transfer.
- [ ] Keep reflexes, late binding, action validation, lifecycle fallback, and
      target replacement host-owned; a plan never supplies physical actions.

M7 exit: plan-conditioned policies produce reproducibly distinct, intended
behavior under counterfactual plans for the same initial state and outperform
the no-plan ablation on frozen objective-completion metrics.

### M8 — unit-level CTDE / MAPPO

Goal: add decentralized execution only after centralized plan-conditioned PPO
works.

- [ ] Add `SnowGymUnitParallelEnv` as a new PettingZoo environment; retain the
      existing team-level environment and version.
- [ ] Begin with global actor observations, then local observations, then local
      observations plus latency. Change only one observability condition per
      experiment.
- [ ] Implement parameter-shared unit actors and a centralized critic over
      global state, assignments, and active plan. Execution must use actor-local
      inputs only.
- [ ] Gate 3v3 before 5v5, and fixed rosters before variable roster transfer.

M8 exit: MAPPO beats its unit-random baseline in frozen 3v3 and 5v5 suites and
the command-conditioned shared actor remains valid under local observations.

### M9 — slow commander over a learned team

Goal: connect the already-tested asynchronous commander only after freezing a
competent learned executor.

- [ ] Hold the learned executor checkpoint fixed while comparing no commander,
      random valid plan, rule commander, high-level RL, online LLM, static
      LLM-generated doctrine, and distilled commander baselines through the
      same `CommandPlan` IR.
- [ ] Reuse scheduler coalescing, timeout, fallback, stale-plan reconciliation,
      trajectory triggers, token/latency accounting, and replay trace alignment;
      do not rebuild these inside the trainer.
- [ ] Prove commander failure or latency cannot block the learned 10 Hz executor
      and cannot bypass action validation.

M9 exit: real commander plans drive the fixed learned executor with aligned
replay/trace artifacts, while cheaper baselines and provider cost/latency are
reported on the same held-out scenarios.

### M10 — latency and generalization benchmark

- [ ] Sweep simulated commander latency at 0, 100, 250, 500 ms and 1, 2, 4,
      and 8 seconds; run separately authorized real-latency checks only after
      deterministic sweeps pass.
- [ ] Compare reject, activate-unchanged, and reconcile/late-bind handling for
      plans produced from stale state.
- [ ] Compare exact IDs/coordinates against symbolic late-bound groups without
      exposing forbidden physical details to the production commander path.
- [ ] Evaluate 3v3/5v5-trained executors and proportional group plans on 10v10.
- [ ] Report win/draw rate, objective completion, rejection/repair rate,
      trajectory quality, plan validity, token count, and end-to-end latency.

M10 exit: the benchmark quantifies where the slow-command/fast-executor
hierarchy fails and whether late-bound group plans degrade more gracefully than
individual exact assignments.

### Immediate commit sequence and gates

1. Training scaffold, trajectory schema, semantic-action inversion, and small
   scripted exporter.
2. Entity model, hybrid losses, deterministic BC trainer, checkpoint schema,
   evaluator, and learned replay.
3. Persistent multi-environment batch host, Python client, exact HTTP parity,
   and throughput benchmark.
4. Centralized PPO and frozen 1v1/3v3 curriculum gates.
5. Synthetic plan generator, directive encoder, and plan-conditioned ablation.
6. Unit-level PettingZoo adapter and MAPPO.
7. Fixed learned executor under commander baselines and latency/generalization
   evaluation.

Each commit is accepted only after targeted tests plus the full milestone gate:

```bash
npm test
npm run build
cd snowgym/python && .venv/bin/python -m pytest -q
```

Training-package commits additionally run their own unit tests, deterministic
CPU smoke, dataset audit, and checkpoint/evaluation replay gate. Provider-backed
LLM calls, large dataset generation, and long training runs are opt-in and are
never part of the default deterministic suite.

## Guardrails

- No RL, Python, transport, or SnowGym imports from `src/`.
- Policies receive observations and return actions; they never hold engine
  entities or mutate world state.
- Canonical benchmark reward stays terminal-only until experiments explicitly
  choose a shaped reward.
- Every change outside `snowgym/` is recorded in `UPSTREAM_PATCHES.md`.
- Human browser behavior must continue to pass the existing test/build/smoke
  suite after each milestone.
- Repository agents should follow the root `AGENTS.md` and the repo-local
  `.agents/skills/snowgym/SKILL.md` guarded workflows.
