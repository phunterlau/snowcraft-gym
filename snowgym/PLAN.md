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
| RL contract            | Canonical reset/step, masked fixed-shape Gym spaces, configurable rosters, terrain observations, and persistent batch worlds are implemented | Keep HTTP as the reference transport and the batch subprocess as the authoritative high-throughput path |

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

M6.2 gate-3 preparation (2026-09-02): a provenance-bound PPO transfer from the
accepted gate-2 update-10 checkpoint stayed at 0/8 wins at all retained
development updates 1, 5, and 10, matching masked-random; the scripted teacher
won 8/8 in 53 decisions. The committed `teacher_3v3_random_v0` suite therefore
freezes a separate 3v3 corpus with train, validation, BC-evaluation, and PPO
evaluation seeds all disjoint. Its 424-transition audited training export has
dataset digest
`sha256:616494b021b437d6b8b641bae03255ffe525607985169b24e218407a4daf5dff`.
The deterministic `bc_3v3_random_v0` initializer won both BC evaluation
episodes in 105 decisions with all blue units alive, versus 0/2 masked-random
and 2/2 teacher, with zero rejected actions. This is initializer evidence;
gate-3 PPO configuration and qualification remain open.

Gate-3 development then tested exploration explicitly. BC-seeded PPO with the
historical log-std `-1` and learning rate `3e-5` collapsed to 0/8 after update
1; narrowing target/power log-std to `-3` did not by itself prevent the
categorical policy from crossing a fragile argmax boundary. The predeclared
`ppo_3v3_random_bc_v0` stability candidate combines log-std `-3` with learning
rate `1e-8` and passed 8/8 at every retained update 1/5/10 versus 0/8
masked-random. Its deterministic behavior remains 105 decisions, equal to the
BC initializer. This is an honest retention result through the PPO pipeline,
not material reward-driven improvement; qualification remains open.

M6.2 gate-3 acceptance (2026-09-02): the post-commit qualifying
`ppo_3v3_random_bc_v0` series reproduces the frozen development curve. Updates
1, 5, and 10 each won 8/8 held-out episodes in 105 decisions, versus 0/8
masked-random and 8/8 scripted teacher, with zero rejected actions. The semantic
auditor verifies the series and all nested checkpoint/evaluation digests; the
series is bound to source commit `1829e6a` and config digest
`sha256:f793476d5297c6ebc0c542dfd9a9bd662c81de43e05b92a1aac531ba9c57e341`.
The first three gates are advanced and the stated M6.2 held-out comparison is
met. Later 3v3 scripted/terrain and 5v5/10v10 curriculum gates remain open, and
the 3v3 result remains explicitly BC retention rather than improvement over BC.

Gate-4 preparation found that direct gate-3 PPO transfer loses all eight
`3v3-scripted` evaluation episodes. An eight-episode open-loop teacher corpus
audited successfully, but its first BC checkpoint lost both disjoint BC
evaluation episodes; target-precision weighting improved one development
variant to 1/2, while simply doubling the corpus did not close the gap. This is
treated as closed-loop covariate shift rather than grounds for further blind
hyperparameter sweeps. The server now exposes read-only `GET /teacher-action`,
and `snowgym-export-dagger` records those oracle labels on states visited by a
provenance-bound learned checkpoint while learner steps remain state-hash
guarded. Dataset aggregation and a new gate-4 initializer remain open.

The first recovery-only DAgger development checkpoint also lost both held-out
episodes, confirming that learner-state labels must augment rather than replace
expert trajectories. `snowgym-merge-trajectories` now provides deterministic
ordered aggregation with repeated-input integer weighting, compatibility checks,
source-digest provenance, episode remapping, and a full output re-audit. A
mixed expert/recovery gate-4 initializer remains open.

A 2:1 ordered expert/recovery aggregate then succeeded in development. It
contains 1,745 transitions and has portable digest
`sha256:2fc2770ae2385c16adb14cffde01104a5e4165a4f8154d92021d40ae1fa3e7e4`;
the 10,000-step precision-weighted checkpoint won both held-out BC episodes,
with one and two blue survivors respectively. `bc_3v3_scripted_v0` now freezes
that optimization contract before provenance-valid regeneration. Gate-4 PPO
qualification remains open.

The post-commit `bc_3v3_scripted_v0` regeneration binds source commit
`c17751d` and the portable aggregate digest, and reproduces 2/2 held-out BC
wins with zero rejected actions. The baseline, checkpoint, and evaluation are
now committed evidence; `3v3-scripted` still requires its eight-seed PPO series.

The frozen `ppo_3v3_scripted_bc_v0` development series retains 5/8 wins at
updates 1/5/10 versus 0/8 masked-random, with three red-win seeds and mean
119.875 decisions. It uses the same narrow-exploration `1e-8` stability regime
as gate 3 and does not claim improvement over the DAgger initializer.
Post-commit gate-4 qualification remains open.

M6.2 gate-4 acceptance (2026-09-02): the committed
`ppo_3v3_scripted_bc_v0` qualifying series reproduces 5/8 wins at updates 1,
5, and 10, versus 0/8 masked-random and 8/8 teacher, with zero rejected
actions. All three red-win seeds remain visible; no best-checkpoint selection
was performed. The semantic auditor binds source commit `eef57e4`, config
digest `sha256:de8e80f22a1ebe0d96c96583e3f01de586215a14c932dcbb2faf7965eeebc87b`,
and all nested artifacts. `3v3-scripted` is advanced; terrain and larger-roster
gates remain open.

M6.2 gate-5 and gate-6 acceptance (2026-09-02): the committed
`ppo_3v3_terrain_bc_v0` and `ppo_5v5_terrain_bc_v0` qualifying series each pass
all eight held-out episodes at updates 1/5/10, versus 0/8 masked-random, with
zero rejected actions. Gate 5 introduced terrain teacher data; gate 6 added
action-conditioned movement and throw target heads so the shared continuous
head no longer had to fit incompatible target distributions. Both results are
BC-initialized PPO retention gates, not cold-start PPO claims.

Gate-7 initializer acceptance (2026-09-02): the first 10v10 BC initializer
reliably eliminated nine opponents but targeted defeated roster slots because
`enemy_mask` represents slot presence, not life state. Relational selection now
combines that mask with the encoded alive bit. The frozen
`bc_10v10_terrain_relational_v0` code/policy hybrid keeps neural action
selection, exact nearest-living-enemy throw targets, and a final-opponent move
target while retaining the learned movement head as an auxiliary BC objective.
Its provenance-valid checkpoint won both disjoint BC evaluation episodes in
145 decisions with all ten blue units alive and zero rejected actions, versus
0/2 masked-random and 2/2 scripted teacher at 146 decisions. Gate-7 PPO
qualification remains open.

M6.2 gate-7 acceptance (2026-09-02): the committed
`ppo_10v10_terrain_relational_bc_v0` qualifying series passes all eight
held-out episodes at updates 1/5/10 in 145 decisions, versus 0/8 masked-random
and 8/8 scripted teacher, with zero rejected actions. The semantic auditor
binds source commit `448f8ba`, config digest
`sha256:39bacab64fca11007617d0698b782a3baf3023e9215a562f851437876525ff47`,
series digest
`sha256:e5c1b9b540a54c787208a7a4270698846d136132b2207ca9dafc9f3d422e3034`,
and every nested checkpoint/evaluation artifact. All seven centralized PPO
curriculum gates are now advanced. Gate 7 is explicitly BC-initialized PPO
retention; cold-start learning and material reward-driven improvement are not
claimed.

### M7 — plan-conditioned learned executor

Goal: train the fast learned controller to follow the existing slow
`CommandPlan` language without online LLM calls.

- [x] Add a deterministic synthetic plan curriculum that emits only schema-
      valid plans and records grounded assignments and source seeds.
- [x] Add a fixed-size `PlanTensorEncoder` for mission, approach, posture, fire
      policy, preferred range, cohesion, relative objective/group geometry,
      group fractions, support relation, and plan age. JSON remains canonical;
      the tensor is only an RL adapter.
- [x] Train matched executor architectures with and without plan input under
      the same data and optimization budget.
- [ ] Evaluate direct versus flank trajectories, focus versus distributed fire,
      hold/support/withdraw behavior, unseen valid directive combinations, and
      3v3/5v5 training to 10v10 transfer.
- [ ] Keep reflexes, late binding, action validation, lifecycle fallback, and
      target replacement host-owned; a plan never supplies physical actions.

M7 exit: plan-conditioned policies produce reproducibly distinct, intended
behavior under counterfactual plans for the same initial state and outperform
the no-plan ablation on frozen objective-completion metrics.

M7 synthetic curriculum foundation (2026-09-02):
`training/plan/SyntheticPlanCurriculum.ts` deterministically samples all five
mission types and the bounded approach/fire vocabularies from explicit source
seeds. Every sample is canonicalized by the production `CommandPlan` parser and
grounded by the production `PlanGrounder`; the resulting record retains the
symbolic plan, stable unit assignments, seed, plan ID, arena/roster provenance,
and optional source-state hash. The pure core performs no file or provider I/O.
Tests prove repeatability, schema validity, complete non-overlapping assignment,
directive coverage, and rejection of unsafe seed ranges or undersized rosters.
The tensor export/data-join path remains next.

M7 plan-tensor foundation (2026-09-02): `training/plan/PlanTensorEncoder.ts`
maps a production `PlanSnapshot` into three stable role slots, each with 38
bounded features and a separate presence mask. It includes role, mission,
approach, posture, fire, range, cohesion and objective-kind one-hots; tactical-
frame-relative objective/group geometry; requested and live assigned fractions;
support relation; and plan age. Raw unit IDs remain host-owned and are not
learnable features. Tests cover exact shape, slot stability, fractions, support,
age normalization, bounds, and counterfactual engage-versus-hold separation on
the same observation. JSON export/data joining and model ablations remain open.

M7 plan-tensor export foundation (2026-09-02):
`training/plan/PlanTensorDataset.ts` resets the authoritative headless
`SnowEnvironment`, binds simulator/hash provenance and the public source-state
hash, aligns every validated curriculum sample with its `[3,38]` tensor, and
computes a canonical SHA-256 dataset digest. Its auditor rejects plan, seed,
shape, value, or digest corruption. The `export-plan-tensors.ts` CLI supports
configurable maps, rosters, environment/plan seeds, counts, safe overwrite, and
machine-readable summaries. Two independent Winter Front 10v10 exports were
byte-identical. The Python data join and paired model ablation remain next.

M7 Python plan-data bridge (2026-09-02): `training/plan_data.py` verifies the
TypeScript semantic digest, converts aligned samples to immutable NumPy
`float32 [samples,3,38]` tensors plus `int8` masks and `int64` source seeds, and
returns detached copies when plan indices are joined to trajectory transitions.
Unit tests cover dtypes, shapes, repeated-plan alignment, immutability, index
validation, bounds and corruption. A live cross-language load reproduced the
exported 10v10 digest exactly. Adding plan inputs to the model and running the
paired with/without-plan ablation remain next.

M7 model adapter foundation (2026-09-02): the shared `EntityPolicy` now has an
opt-in `plan_conditioned` architecture flag. Enabled models encode the masked
`[3,38]` plan rows plus role-presence mask into one global embedding appended to
the unchanged physical entity context; disabled models retain the exact legacy
parameter shape and metadata. `TorchPolicy` requires and shape-checks the plan
tensors only for conditioned checkpoints. Tests prove missing/malformed inputs
fail closed, masked rows are removed from the adapter input, counterfactual
plans change hidden state for the same physical observation, and gradients are
finite. Plan-caused trajectory collection and the paired training ablation
remain open, so the M7 training checklist item is not yet advanced.

M7 plan-caused rollout foundation (2026-09-02):
`training/plan/PlanRolloutDataset.ts` executes every synthetic plan through the
production `PlanGrounder`, `PlanStore`, `PlanAwareTeamController`, and
`ReactiveUnitPolicy`, restarting the authoritative headless environment from
the same seed and public state hash for each counterfactual plan. Every decision
retains the detached observation, semantic action, dynamic plan tensor, reward,
pre/post state hashes, and physical action acceptance. Its semantic auditor
checks curriculum/episode alignment, observation hashes, tensor bounds, action
results, outcomes, and a canonical dataset digest. Tests prove byte-level
repeatability, same-state restarts, plan/action diversity, zero rejected actions,
and rejection of state, tensor, result, or digest corruption. A portable CLI,
Python trajectory conversion, and the paired training ablation remain next.

M7 portable rollout bridge (2026-09-02): `export-plan-rollouts.ts` exposes the
plan-caused collector for configurable map/open arenas, rosters, environment and
plan seeds, decision horizon/frequency, and red difficulty, with guarded
overwrite and machine-readable outcome summaries. Two independent exports were
byte-identical. `snowgym-convert-plan-rollouts` independently verifies the
ECMAScript canonical digest, public observation hashes, continuity, tensor
bounds, and action acceptance before using the shared Gym encoders to emit
audited `snowgym.trajectory.v0` shards. Each transition carries aligned
`plan_groups [3,38]`, `plan_group_mask [3]`, and the plan source seed. The
training loader exposes these arrays only for provenance-marked datasets, and
conditioned training fails closed when they are absent; the no-plan ablation can
consume the exact same shards and ignore the extras. The canonicalizer now also
matches `JSON.stringify` object semantics for omitted `undefined` fields, with
a write/read digest regression test. Paired training configurations and frozen
behavior metrics remain next.

M7 matched-training runner foundation (2026-09-02):
`snowgym-run-plan-ablation` accepts one audited plan trajectory dataset and one
shared architecture/optimization contract, then deterministically trains a
no-plan and plan-conditioned policy whose generated configs differ only by the
plan adapter flag. The result manifest binds the source config and dataset,
both checkpoint/state digests, architectures, and training losses. Its auditor
reloads each restricted checkpoint and rejects unmatched dataset, optimizer,
loss, seed, step, adapter flags, or child digest. Unit tests run the full pair
twice, prove exact reproducibility, and reject checkpoint-metadata corruption.
A frozen non-smoke data/config split and counterfactual trajectory evaluation
remain required before advancing the paired-training checklist item.

M7 counterfactual evaluator foundation (2026-09-02):
`snowgym-evaluate-plan-ablation` loads both restricted child checkpoints and a
separate audited plan trajectory dataset, then reports overall and same-state
first-decision action accuracy. At the latter boundary it cyclically swaps only
the plan tensors while preserving each physical observation and measures
correct-versus-shuffled action NLL, discrete action changes, and target deltas.
The no-plan control is required to remain exactly insensitive by construction;
conditioned sensitivity alone is not treated as following without improved
correct-plan fit. The evaluation artifact binds the ablation and dataset
digests, and its auditor rejects result or provenance drift. Unit tests execute
the full collect/convert/train/evaluate chain and prove the swap cannot affect
the no-plan model. Frozen held-out data, thresholds, and closed-loop objective
metrics remain next.

M7 development experiment freeze (2026-09-02):
`configs/plan_bc_ablation_dev_v0.json` fixes identical 3,000-step CPU BC
budgets for the first no-plan/conditioned comparison before its outputs are
observed. Development data will use Winter Front map-backed 6v6 rollouts with
environment seed 4200, plan seeds 120-143, and an 80-decision horizon;
evaluation uses disjoint environment seed 5200 and plan seeds 600-611 under the
same roster/map/horizon. This run is diagnostic and has no retroactive pass
threshold. Its outcome will determine whether data balance, model capacity, or
metrics need a separately documented development revision before freezing a
disjoint qualification configuration.

M7 development outcome (2026-09-02): the post-freeze run collected 1,920
Winter Front training transitions and 960 disjoint evaluation transitions.
Their dataset digests are respectively
`sha256:d241e366a465b5920e3257896ec0b8c2c6fd0292e4920226cf109ee2f90b7e4f`
and
`sha256:1ebef435923485f7b26c22d892bd1d5147d31f35bab7d98396bd2d2d636c7069`.
The matched result digest is
`sha256:f838b542ab94c1b37ef763afbfc670afd6bb057844882e12bfe30c865845d8df`.
On held-out first decisions, plan conditioning reduced correct-plan target MSE
from `0.272753` to `0.048208`; swapping only plans raised conditioned target
MSE by `0.306098`, while the no-plan target/action deltas were exactly zero.
Overall conditioned action-type accuracy was lower (`0.955035` versus
`0.973090`), and first-decision action types did not vary, making action NLL
non-discriminative there. The evaluator now explicitly records correct and
shuffled target MSE. These are strong target-following development results, not
an M7 exit: a disjoint predeclared qualification suite and closed-loop
directive/objective metrics remain open.

M7 qualification freeze (2026-09-02): before collecting qualification data,
`configs/plan_qualification_v0.json` fixes Winter Front 6v6 training at
environment seed 6200 / plan seeds 1200-1223 and evaluation at disjoint
environment seed 7200 / plan seeds 1800-1811, both with 80-decision horizons.
It binds the exact 3,000-step `plan_bc_ablation_qual_v0` config digest and
requires every check: conditioned target MSE at most `0.10` and half the
no-plan MSE, shuffled-plan target-MSE increase at least `0.10`, mean target
change at least `0.20`, action-accuracy deficit at most `0.03`, and all no-plan
sensitivity measures at most `1e-12`. The qualification tool validates seed
separation and config binding, verifies the evaluation digest, records every
check, and never performs checkpoint selection. Qualification execution is
next; even a pass advances only the paired offline target-following evidence,
not the closed-loop M7 exit.

M7 qualification-v0 outcome (2026-09-02): the frozen post-commit run is
retained as a failure, not retuned. It used 1,920 training transitions (digest
`sha256:e252f624d0b45cf240af1c9c06709c4bf06b268040db48feb03d109611b0a902`)
and 960 disjoint evaluation transitions (digest
`sha256:b388d20c5f27dbee2984fce34df67d2b391b461b3e81333faf9c7bc8f79a3e78`).
Seven of eight predeclared checks passed: conditioned correct-plan target MSE
was `0.064513` versus no-plan `0.239327`, plan swapping added `0.314999` target
MSE and changed targets by `0.369133`, and no-plan sensitivity remained zero.
The action-accuracy check failed: conditioned accuracy `0.875694` trailed
no-plan `0.924132` by `0.048438`, exceeding the frozen `0.03` allowance. The
result digest is
`sha256:9b8a91fcb91657946f75317015d039410f2c0a6e09a7bc21e1273343145189e2`
and qualification digest is
`sha256:2b53b246fbc236770b2c59a2e0bfcec82b278192dd71c4acc673864bb3524c77`.
The gate stays closed. Next development must isolate plan-conditioned target
features from the action classifier or otherwise address multi-task
interference, then freeze a new disjoint qualification rather than modifying
v0.

M7 action-interference isolation foundation (2026-09-02): the optional
`plan_target_only` plus `separate_target_actor` architecture routes physical
context through the unchanged action actor and routes detached physical
features plus the encoded commander plan through a separate target actor.
Target loss cannot update the shared action feature path; action logits are
therefore counterfactually plan-invariant while movement/throw targets and
power remain plan-conditioned. The option requires plan conditioning and
action-conditioned target heads; legacy and fully shared plan-conditioned
checkpoint shapes remain unchanged. The paired runner
strips this plan-only routing flag from its no-plan control. Tests prove exact
action-logit/hidden invariance under counterfactual plans, distinct target-path
representations, configuration rejection, and finite end-to-end training. A
new development run on the retained v0 data is next; qualification v0 remains
permanently failed.

M7 target-only development freeze (2026-09-02):
`plan_bc_ablation_target_only_dev_v1.json` reuses qualification-v0's retained
training/evaluation datasets, seed, 3,000-step budget, loss weights, and physical
architecture, adding only `plan_target_only` to the conditioned branch. This is
an explicit diagnosis on already observed data, not a qualification retry and
has no pass threshold. If it restores action accuracy while preserving target
following, a v1 qualification must be frozen with new disjoint environment and
plan seeds.

M7 target-only development outcome (2026-09-02): the corrected post-commit run
on retained qualification-v0 data produced bit-identical action behavior for
both branches: overall accuracy `0.928299`, first-decision accuracy `1.0`, and
first-decision action NLL `0.00335723`. Conditioned correct-plan target MSE was
`0.058109` versus no-plan `0.242607`; swapping only the plans added `0.352305`
target MSE and changed predicted targets by `0.405377`, while every no-plan
sensitivity stayed zero. The matched result digest is
`sha256:f3925f9512d8339e7aadb790430b5ebee5d7ab7c4cf3fc569fbcaed812a63a10`
and evaluation digest is
`sha256:e4d186c40bf828db756e01a7afccb619a5396a98a50f119697943500f1b231d0`.
This resolves the observed action-head interference on development data. A v1
qualification with new seeds must still be frozen and passed.

M7 qualification-v1 freeze (2026-09-02): before observing any new outputs,
`plan_qualification_v1.json` retains every v0 numerical threshold and binds the
target-only/separate-actor 3,000-step config digest. Training uses new
environment seed 8200 and plan seeds 2400-2423; evaluation uses disjoint seed
9200 and plan seeds 3000-3011. Map, 6v6 roster, red difficulty, sample counts,
and 80-decision horizon remain unchanged so the architecture fix is the only
intentional experimental change. Execution is next; v0 remains failed
regardless of the v1 outcome.

M7 qualification-v1 acceptance (2026-09-02): the post-freeze run passed every
predeclared check on new seeds. The converted training and evaluation dataset
digests are
`sha256:f64e30e6458ca3d9c9a6e110aae7ae3248e6e2715a938ce323732b1e658d61e6`
and
`sha256:4e7f90f323ad8b3a660b7c6ff26cb91d286d2c64b685a612dd1c3e0622bc92c1`.
Both branches produced exactly identical action accuracy (`0.963021`),
first-decision accuracy (`1.0`), and action NLL (`0.000107431`). Conditioned
correct-plan target MSE was `0.044480` versus no-plan `0.264987`; plan swapping
added `0.353375` target MSE and changed targets by `0.372476`, with every
no-plan sensitivity equal to zero. The matched run, evaluation, and
qualification digests are respectively
`sha256:7583be365ffddcb88d40b4bfb3d3c00dca775b4f760fbe5f5453fff44270011b`,
`sha256:780f7c863baf352ff7de46695457114abb404f16593d879ca237abcf20b86fdf`,
and
`sha256:ea93ce6ff71029e932c1ec4bd3493fce83e80f71185fced6f123d5e6e0b3a12d`.
The matched offline training checklist item is advanced. Direct/flank,
focus/distributed, hold/support/withdraw, unseen-combination, roster-transfer,
and closed-loop objective metrics remain open; M7 exit is not yet claimed.

M7 authoritative online-plan bridge (2026-09-02): the guarded server API now
activates schema-valid symbolic plans and emits current fixed-size plan tensors
without advancing physics. Grounding, stable group assignments, late-bound
objective resolution, tactical geometry, living fractions, plan age, reset
invalidation, and state-hash/idempotency protection remain in TypeScript. The
persistent batch host exposes exact per-world parity, and `SnowGymBatchEnv`
validates and stacks `[B,3,38]` plan tensors for the next learned closed-loop
runner. Service, batch-isolation, validation, age, and live subprocess tests
cover the bridge; closed-loop objective evaluation remains next.

M7 closed-loop evaluation freeze (2026-09-02):
`configs/plan_closed_loop_v0.json` fixes a same-seed 6v6 Winter Front comparison
between direct/focus and left-flank/distributed plans before seeing execution
results. `plan_closed_loop.py` runs the qualified conditioned checkpoint and its
matched no-plan control as real blue policies, fetching fresh host-owned plan
tensors at every decision. Its hashed result reports terminal outcomes, rejected
actions, normalized objective progress, first-action target divergence, and
final group-position separation. The one-decision real-subprocess smoke proves
the complete checkpoint-to-authoritative-world path; the frozen full run is
next and thresholds have not been retrofitted.

M7 closed-loop development outcome (2026-09-02): the frozen v0 run completed
four real executions with zero rejected actions. Both policies truncated at the
predeclared 900-tick horizon and neither won, so this is not an M7 exit. In both
direct and flank cases, the conditioned policy finished with four blue survivors
versus two for no-plan. Its objective-progress advantage was `0.194498` and
`0.214476`; first-action target mean absolute deltas were `0.230875` and
`0.235246`, and final group-position distances were `0.270414` and `0.219296`.
The retained result digest is
`sha256:707864e8d39522f1c1f051d9090f104d956ce10bf8e42b657a2fadefe3a2e767`.
The next revision needs longer terminal evaluation plus explicit hold/support/
withdraw and multi-group objective metrics; this v0 result must not be tuned or
reinterpreted as a win-rate qualification.

M7 behavior-suite-v1 freeze (2026-09-02): before running new episodes,
`configs/plan_closed_loop_behaviors_v1.json` fixes same-seed 6v6 hold-current,
withdraw-backfield, and main-plus-reserve-support cases at a 3,600-tick / 600-
decision horizon. This suite is additive to the immutable v0 direct/flank run;
it tests one-group and multi-group behaviors without changing the qualified
checkpoints. Terminal outcomes and per-role behavior metrics remain unseen.

M7 mission-metric foundation (2026-09-02): the closed-loop evaluator now
retains objective distance, progress, start-to-final displacement, and terminal
position separately for each stable role slot. Case comparisons expose per-role
progress and displacement deltas, so hold can be judged by movement from its
activation position and reserve support is no longer averaged into the main
body. Existing result artifacts remain immutable; a new audited metrics result
must use a distinct path.

M7 behavior metrics-v1 outcome (2026-09-02): the immutable rerun at
`evaluations/plan_closed_loop_behaviors_metrics_v1.json` has digest
`sha256:060c33d396221ee3bf4d0b21e93f65462c0ae27c531feaae991585c18128e2d3`.
Conditioning reduced hold displacement by `0.552726`, which is directionally
correct, but also reduced withdraw displacement by `0.427437`. In the two-group
support case, main and reserve objective-progress deltas were `-0.151835` and
`-0.347228`. The model therefore demonstrates a defensive movement-suppression
effect, not yet distinct hold/withdraw/support competence. The next training
revision must balance these missions and qualify per-role behavior rather than
optimizing the aggregate objective-distance metric.

M7 plan-oracle foundation (2026-09-02): the server and persistent batch host
now expose a read-only plan-aware teacher action at the exact current state
hash. It uses the production `PlanAwareTeamController` and reactive unit policy,
does not advance physics, and fails closed until a plan is active. Python
validates per-world state-hash alignment before returning semantic labels. This
is the host-owned oracle seam for the next plan-conditioned DAgger collector;
learner-state collection and retraining remain open.

M7 plan-DAgger collector foundation (2026-09-02):
`export_plan_dagger.py` now runs the qualified plan-conditioned checkpoint in
persistent authoritative worlds, fetches a fresh host-resolved plan tensor and
same-hash production plan-teacher action at every learner-visited state, then
executes only the learner action. It rejects state drift, semantic-action
round-trip differences, physical action rejection, non-plan checkpoints, and
invalid plan episodes. Audited trajectory shards retain aligned plan tensors,
teacher labels, rollout-checkpoint provenance, split seeds, and complete plans.
A real two-decision integration test covers the full path. A frozen multi-
mission collection spec, merged training corpus, and retraining ablation remain
next.

M7 plan-DAgger-v0 freeze (2026-09-02): the catalog-based
`configs/plan_dagger_v0.json` freezes direct, flank, hold, withdraw, and two-
group support templates; ten training episodes and five each for validation and
evaluation use disjoint seeds. All run on 6v6 Winter Front against easy scripted
red with a 1,800-tick horizon. The catalog and episode references are validated
before the batch host starts. Collection results and retraining outcomes remain
unseen; this freeze must be committed before generation.

M7 plan-DAgger-v0 collection (2026-09-02): the post-freeze headless run produced
2,312 training, 1,191 validation, and 1,210 evaluation transitions with dataset
digests `sha256:8d4138dc4eea7a83f7af3273996a026cba2503268e89b468e08a0f490f08d182`,
`sha256:900a2872af7e730f81535b91356c8f70e2a78ef6b9972116191d14efb3d8700e`,
and `sha256:c8497125d4c62e86b2e0bbc92b1b0dae1275db42049c8149073079ffca2c2b85`.
Every direct, flank, hold, and support rollout lost; withdraw reached the frozen
300-decision limit. These are learner-state correction labels, not performance
evidence. Before combining them with expert-state data, the aggregate writer
must retain plan-conditioning metadata and explicitly support independently
seeded source specs.

M7 plan-aware aggregate foundation (2026-09-02): trajectory merging now has an
explicit independent-source mode for combining expert-state and learner-state
corpora. It requires matching split, capacity, simulator versions, and plan-
conditioning status; rejects any seed overlap; retains only common audited
tensor fields; records dropped auxiliaries per source; and preserves plan tensor
visibility for training. A real qualification-plus-DAgger smoke retained 4,232
transitions and both required plan arrays while dropping only the original
`plan_source_seed` auxiliary. Default same-spec merging remains strict.

M7 safe correction-training foundation (2026-09-02): behavior cloning now
distinguishes exact resume from initialization on a new dataset. The optional
`plan-target-path` mode requires the target-only/separate-target architecture
and freezes entity encoders, the physical action actor, and action head while
training only the plan encoder, target actor/heads, and power head. Checkpoints
bind initializer checkpoint/state digests and the new dataset independently. A
real plan-DAgger smoke proves the action head remains bit-identical while the
plan encoder updates. A frozen correction config and post-freeze run are next.

M7 plan-DAgger correction-v0 freeze (2026-09-02): the audited expert-plus-
learner aggregate contains 4,232 transitions with digest
`sha256:297be4717a9f33a44f374c006e6aca73aed804149c2f781ebd69508b79497bdd`.
`configs/plan_dagger_correction_v0.json` fixes qualified-checkpoint
initialization, plan-target-path-only training, seed 84001, 1,500 steps, batch
64, learning rate `0.001`, and the existing matched loss weights. The physical
action path is frozen. Training and closed-loop outcomes remain unseen.

M7 correction-evaluation bridge (2026-09-02): closed-loop evaluation can now
accept an explicit conditioned checkpoint while retaining the original matched
ablation's no-plan control. Results bind both checkpoint and state digests, and
the semantic auditor reloads the supplied override. This avoids copying a
checkpoint or manufacturing a new ablation manifest for post-DAgger evaluation.

M7 plan-DAgger correction-v0 outcome (2026-09-02): the frozen 1,500-step run
produced checkpoint digest
`sha256:54439be3493c17aa5fdd6a5f21f3698c616784e33c9e0a1ce00bf7bcd932cb29`.
All physical action-path tensors remained bit-identical and all 14 permitted
target-path tensors changed. The correction did not pass closed-loop acceptance:
direct ended with 3 blue / 5 red rather than the prior 4 / 4; hold and withdraw
lost sooner; and support regressed from eliminating two red units to eliminating
none. Evaluation digests are
`sha256:3b1a65956f4c3be56b0654091b021d52e6010949eabbba10ac4c6ae4099192e1`
and
`sha256:dfcfbf1418b7171ce37c62ecd5c4cb3dada316ab32dabd2e2c388e3a99585965`.
The checkpoint is retained as failed evidence and must not replace qualification
v1. Target-only learner-state correction cannot fix mission-dependent action
timing; the next architecture must condition action decisions while preserving
a matched no-plan control and explicit action-accuracy bounds.

M7 residual action-adapter foundation (2026-09-02): the optional
`plan_action_adapter` adds a plan-conditioned residual to action logits without
changing the inherited physical actor shape. Its output layer starts at exact
zero; a target-only qualification checkpoint therefore produces bit-identical
initial logits when loaded into the expanded model. The corresponding
`plan-action-target-path` training mode freezes all inherited entity encoders,
the actor, and action head while updating only the new adapter plus the existing
plan/target path. Tests prove zero-init invariance, compatible initialization,
frozen action-head identity, and adapter learning. A matched frozen experiment
and action-accuracy gate remain next.

M7 single-checkpoint offline gate foundation (2026-09-02):
`plan_checkpoint_evaluate.py` applies the existing action, target, and same-state
counterfactual metrics to one checkpoint on any audited aligned plan dataset.
Its semantic result binds checkpoint/state and dataset digests and is independently
auditable. This supplies the action-accuracy safety measurement needed before a
residual adapter can be judged on closed-loop outcomes.

M7 residual action-adapter-v0 freeze (2026-09-02): the qualified checkpoint's
pre-run learner-state baseline has action accuracy `0.631130`, first-decision
accuracy `1.0`, target MSE `0.121232`, and evaluation digest
`sha256:4e97e2f7686f56559819395d6df9b3b2f09b71bfd4c957664ffb8de0a6440789`.
The frozen run uses seed 85001, 1,500 steps, batch 64, learning rate `0.0003`,
qualified initialization, and plan-action-target-path training. Its conjunctive
spec requires action accuracy at least `0.681130`, first-decision accuracy at
least `0.95`, target MSE at most `0.13`, action counterfactual change at least
`0.05`, target sensitivity at least `0.2`, zero rejected actions, direct/flank
blue survivors at least 4 each, hold/withdraw duration at least 267/341, and
support red survivors at most 4. No adapter training outcome has been observed.

M7 residual action-adapter-v0 outcome (2026-09-02): the frozen run is retained
as a failed gate with checkpoint digest
`sha256:4079ba7186995f293731887bd99073a70c0c6c1136d6bb853034a828b0ffd1b3`.
Offline action accuracy improved to `0.854821` and target MSE to `0.018053`,
while hold and withdraw exceeded their duration thresholds and all actions were
accepted. Direct and flank preserved three rather than four blue units, support
eliminated no red units, and the first-decision counterfactual action-change
rate remained zero. Seven of eleven frozen checks passed; qualification digest
`sha256:4482dcd8585932e62dcc5d0605cbc3f1d62a4eb76f2edeae45ee76d29dd42777`
records the failure. Dataset inspection showed every evaluated plan labels all
six first-decision units as `move`, so that first-decision-only action-change
criterion cannot identify correct mission-dependent timing. The next revision
must collect host-generated counterfactual plan/action labels on the same later
learner-visited physical states; v0 thresholds and results remain immutable.

M7 counterfactual plan-preview foundation (2026-09-02): the service and batch
host can now ground an arbitrary schema-valid plan against the current detached
world, return its host-resolved `[3,38]` tensor and production plan-teacher
action, and discard the temporary plan without changing the active plan or
advancing physics. Requests are guarded by the current public-state hash. This
read-only intervention seam enables paired later-state labels without resetting
plan age or corrupting the executed learner trajectory. Dataset export,
counterfactual training loss, and a disjoint v1 gate remain next.

M7 same-state counterfactual DAgger foundation (2026-09-02):
`snowgym.plan-dagger-export.v1` requires each rollout plan to name a distinct
catalog counterfactual. At every truthful learner transition, the collector
uses read-only plan preview to attach an alternate `[3,38]` tensor and complete
production teacher action for the identical pre-step physical state. The
trajectory auditor validates the extra tensors, legal masked actions, numeric
bounds, and explicit same-state provenance. Behavior cloning can apply a
bounded `counterfactualLossWeight` to a second hybrid loss after replacing only
the plan tensor and label; zero or omission preserves all legacy runs. A frozen
collection schedule, an all-transition paired evaluator, and a v1 residual-
adapter experiment remain next.

M7 all-transition counterfactual evaluator foundation (2026-09-02):
`plan_counterfactual_evaluate.py` runs one checkpoint under both authoritative
plan tensors for every paired state. Its audited result reports whether the
teacher action actually changes, accuracy and NLL under each plan, the model's
action-change rate, recall and strict pair accuracy where the teacher differs,
and target error/sensitivity. Checkpoint/state and dataset digests are bound in
the artifact. This separates an uninformative counterfactual set from a model
that ignores informative plan interventions. Frozen data and acceptance
thresholds remain next.

M7 counterfactual DAgger-v1 collection freeze (2026-09-02): before generating
new labels, `configs/plan_counterfactual_dagger_v1.json` fixes ten training,
five validation, and five untouched evaluation episodes on 6v6 Winter Front
against easy scripted red, each with a 1,800-tick horizon. Seeds 14201–14210,
14301–14305, and 14401–14405 are disjoint from each other and v0. Every direct,
flank, hold, withdraw, and support primary is paired with a distinct plan chosen
to contrast offensive timing, defensive posture, or group support. Training
teacher-diversity diagnostics may inform a separately committed optimization
freeze; validation/evaluation metrics must remain unseen until that freeze.

M7 counterfactual DAgger-v1 collection (2026-09-02): the post-freeze run
produced 2,393 training, 1,238 validation, and 1,200 evaluation transitions
with respective digests
`sha256:c3bd756af35908cfe972c43e11f0292ab5cab3dca0e3375e2bed436725749df5`,
`sha256:d0ec0683697f159e2e14983ee4f19c329fd4ae6778f29f3084185d4ebe237576`,
and `sha256:f5598b7fa17f03fce5e2b7af2814504796e7eea8ea16b36c847ebc955efd22fb`.
One training and one validation episode ended when attrition made their
two-group alternate plan inapplicable; both manifests record the bounded stop,
and evaluation completed all episodes. The permitted training-only diagnostic
found 1,538 teacher-changed unit decisions out of 14,358 (`0.107118`), while
the inherited target-only checkpoint remained exactly action-invariant. This
establishes identifiable supervision without inspecting validation/evaluation
metrics. Optimization and acceptance thresholds must be frozen next.

M7 counterfactual residual-adapter-v1 freeze (2026-09-02): before training or
reading validation/evaluation metrics, `plan_action_adapter_v1.json` fixes
qualified-checkpoint initialization, seed 86001, 2,000 steps, batch 64,
learning rate `0.0003`, unchanged hybrid loss weights, and equal weight on the
same-state counterfactual loss. The v1 qualification spec binds all three
corpus digests, initializer, and training-only baseline. Its paired evaluation
requires teacher diversity at least `0.05`, primary/counterfactual accuracy at
least `0.75`, predicted action changes at least `0.05`, changed-teacher recall
at least `0.50`, strict changed-pair accuracy at least `0.35`, and both target
MSE values at most `0.13`. It retains every v0 closed-loop mission threshold.
The qualification runner audits input/result and checkpoint provenance and
applies all checks conjunctively. Training and held-out results remain unseen.

M7 counterfactual residual-adapter-v1 outcome (2026-09-02): the frozen run is
retained as a failed gate with checkpoint digest
`sha256:4e7c4aaf9644ce63179bbecff2ebdca05e3dee64ad24b40efc6bb96a524348f5`.
On untouched evaluation data, primary/counterfactual action accuracy reached
`0.841111`/`0.860139`, target MSE `0.006856`/`0.009023`, and predicted action
change `0.084583`; teacher change was `0.090417`. Changed-teacher recall and
strict pair accuracy were only `0.165899` and `0.084485`, failing their frozen
thresholds. Closed loop, flank preserved five blue, hold lasted 338 decisions,
and support achieved the first learned plan-conditioned blue win at 6–0 with
zero rejected actions. Direct preserved three blue and withdraw lasted 259
decisions, so those checks failed. Ten of fourteen checks passed; qualification
digest is
`sha256:075a610c3f60ac2a77b112a5dd1c67506b991ae29ad56db2cda1f21f0a81a5c0`.
The next revision must weight the identifiable teacher-changed positions rather
than letting the roughly 90% unchanged pairs dominate the counterfactual loss.

M7 changed-action loss foundation (2026-09-02): behavior cloning now accepts a
bounded `counterfactualChangedActionWeight` only for datasets with audited
same-state labels. It adds symmetric primary/alternate cross-entropy solely on
present unit-states where the production teacher action types differ, retaining
the existing full primary and counterfactual hybrid objectives. Empty changed
batches contribute an exact finite zero. Legacy configs are unchanged. A
separately frozen v2 weight/budget and unchanged held-out gate are next.

M7 changed-action-v2 development freeze (2026-09-02): before training,
`plan_action_adapter_v2_dev.json` fixes the v1 architecture, corpus, 2,000-step
budget, batch size, learning rate, full counterfactual weight, and initializer,
changing only the training seed to 87001 and adding changed-action weight
`5.0`. This is a diagnostic on already observed v1 validation/evaluation and
closed-loop suites, not a qualification retry. If it materially improves
changed-teacher alignment without losing v1's support/flank progress, a new
counterfactual corpus with fresh seeds and a separately frozen gate is required.

M7 changed-action-v2 development outcome (2026-09-02): the fixed weight-5 run
produced checkpoint digest
`sha256:00b63d2e8f99e3557211b90b38694bf0a6dbc0333bcd9d587125e6e69148013d`.
On the reused evaluation split, changed-teacher recall rose from v1's `0.165899`
to `0.761905` and strict pair accuracy from `0.084485` to `0.697389`, but
predicted change increased to `0.395417` versus teacher change `0.090417`.
Closed loop preserved five blue on flank and kept withdraw alive for all 600
decisions at 2 blue / 1 red, while direct preserved two, hold lost after 205,
and support lost 0–3 rather than v1's 6–0 win. All actions remained accepted.
The loss fixes alignment but is overweighted. A predeclared intermediate-weight
development sweep is required before spending fresh qualification seeds.

M7 changed-action weight-sweep-v0 freeze (2026-09-02): before running any new
checkpoints, three configs fix weights `1`, `2`, and `3` with identical seed
88001, 2,000-step budget, data, initializer, model, and other losses. The frozen
development checks require paired accuracy at least `0.75`, changed recall/pair
accuracy at least `0.40`/`0.30`, predicted change in `[0.05,0.20]`, and the
existing five mission thresholds. Selection maximizes checks passed, then
changed-pair accuracy, then prefers lower weight. These reused suites select a
development candidate only; fresh data and a new gate remain mandatory.

M7 changed-action weight-sweep-v0 outcome (2026-09-02): the audited sweep
result digest is
`sha256:e83bff1be89aad0345b0b0336ea7a5caa5bbdb5edabe389ac5c5adb18548054b`.
Weights 1, 2, and 3 each passed 7/11 development checks. Weight 1 alone kept
predicted action change under `0.20` (`0.179456`) but narrowly missed changed
recall (`0.398411`); weights 2/3 improved strict pair accuracy to `0.640182`/
`0.708286` while over-changing `0.304389`/`0.373452`. All three missed flank,
hold, and withdraw thresholds. The predeclared tie-break selects weight 3, but
7/11 is not qualification readiness and no fresh seeds are consumed. Scalar
weighting cannot resolve the observed mission tradeoff; the next development
revision must balance supervision by primary mission and group role.

M7 unit-role conditioning foundation (2026-09-02): the Python batch adapter now
maps host-owned plan assignments onto stable current ally slots as fixed
`[B,U,3]` main/maneuver/reserve one-hot tensors, ignoring dead assignment IDs
without exposing them to the learner. `snowgym.plan-dagger-export.v2` retains
both primary and same-state counterfactual unit-role tensors with audited
one-hot, presence, and coverage rules. The optional `plan_role_conditioned`
model path requires the residual plan adapter and injects per-unit role into a
zero-initialized action residual and a separate zero-initialized target
residual, preserving qualified initializer outputs exactly. Tests cover live
batch encoding, preview roles, schema audit, malformed roles, compatible
initialization, and a training step. A fresh role-aware collection freeze and
development run remain next.

M7 role-aware-v0 development freeze (2026-09-02): before collection,
`plan_role_dagger_v2.json` fixes fresh 15201–15210 training, 15301–15305
validation, and sealed 15401–15405 evaluation seeds with the existing balanced
mission-pair schedule. `plan_role_adapter_v0_dev.json` fixes qualified
initialization, seed 89001, the established 2,000-step budget and losses,
changed-action weight `1.0`, and enables only the new unit-role residuals.
Development may inspect training/validation plus the existing closed-loop
suites; the v2 evaluation split remains unopened until a new gate is frozen.

M7 role-aware-v0 development outcome (2026-09-02): after correcting role
encoding to cover living ally slots only, the audited train/validation corpora
contain 2,358/1,152 transitions with digests `173ad8a4...`/`0ba2ad5e...`.
The frozen 2,000-step run produced checkpoint `490f88c0...`. Validation primary
action accuracy improved to `0.863426`, but predicted plan-change rate
`0.227865` still exceeded the teacher's `0.130642`; changed-teacher recall and
strict pair accuracy were only `0.559247`/`0.462901`. Closed loop preserved
four blue in both direct/flank cases but completed neither, while hold,
withdraw, and support all lost; support regressed to 0 blue versus 6 red.
The checkpoint is retained as a failed development result and the sealed
15401–15405 evaluation split remains unopened. The next M7 revision must
condition each unit on its own resolved group directive, not merely its
main/maneuver/reserve category, while retaining the qualified initializer and
same-state counterfactual evaluation.

M7 per-unit directive foundation (2026-09-02): the optional
`plan_unit_directive_conditioned` architecture derives a deterministic
`[B,U,38]` tensor by selecting each living unit's host-resolved group row with
its audited one-hot assignment. The full local directive now enters both the
zero-initialized action and target residuals while the global plan encoder is
retained. It requires role conditioning, adds no new observation schema, and
keeps older checkpoints and v2 corpora compatible. Tests cover configuration
constraints, exact group-row selection, initialized inference, and the
role-aware closed-loop policy bridge. A frozen development config and run on
the retained train/validation corpus are next; sealed evaluation seeds remain
untouched.

M7 per-unit-directive-v0 development freeze (2026-09-02):
`plan_unit_directive_adapter_v0_dev.json` reuses the corrected v2 training and
validation corpora plus the qualified plan-conditioned initializer. It holds
seed 89001, 2,000 steps, batch 64, learning rate `0.0003`, and both
counterfactual weights at `1.0`, exactly matching role-aware-v0. The sole model
change is `plan_unit_directive_conditioned: true`. Only validation and the
existing closed-loop suites may be inspected; evaluation seeds 15401–15405
remain sealed.

M7 per-unit-directive-v0 development outcome (2026-09-02): the frozen run
produced checkpoint `3855cc78...`. Validation changed-teacher recall/pair
accuracy improved over role-aware-v0 to `0.599114`/`0.503876`, while primary
accuracy fell slightly to `0.855758` and predicted plan changes remained high
at `0.239873`. The withdraw case improved materially to the full 600-decision
horizon with five blue versus four red, but direct/flank retained only three/two
blue and support still lost 0–6. This is retained partial evidence, not a
qualification candidate; evaluation seeds remain sealed. Richer directive
features alone do not resolve the mission tradeoff. The next revision must
balance correction supervision across primary missions and assigned roles.

M7 mission/role balancing foundation (2026-09-02): behavior-cloning configs may
now opt into deterministic `plan-mission-uniform` sampling, which selects plan
names uniformly before choosing transitions, and `roleBalancedLoss`, which
applies mean-one inverse-frequency unit weights to action, target, power, and
teacher-changed losses. Defaults preserve every prior run. Weighting covers
observed roles and gives absent roles zero weight; auditing the retained corpus
revealed 11,488 main, 790 reserve, and zero maneuver labels in train (5,754,
335, and zero in validation). Thus balancing can correct main/reserve skew, but
a future curriculum must add a genuine maneuver-group plan before claiming
three-role transfer. Tests cover deterministic mission balance, equal aggregate
weight across observed roles, malformed loss weights, and a real balanced
training step.

M7 balanced-unit-directive-v1 development freeze (2026-09-02):
`plan_unit_directive_balanced_v1_dev.json` holds the v0 directive architecture,
initializer, seed, 2,000-step budget, batch size, optimizer, and loss weights
fixed. Its only changes are `sampling: plan-mission-uniform` and
`roleBalancedLoss: true`. The run may inspect the retained train/validation
corpora and existing closed-loop suites only; evaluation seeds remain sealed.

M7 balanced-unit-directive-v1 development outcome (2026-09-02): the frozen run
produced checkpoint `3106db06...`. Direct improved to five blue versus four red,
hold extended to 317 decisions, and withdraw again reached 600 decisions with
five blue while reducing red to one. However flank fell to one blue versus five
red and support remained a 0–6 loss. Validation primary/counterfactual accuracy
regressed to `0.826678`/`0.798177`, predicted changes rose to `0.280961`, and
strict changed-pair accuracy fell to `0.447398`. The result is retained but not
promoted, and sealed evaluation remains untouched. The next collection must add
real maneuver-group assignments and more multi-group support supervision before
another controlled training run.

M7 multi-group DAgger-v3 collection freeze (2026-09-02):
`plan_multigroup_dagger_v3.json` defines six plan families on the unchanged 6v6
arena: direct, hold, withdraw, a true main+maneuver left flank, main+reserve
support, and main+maneuver+reserve support. Train seeds 15501–15512 and
validation seeds 15601–15606 are disjoint from sealed evaluation seeds
15701–15706. Each plan appears twice in train and once in validation/evaluation;
all pairs use a distinct same-state counterfactual. Only train/validation may be
collected until a later gate is frozen.

M7 multi-group DAgger-v3 collection outcome (2026-09-02): the audited train
and validation splits contain 2,763 and 1,396 transitions with digests
`5bf6a6a2...` and `ed8dceda...`. Primary role counts are 11,838 main, 1,512
maneuver, 952 reserve in train and 5,852/839/439 in validation; counterfactual
roles also cover all three categories. Attrition-bounded episodes are retained
and explicitly marked. No evaluation artifact exists. A training freeze on
this immutable corpus is next.

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
