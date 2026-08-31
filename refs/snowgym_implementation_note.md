# SnowGym Fork Implementation Note

## Goal

Build a fork of the Khadgar SnowCraft engine that supports:

- Multiple blue-team units
- Autonomous blue-vs-red play
- An RL-compatible simulation interface
- Future Gymnasium and PettingZoo adapters
- Headless simulation
- Scripted, RL, LLM, and network-controlled agents

The fork should remain easy to compare with and update from upstream.

The primary engineering rule is:

> **Keep all new SnowGym development inside its own top-level directory, and make the smallest possible changes to the original SnowCraft codebase.**

The original game should continue to run normally.

---

# 1. Forking Principle

Treat the upstream SnowCraft code as an embedded engine rather than a codebase to rewrite.

Preferred structure:

```text
snowcraft/
├── src/                         # upstream SnowCraft code
├── public/
├── tests/
├── package.json
├── ...
│
└── snowgym/                     # ALL new development goes here
    ├── README.md
    ├── package.json             # optional workspace package
    ├── tsconfig.json
    │
    ├── core/
    ├── adapters/
    ├── agents/
    ├── scenarios/
    ├── observations/
    ├── actions/
    ├── rewards/
    ├── headless/
    ├── protocol/
    ├── gym/
    ├── pettingzoo/
    ├── benchmarks/
    ├── tests/
    └── tools/
```

Do not move existing upstream files unless absolutely necessary.

Do not rename upstream classes or modules merely to make SnowGym cleaner.

Do not convert upstream architecture into a monorepo during the first milestones.

The first goal is a **good fork**, not a perfect rewrite.

---

# 2. Upstream Change Budget

Any modification outside `snowgym/` should satisfy at least one of these conditions:

1. SnowGym cannot observe necessary simulation state otherwise.
2. SnowGym cannot submit actions without directly mutating engine internals.
3. The existing engine assumes exactly one blue unit and this prevents multiple units from existing.
4. A minimal export or hook dramatically reduces duplicated logic.

Prefer changes such as:

```ts
export { SomeExistingClass };
```

or:

```ts
interface GameHooks {
  onBeforeStep?: (...) => void;
  onAfterStep?: (...) => void;
}
```

over structural rewrites.

Avoid changes such as:

```text
src/core/        → packages/sim/
src/render/      → packages/browser/
src/game/...     → redesigned ECS hierarchy
```

during the initial fork.

Those may be considered much later if the SnowGym work proves valuable.

---

# 3. Compatibility Rule

The original game must continue to work.

The expected dependency direction is:

```text
UPSTREAM SNOWCRAFT
        ▲
        │ imports
        │
     SNOWGYM
```

Never:

```text
UPSTREAM SNOWCRAFT
        │
        ▼
     SNOWGYM
```

The original engine should not depend on Gym, RL, Python, networking, PettingZoo, or SnowGym-specific concepts.

SnowGym is an extension layer.

---

# 4. First Milestone: 3 Blue AI vs 3 Red AI

Before Gymnasium, Python, networking, or training, prove that the current engine can support multiple blue units.

## Target scenario

```text
Blue AI team

B0      B1      B2


R0      R1      R2

Red AI team
```

Requirements:

- 3 blue units
- 3 red units
- no human input required
- no respawning
- fixed spawn locations
- fixed RNG seed
- no buffs initially
- no cover initially
- both teams can move
- both teams can throw
- projectiles identify owner/team
- units can damage opponents
- dead units are removed or disabled correctly
- round ends when one team has no living units

Acceptance test:

> Given the same seed and scenario configuration, three scripted Blue agents and three Red AI agents autonomously complete a battle and the simulator deterministically declares a winner.

---

# 5. Do Not Build `BlueAISystem`

The multi-unit work should not create separate game logic for Blue and Red.

Avoid:

```ts
BlueAISystem
RedAISystem
```

Prefer a team-neutral model:

```ts
updateAI(unit, opponents)
```

where:

```ts
const opponents = units.filter(
  other =>
    other.team !== unit.team &&
    other.alive
);
```

AI behavior may differ between controllers, but combat mechanics must not.

---

# 6. Introduce an RL-Compatible Boundary Immediately

Even though the first Blue controller is hard-coded, it should interact with the game as if it were an RL policy.

The simulator must not know whether an action came from:

- scripted AI
- human input
- PPO
- MAPPO
- LLM
- remote network client
- replay file

The desired logical boundary is:

```text
Controller
    │
    ▼
Observation
    │
    ▼
Policy
    │
    ▼
Action
    │
    ▼
SnowGym Adapter
    │
    ▼
SnowCraft Simulation
```

Agents must not directly modify entity state.

---

# 7. Canonical Action Interface

Create inside:

```text
snowgym/actions/
```

Initial action types should be simple and engine-independent.

Example:

```ts
export type UnitAction =
  | {
      type: "noop";
      unitId: number;
    }
  | {
      type: "move";
      unitId: number;
      x: number;
      y: number;
    }
  | {
      type: "throw";
      unitId: number;
      x: number;
      y: number;
      power: number;
    };

export interface TeamAction {
  actions: UnitAction[];
}
```

Do not expose mouse coordinates, drag selection, key presses, or UI events as the RL action space.

Human UI commands may eventually be translated into the same `UnitAction` representation.

---

# 8. Canonical Observation Interface

Create inside:

```text
snowgym/observations/
```

The canonical observation should represent simulation entities rather than rendered objects.

Example:

```ts
export interface UnitObservation {
  id: number;
  team: number;

  x: number;
  y: number;

  vx: number;
  vy: number;

  health: number;
  alive: boolean;

  throwCooldown: number;
  charge: number;
}

export interface ProjectileObservation {
  id: number;
  ownerId: number;
  team: number;

  x: number;
  y: number;

  vx: number;
  vy: number;
}

export interface Observation {
  tick: number;
  selfTeam: number;

  allies: UnitObservation[];
  enemies: UnitObservation[];
  projectiles: ProjectileObservation[];

  match: {
    blueAlive: number;
    redAlive: number;
    timeRemaining?: number;
  };
}
```

Do not make Three.js objects part of observations.

Do not return meshes, scene nodes, camera state, DOM elements, or screen coordinates.

---

# 9. RL Environment Contract

Create the first environment abstraction in:

```text
snowgym/core/
```

Suggested interface:

```ts
export interface StepResult {
  observation: Observation;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;
}

export interface SnowEnvironment {
  reset(seed?: number): Observation;

  observe(team: number): Observation;

  step(actions: {
    blue: TeamAction;
    red: TeamAction;
  }): StepResult;
}
```

A later version can provide per-team results separately.

The important Gym-like semantics are:

```text
reset()
observe()
step()
reward
terminated
truncated
info
```

---

# 10. Separate Physics Frequency from Agent Frequency

The game simulation should continue using its existing fixed simulation timestep.

Do not make one RL step equal one physics tick.

Example:

```text
physics            60 Hz
agent decisions    10 Hz
```

One RL step can therefore advance six simulation ticks.

Suggested configuration:

```ts
interface EnvironmentConfig {
  simulationHz: 60;
  decisionHz: number;
}
```

Internally:

```ts
ticksPerDecision =
  simulationHz / decisionHz;
```

Initial default:

```text
simulationHz = 60
decisionHz   = 10
```

Later benchmark values may include:

```text
60 Hz
20 Hz
10 Hz
5 Hz
2 Hz
1 Hz
```

This is important for future temporal-abstraction and inference-latency research.

---

# 11. Team Controller Interface

Create:

```text
snowgym/agents/
```

Suggested interface:

```ts
export interface TeamController {
  act(observation: Observation): TeamAction;
}
```

First implementations:

```text
snowgym/agents/
├── RandomAgent.ts
├── SimpleBlueAgent.ts
├── ExistingRedAIAdapter.ts
└── PassiveAgent.ts
```

The first Blue AI should intentionally remain simple.

Example strategy:

```text
for each living blue unit:

    if incoming projectile is dangerous:
        dodge

    else if target is in range:
        throw

    else:
        move toward nearest enemy
```

Do not optimize gameplay intelligence yet.

The purpose of the first agent is to validate the environment boundary.

---

# 12. Existing Red AI

Avoid rewriting Red AI initially.

If possible, wrap the existing Red behavior:

```text
ExistingRedAI
      │
      ▼
ExistingRedAIAdapter
      │
      ▼
TeamAction
```

If the current Red AI directly mutates entities, it is acceptable during the first prototype to leave it intact while Blue uses the new interface.

However, the next milestone should migrate both sides toward the same action boundary.

Desired state:

```text
Blue scripted controller ─┐
                          │
Red scripted controller ──┼──> TeamAction[]
                          │
RL controller ────────────┘
```

---

# 13. Multi-Blue Changes to Upstream

The first implementation should search for singleton assumptions such as:

```text
player
thePlayer
playerEntity
selectedPlayer
localPlayer
```

Likely minimal upstream changes may include:

## Player creation

Allow multiple player-like entities.

Prefer:

```ts
spawnPlayer({
  team,
  position
});
```

instead of adding SnowGym-specific spawning logic inside upstream.

## Team field

Every combat-capable unit should expose a team identifier.

Example:

```ts
enum Team {
  Blue = 0,
  Red = 1
}
```

Reuse existing team data if already present.

## Projectile ownership

Each projectile should expose:

```ts
ownerId
team
```

This is required for:

- collision rules
- reward calculation
- kill attribution
- observations
- replays

## Victory logic

Move from:

```text
player alive?
enemies remaining?
```

toward a minimal generic check:

```text
living units on Team Blue
living units on Team Red
```

For the prototype:

```text
Blue alive == 0 → Red wins
Red alive == 0  → Blue wins
```

## Input

AI-vs-AI mode must work with no selection state and no human input.

If game simulation currently depends on selection state, isolate that dependency with the smallest possible patch.

---

# 14. Scenario Definition

Create:

```text
snowgym/scenarios/
```

Example:

```ts
export interface Scenario {
  name: string;
  seed: number;

  blueSpawns: Vec2[];
  redSpawns: Vec2[];

  respawn: boolean;
  buffs: boolean;

  maxTicks?: number;
}
```

Initial scenario:

```text
snowgym/scenarios/three_vs_three_open.ts
```

Properties:

```text
3 blue
3 red
open arena
no obstacles
no pickups
no respawn
fixed seed
fixed positions
```

This should become the canonical regression scenario.

---

# 15. Rewards

Create:

```text
snowgym/rewards/
```

The simulator should expose events sufficient to compute rewards without embedding RL reward logic inside upstream SnowCraft.

Useful events:

```text
ProjectileThrown
ProjectileHit
DamageApplied
UnitKilled
RoundEnded
```

Canonical benchmark reward:

```text
win   +1
loss  -1
draw   0
```

Optional diagnostic reward breakdown:

```ts
interface RewardBreakdown {
  damageDealt: number;
  damageTaken: number;
  kills: number;
  deaths: number;
  terminal: number;
}
```

Avoid deeply shaped rewards initially.

Do not reward:

```text
being near cover
moving toward enemy
throwing frequently
dodging
```

as canonical benchmark behavior.

Those may be experimental shaping signals later.

---

# 16. Event Adapter Instead of Engine Rewrite

If SnowCraft already emits events, consume them from SnowGym.

If not, prefer adding very small event hooks upstream.

Example:

```ts
gameEvents.emit("damage", {
  sourceId,
  targetId,
  amount
});
```

instead of moving reward logic into the engine.

SnowGym can subscribe:

```ts
rewardTracker.onDamage(...)
```

This keeps RL concepts outside upstream.

---

# 17. Debug Rendering

The original renderer remains useful for validating the simulator.

Add SnowGym-specific debugging in:

```text
snowgym/adapters/browser/
```

where possible.

Useful overlays:

```text
B0 → R1
B1 → R1
B2 → R2

R0 → B0
R1 → B2
R2 → B0
```

Optional visualizations:

- target lines
- unit IDs
- team IDs
- current action
- health
- cooldown
- projectile owner
- tick counter
- seed

Do not permanently modify normal game graphics just to support RL debugging.

---

# 18. Headless Milestone

Only after 3v3 autonomous play works in the browser, add:

```text
snowgym/headless/
```

Target API:

```ts
const env = createSnowEnvironment(config);

let obs = env.reset(42);

while (true) {
  const blue = blueAgent.act(obs.blue);
  const red = redAgent.act(obs.red);

  const result = env.step({
    blue,
    red
  });

  if (result.terminated || result.truncated)
    break;

  obs = result.observation;
}
```

The headless runner must not require:

```text
DOM
Canvas
WebGL
Three.js renderer
browser
Chromium
```

Three.js types should not leak into the RL API.

---

# 19. Python / Gymnasium Phase

Do not introduce Python until the TypeScript environment contract works reliably.

Later:

```text
snowgym/gym/
```

can expose a transport layer to:

```text
python/snowgym/
```

or eventually become:

```text
snowgym/python/
```

depending on repository layout.

Target:

```python
import gymnasium as gym

env = gym.make(
    "SnowGym/Squad-v0",
    blue_units=3,
    red_units=3,
    decision_hz=10,
)

obs, info = env.reset(seed=42)

while True:
    action = policy(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

---

# 20. PettingZoo Phase

Use the same simulator.

Do not create a second MARL simulation implementation.

Team-level Gym view:

```text
one policy
    ↓
Blue squad
```

PettingZoo view:

```text
blue_0
blue_1
blue_2
```

Same underlying entities.

Same physics.

Same actions.

Same observations.

Same seed.

Same scenario.

Only the adapter changes.

---

# 21. Batch Simulation

High-throughput RL training should eventually use batched environments.

Do not use a WebSocket round trip for every local RL step.

Target architecture:

```text
Python trainer
      │
      │ batch actions
      ▼
SnowGym Batch Host
      │
      ├── env 0
      ├── env 1
      ├── env 2
      ├── ...
      └── env N
```

Networking is for remote play.

Batch IPC/direct bindings are for local training.

---

# 22. Future Network Agent Protocol

Networking should be an adapter around the same environment contract.

Eventually:

```text
snowgym/protocol/
```

can support:

```text
human client
remote RL agent
LLM agent
tournament bot
spectator
```

Protocol actions should use the same semantic action types:

```text
noop
move
throw
attack
take_cover
...
```

Never require remote AI agents to simulate mouse input.

---

# 23. Research Features to Preserve in the Architecture

The fork should eventually support experiments over:

## Decision frequency

```text
60 Hz
20 Hz
10 Hz
5 Hz
2 Hz
1 Hz
```

## Observation fidelity

```text
perfect state
partial state
entity observations
semantic raster
pixels
language description
```

## Control abstraction

```text
motor control
unit intent
squad intent
```

## Opponent type

```text
random
scripted
PPO
MAPPO
self-play
LLM
human
```

## Inference latency

```text
0 ms
25 ms
50 ms
100 ms
250 ms
500 ms
1000 ms
```

The environment design should not hard-code assumptions that prevent these later experiments.

---

# 24. Determinism

Deterministic execution should be treated as a first-class requirement.

Every environment reset should accept a seed.

```ts
env.reset(seed);
```

Record:

```text
seed
scenario
initial state
actions
simulation version
upstream commit
```

A replay should ideally reconstruct the same battle from:

```text
seed + action sequence
```

This will be essential for:

- debugging
- regression testing
- benchmark reproducibility
- RL evaluation
- tournament disputes

---

# 25. Test Strategy

All new tests should live in:

```text
snowgym/tests/
```

except for tiny upstream regression tests required by minimal engine patches.

Essential tests:

## Multi-blue spawn

```text
spawn 3 blue
assert 3 blue exist
```

## Team targeting

```text
blue cannot target blue
red cannot target red
```

## Projectile ownership

```text
projectile.ownerId is correct
projectile.team is correct
```

## Team damage

```text
blue projectile damages red
red projectile damages blue
```

## No input dependency

```text
run match without InputManager
match progresses normally
```

## Round termination

```text
redAlive == 0
terminated == true
winner == blue
```

## Seed reproducibility

```text
same seed
same action sequence
same final state
```

## Observation purity

Observation data must not include references to mutable engine entities.

---

# 26. Upstream Patch Tracking

Create:

```text
snowgym/UPSTREAM_PATCHES.md
```

Every modification outside `snowgym/` should be recorded.

Example:

```markdown
## src/game/PlayerFactory.ts

Reason:
Allow creation of more than one Blue player entity.

Change:
Added optional `team` argument.

Upstream behavior:
Unchanged when omitted.

SnowGym dependency:
Required for 3v3 scenarios.
```

This provides a human-readable rebase guide.

Keep this file current.

---

# 27. Git Strategy

Recommended branches:

```text
upstream/main
fork/main
feature/snowgym-multi-blue
feature/snowgym-env-api
feature/snowgym-headless
feature/snowgym-gymnasium
```

Add the original repository as an upstream remote:

```bash
git remote add upstream <original-repository>
```

Periodic update:

```bash
git fetch upstream
git rebase upstream/main
```

Because SnowGym code mostly lives under one top-level directory, conflicts should stay small.

---

# 28. Recommended First PR / Commit Sequence

## Commit 1

```text
Add snowgym directory and architecture note
```

No gameplay changes.

## Commit 2

```text
Add SnowGym scenario and observation/action types
```

Still no gameplay changes.

## Commit 3

```text
Allow spawning multiple Blue units
```

Minimal upstream patch.

## Commit 4

```text
Add 3v3 autonomous SnowGym scenario
```

Scripted Blue AI.

## Commit 5

```text
Add team-neutral projectile ownership and victory tracking
```

Only if required.

## Commit 6

```text
Add SnowEnvironment reset/observe/step interface
```

Browser-backed initially if necessary.

## Commit 7

```text
Add deterministic 3v3 regression tests
```

## Commit 8

```text
Add headless runner
```

At that point, begin Gymnasium work.

---

# 29. M1 Acceptance Criteria

The first major milestone is complete when all of the following are true:

```text
[ ] Three Blue units can exist simultaneously
[ ] Three Red units can exist simultaneously
[ ] No human input is required
[ ] Blue scripted controller uses SnowGym actions
[ ] Blue units independently move and throw
[ ] Red units behave normally
[ ] Projectiles retain owner/team identity
[ ] Damage works in both directions
[ ] Friendly-fire behavior is explicit
[ ] Team elimination determines victory
[ ] Scenario accepts deterministic seed
[ ] Original human-vs-AI mode still works
[ ] Most new code exists under snowgym/
[ ] Every upstream modification is documented
```

---

# 30. M2 Acceptance Criteria

The RL-compatible simulation layer is complete when:

```text
[ ] reset(seed) works
[ ] observe(team) works
[ ] step(actions) works
[ ] step returns reward
[ ] step returns terminated
[ ] step returns truncated
[ ] step returns info
[ ] RL step can advance multiple physics ticks
[ ] action interface contains no UI events
[ ] observation interface contains no renderer objects
[ ] scripted Blue and Red agents can both use the interface
[ ] deterministic replay test passes
```

---

# 31. Architecture Constraint Summary

The fork should obey these rules:

```text
RULE 1
All SnowGym-specific development belongs under /snowgym.

RULE 2
Upstream code must never depend on SnowGym.

RULE 3
Modify upstream only to expose missing generic engine capabilities.

RULE 4
Prefer adapters to rewrites.

RULE 5
Prefer exports/hooks/events to moving code.

RULE 6
Agents never mutate engine state directly.

RULE 7
Actions are semantic game actions, never mouse/keyboard actions.

RULE 8
Observations are simulation state, never rendering state.

RULE 9
Gymnasium and PettingZoo are adapters over one simulator.

RULE 10
The original playable game must remain functional.

RULE 11
Every upstream patch is documented.

RULE 12
Deterministic seeded simulation is mandatory.
```

---

# 32. Long-Term Architecture

If the experiment succeeds, the architecture can grow naturally:

```text
                       ORIGINAL SNOWCRAFT
                              │
                              │ minimal hooks
                              ▼
                        SNOWGYM ADAPTER
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
             Browser       Headless      Replay
                │             │
                │             ▼
                │        Environment API
                │             │
                │     ┌───────┼────────┐
                │     │       │        │
                ▼     ▼       ▼        ▼
              Human Scripted Gym    PettingZoo
                              │        │
                              ▼        ▼
                             PPO     MAPPO/QMIX
                              │
                              ▼
                        Batch Training
                              │
                              ▼
                  Latency / Hierarchy Research
```

The central idea is simple:

> **SnowCraft remains the game engine. SnowGym becomes a thin, isolated research layer around it.**

That gives us the best chance of maintaining a clean fork while still evolving toward a serious RL and multi-agent benchmark.
