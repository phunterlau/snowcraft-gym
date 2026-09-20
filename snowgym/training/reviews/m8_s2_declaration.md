# M8-S2 — `SnowGymUnitParallelEnv` contract repair: termination semantics, action validation, simulator-backed tests

Declared 2026-09-20. Infrastructure repair, same lightened shape as
[M8-S1](m8_s1_declaration.md): declare → implement. No policy is trained, no
simulator-decision budget beyond test runs, no seed band, no sealed archive.

## 0. Why

An external review of the S1 code (`refs/snowgym_handoff_review_and_next_steps_2026-09-20.md`,
local notes) found two contract defects, and one gap in what S1's tests
establish. I re-checked each against the source before declaring this step.

1. **Truncation is reported as termination** (`unit_parallel_env.py`,
   `step()`). `episode_over = any(terms) or any(truncs)` feeds
   `terminations[agent] = episode_over or not unit_alive`. A pure timeout
   therefore returns `terminated=True` *and* `truncated=True` for every
   survivor. The team environment and `SnowEnvironment.stepJoint` report them
   separately. A trainer that zeroes value bootstrapping on `terminated`
   would treat every timeout as a true terminal.
2. **Invalid unit actions are silently coerced.** `_merge_team_actions` runs
   `int(submitted["action_type"])` and `np.asarray(submitted["target"], ...)`
   before the wrapped team env's `action_space.contains` check ever sees the
   values. `action_type=2.7` becomes 2; a scalar `target` broadcasts into both
   coordinates. The action-space error becomes a valid, different action and
   rejection statistics stay clean.
3. **S1's tests are fake-client only.** The PettingZoo checker, death, and
   episode-over tests use `FakeJointClient`. S1's declaration and PLAN.md
   describe simulator-backed determinism and death/removal checks; the
   checked-in tests do not establish them against the real simulator.

S1 has no consumers besides its own test file, so repairing it now is cheap.
It stays cheap only before a trainer is wired to it.

**Non-goals (explicitly deferred, one concern per step):** the v3/plan/option
observation adapter and focal-unit identity (S3), Red action routing for
training (S3), local visibility or latency, parameter-shared training, any
learning claim. Team-level `SnowGymParallelEnv`, `encoding.py`, and
`research_env.py` are read, not edited.

## 1. Termination contract

Let `team_terminated = any(team_terminations.values())`,
`team_truncated = any(team_truncations.values())`,
`unit_alive` = the unit's `alive` flag after the step.

| Field per active agent | Value |
| --- | --- |
| `terminated` | `team_terminated or not unit_alive` |
| `truncated` | `team_truncated` |
| `info["snowgym_unit"]["team_terminated"]` | `team_terminated` |
| `info["snowgym_unit"]["team_truncated"]` | `team_truncated` |
| `info["snowgym_unit"]["unit_alive"]` | `unit_alive` |
| `info["snowgym_unit"]["unit_died"]` | alive before this step, not alive after |

- `terminated` is the PettingZoo *actor lifecycle* flag: this agent will not act
  again. It is **not** a value-bootstrapping boundary. A trainer must derive
  the team value boundary from `info["snowgym_unit"]["team_terminated"]`:
  bootstrap through a unit's death while the team battle continues, and do not
  bootstrap through a true team termination. Truncation is a rollout cut and
  may bootstrap from the final observation. This is stated in the class
  docstring.
- Pure timeout: `terminated=False`, `truncated=True` for every survivor.
- Natural team termination: `terminated=True` for every active agent on both
  teams, `truncated=False`.
- Unit death while the battle continues: only that agent has `terminated=True`;
  it is removed from `agents`; the team's other agents continue.
- Coincident death and timeout: the dead unit has `terminated=True` and
  `truncated=True`; the survivors have `terminated=False`, `truncated=True`.
- `agents` is cleared when the episode ends (`team_terminated or
  team_truncated`), unchanged from S1.
- The existing upstream `info` keys are preserved unchanged; the new fields
  live under the single namespaced key `snowgym_unit` so nothing is clobbered.

## 2. Action validation contract

Before any merging, every submitted action is checked with the same rule the
team environment already applies to its own actions:
`self.action_space(agent).contains(action)`, raising `ValueError` naming the
agent. That rejects non-integer or out-of-range `action_type` (including
`2.7`), wrong-shaped `target` (including a scalar) or `power`, values outside
`[-1, 1]` / `[0, 1]`, NaN, and wrong dtype exactly as the team env does. No
new numeric rules are invented and the action representation is unchanged.
Missing or extra agents still raise the existing exact-key `ValueError`.

## 3. Simulator-backed tests

Run against a real `node --import tsx snowgym/server/main.ts --port <free>`
subprocess started by a pytest fixture (skipped only if `node` is
unavailable). Each is a claim S1 made but did not establish:

1. **Same-action parity.** Unit env and team env, same seed, same 3v3
   scenario, same scripted per-step action sequence (mixed NOOP/MOVE/THROW,
   fixed), N steps: the joint `stateHash` in every step's `info` and every
   team's encoded observation are identical.
2. **Pure timeout.** A small `max_ticks` scenario reaches truncation: survivors
   have `terminated=False`, `truncated=True`; `agents` empties.
3. **Real unit death.** A seed/action script, found by search and then pinned
   in the test with a comment saying how it was found, in which a unit dies
   while the battle continues: that agent terminates exactly once, is removed,
   its slot keeps `alive: false` and is not reindexed, the rest keep acting.
4. **Determinism.** Same seed and actions twice produce identical state hashes
   (S1's determinism test compared reset tensors only).
5. **Rejected actions do not reach the simulator.** A malformed action raises
   before any `step_joint` call; the simulator's tick is unchanged.

The fake-client tests stay; new fake-client unit tests cover §1's table
(timeout, natural termination, death, coincident death and timeout) and §2's
rejection cases (`2.7`, scalar target, out-of-range, NaN, wrong-shaped power).

## 4. Pin and compatibility check

`unit_parallel_env.py` was created in S1 (`234168d`), after the R1m-S1..S6
digest pins, and no test references it besides `test_unit_parallel_env.py`
(`git grep`, checked before declaring). The complete gates below are the
final check: a digest-pin failure would appear there and turn this repair into
a new subclass instead of an in-place edit.

## 5. Verification and what falsifies this

Complete gates before commit: `npm test` (366/367, the one documented R1n-b
seed-preflight exception, unchanged), `npm run build`, Python client
`pytest`, Python training `pytest`. No new seed is reserved; the pinned
death seed is a simulator seed used by a unit test, not a research seed.
Per the seed-preflight rule, the new test file must not declare any JSON seed
band in 630000–630119.

This step is wrong if: the parity test fails (the wrapper changes physics
relative to the team env), the pure-timeout test still shows `terminated=True`,
or a malformed action reaches the simulator. Any of those is reported as a
finding, not patched around.
