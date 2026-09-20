# M8-S1 — `SnowGymUnitParallelEnv`: one PettingZoo agent per unit, global observations, no training

Declared 2026-09-20. This is an infrastructure step, not a collection
experiment: no policy is trained, no simulator-decision budget is spent
beyond ordinary test runs, no seed band is reserved, and no run archive is
sealed. The declare → implement shape below is scaled down from the R1n
track's declare → implement → collect → archive discipline accordingly —
that scaling-down is a decision, not an oversight (§5).

## 0. Why this exists, and the gate it proceeds under

PLAN.md's M8 section (`### M8 — unit-level CTDE / MAPPO`) lists as its
first unchecked item: *"Add `SnowGymUnitParallelEnv` as a new PettingZoo
environment; retain the existing team-level environment and version."*
This is that item.

**Explicit gate note:** PLAN.md's M7c section states *"Only then may M8
local CTDE and M9 online commander comparisons advance"*, and M7c
(full-fight fixed-plan composition) is not complete — its checklist is
still unchecked. This declaration proceeds under a different, later
decision: the user's 2026-09-14 red-agent roadmap agreement (recorded in
project memory) that *"M8 (unit-level MAPPO) lands regardless"* of M7c,
validated first against a known-good baseline on blue. That is a scope
decision for this research track, not a claim that M7c's own exit
criterion has been met — PLAN.md's M7c section is unchanged by this work,
and M7c's own checklist remains the gate for whatever depends on it
specifically (M9's commander comparisons, the roster/composition mission
gates). This declaration's M8 entry in PLAN.md states the same override
explicitly (§6).

R1n-b through R1n-i (all 1v1 Engage, `ROSTER = 1` hardcoded in the
digest-pinned `full_authority_train.py`) motivate M8 directly: the
existing `FullAuthorityPolicyV1` actor already computes per-unit action
logits from a shared team-observation tensor (`features()`'s pooled
egocentric encoding, per unit) and a single team-scalar centralized
critic (`RoleAwareCentralCritic`/`OptionCentralCritic`) already exists —
but it has only ever been exercised at roster size 1, as one monolithic
per-team joint action, trained as a single-agent PPO problem with one
team-level reward. What does not exist is a contract that treats each
unit as its own decentralized-execution agent — the thing M8's own exit
criterion needs ("the command-conditioned shared actor remains valid
under local observations," which presupposes per-unit action/observation
boundaries the current team-level env does not have).

**M8-S1 decides nothing about R1 qualification, trains nothing, and
authorizes no MAPPO update.** It is one new environment class plus its
own test suite.

## 1. What already exists (reused, not rebuilt)

Verified directly from source before design, not assumed:

- **Multi-unit rosters already work at the environment/encoding layer.**
  `MAX_CONFIGURABLE_TEAM_UNITS = 10`; `SnowGym/Squad-v2` is already
  registered at `max_team_units=10`, and `test_ten_unit_environment_
  supports_a_10v10_roster` already passes. 3v3/5v5 need no new
  environment capacity, only a new roster configuration.
- **A team-level PettingZoo env already exists and is tested.**
  `SnowGymParallelEnv` (`parallel_env.py`) exposes `possible_agents =
  ["blue", "red"]`, one joint action per team per step, and already
  passes `pettingzoo.test.parallel_api_test`
  (`test_parallel_environment_passes_pettingzoo_checker`).
- **The local-observation/latency axis M8's checklist names is already
  built, at team level.** `SnowGymResearchParallelEnv` (`research_env.py`)
  wraps `SnowGymParallelEnv` and already implements `visibility_radius`,
  `action_delay_steps`, and `observation_delay_steps` — exactly M8's
  "global → local → local+latency" progression, just not yet at
  per-unit granularity. Reusing this pattern (not this class directly, it
  wraps the team-level env) is the natural next M8 stage after this one;
  out of scope here (§3).
- **`encode_action` already treats a dead unit's submitted action as a
  no-op regardless of its value** (`if not bool(unit.get("alive", False))
  or action_type == ACTION_NOOP: ... {"type": "noop", ...}`,
  `encoding.py:234`). A per-unit-to-team-action merge does not need
  special-case dead-unit handling beyond filling an arbitrary placeholder
  for slots without a live agent.
- **Unit slot identity is stable for an entire episode.** Verified in
  `observeWorld` (`observations/Observation.ts:112-136`): `units =
  world.players.map(...).sort((a, b) => a.id - b.id)`, with no `alive`
  filter — dead players remain in the sorted list at their fixed
  position, marked `alive: false`, never removed or reindexed (consistent
  with the no-respawn scenario convention in PLAN.md's validated-state
  table). `encode_observation`'s `for index, unit in enumerate(allies_raw)`
  therefore assigns the same slot index to the same player for the whole
  episode. A PettingZoo agent id of `f"{team}-{slot index}"` is stable by
  construction — no separate id-tracking mechanism is needed.
- **The Engage option is not hardcoded to one assigned unit or one
  enemy target.** `FrozenEngageTracker` (`engage_v1.py`) already sums
  target health over `self.activated_target_ids` (plural) and counts
  `living = sum(u["alive"] and u["id"] in self.assigned_ids for u in
  observation["allies"])` (plural `assigned_ids`) — this bookkeeping is
  already N-unit-general. `ROSTER = 1` is a training-config choice in a
  separate, pinned file, not a constraint the option/tracker layer itself
  imposes. M8-S1 does not touch the option layer at all (§3), but this
  means a later M8 stage reusing Engage for 3v3 does not need a new
  option spec first.

## 2. Contract (declaration for `unit_parallel_env.py`, new file)

`SnowGymUnitParallelEnv` wraps a `SnowGymParallelEnv` instance by
composition (`self.environment = environment or SnowGymParallelEnv(...)`,
the same pattern `SnowGymResearchParallelEnv` already uses) — it does not
subclass or edit it. Both existing environment files (`parallel_env.py`,
`encoding.py`, `research_env.py`) are read, not modified.

- **`possible_agents`:** `f"{team}-{slot}"` for `slot in range(blue_units)`
  and `slot in range(red_units)`, both teams symmetric — the env does not
  bake in which team is "the agent"; that is a training-harness decision,
  matching how `SnowGymParallelEnv` already exposes both teams
  symmetrically and opponent control is decided outside the env.
- **`self.agents`:** starts as `list(possible_agents)` on reset. A unit's
  agent is removed from `self.agents` the step *after* its `alive` flag
  first reads `False` — its final observation/reward/`terminated=True`
  for that step is still delivered (the PettingZoo contract), and it does
  not submit an action on any later step.
- **Observations (first cut — "global," per M8's own checklist order):**
  every living unit-agent on a team receives that team's full
  `encode_observation` output unchanged — the same team-level observation
  `SnowGymParallelEnv` already produces, broadcast identically to every
  living agent on that team. No new observation encoding. Per-unit
  egocentric framing (what `FullAuthorityPolicyV1.pair_features` already
  computes from a team tensor) stays a model concern, not an env concern,
  matching the existing architecture's own division of labor.
- **Actions:** a new, genuinely per-unit action space — one
  `action_type` (`Discrete(ACTION_TYPE_COUNT)`), one `target`
  (`Box(-1,1,shape=(2,))`), one `power` (`Box(0,1,shape=(1,))`) per agent,
  not the team-shaped `MultiDiscrete`/`Box` arrays `make_action_space`
  builds. `step()` merges the current living agents' individual actions
  into one team-level `GymAction` per team (dead/absent slots filled with
  `ACTION_NOOP`, which `encode_action` already treats identically to a
  dead unit's actual submission) and calls the wrapped
  `SnowGymParallelEnv.step()` once per team, unchanged.
- **Rewards:** every living unit-agent on a team receives that team's
  existing scalar team reward, unchanged, for this first cut — the
  standard MAPPO default (shared team reward, individual observations)
  and the honest starting point given the server emits no per-unit
  reward signal today. Inventing per-unit credit is explicitly out of
  scope (§3).
- **Terminations/truncations:** an agent's `terminated` is `True` on the
  step its `alive` flag first reads `False`; `truncated` mirrors the
  wrapped team env's truncation for every remaining living agent on that
  team simultaneously (episode-level horizon/timeout, not a per-unit
  concept).

## 3. Explicit non-goals for M8-S1

- No local-visibility or decision-latency restriction (M8's checklist
  next items) — this stage is "global observations" only.
- No new option spec, no Engage/scenario changes, no training config.
- No per-unit reward shaping or credit assignment beyond the shared team
  reward.
- No training, no MAPPO update, no policy, no PPO loss change.
- No R1 qualification claim.

## 4. Exit criterion

- `pettingzoo.test.parallel_api_test` passes against
  `SnowGymUnitParallelEnv`, for at least one roster config with unit
  count > 1 per team (the same checker `test_parallel_environment_
  passes_pettingzoo_checker` already runs against the team-level env,
  `test_env.py:170`).
- A determinism check: two `reset()` calls with the same seed produce
  identical per-agent observations, and identical action sequences from
  identical resets produce identical trajectories (matching the existing
  environment's `nondeterministic=False` contract).
- A live death-and-removal check: in a roster where a unit can die within
  the episode horizon, its agent id leaves `self.agents` on schedule and
  no further action is required from it, without breaking the checker or
  the remaining agents' stepping.

No budget, no seed band, no sealed manifest — this stage collects no
data and trains nothing; there is nothing to bound or seal.

**Outcome (2026-09-20): all three pass.** `parallel_api_test` passes
against a 3v3 `SnowGymUnitParallelEnv` (`test_unit_parallel_environment_
passes_pettingzoo_checker`). Reset is deterministic for a fixed seed
(`test_reset_is_deterministic_for_the_same_seed`). A unit-death test
(`test_a_dying_unit_leaves_agents_and_is_terminated_exactly_once`)
confirms an agent is removed on schedule and the next `step()` neither
requires nor accepts an action from it. A team-level-termination test
(added during pre-commit advisor review, §6 A1) confirms a termination
unrelated to any single unit's death still ends every remaining agent on
both teams in the same step, not just the terminating team's own units.
10 new tests total, all passing; see §6 for two implementation
corrections made before commit.

## 5. Verification plan

- New tests in `snowgym/python/tests/test_unit_parallel_env.py`,
  parallel to the existing `test_env.py`/`parallel_env.py` coverage.
- Full python client suite, full training suite, `npm run build`,
  `npm test` (documented R1n-b exception only — no seeds touched by this
  stage at all).

## 6. Amendments

**A1 (2026-09-20, before commit, advisor review):** two corrections to
`unit_parallel_env.py`, made before any commit:

1. The reset roster-mismatch guard (§2) originally read the red unit
   count from `raw_observations["blue"]["enemies"]`, while `_unit_alive`
   separately read red's own aliveness from `raw_observations["red"]
   ["allies"]` — two different paths to the same fact, correct only
   because blue's `enemies` list and red's `allies` list happen to be
   filtered from the same globally-id-sorted list in the same order,
   which nothing enforced or tested. Both now read
   `raw_observations[team]["allies"]` — one source of truth for roster
   size and slot ordering.
2. `episode_over` (§2, any team's termination or truncation) is not the
   same as any *single* unit's death — a team-level termination should
   end every remaining agent on *both* teams in that step, not just the
   terminating team's own. The implementation already did this
   correctly, but no test exercised it (the fake client used through
   implementation never terminated). Added
   `test_a_team_level_termination_ends_every_remaining_agent_on_both_teams`.

Also: `split_unit_agent` now validates the team half of an agent id
(`rpartition("-")` alone would silently accept a malformed id like
`"blue-1-2"` as `("blue-1", 2)`); a dead comment inside
`_merge_team_actions` that sat after its `return`/`continue` statement
(unreachable-looking) was moved above the branch it explains.
