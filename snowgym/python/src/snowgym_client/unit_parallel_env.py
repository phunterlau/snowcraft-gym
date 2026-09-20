"""M8-S1: one PettingZoo agent per unit, wrapping the team-level parallel env.

No local-observation or latency restriction yet (M8's own checklist puts those
after "begin with global observations"); every living unit-agent on a team
receives that team's full team-level observation unchanged. See
`reviews/m8_s1_declaration.md`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .encoding import ACTION_NOOP, ACTION_TYPE_COUNT, make_observation_space
from .parallel_env import AgentId, SnowGymParallelEnv

UnitAgentId = str


def unit_agents(blue_units: int, red_units: int) -> list[UnitAgentId]:
    return [f"blue-{i}" for i in range(blue_units)] + [f"red-{i}" for i in range(red_units)]


def split_unit_agent(agent: UnitAgentId) -> tuple[AgentId, int]:
    team, _, slot = agent.rpartition("-")
    if team not in ("blue", "red"):
        raise ValueError(f"not a unit-agent id: {agent!r}")
    return team, int(slot)


class SnowGymUnitParallelEnv(ParallelEnv[UnitAgentId, dict[str, Any], dict[str, Any]]):
    """Wraps a `SnowGymParallelEnv` by composition (not a subclass — that file is
    read, never edited). Each living unit is its own PettingZoo agent; the
    underlying team-level step/reset contract, observation encoding, and action
    encoding are reused unchanged."""

    metadata = {
        "name": "SnowGym/UnitParallelSquad-v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        environment: SnowGymParallelEnv | None = None,
        *,
        render_mode: None = None,
        **environment_kwargs: Any,
    ):
        if environment is not None and environment_kwargs:
            raise ValueError("environment kwargs cannot be combined with an environment")
        if render_mode is not None:
            raise ValueError("SnowGym unit-parallel environments support render_mode=None only")
        self.render_mode = render_mode
        self.environment = environment or SnowGymParallelEnv(**environment_kwargs)
        # SnowGymParallelEnv validates and stores the roster synchronously at
        # construction, before any reset/server round-trip; there is no public
        # accessor for it (only `max_team_units`, the *capacity*, is public), so
        # this reads the same private config the constructor already validated,
        # rather than duplicating and risking divergence from a caller-supplied
        # `environment=`.
        config = self.environment._scenario_config
        self._blue_units, self._red_units = int(config["blueUnits"]), int(config["redUnits"])
        self.possible_agents: list[UnitAgentId] = unit_agents(self._blue_units, self._red_units)

        def unit_action_space() -> spaces.Dict:
            return spaces.Dict(
                {
                    "action_type": spaces.Discrete(ACTION_TYPE_COUNT),
                    "target": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    "power": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
                }
            )

        # A fresh space instance per agent (not one shared object across every
        # unit on a team) — matching how SnowGymParallelEnv itself builds one
        # `make_observation_space(...)` call per agent, not a single shared
        # instance, so per-agent `.seed()` calls (e.g. from the PettingZoo
        # checker) cannot interfere with each other through a shared object.
        self.action_spaces = {agent: unit_action_space() for agent in self.possible_agents}
        self.observation_spaces = {
            agent: make_observation_space(self.environment.max_team_units, include_unit_masks=True)
            for agent in self.possible_agents
        }
        self.agents: list[UnitAgentId] = []
        self.np_random, _ = seeding.np_random(None)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[UnitAgentId, dict[str, Any]], dict[UnitAgentId, dict[str, Any]]]:
        self.np_random, _ = seeding.np_random(seed)
        team_observations, team_infos = self.environment.reset(seed, options)
        # `possible_agents` is fixed at construction from the roster the wrapped env
        # validated then (declaration §2). A reset `options["scenario"]` that changes
        # blueUnits/redUnits would silently desync possible_agents from the actual
        # roster; fail loud instead of producing a wrong agent set. Reads each team's
        # own "allies" list, the same path `_unit_alive` uses — not blue's "enemies"
        # list for the red count — so there is exactly one source of truth for roster
        # size and slot ordering, not two paths that happen to agree today.
        actual_blue = len(self.environment.raw_observations["blue"]["allies"])
        actual_red = len(self.environment.raw_observations["red"]["allies"])
        if (actual_blue, actual_red) != (self._blue_units, self._red_units):
            raise ValueError(
                "reset() roster does not match the roster this environment was constructed "
                f"with ({self._blue_units}v{self._red_units}); a scenario override changing "
                f"unit counts is not supported (got {actual_blue}v{actual_red})"
            )
        self.agents = [agent for agent in self.possible_agents if self._unit_alive(agent)]
        observations = {agent: team_observations[split_unit_agent(agent)[0]] for agent in self.agents}
        infos = {agent: dict(team_infos[split_unit_agent(agent)[0]]) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[UnitAgentId, dict[str, Any]]
    ) -> tuple[
        dict[UnitAgentId, dict[str, Any]],
        dict[UnitAgentId, float],
        dict[UnitAgentId, bool],
        dict[UnitAgentId, bool],
        dict[UnitAgentId, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("reset() must be called before step()")
        if set(actions) != set(self.agents):
            raise ValueError(f"actions must contain exactly: {', '.join(self.agents)}")
        active_before = list(self.agents)
        team_actions = self._merge_team_actions(actions)
        team_observations, team_rewards, team_terms, team_truncs, team_infos = self.environment.step(team_actions)
        episode_over = any(team_terms.values()) or any(team_truncs.values())

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in active_before:
            team, _ = split_unit_agent(agent)
            unit_alive = self._unit_alive(agent)
            observations[agent] = team_observations[team]
            rewards[agent] = team_rewards[team]
            terminations[agent] = episode_over or not unit_alive
            truncations[agent] = team_truncs[team]
            infos[agent] = dict(team_infos[team])

        self.agents = [] if episode_over else [a for a in self.agents if self._unit_alive(a)]
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: UnitAgentId):
        return self.observation_spaces[agent]

    def action_space(self, agent: UnitAgentId):
        return self.action_spaces[agent]

    def close(self) -> None:
        self.environment.close()
        self.agents = []

    def _unit_alive(self, agent: UnitAgentId) -> bool:
        team, slot = split_unit_agent(agent)
        return bool(self.environment.raw_observations[team]["allies"][slot]["alive"])

    def _merge_team_actions(self, actions: dict[UnitAgentId, dict[str, Any]]) -> dict[AgentId, dict[str, Any]]:
        capacity = self.environment.max_team_units
        merged: dict[AgentId, dict[str, Any]] = {}
        for team, roster in (("blue", self._blue_units), ("red", self._red_units)):
            action_type = np.full(capacity, ACTION_NOOP, dtype=np.int64)
            target = np.zeros((capacity, 2), dtype=np.float32)
            power = np.zeros(capacity, dtype=np.float32)
            for slot in range(roster):
                agent = f"{team}-{slot}"
                submitted = actions.get(agent)
                if submitted is None:
                    # Dead this step, no agent submitted — stays ACTION_NOOP, which
                    # encode_action already applies to a dead unit's slot regardless
                    # of its value (encoding.py's encode_action).
                    continue
                action_type[slot] = int(submitted["action_type"])
                target[slot] = np.asarray(submitted["target"], dtype=np.float32)
                power[slot] = float(submitted["power"])
            merged[team] = {"action_type": action_type, "target": target, "power": power}
        return merged
