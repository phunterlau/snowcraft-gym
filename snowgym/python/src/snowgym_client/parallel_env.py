"""PettingZoo ParallelEnv backed by the authoritative SnowGym server."""

from __future__ import annotations

from typing import Any

from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .client import SnowGymClient, SnowGymHttpClient
from .encoding import (
    MAX_CONFIGURABLE_TEAM_UNITS,
    GymAction,
    GymObservation,
    encode_action,
    encode_observation,
    make_action_space,
    make_observation_space,
)
from .env import require_payload_info, require_state_hash, validate_scenario_config

AgentId = str


class SnowGymParallelEnv(ParallelEnv[AgentId, GymObservation, GymAction]):
    """Two team-level agents submit simultaneous actions to one simulation."""

    metadata = {
        "name": "SnowGym/ParallelSquad-v0",
        "render_modes": [],
        "is_parallelizable": True,
    }
    possible_agents = ["blue", "red"]

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8787",
        timeout: float = 10.0,
        client: SnowGymClient | None = None,
        max_team_units: int = MAX_CONFIGURABLE_TEAM_UNITS,
        blue_units: int = 3,
        red_units: int = 3,
        arena_width: float = 40.0,
        arena_height: float = 30.0,
        max_ticks: int = 60 * 180,
        decision_hz: int = 10,
        map: str | None = None,
        render_mode: None = None,
    ):
        if render_mode is not None:
            raise ValueError("SnowGym parallel environments support render_mode=None only")
        self.render_mode = render_mode
        self._max_team_units = max_team_units
        self._map = map
        self._scenario_config = validate_scenario_config(
            {
                "blueUnits": blue_units,
                "redUnits": red_units,
                "arenaWidth": arena_width,
                "arenaHeight": arena_height,
                "maxTicks": max_ticks,
                "decisionHz": decision_hz,
                "redDifficulty": "normal",
                "redController": "scripted",
            },
            max_team_units,
        )
        action_space = make_action_space(max_team_units)
        observation_space = make_observation_space(max_team_units, include_unit_masks=True)
        self.action_spaces = {agent: action_space for agent in self.possible_agents}
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        self._client = client or SnowGymHttpClient(server_url, timeout)
        self._raw_observations: dict[AgentId, dict[str, Any]] = {}
        self._state_hash: str | None = None
        self.agents: list[AgentId] = []
        self.np_random, _ = seeding.np_random(None)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentId, GymObservation], dict[AgentId, dict[str, Any]]]:
        self.np_random, _ = seeding.np_random(seed)
        server_seed = options.get("server_seed") if options else None
        if server_seed is None:
            server_seed = seed
        if server_seed is None:
            server_seed = int(self.np_random.integers(0, 2**32))

        scenario_override = options.get("scenario") if options else None
        if scenario_override is not None and not isinstance(scenario_override, dict):
            raise ValueError("reset option scenario must be an object")
        scenario = self._reset_scenario(scenario_override or {})
        payload = self._client.reset(int(server_seed), scenario)
        self._raw_observations = require_team_observations(payload)
        status = require_payload_info(payload, "status")
        self._state_hash = require_state_hash(status)
        self.agents = list(self.possible_agents)
        observations = self._encode_observations()
        infos = {agent: dict(status) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[AgentId, GymAction]
    ) -> tuple[
        dict[AgentId, GymObservation],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("reset() must be called before step()")
        if set(actions) != set(self.agents):
            raise ValueError(f"actions must contain exactly: {', '.join(self.agents)}")
        semantic = {}
        for agent in self.agents:
            action = actions[agent]
            if not self.action_space(agent).contains(action):
                raise ValueError(f"action for {agent} is outside SnowGym action_space")
            semantic[agent] = encode_action(
                action,
                self._raw_observations[agent],
                self._max_team_units,
            )

        payload = self._client.step_joint(
            semantic,
            expected_state_hash=self._state_hash,
        )
        self._raw_observations = require_team_observations(payload)
        info = require_payload_info(payload, "info")
        self._state_hash = require_state_hash(info)
        observations = self._encode_observations()
        rewards = require_agent_scalars(payload, "rewards", float)
        terminations = require_agent_scalars(payload, "terminations", bool)
        truncations = require_agent_scalars(payload, "truncations", bool)
        infos = {agent: dict(info) for agent in self.agents}
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: AgentId):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentId):
        return self.action_spaces[agent]

    @property
    def raw_observations(self) -> dict[AgentId, dict[str, Any]]:
        return self._raw_observations

    def close(self) -> None:
        self.agents = []
        self._raw_observations = {}
        self._state_hash = None

    def _encode_observations(self) -> dict[AgentId, GymObservation]:
        return {
            agent: encode_observation(
                self._raw_observations[agent],
                self._max_team_units,
                include_unit_masks=True,
            )
            for agent in self.possible_agents
        }

    def _reset_scenario(self, override: dict[str, Any]) -> dict[str, Any]:
        scenario = dict(override)
        map_id = scenario.pop("map", self._map)
        if map_id is not None:
            if not isinstance(map_id, str):
                raise ValueError("scenario.map must be a string map id")
            result = {
                key: scenario.get(key, self._scenario_config[key])
                for key in ("decisionHz", "maxTicks")
            }
            for key in ("blueUnits", "redUnits"):
                result[key] = scenario.get(key, self._scenario_config[key])
            result["map"] = map_id
            return result
        return validate_scenario_config(
            self._scenario_config | scenario,
            self._max_team_units,
        )


def require_team_observations(payload: dict[str, Any]) -> dict[AgentId, dict[str, Any]]:
    observations = payload.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("SnowGym response is missing observations")
    result = {}
    for agent in SnowGymParallelEnv.possible_agents:
        observation = observations.get(agent)
        if not isinstance(observation, dict):
            raise ValueError(f"SnowGym response is missing observation for {agent}")
        result[agent] = observation
    return result


def require_agent_scalars(
    payload: dict[str, Any], key: str, value_type: type
) -> dict[AgentId, Any]:
    values = payload.get(key)
    if not isinstance(values, dict):
        raise ValueError(f"SnowGym response is missing {key}")
    result = {}
    for agent in SnowGymParallelEnv.possible_agents:
        value = values.get(agent)
        if value_type is float:
            valid = isinstance(value, int | float) and not isinstance(value, bool)
        else:
            valid = isinstance(value, value_type)
        if not valid:
            raise ValueError(f"SnowGym response has invalid {key}.{agent}")
        result[agent] = value_type(value)
    return result
