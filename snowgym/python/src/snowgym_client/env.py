"""Gymnasium environment backed by the SnowGym TypeScript server."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .client import SnowGymClient, SnowGymHttpClient
from .encoding import (
    MAX_CONFIGURABLE_TEAM_UNITS,
    MAX_TEAM_UNITS,
    GymAction,
    GymObservation,
    encode_action,
    encode_observation,
    make_action_space,
    make_observation_space,
)


class SnowGymEnv(gym.Env[GymObservation, GymAction]):
    """Single-policy blue-squad environment; the server controls the red team."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8787",
        timeout: float = 10.0,
        client: SnowGymClient | None = None,
        max_team_units: int = MAX_TEAM_UNITS,
        configurable: bool = False,
        blue_units: int = 3,
        red_units: int = 3,
        arena_width: float = 40.0,
        arena_height: float = 30.0,
        max_ticks: int = 60 * 180,
        decision_hz: int = 10,
        red_difficulty: str = "normal",
        red_controller: str = "scripted",
        map: str | None = None,
        render_mode: None = None,
    ):
        if render_mode is not None:
            raise ValueError("SnowGym squad environments support render_mode=None only")
        self.render_mode = render_mode
        self._max_team_units = validate_capacity(max_team_units)
        self._configurable = configurable
        self._map = map
        self._scenario_config = validate_scenario_config(
            {
                "blueUnits": blue_units,
                "redUnits": red_units,
                "arenaWidth": arena_width,
                "arenaHeight": arena_height,
                "maxTicks": max_ticks,
                "decisionHz": decision_hz,
                "redDifficulty": red_difficulty,
                "redController": red_controller,
            },
            self._max_team_units,
        )
        if not configurable and self._scenario_config != default_scenario_config():
            raise ValueError("custom squad configuration requires a configurable SnowGym squad environment")
        self.action_space = make_action_space(self._max_team_units)
        self.observation_space = make_observation_space(
            self._max_team_units,
            include_unit_masks=configurable,
        )
        self._client = client or SnowGymHttpClient(server_url, timeout)
        self._raw_observation: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[GymObservation, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            unknown = sorted(set(options) - {"server_seed", "scenario"})
            if unknown:
                raise ValueError(f"unsupported reset options: {unknown}")
        server_seed = options.get("server_seed") if options else None
        if server_seed is None:
            server_seed = seed
        if server_seed is None:
            server_seed = int(self.np_random.integers(0, 2**32))

        scenario_override = options.get("scenario") if options else None
        if scenario_override is not None and not self._configurable:
            raise ValueError("scenario reset options require a configurable SnowGym squad environment")
        scenario = None
        if self._configurable:
            if scenario_override is not None and not isinstance(scenario_override, dict):
                raise ValueError("reset option scenario must be an object")
            scenario = dict(scenario_override or {})
            map_id = scenario.pop("map", self._map)
            if map_id is not None:
                if not isinstance(map_id, str):
                    raise ValueError("scenario.map must be a string map id")
                # Map fixes terrain + rosters; only tuning knobs ride along.
                scenario = {
                    key: scenario.get(key, self._scenario_config[key])
                    for key in ("decisionHz", "redDifficulty", "redController", "maxTicks")
                }
                scenario["map"] = map_id
            else:
                scenario = validate_scenario_config(
                    self._scenario_config | scenario,
                    self._max_team_units,
                )

        payload = self._client.reset(int(server_seed), scenario)
        raw = require_payload_observation(payload)
        status = require_payload_info(payload, "status")
        self._raw_observation = raw
        return self._encode_observation(raw), dict(status)

    def step(
        self, action: GymAction
    ) -> tuple[GymObservation, float, bool, bool, dict[str, Any]]:
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action is outside SnowGym action_space: {action!r}")

        semantic_action = encode_action(action, self._raw_observation, self._max_team_units)
        return self._consume_step(self._client.step(semantic_action))

    def step_scripted(
        self,
    ) -> tuple[GymObservation, float, bool, bool, dict[str, Any]]:
        """Advance one decision using the server's reference blue policy."""
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before step_scripted()")
        return self._consume_step(self._client.step_scripted())

    def _consume_step(
        self, payload: dict[str, Any]
    ) -> tuple[GymObservation, float, bool, bool, dict[str, Any]]:
        raw = require_payload_observation(payload)
        info = require_payload_info(payload, "info")
        self._raw_observation = raw
        return (
            self._encode_observation(raw),
            float(payload.get("reward", 0.0)),
            bool(payload.get("terminated", False)),
            bool(payload.get("truncated", False)),
            dict(info),
        )

    @property
    def raw_observation(self) -> dict[str, Any] | None:
        """Last detached JSON observation, useful for debugging and policies."""
        return self._raw_observation

    def close(self) -> None:
        self._raw_observation = None

    def _encode_observation(self, raw: dict[str, Any]) -> GymObservation:
        return encode_observation(
            raw,
            self._max_team_units,
            include_unit_masks=self._configurable,
        )


def require_payload_observation(payload: dict[str, Any]) -> dict[str, Any]:
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("SnowGym response is missing observation")
    return observation


def require_payload_info(payload: dict[str, Any], key: str) -> dict[str, Any]:
    info = payload.get(key)
    if not isinstance(info, dict):
        raise ValueError(f"SnowGym response is missing {key}")
    return info


def default_scenario_config() -> dict[str, Any]:
    return {
        "blueUnits": 3,
        "redUnits": 3,
        "arenaWidth": 40.0,
        "arenaHeight": 30.0,
        "maxTicks": 60 * 180,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "scripted",
    }


def validate_capacity(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("max_team_units must be an integer")
    if value < 1 or value > MAX_CONFIGURABLE_TEAM_UNITS:
        raise ValueError(
            f"max_team_units must be in [1, {MAX_CONFIGURABLE_TEAM_UNITS}]"
        )
    return value


def validate_scenario_config(
    value: dict[str, Any], max_team_units: int
) -> dict[str, Any]:
    allowed = set(default_scenario_config()) | {"blueSpawns", "redSpawns"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario fields: {unknown}")

    result = dict(value)
    for key in ("blueUnits", "redUnits"):
        count = result.get(key)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= max_team_units:
            raise ValueError(f"{key} must be in [1, {max_team_units}]")
    for key in ("arenaWidth", "arenaHeight"):
        size = result.get(key)
        if not isinstance(size, int | float) or isinstance(size, bool) or not 12 <= size <= 120:
            raise ValueError(f"{key} must be in [12, 120]")
    max_ticks = result.get("maxTicks")
    if not isinstance(max_ticks, int) or isinstance(max_ticks, bool) or max_ticks <= 0:
        raise ValueError("maxTicks must be a positive integer")
    decision_hz = result.get("decisionHz")
    if (
        not isinstance(decision_hz, int)
        or isinstance(decision_hz, bool)
        or decision_hz <= 0
        or decision_hz > 60
        or 60 % decision_hz != 0
    ):
        raise ValueError("decisionHz must be a positive divisor of 60")
    if result.get("redDifficulty") not in {"easy", "normal", "hard"}:
        raise ValueError("redDifficulty must be easy, normal, or hard")
    if result.get("redController") not in {"scripted", "random"}:
        raise ValueError("redController must be scripted or random")
    for key, count_key in (("blueSpawns", "blueUnits"), ("redSpawns", "redUnits")):
        if key in result:
            spawns = result[key]
            if not isinstance(spawns, list) or len(spawns) != result[count_key]:
                raise ValueError(f"{key} must contain {result[count_key]} positions")
    return result
