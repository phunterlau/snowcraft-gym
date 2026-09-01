"""Deterministic research transforms for the SnowGym PettingZoo environment."""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .encoding import GymAction, GymObservation, encode_observation
from .parallel_env import AgentId, SnowGymParallelEnv

SEMANTIC_RASTER_CHANNELS = (
    "allies",
    "enemies",
    "friendly_projectiles",
    "hostile_projectiles",
    "obstacles",
)


class SnowGymResearchParallelEnv(ParallelEnv[AgentId, GymObservation, GymAction]):
    """Adds explicit local visibility and decision-latency profiles.

    The wrapped environment still owns transport, state hashes, rewards, and
    physics. Delays are counted in team decisions and are deterministic.
    """

    metadata = {
        "name": "SnowGym/ResearchParallelSquad-v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        environment: SnowGymParallelEnv | None = None,
        *,
        visibility_radius: float | None = None,
        action_delay_steps: int = 0,
        observation_delay_steps: int = 0,
        semantic_raster_size: int | None = None,
        **environment_kwargs: Any,
    ):
        if environment is not None and environment_kwargs:
            raise ValueError("environment kwargs cannot be combined with an environment")
        self.environment = environment or SnowGymParallelEnv(**environment_kwargs)
        self.visibility_radius = optional_positive(visibility_radius, "visibility_radius")
        self.action_delay_steps = non_negative_integer(
            action_delay_steps, "action_delay_steps"
        )
        self.observation_delay_steps = non_negative_integer(
            observation_delay_steps, "observation_delay_steps"
        )
        self.semantic_raster_size = optional_raster_size(semantic_raster_size)
        self.possible_agents = list(self.environment.possible_agents)
        self.agents: list[AgentId] = []
        self.action_spaces = self.environment.action_spaces
        self.observation_spaces = {
            agent: research_observation_space(
                self.environment.observation_spaces[agent],
                self.semantic_raster_size,
            )
            for agent in self.possible_agents
        }
        self.render_mode = self.environment.render_mode
        self._action_queues: dict[AgentId, deque[tuple[GymAction, int | None]]] = {}
        self._observation_queues: dict[AgentId, deque[GymObservation]] = {}
        self._observation_source_ticks: dict[AgentId, int] = {}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentId, GymObservation], dict[AgentId, dict[str, Any]]]:
        _, infos = self.environment.reset(seed=seed, options=options)
        self.agents = list(self.environment.agents)
        observations = self._current_observations()
        self._action_queues = {
            agent: deque(
                (noop_action(self.environment.max_team_units), None)
                for _ in range(self.action_delay_steps)
            )
            for agent in self.possible_agents
        }
        self._observation_queues = {
            agent: deque(
                clone_observation(observations[agent])
                for _ in range(self.observation_delay_steps)
            )
            for agent in self.possible_agents
        }
        self._observation_source_ticks = {
            agent: int(observations[agent]["tick"][0])
            for agent in self.possible_agents
        }
        return observations, self._research_infos(infos, observations, None)

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
        applied = {}
        applied_source_ticks: dict[AgentId, int | None] = {}
        for agent in self.agents:
            action = clone_action(actions[agent])
            queue = self._action_queues[agent]
            queue.append((action, self._observation_source_ticks[agent]))
            applied[agent], applied_source_ticks[agent] = queue.popleft()

        _, rewards, terminations, truncations, infos = self.environment.step(applied)
        current = self._current_observations()
        returned = {}
        for agent in self.possible_agents:
            queue = self._observation_queues[agent]
            queue.append(clone_observation(current[agent]))
            returned[agent] = queue.popleft()
        self._observation_source_ticks = {
            agent: int(returned[agent]["tick"][0])
            for agent in self.possible_agents
        }
        self.agents = list(self.environment.agents)
        return (
            returned,
            rewards,
            terminations,
            truncations,
            self._research_infos(infos, returned, applied_source_ticks),
        )

    def observation_space(self, agent: AgentId):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentId):
        return self.action_spaces[agent]

    def close(self) -> None:
        self.environment.close()
        self.agents = []
        self._action_queues = {}
        self._observation_queues = {}
        self._observation_source_ticks = {}

    def _current_observations(self) -> dict[AgentId, GymObservation]:
        observations = {}
        for agent in self.possible_agents:
            raw = local_observation(
                self.environment.raw_observations[agent],
                self.visibility_radius,
            )
            encoded = encode_observation(
                raw,
                self.environment.max_team_units,
                include_unit_masks=True,
            )
            if self.semantic_raster_size is not None:
                encoded["semantic_raster"] = semantic_raster(
                    raw, self.semantic_raster_size
                )
            observations[agent] = encoded
        return observations

    def _research_infos(
        self,
        infos: dict[AgentId, dict[str, Any]],
        observations: dict[AgentId, GymObservation],
        applied_source_ticks: dict[AgentId, int | None] | None,
    ) -> dict[AgentId, dict[str, Any]]:
        return {
            agent: {
                **infos[agent],
                "research": {
                    "visibilityRadius": self.visibility_radius,
                    "actionDelaySteps": self.action_delay_steps,
                    "observationDelaySteps": self.observation_delay_steps,
                    "semanticRasterSize": self.semantic_raster_size,
                    "observationSourceTick": int(observations[agent]["tick"][0]),
                    "appliedActionSourceTick": (
                        applied_source_ticks[agent]
                        if applied_source_ticks is not None
                        else None
                    ),
                },
            }
            for agent in self.possible_agents
        }


def local_observation(
    raw: dict[str, Any], visibility_radius: float | None
) -> dict[str, Any]:
    """Returns an entity-detached local view without mutating server state."""
    result = copy.deepcopy(raw)
    if visibility_radius is None:
        return result
    allies = [unit for unit in result["allies"] if unit.get("alive", False)]
    result["enemies"] = [
        unit for unit in result["enemies"] if visible(unit, allies, visibility_radius)
    ]
    self_team = result.get("selfTeam")
    result["projectiles"] = [
        projectile
        for projectile in result["projectiles"]
        if projectile.get("team") == self_team
        or visible(projectile, allies, visibility_radius)
    ]
    return result


def visible(entity: dict[str, Any], allies: list[dict[str, Any]], radius: float) -> bool:
    return any(
        math.hypot(float(entity["x"]) - float(ally["x"]), float(entity["y"]) - float(ally["y"]))
        <= radius
        for ally in allies
    )


def semantic_raster(raw: dict[str, Any], size: int) -> np.ndarray:
    """Rasterizes detached semantic entities; it never invokes the renderer."""
    validated_size = optional_raster_size(size)
    if validated_size is None:
        raise ValueError("semantic raster size is required")
    size = validated_size
    arena = raw["arena"]
    width = float(arena["width"])
    height = float(arena["height"])
    raster = np.zeros((len(SEMANTIC_RASTER_CHANNELS), size, size), dtype=np.float32)

    for unit in raw["allies"]:
        if unit.get("alive", False):
            mark_point(raster[0], unit, width, height)
    for unit in raw["enemies"]:
        if unit.get("alive", False):
            mark_point(raster[1], unit, width, height)
    self_team = raw.get("selfTeam")
    for projectile in raw["projectiles"]:
        channel = 2 if projectile.get("team") == self_team else 3
        mark_point(raster[channel], projectile, width, height)
    for obstacle in raw["obstacles"]:
        mark_obstacle(raster[4], obstacle, width, height)
    return raster


def research_observation_space(
    base: spaces.Space, raster_size: int | None
) -> spaces.Space:
    if raster_size is None:
        return base
    if not isinstance(base, spaces.Dict):
        raise TypeError("SnowGym research observations require a Dict space")
    return spaces.Dict(
        {
            **base.spaces,
            "semantic_raster": spaces.Box(
                0.0,
                1.0,
                shape=(len(SEMANTIC_RASTER_CHANNELS), raster_size, raster_size),
                dtype=np.float32,
            ),
        }
    )


def mark_point(
    channel: np.ndarray,
    entity: dict[str, Any],
    width: float,
    height: float,
) -> None:
    row, column = raster_cell(
        float(entity["x"]),
        float(entity["y"]),
        width,
        height,
        channel.shape[0],
    )
    channel[row, column] = 1.0


def mark_obstacle(
    channel: np.ndarray,
    obstacle: dict[str, Any],
    width: float,
    height: float,
) -> None:
    half_width = float(obstacle["halfWidth"])
    half_height = float(obstacle["halfHeight"])
    min_row, min_column = raster_cell(
        float(obstacle["x"]) - half_width,
        float(obstacle["y"]) - half_height,
        width,
        height,
        channel.shape[0],
    )
    max_row, max_column = raster_cell(
        float(obstacle["x"]) + half_width,
        float(obstacle["y"]) + half_height,
        width,
        height,
        channel.shape[0],
    )
    channel[
        min(min_row, max_row) : max(min_row, max_row) + 1,
        min(min_column, max_column) : max(min_column, max_column) + 1,
    ] = 1.0


def raster_cell(
    x: float, y: float, width: float, height: float, size: int
) -> tuple[int, int]:
    column = int(np.floor((x / width + 0.5) * size))
    row = int(np.floor((y / height + 0.5) * size))
    return (
        int(np.clip(row, 0, size - 1)),
        int(np.clip(column, 0, size - 1)),
    )


def noop_action(max_team_units: int) -> GymAction:
    return {
        "action_type": np.zeros(max_team_units, dtype=np.int64),
        "target": np.zeros((max_team_units, 2), dtype=np.float32),
        "power": np.zeros(max_team_units, dtype=np.float32),
    }


def clone_action(action: GymAction) -> GymAction:
    return {key: np.array(value, copy=True) for key, value in action.items()}


def clone_observation(observation: GymObservation) -> GymObservation:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def non_negative_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def optional_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def optional_raster_size(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 8 <= value <= 128:
        raise ValueError("semantic_raster_size must be an integer in [8, 128]")
    return value
