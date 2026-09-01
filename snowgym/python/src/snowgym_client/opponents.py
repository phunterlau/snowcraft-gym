"""Opponent policy adapters and a single-team wrapper over joint SnowGym play."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .encoding import ACTION_NOOP, GymAction, GymObservation, make_action_space
from .parallel_env import AgentId, SnowGymParallelEnv
from .research_env import SnowGymResearchParallelEnv

REMOTE_OBSERVATION_FORMAT = "snowgym.opponent-observation.v0"
REMOTE_ACTION_FORMAT = "snowgym.opponent-action.v0"


class OpponentPolicy(Protocol):
    """Policy boundary used by :class:`SnowGymSingleTeamEnv`."""

    def reset(self, *, seed: int, agent: AgentId) -> None: ...

    def act(
        self, observation: GymObservation, info: dict[str, Any]
    ) -> GymAction: ...


class RemoteOpponentClient(Protocol):
    """Provider-neutral transport boundary for a stateless remote policy."""

    def act(self, request: dict[str, Any]) -> dict[str, Any]: ...


class NoopOpponent:
    def reset(self, *, seed: int, agent: AgentId) -> None:
        del seed, agent

    def act(self, observation: GymObservation, info: dict[str, Any]) -> GymAction:
        del info
        return noop_action(len(observation["ally_mask"]))


class MaskedRandomOpponent:
    """Deterministic baseline that samples only actions visible as legal."""

    def __init__(self) -> None:
        self._generator: np.random.Generator | None = None

    def reset(self, *, seed: int, agent: AgentId) -> None:
        agent_index = SnowGymParallelEnv.possible_agents.index(agent)
        self._generator = np.random.default_rng(
            np.random.SeedSequence([seed, agent_index, 0x534E4F57])
        )

    def act(self, observation: GymObservation, info: dict[str, Any]) -> GymAction:
        del info
        if self._generator is None:
            raise RuntimeError("opponent reset() must be called before act()")
        return masked_random_action(observation, self._generator)


class LearnedOpponent:
    """Adapts an in-process learned-policy callable to the opponent contract."""

    def __init__(
        self,
        policy: Callable[[GymObservation, dict[str, Any]], GymAction],
    ) -> None:
        self.policy = policy

    def reset(self, *, seed: int, agent: AgentId) -> None:
        del seed, agent

    def act(self, observation: GymObservation, info: dict[str, Any]) -> GymAction:
        return self.policy(clone_observation(observation), copy.deepcopy(info))


class RemoteOpponent:
    """Converts fixed tensors to versioned JSON and validates the response shape."""

    def __init__(self, client: RemoteOpponentClient) -> None:
        self.client = client
        self._seed: int | None = None
        self._agent: AgentId | None = None
        self._decision = 0

    def reset(self, *, seed: int, agent: AgentId) -> None:
        self._seed = seed
        self._agent = agent
        self._decision = 0

    def act(self, observation: GymObservation, info: dict[str, Any]) -> GymAction:
        if self._seed is None or self._agent is None:
            raise RuntimeError("opponent reset() must be called before act()")
        request = {
            "format": REMOTE_OBSERVATION_FORMAT,
            "agent": self._agent,
            "seed": self._seed,
            "decision": self._decision,
            "tick": int(observation["tick"][0]),
            "stateHash": info.get("stateHash"),
            "observation": json_value(observation),
        }
        response = self.client.act(request)
        if not isinstance(response, dict) or response.get("format") != REMOTE_ACTION_FORMAT:
            raise ValueError(f"remote opponent response format must be {REMOTE_ACTION_FORMAT}")
        action = response.get("action")
        if not isinstance(action, dict) or set(action) != {"action_type", "target", "power"}:
            raise ValueError("remote opponent action must contain action_type, target, and power")
        converted = {
            "action_type": np.asarray(action["action_type"], dtype=np.int64),
            "target": np.asarray(action["target"], dtype=np.float32),
            "power": np.asarray(action["power"], dtype=np.float32),
        }
        if not make_action_space(len(observation["ally_mask"])).contains(converted):
            raise ValueError("remote opponent action is outside SnowGym action_space")
        self._decision += 1
        return converted


class SnowGymSingleTeamEnv(gym.Env[GymObservation, GymAction]):
    """Gym wrapper controlling one team while an adapter controls the other."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: SnowGymParallelEnv | SnowGymResearchParallelEnv | None = None,
        *,
        controlled_agent: AgentId = "blue",
        opponent: OpponentPolicy | None = None,
        **environment_kwargs: Any,
    ) -> None:
        if environment is not None and environment_kwargs:
            raise ValueError("environment kwargs cannot be combined with an environment")
        self.environment = environment or SnowGymParallelEnv(**environment_kwargs)
        if controlled_agent not in self.environment.possible_agents:
            raise ValueError("controlled_agent must be blue or red")
        self.controlled_agent = controlled_agent
        self.opponent_agent = next(
            agent
            for agent in self.environment.possible_agents
            if agent != controlled_agent
        )
        self.opponent = opponent or MaskedRandomOpponent()
        self.action_space = self.environment.action_space(controlled_agent)
        self.observation_space = self.environment.observation_space(controlled_agent)
        self.render_mode = self.environment.render_mode
        self._observations: dict[AgentId, GymObservation] = {}
        self._infos: dict[AgentId, dict[str, Any]] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[GymObservation, dict[str, Any]]:
        super().reset(seed=seed)
        observations, infos = self.environment.reset(seed=seed, options=options)
        server_seed = infos[self.controlled_agent].get("seed")
        if not isinstance(server_seed, int) or isinstance(server_seed, bool):
            raise ValueError("SnowGym reset info is missing an integer seed")
        self.opponent.reset(seed=server_seed, agent=self.opponent_agent)
        self._observations = observations
        self._infos = infos
        return observations[self.controlled_agent], dict(infos[self.controlled_agent])

    def step(
        self, action: GymAction
    ) -> tuple[GymObservation, float, bool, bool, dict[str, Any]]:
        if not self.environment.agents:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError("controlled action is outside SnowGym action_space")
        opponent_action = self.opponent.act(
            self._observations[self.opponent_agent],
            self._infos[self.opponent_agent],
        )
        opponent_space = self.environment.action_space(self.opponent_agent)
        if not opponent_space.contains(opponent_action):
            raise ValueError("opponent action is outside SnowGym action_space")
        actions = {self.controlled_agent: action, self.opponent_agent: opponent_action}
        observations, rewards, terminations, truncations, infos = self.environment.step(
            actions
        )
        self._observations = observations
        self._infos = infos
        return (
            observations[self.controlled_agent],
            rewards[self.controlled_agent],
            terminations[self.controlled_agent],
            truncations[self.controlled_agent],
            dict(infos[self.controlled_agent]),
        )

    def close(self) -> None:
        self.environment.close()
        self._observations = {}
        self._infos = {}


def noop_action(capacity: int) -> GymAction:
    return {
        "action_type": np.full(capacity, ACTION_NOOP, dtype=np.int64),
        "target": np.zeros((capacity, 2), dtype=np.float32),
        "power": np.zeros(capacity, dtype=np.float32),
    }


def masked_random_action(
    observation: GymObservation, generator: np.random.Generator
) -> GymAction:
    action = noop_action(len(observation["ally_mask"]))
    for index in range(len(observation["ally_mask"])):
        if not observation["ally_mask"][index]:
            continue
        valid = np.flatnonzero(observation["unit_action_mask"][index])
        if valid.size:
            action["action_type"][index] = int(generator.choice(valid))
        action["target"][index] = generator.uniform(-1.0, 1.0, size=2).astype(
            np.float32
        )
        action["power"][index] = np.float32(generator.random())
    return action


def clone_observation(observation: GymObservation) -> GymObservation:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
