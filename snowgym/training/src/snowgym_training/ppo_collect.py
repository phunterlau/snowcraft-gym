"""Deterministic persistent-batch rollout collection for SnowGym PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchEnv

from .ppo import HybridActorCritic, PPOConfig, PPORollout, RolloutBuffer


@dataclass
class SeedSchedule:
    minimum: int
    maximum: int
    next_seed: int | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (self.minimum, self.maximum)
        ):
            raise ValueError("seed schedule bounds must be integers")
        if self.minimum > self.maximum:
            raise ValueError("seed schedule minimum must not exceed maximum")
        if self.next_seed is None:
            self.next_seed = self.minimum
        if not self.minimum <= self.next_seed <= self.maximum + 1:
            raise ValueError("seed schedule cursor is outside its range")

    def take(self, count: int) -> list[int]:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("seed count must be a positive integer")
        assert self.next_seed is not None
        end = self.next_seed + count
        if end > self.maximum + 1:
            raise RuntimeError("training seed schedule exhausted")
        result = list(range(self.next_seed, end))
        self.next_seed = end
        return result

    def state(self) -> dict[str, int]:
        assert self.next_seed is not None
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "nextSeed": self.next_seed,
        }


@dataclass(frozen=True)
class RolloutCollection:
    rollout: PPORollout
    episode_seeds: tuple[int, ...]
    completed_episodes: int
    rejected_actions: int
    boundary_truncations: int
    seed_schedule: dict[str, int]


def collect_rollout(
    environment: SnowGymBatchEnv,
    model: HybridActorCritic,
    *,
    scenario: dict[str, Any],
    seed_schedule: SeedSchedule,
    rollout_steps: int,
    config: PPOConfig,
) -> RolloutCollection:
    """Collect one restartable rollout; every call begins at an episode boundary."""
    seeds = seed_schedule.take(environment.batch_size)
    episode_seeds = list(seeds)
    scenarios = [dict(scenario) for _ in range(environment.batch_size)]
    observation, _ = environment.reset(seeds, scenarios)
    buffer = RolloutBuffer(rollout_steps, environment.batch_size)
    completed_episodes = 0
    rejected_actions = 0
    boundary_truncations = 0
    model.eval()

    for step in range(rollout_steps):
        tensor_observation = tensor_dict(observation)
        with torch.no_grad():
            action, log_probability, value = model.act(tensor_observation)
        next_observation, reward, terminated, truncated, infos = environment.step(
            numpy_actions(action)
        )
        with torch.no_grad():
            next_value = model(tensor_dict(next_observation))["value"]
        is_boundary = step == rollout_steps - 1
        stored_truncated = np.asarray(truncated, dtype=np.bool_).copy()
        if is_boundary:
            artificial = ~(np.asarray(terminated, dtype=np.bool_) | stored_truncated)
            boundary_truncations += int(artificial.sum())
            stored_truncated |= artificial
        buffer.add(
            observation=tensor_observation,
            action=action,
            log_probability=log_probability,
            value=value,
            reward=torch.as_tensor(reward, dtype=torch.float32),
            terminated=torch.as_tensor(terminated, dtype=torch.bool),
            truncated=torch.as_tensor(stored_truncated, dtype=torch.bool),
            next_value=next_value,
        )
        rejected_actions += sum(
            result.get("accepted") is False
            for info in infos
            for result in info.get("actionResults", [])
            if isinstance(result, dict)
        )
        done = np.asarray(terminated, dtype=np.bool_) | np.asarray(
            truncated, dtype=np.bool_
        )
        completed_episodes += int(done.sum())
        if not is_boundary and bool(done.any()):
            indices = np.flatnonzero(done).tolist()
            reset_seeds = seed_schedule.take(len(indices))
            episode_seeds.extend(reset_seeds)
            reset_observation, _ = environment.reset_indices(
                indices,
                reset_seeds,
                [dict(scenario) for _ in indices],
            )
            next_observation = replace_rows(next_observation, indices, reset_observation)
        observation = next_observation

    return RolloutCollection(
        rollout=buffer.finish(config),
        episode_seeds=tuple(episode_seeds),
        completed_episodes=completed_episodes,
        rejected_actions=rejected_actions,
        boundary_truncations=boundary_truncations,
        seed_schedule=seed_schedule.state(),
    )


def tensor_dict(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(value).clone() for name, value in observation.items()}


def numpy_actions(action: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "action_type": action["action_type"].detach().cpu().numpy(),
        "target": action["target"].detach().cpu().numpy().astype(np.float32),
        "power": action["power"].detach().cpu().numpy().astype(np.float32),
    }


def replace_rows(
    observation: dict[str, np.ndarray],
    indices: list[int],
    replacement: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value).copy() for name, value in observation.items()}
    if set(result) != set(replacement):
        raise RuntimeError("selective reset observation keys changed")
    for name, values in replacement.items():
        result[name][indices] = values
    return result
