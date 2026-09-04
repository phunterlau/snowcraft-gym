"""Deterministic persistent-batch rollout collection for SnowGym PPO."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchEnv

from .ppo import (
    HybridActorCritic,
    PPOConfig,
    PPORollout,
    RolloutBuffer,
    health_potential,
    potential_shaped_reward,
)
from .trajectory import json_digest


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


@dataclass
class PlanSchedule:
    """Immutable fixed-plan sequence with a checkpointable exact-resume cursor."""

    plans: tuple[dict[str, Any], ...]
    prefix: str = "plan-ppo"
    next_index: int = 0

    def __post_init__(self) -> None:
        if not self.plans or any(not isinstance(plan, dict) for plan in self.plans):
            raise ValueError("plan schedule requires at least one object plan")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("plan schedule prefix must be non-empty")
        if (
            not isinstance(self.next_index, int)
            or isinstance(self.next_index, bool)
            or not 0 <= self.next_index <= len(self.plans)
        ):
            raise ValueError("plan schedule cursor is outside its range")

    @property
    def digest(self) -> str:
        return json_digest({"prefix": self.prefix, "plans": self.plans})

    def take(self, count: int) -> tuple[list[str], list[dict[str, Any]]]:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("plan count must be a positive integer")
        end = self.next_index + count
        if end > len(self.plans):
            raise RuntimeError("plan schedule exhausted")
        indices = range(self.next_index, end)
        identifiers = [f"{self.prefix}-{index:06d}" for index in indices]
        values = [deepcopy(self.plans[index]) for index in indices]
        self.next_index = end
        return identifiers, values

    def state(self) -> dict[str, Any]:
        return {
            "format": "snowgym.plan-schedule.v0",
            "digest": self.digest,
            "prefix": self.prefix,
            "length": len(self.plans),
            "nextIndex": self.next_index,
        }

    @classmethod
    def restore(
        cls, plans: tuple[dict[str, Any], ...], state: dict[str, Any]
    ) -> PlanSchedule:
        if not isinstance(state, dict) or set(state) != {
            "format", "digest", "prefix", "length", "nextIndex"
        }:
            raise ValueError("plan schedule state fields are invalid")
        if state["format"] != "snowgym.plan-schedule.v0":
            raise ValueError("plan schedule state format is invalid")
        schedule = cls(plans, prefix=state["prefix"], next_index=state["nextIndex"])
        if state["length"] != len(plans) or state["digest"] != schedule.digest:
            raise ValueError("plan schedule contents do not match checkpoint state")
        return schedule


@dataclass(frozen=True)
class RolloutCollection:
    rollout: PPORollout
    episode_seeds: tuple[int, ...]
    completed_episodes: int
    rejected_actions: int
    boundary_truncations: int
    canonical_reward_sum: float
    training_reward_sum: float
    seed_schedule: dict[str, int]
    plan_schedule: dict[str, Any] | None = None
    episode_plan_ids: tuple[str, ...] = ()


def collect_rollout(
    environment: SnowGymBatchEnv,
    model: HybridActorCritic,
    *,
    scenario: dict[str, Any],
    seed_schedule: SeedSchedule,
    rollout_steps: int,
    config: PPOConfig,
    reward_mode: str = "canonical",
) -> RolloutCollection:
    """Collect one restartable rollout; every call begins at an episode boundary."""
    if reward_mode not in {"canonical", "health-potential"}:
        raise ValueError("reward_mode must be canonical or health-potential")
    seeds = seed_schedule.take(environment.batch_size)
    episode_seeds = list(seeds)
    scenarios = [dict(scenario) for _ in range(environment.batch_size)]
    observation, _ = environment.reset(seeds, scenarios)
    buffer = RolloutBuffer(rollout_steps, environment.batch_size)
    completed_episodes = 0
    rejected_actions = 0
    boundary_truncations = 0
    canonical_reward_sum = 0.0
    training_reward_sum = 0.0
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
        canonical_reward = torch.as_tensor(reward, dtype=torch.float32)
        training_reward = canonical_reward
        if reward_mode == "health-potential":
            training_reward = potential_shaped_reward(
                canonical_reward,
                health_potential(tensor_observation),
                health_potential(tensor_dict(next_observation)),
                torch.as_tensor(terminated, dtype=torch.bool),
                gamma=config.gamma,
            )
        canonical_reward_sum += float(canonical_reward.sum())
        training_reward_sum += float(training_reward.sum())
        buffer.add(
            observation=tensor_observation,
            action=action,
            log_probability=log_probability,
            value=value,
            reward=training_reward,
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
        canonical_reward_sum=canonical_reward_sum,
        training_reward_sum=training_reward_sum,
        seed_schedule=seed_schedule.state(),
    )


def collect_plan_rollout(
    environment: SnowGymBatchEnv,
    model: HybridActorCritic,
    *,
    scenario: dict[str, Any],
    seed_schedule: SeedSchedule,
    plan_schedule: PlanSchedule,
    rollout_steps: int,
    config: PPOConfig,
    reward_mode: str = "canonical",
) -> RolloutCollection:
    """Collect fixed-plan v3 PPO data while refreshing host-owned tensors."""
    if environment.observation_version != 3:
        raise ValueError("plan PPO collection requires observation version 3")
    if not model.policy.config.plan_ppo_residuals:
        raise ValueError("plan PPO collection requires the plan residual architecture")
    if reward_mode not in {"canonical", "health-potential"}:
        raise ValueError("reward_mode must be canonical or health-potential")
    seeds = seed_schedule.take(environment.batch_size)
    episode_seeds = list(seeds)
    scenarios = [dict(scenario) for _ in range(environment.batch_size)]
    physical_observation, _ = environment.reset(seeds, scenarios)
    plan_ids, plans = plan_schedule.take(environment.batch_size)
    episode_plan_ids = list(plan_ids)
    environment.activate_plans(plan_ids, plans)
    buffer = RolloutBuffer(rollout_steps, environment.batch_size)
    completed_episodes = 0
    rejected_actions = 0
    boundary_truncations = 0
    canonical_reward_sum = 0.0
    training_reward_sum = 0.0
    model.eval()

    for step in range(rollout_steps):
        plan_observation, _ = environment.plan_observations()
        observation = merge_observations(physical_observation, plan_observation)
        tensor_observation = tensor_dict(observation)
        with torch.no_grad():
            action, log_probability, value = model.act(tensor_observation)
        next_physical, reward, terminated, truncated, infos = environment.step(
            numpy_actions(action)
        )
        next_plan, _ = environment.plan_observations()
        next_observation = merge_observations(next_physical, next_plan)
        with torch.no_grad():
            next_value = model(tensor_dict(next_observation))["value"]
        is_boundary = step == rollout_steps - 1
        stored_truncated = np.asarray(truncated, dtype=np.bool_).copy()
        if is_boundary:
            artificial = ~(np.asarray(terminated, dtype=np.bool_) | stored_truncated)
            boundary_truncations += int(artificial.sum())
            stored_truncated |= artificial
        canonical_reward = torch.as_tensor(reward, dtype=torch.float32)
        training_reward = canonical_reward
        if reward_mode == "health-potential":
            training_reward = potential_shaped_reward(
                canonical_reward,
                health_potential(tensor_observation),
                health_potential(tensor_dict(next_observation)),
                torch.as_tensor(terminated, dtype=torch.bool),
                gamma=config.gamma,
            )
        canonical_reward_sum += float(canonical_reward.sum())
        training_reward_sum += float(training_reward.sum())
        buffer.add(
            observation=tensor_observation,
            action=action,
            log_probability=log_probability,
            value=value,
            reward=training_reward,
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
            replacement_ids, replacement_plans = plan_schedule.take(len(indices))
            episode_plan_ids.extend(replacement_ids)
            environment.activate_plan_indices(
                indices, replacement_ids, replacement_plans
            )
            next_physical = replace_rows(next_physical, indices, reset_observation)
        physical_observation = next_physical

    return RolloutCollection(
        rollout=buffer.finish(config),
        episode_seeds=tuple(episode_seeds),
        completed_episodes=completed_episodes,
        rejected_actions=rejected_actions,
        boundary_truncations=boundary_truncations,
        canonical_reward_sum=canonical_reward_sum,
        training_reward_sum=training_reward_sum,
        seed_schedule=seed_schedule.state(),
        plan_schedule=plan_schedule.state(),
        episode_plan_ids=tuple(episode_plan_ids),
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


def merge_observations(
    physical: dict[str, np.ndarray], plan: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    overlap = set(physical) & set(plan)
    if overlap:
        raise RuntimeError(f"physical and plan observation keys overlap: {sorted(overlap)}")
    batches = {value.shape[0] for value in (*physical.values(), *plan.values())}
    if len(batches) != 1:
        raise RuntimeError("physical and plan observation batches differ")
    return {**physical, **plan}
