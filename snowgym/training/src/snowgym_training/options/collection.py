"""Deterministic selective-reset PPO collection for fixed mission options."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..ppo import HybridActorCritic, PPOConfig, RolloutBuffer
from ..ppo_collect import SeedSchedule, numpy_actions, replace_rows, tensor_dict
from ..trajectory import json_digest
from .definitions import OptionSpec
from .environment import FixedPlanOptionBatchEnv


@dataclass(frozen=True)
class OptionEntry:
    plan: dict[str, Any]
    spec: OptionSpec


@dataclass
class OptionSchedule:
    entries: tuple[OptionEntry, ...]
    prefix: str = "option-ppo"
    next_index: int = 0

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("option schedule requires at least one entry")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("option schedule prefix must be non-empty")
        if (
            not isinstance(self.next_index, int)
            or isinstance(self.next_index, bool)
            or not 0 <= self.next_index <= len(self.entries)
        ):
            raise ValueError("option schedule cursor is outside its range")

    @property
    def digest(self) -> str:
        return json_digest(
            {
                "prefix": self.prefix,
                "entries": [
                    {
                        "plan": entry.plan,
                        "spec": {
                            "name": entry.spec.name,
                            "horizon": entry.spec.horizon,
                            "role": entry.spec.role,
                        },
                    }
                    for entry in self.entries
                ],
            }
        )

    def take(
        self, count: int
    ) -> tuple[list[str], list[dict[str, Any]], list[OptionSpec]]:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("option count must be a positive integer")
        end = self.next_index + count
        if end > len(self.entries):
            raise RuntimeError("option schedule exhausted")
        selected = self.entries[self.next_index : end]
        identifiers = [
            f"{self.prefix}-{index:06d}" for index in range(self.next_index, end)
        ]
        self.next_index = end
        return (
            identifiers,
            [deepcopy(entry.plan) for entry in selected],
            [entry.spec for entry in selected],
        )

    def state(self) -> dict[str, Any]:
        return {
            "format": "snowgym.option-schedule.v0",
            "digest": self.digest,
            "prefix": self.prefix,
            "length": len(self.entries),
            "nextIndex": self.next_index,
        }

    @classmethod
    def restore(
        cls, entries: tuple[OptionEntry, ...], state: dict[str, Any]
    ) -> OptionSchedule:
        if not isinstance(state, dict) or set(state) != {
            "format", "digest", "prefix", "length", "nextIndex"
        }:
            raise ValueError("option schedule state fields are invalid")
        if state["format"] != "snowgym.option-schedule.v0":
            raise ValueError("option schedule state format is invalid")
        schedule = cls(entries, prefix=state["prefix"], next_index=state["nextIndex"])
        if state["length"] != len(entries) or state["digest"] != schedule.digest:
            raise ValueError("option schedule contents do not match checkpoint state")
        return schedule


@dataclass(frozen=True)
class OptionRolloutCollection:
    rollout: Any
    teacher_actions: dict[str, torch.Tensor]
    episode_seeds: tuple[int, ...]
    episode_option_ids: tuple[str, ...]
    completed_options: int
    successful_options: int
    rejected_actions: int
    boundary_truncations: int
    reward_sums: dict[str, float]
    seed_schedule: dict[str, int]
    option_schedule: dict[str, Any]


def collect_option_rollout(
    environment: FixedPlanOptionBatchEnv,
    model: HybridActorCritic,
    *,
    scenario: dict[str, Any],
    seed_schedule: SeedSchedule,
    option_schedule: OptionSchedule,
    rollout_steps: int,
    config: PPOConfig,
) -> OptionRolloutCollection:
    if not model.policy.config.plan_ppo_residuals:
        raise ValueError("option PPO collection requires plan residual architecture")
    seeds = seed_schedule.take(environment.batch_size)
    episode_seeds = list(seeds)
    plan_ids, plans, specs = option_schedule.take(environment.batch_size)
    episode_option_ids = list(plan_ids)
    observation, _ = environment.reset(
        seeds,
        [dict(scenario) for _ in seeds],
        plan_ids,
        plans,
        specs,
    )
    buffer = RolloutBuffer(rollout_steps, environment.batch_size)
    teachers: list[dict[str, torch.Tensor]] = []
    completed = 0
    successful = 0
    rejected = 0
    boundary_truncations = 0
    reward_sums = {
        "mission": 0.0,
        "combat": 0.0,
        "shaping": 0.0,
        "canonical": 0.0,
        "executor": 0.0,
    }
    model.eval()
    for step_index in range(rollout_steps):
        tensor_observation = tensor_dict(observation)
        teachers.append(
            tensor_dict(environment.environment.plan_teacher_tensor_actions())
        )
        with torch.no_grad():
            action, log_probability, value = model.act(tensor_observation)
        next_observation, reward, terminated, truncated, infos = environment.step(
            numpy_actions(action)
        )
        with torch.no_grad():
            next_value = model(tensor_dict(next_observation))["value"]
        is_boundary = step_index == rollout_steps - 1
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
        for info in infos:
            option = info["option"]
            for name, amount in option["rewards"].items():
                reward_sums[name] += float(amount)
            rejected += sum(
                result.get("accepted") is False
                for result in info.get("actionResults", [])
                if isinstance(result, dict)
            )
        done = np.asarray(terminated, dtype=np.bool_) | np.asarray(
            truncated, dtype=np.bool_
        )
        completed += int(done.sum())
        successful += sum(
            bool(info["option"]["success"])
            for index, info in enumerate(infos)
            if done[index]
        )
        if not is_boundary and bool(done.any()):
            indices = np.flatnonzero(done).tolist()
            reset_seeds = seed_schedule.take(len(indices))
            episode_seeds.extend(reset_seeds)
            replacement_ids, replacement_plans, replacement_specs = option_schedule.take(
                len(indices)
            )
            episode_option_ids.extend(replacement_ids)
            replacement, _ = environment.reset_indices(
                indices,
                reset_seeds,
                [dict(scenario) for _ in indices],
                replacement_ids,
                replacement_plans,
                replacement_specs,
            )
            next_observation = replace_rows(next_observation, indices, replacement)
        observation = next_observation
    return OptionRolloutCollection(
        rollout=buffer.finish(config),
        teacher_actions={
            name: torch.stack([batch[name] for batch in teachers])
            for name in teachers[0]
        },
        episode_seeds=tuple(episode_seeds),
        episode_option_ids=tuple(episode_option_ids),
        completed_options=completed,
        successful_options=successful,
        rejected_actions=rejected,
        boundary_truncations=boundary_truncations,
        reward_sums=reward_sums,
        seed_schedule=seed_schedule.state(),
        option_schedule=option_schedule.state(),
    )
