"""Persistent fixed-plan batch option environment for M7b PPO."""

from __future__ import annotations

from typing import Any

import numpy as np

from snowgym_client.batch import SnowGymBatchEnv

from ..ppo_collect import merge_observations
from .definitions import OptionSpec
from .tracker import FixedOptionTracker, OptionStep


class FixedPlanOptionBatchEnv:
    """Adds option termination and decomposed rewards to authoritative worlds."""

    def __init__(self, environment: SnowGymBatchEnv, *, gamma: float) -> None:
        if environment.observation_version != 3:
            raise ValueError("fixed-plan options require observation version 3")
        if not 0 < gamma <= 1:
            raise ValueError("fixed-plan option gamma must be in (0, 1]")
        self.environment = environment
        self.gamma = gamma
        self.trackers: list[FixedOptionTracker | None] = [None] * environment.batch_size

    @property
    def batch_size(self) -> int:
        return self.environment.batch_size

    def reset(
        self,
        seeds: list[int],
        scenarios: list[dict[str, Any]],
        plan_ids: list[str],
        plans: list[dict[str, Any]],
        specs: list[OptionSpec],
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if len(specs) != self.batch_size:
            raise ValueError("fixed option specs must match batch_size")
        physical, infos = self.environment.reset(seeds, scenarios)
        bodies = self.environment.activate_plans(plan_ids, plans)
        self._install_trackers(list(range(self.batch_size)), plans, specs, bodies)
        plan_tensors, _ = self.environment.plan_observations()
        return merge_observations(physical, plan_tensors), infos

    def reset_indices(
        self,
        indices: list[int],
        seeds: list[int],
        scenarios: list[dict[str, Any]],
        plan_ids: list[str],
        plans: list[dict[str, Any]],
        specs: list[OptionSpec],
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if not (
            len(indices) == len(seeds) == len(scenarios) == len(plan_ids) == len(plans) == len(specs)
        ):
            raise ValueError("selective fixed-option reset inputs must have equal length")
        physical, infos = self.environment.reset_indices(indices, seeds, scenarios)
        bodies = self.environment.activate_plan_indices(indices, plan_ids, plans)
        self._install_trackers(indices, plans, specs, bodies)
        plan_tensors, _ = self.environment.plan_observations(indices)
        return merge_observations(physical, plan_tensors), infos

    def step(
        self, actions: dict[str, np.ndarray] | None = None, *, teacher: bool = False
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        if teacher == (actions is not None):
            raise ValueError("provide actions or select teacher, exclusively")
        if teacher:
            physical, canonical, environment_terminated, environment_truncated, infos = (
                self.environment.step_team_actions(self.environment.plan_teacher_actions())
            )
        else:
            assert actions is not None
            physical, canonical, environment_terminated, environment_truncated, infos = (
                self.environment.step(actions)
            )
        plan_tensors, bodies = self.environment.plan_observations()
        steps: list[OptionStep] = []
        for index, tracker in enumerate(self.trackers):
            if tracker is None:
                raise RuntimeError("fixed option batch slot is not initialized")
            raw = self.environment.raw_observations[index]
            if raw is None:
                raise RuntimeError("fixed option batch raw state is missing")
            steps.append(
                tracker.update(
                    raw,
                    bodies[index],
                    canonical_reward=float(canonical[index]),
                    gamma=self.gamma,
                    environment_done=bool(
                        environment_terminated[index] or environment_truncated[index]
                    ),
                )
            )
        option_terminated = np.asarray(
            [step.done for step in steps], dtype=np.bool_
        )
        option_truncated = np.asarray(
            [False for _ in steps], dtype=np.bool_
        )
        enriched = []
        for info, step in zip(infos, steps, strict=True):
            enriched.append(
                {
                    **info,
                    "option": {
                        "decision": step.decision,
                        "success": step.success,
                        "failed": step.failed,
                        "timedOut": step.timed_out,
                        "progress": step.progress,
                        "rewards": {
                            "mission": step.mission_reward,
                            "combat": step.combat_reward,
                            "shaping": step.shaping_reward,
                            "canonical": step.canonical_reward,
                            "executor": step.executor_reward,
                        },
                        "metrics": step.metrics,
                    },
                }
            )
        return (
            merge_observations(physical, plan_tensors),
            np.asarray([step.executor_reward for step in steps], dtype=np.float32),
            option_terminated,
            option_truncated,
            enriched,
        )

    def _install_trackers(
        self,
        indices: list[int],
        plans: list[dict[str, Any]],
        specs: list[OptionSpec],
        bodies: list[dict[str, Any]],
    ) -> None:
        for index, plan, spec, body in zip(indices, plans, specs, bodies, strict=True):
            raw = self.environment.raw_observations[index]
            if raw is None:
                raise RuntimeError("fixed option reset raw state is missing")
            self.trackers[index] = FixedOptionTracker(spec, plan, raw, body)
