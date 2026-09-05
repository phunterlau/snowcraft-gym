"""Versioned fixed-identity Engage observations for new scoped option PPO.

Historical option wrappers retain their original tensor/reward contract. New
movement-only PPO requires this wrapper and authoritative activation targets.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .environment import FixedPlanOptionBatchEnv
from .tracker import FixedOptionTracker
from ..ppo_collect import merge_observations

OPTION_STATE_VERSION = "snowgym.engage-option-state.v1"
OPTION_STATE_FIELDS = ("remainingFraction", "activatedTargetHealthFraction", "assignedLivingFraction")


class FrozenEngageTracker(FixedOptionTracker):
    def __init__(self, spec, plan, observation, plan_observation):
        if spec.name != "engage":
            raise ValueError("Engage v1 does not implement temporal or composed missions")
        objectives = plan_observation.get("activationObjectives", [])
        selected = next((item for item in objectives if item["role"] == spec.role), None)
        if selected is None or selected.get("kind") != "enemy_cluster":
            raise ValueError("Engage v1 requires authoritative activated target membership")
        ids = selected.get("enemyIds", [])
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
            raise ValueError("activated target membership is invalid")
        enemies = {u["id"]: u for u in observation["enemies"]}
        if any(i not in enemies for i in ids):
            raise ValueError("activated target is absent from activation observation")
        self.activated_target_ids = tuple(ids)
        self.activated_target_health = sum(max(0., float(enemies[i]["health"])) for i in ids)
        if self.activated_target_health <= 0:
            raise ValueError("cannot start Engage with an already eliminated objective")
        super().__init__(spec, plan, observation, plan_observation)
        self.finished = False

    def update(self, *args, **kwargs):
        if self.finished:
            raise RuntimeError("Engage option is already complete")
        step = super().update(*args, **kwargs)
        self.finished = step.done
        return step

    def target_health_fraction(self, observation):
        current = {u["id"]: max(0., float(u["health"])) if u["alive"] else 0. for u in observation["enemies"]}
        return float(np.clip(sum(current.get(i, 0.) for i in self.activated_target_ids)/self.activated_target_health, 0, 1))

    def _mission_snapshot(self, observation, plan_observation):
        snapshot = super()._mission_snapshot(observation, plan_observation)
        return replace(snapshot, objective_health=self.target_health_fraction(observation))

    def option_state(self, observation):
        living = sum(u["alive"] and u["id"] in self.assigned_ids for u in observation["allies"])
        return np.asarray([max(0., 1-self.decision/self.spec.horizon),
                           self.target_health_fraction(observation), living/self.initial_assigned], dtype=np.float32)


class EngageOptionBatchV1(FixedPlanOptionBatchEnv):
    def _install_trackers(self, indices, plans, specs, bodies):
        for index, plan, spec, body in zip(indices, plans, specs, bodies, strict=True):
            raw = self.environment.raw_observations[index]
            if raw is None:
                raise RuntimeError("missing activation observation")
            self.trackers[index] = FrozenEngageTracker(spec, plan, raw, body)

    def _augment(self, observation, indices=None):
        indices = list(range(self.batch_size)) if indices is None else indices
        values = []
        for index in indices:
            tracker = self.trackers[index]
            raw = self.environment.raw_observations[index]
            if not isinstance(tracker, FrozenEngageTracker) or raw is None:
                raise RuntimeError("Engage v1 slot is uninitialized")
            values.append(tracker.option_state(raw))
        return {**observation, "option_state": np.stack(values)}

    def reset(self, *args, **kwargs):
        observation, infos = super().reset(*args, **kwargs)
        return self._augment(observation), infos

    def reset_indices(self, indices, *args, **kwargs):
        observation, infos = super().reset_indices(indices, *args, **kwargs)
        return self._augment(observation, indices), infos

    def step(self, actions):
        return self.step_indices(list(range(self.batch_size)), actions)

    def step_indices(self, indices, actions):
        if any(self.trackers[index] is None or self.trackers[index].finished for index in indices):
            raise RuntimeError("reset a completed option before stepping it")
        physical, canonical, terminated, truncated, infos = self.environment.step_indices(indices, actions)
        tensors, bodies = self.environment.plan_observations(indices)
        steps, enriched = [], []
        for row, index in enumerate(indices):
            tracker = self.trackers[index]
            step = tracker.update(self.environment.raw_observations[index], bodies[row],
                                  canonical_reward=float(canonical[row]), gamma=self.gamma,
                                  environment_done=bool(terminated[row] or truncated[row]))
            steps.append(step)
            enriched.append({**infos[row], "optionStateVersion": OPTION_STATE_VERSION,
                "option": {"decision": step.decision, "success": step.success,
                    "failed": step.failed, "timedOut": step.timed_out, "progress": step.progress,
                    "rewards": {"mission": step.mission_reward, "combat": step.combat_reward,
                        "shaping": step.shaping_reward, "canonical": step.canonical_reward,
                        "executor": step.executor_reward}, "metrics": step.metrics}})
        return (self._augment(merge_observations(physical, tensors), indices),
                np.asarray([step.executor_reward for step in steps], dtype=np.float32),
                np.asarray([step.done for step in steps], dtype=bool),
                np.zeros(len(indices), dtype=bool), enriched)
