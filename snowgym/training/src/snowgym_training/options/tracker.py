"""Deterministic reward and completion state for one fixed symbolic option."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .definitions import OptionSpec

ROLE_INDEX = {"main": 0, "maneuver": 1, "reserve": 2}
MEANINGFUL_DAMAGE_FRACTION = 0.1


@dataclass(frozen=True)
class OptionStep:
    mission_reward: float
    combat_reward: float
    shaping_reward: float
    canonical_reward: float
    executor_reward: float
    progress: float
    success: bool
    failed: bool
    done: bool
    timed_out: bool
    decision: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class MissionSnapshot:
    living: tuple[dict[str, Any], ...]
    living_fraction: float
    objective_health: float
    advance_inside: bool
    advance_fraction: float
    hold_inside: bool
    withdraw_inside: bool
    support_qualified: bool
    signed_lateral: float
    support_distance: float | None


class FixedOptionTracker:
    """Evaluate an immutable plan without commander lifecycle replacement."""

    def __init__(
        self,
        spec: OptionSpec,
        plan: dict[str, Any],
        observation: dict[str, Any],
        plan_observation: dict[str, Any],
    ) -> None:
        self.spec = spec
        groups = plan.get("groups")
        if not isinstance(groups, list):
            raise ValueError("fixed option plan groups are invalid")
        self.command = next(
            (group for group in groups if group.get("role") == spec.role), None
        )
        if not isinstance(self.command, dict):
            raise ValueError(f"fixed option plan has no {spec.role} group")
        self._validate_command()
        assignments = plan_observation.get("assignments")
        if not isinstance(assignments, list):
            raise ValueError("fixed option assignments are missing")
        assignment = next(
            (item for item in assignments if item.get("role") == spec.role), None
        )
        if not isinstance(assignment, dict) or not isinstance(assignment.get("unitIds"), list):
            raise ValueError("fixed option assignment is invalid")
        self.assigned_ids = tuple(int(value) for value in assignment["unitIds"])
        if not self.assigned_ids:
            raise ValueError("fixed option assignment is empty")
        self.initial_assigned = len(self.assigned_ids)
        self.initial_blue_health = _team_max_health(observation["allies"])
        self.initial_red_health = _team_max_health(observation["enemies"])
        self.previous_blue_health = _health_by_id(observation["allies"])
        self.previous_red_health = _health_by_id(observation["enemies"])
        self.target_damage = {unit_id: 0.0 for unit_id in self.previous_red_health}
        self.mean_enemy_max_health = self.initial_red_health / max(len(self.target_damage), 1)
        self.activation_anchor = _centroid(_assigned(observation, self.assigned_ids, living=True))
        self.objective_anchor = self._objective_anchor(observation, plan_observation)
        supported_role = self.command["order"].get("objective", {}).get("role")
        supported = next(
            (item for item in assignments if item.get("role") == supported_role), None
        )
        self.supported_ids = tuple(int(value) for value in supported.get("unitIds", [])) if supported else ()
        self.initial_supported = len(self.supported_ids)
        own = _centroid([unit for unit in observation["allies"] if unit.get("alive")])
        enemy = _centroid([unit for unit in observation["enemies"] if unit.get("alive")])
        dx, dy = enemy[0] - own[0], enemy[1] - own[1]
        length = math.hypot(dx, dy)
        self.forward = (1.0, 0.0) if length <= 1e-9 else (dx / length, dy / length)
        self.left = (-self.forward[1], self.forward[0])
        width = float(observation["arena"]["width"])
        height = float(observation["arena"]["height"])
        self.diagonal = math.hypot(width, height)
        self.lateral_extent = (abs(self.left[0]) * width + abs(self.left[1]) * height) / 2
        self.decision = 0
        self.inside_count = 0
        self.consecutive = 0
        self.flank_reached_at: int | None = None
        self.flank_damage_at_reach = 0.0
        self.previous_progress = self._progress(
            self._mission_snapshot(observation, plan_observation)
        )

    def update(
        self,
        observation: dict[str, Any],
        plan_observation: dict[str, Any],
        *,
        canonical_reward: float,
        gamma: float,
        environment_done: bool = False,
    ) -> OptionStep:
        if not 0 < gamma <= 1:
            raise ValueError("option shaping gamma must be in (0, 1]")
        self.decision += 1
        current_blue = _health_by_id(observation["allies"])
        current_red = _health_by_id(observation["enemies"])
        damage_dealt = sum(
            max(0.0, health - current_red.get(unit_id, 0.0))
            for unit_id, health in self.previous_red_health.items()
        )
        damage_received = sum(
            max(0.0, health - current_blue.get(unit_id, 0.0))
            for unit_id, health in self.previous_blue_health.items()
        )
        for unit_id, health in self.previous_red_health.items():
            self.target_damage[unit_id] += max(0.0, health - current_red.get(unit_id, 0.0))
        self.previous_blue_health = current_blue
        self.previous_red_health = current_red
        combat = float(
            np.clip(
                damage_dealt / max(self.initial_red_health, 1e-9)
                - damage_received / max(self.initial_blue_health, 1e-9),
                -1,
                1,
            )
        )
        snapshot = self._mission_snapshot(observation, plan_observation)
        self._advance_temporal_state(snapshot)
        success = self._success(snapshot)
        progress = self._progress(snapshot)
        living = len(snapshot.living)
        timed_out = self.decision >= self.spec.horizon and not success
        failed = living == 0 or timed_out
        if environment_done and not success:
            failed = True
        done = success or failed
        mission_reward = 1.0 if success else -1.0 if failed else 0.0
        next_potential = 0.0 if done else progress
        shaping = gamma * next_potential - self.previous_progress
        self.previous_progress = next_potential
        return OptionStep(
            mission_reward=mission_reward,
            combat_reward=combat,
            shaping_reward=shaping,
            canonical_reward=float(canonical_reward),
            executor_reward=mission_reward + 0.1 * combat + shaping,
            progress=progress,
            success=success,
            failed=failed,
            done=done,
            timed_out=timed_out,
            decision=self.decision,
            metrics=self._metrics(snapshot),
        )

    def _validate_command(self) -> None:
        order = self.command.get("order")
        if not isinstance(order, dict):
            raise ValueError("fixed option command order is invalid")
        name = self.spec.name
        mission = order.get("mission")
        expected = name if name in {"engage", "advance", "hold", "withdraw", "support"} else "engage"
        if mission != expected:
            raise ValueError(f"{name} option requires {expected} mission")
        if name == "flank" and order.get("approach") not in {"left_flank", "right_flank"}:
            raise ValueError("flank option requires a signed flank approach")
        fire = order.get("engagement", {}).get("fire")
        if name in {"focus", "distributed"} and fire != name:
            raise ValueError(f"{name} option requires {name} fire policy")

    def _objective_anchor(
        self, observation: dict[str, Any], plan_observation: dict[str, Any]
    ) -> tuple[float, float]:
        row = _role_row(plan_observation, self.spec.role)
        center = _centroid(_assigned(observation, self.assigned_ids, living=True))
        return (
            center[0] + row[8] * float(observation["arena"]["width"]),
            center[1] + row[9] * float(observation["arena"]["height"]),
        )

    def _mission_snapshot(
        self, observation: dict[str, Any], plan_observation: dict[str, Any]
    ) -> MissionSnapshot:
        living = tuple(_assigned(observation, self.assigned_ids, living=True))
        advance_fraction = _fraction_within(
            list(living), self.objective_anchor, 0.1 * self.diagonal
        )
        support_distance = self._support_distance(observation)
        supported_living = len(_assigned(observation, self.supported_ids, living=True))
        living_fraction = len(living) / self.initial_assigned
        support_healthy = living_fraction >= 0.5 and (
            self.initial_supported > 0
            and supported_living / self.initial_supported >= 0.5
        )
        return MissionSnapshot(
            living=living,
            living_fraction=living_fraction,
            objective_health=_role_row(plan_observation, self.spec.role)[11],
            advance_inside=advance_fraction >= 0.8,
            advance_fraction=advance_fraction,
            hold_inside=_fraction_within(
                list(living), self.activation_anchor, 0.08 * self.diagonal
            ) >= 1.0,
            withdraw_inside=advance_fraction >= 0.8,
            support_qualified=support_healthy
            and support_distance is not None
            and 0.08 * self.diagonal <= support_distance <= 0.18 * self.diagonal,
            signed_lateral=self._signed_lateral(list(living)),
            support_distance=support_distance,
        )

    def _advance_temporal_state(self, snapshot: MissionSnapshot) -> None:
        name = self.spec.name
        if name == "advance":
            self.consecutive = self.consecutive + 1 if snapshot.advance_inside else 0
        elif name == "hold":
            self.inside_count += int(snapshot.hold_inside)
        elif name == "withdraw":
            self.consecutive = self.consecutive + 1 if snapshot.withdraw_inside else 0
        elif name == "support":
            self.consecutive = self.consecutive + 1 if snapshot.support_qualified else 0
        elif name == "flank" and (
            self.flank_reached_at is None
            and snapshot.signed_lateral >= 0.2 * self.lateral_extent
        ):
            self.flank_reached_at = self.decision
            self.flank_damage_at_reach = self._total_damage()

    def _progress(self, snapshot: MissionSnapshot) -> float:
        name = self.spec.name
        if name == "engage":
            return float(np.clip(1 - snapshot.objective_health, 0, 1))
        if name == "advance":
            return snapshot.advance_fraction
        if name == "hold":
            return self.inside_count / self.spec.horizon
        if name == "withdraw":
            return min(self.consecutive / 20, 1.0)
        if name == "flank":
            geometry = float(np.clip(snapshot.signed_lateral / max(0.2 * self.lateral_extent, 1e-9), 0, 1))
            return 0.5 * geometry + (0.5 if self._total_damage() > self.flank_damage_at_reach and self.flank_reached_at is not None else 0)
        if name == "focus":
            return self._hhi() * min(self._total_damage() / max(self._meaningful_damage(), 1e-9), 1)
        if name == "distributed":
            return self._entropy()
        if name == "support":
            return min(self.consecutive / 30, 1.0)
        raise AssertionError(name)

    def _success(self, snapshot: MissionSnapshot) -> bool:
        name = self.spec.name
        if name == "engage":
            return snapshot.objective_health <= 0.2
        if name == "advance":
            return self.consecutive >= 10
        if name == "hold":
            return (
                self.decision >= self.spec.horizon
                and self.inside_count / self.spec.horizon >= 0.9
                and snapshot.living_fraction >= 0.5
            )
        if name == "withdraw":
            return self.consecutive >= 20 and snapshot.living_fraction >= 0.5
        if name == "flank":
            if self.flank_reached_at is None:
                return False
            if self.decision - self.flank_reached_at > 50:
                return False
            return self._total_damage() > self.flank_damage_at_reach
        if name == "focus":
            return self._total_damage() >= self._meaningful_damage() and self._hhi() >= 0.65
        if name == "distributed":
            damaged = sum(damage > 0 for damage in self.target_damage.values())
            return damaged >= 2 and self._total_damage() >= 2 * self._meaningful_damage() and self._entropy() >= 0.65
        if name == "support":
            return self.consecutive >= 30
        raise AssertionError(name)

    def _signed_lateral(self, living: list[dict[str, Any]]) -> float:
        center = _centroid(living)
        displacement = (
            center[0] - self.activation_anchor[0],
            center[1] - self.activation_anchor[1],
        )
        lateral = displacement[0] * self.left[0] + displacement[1] * self.left[1]
        approach = self.command["order"]["approach"]
        return lateral if approach == "left_flank" else -lateral

    def _support_in_band(self, observation: dict[str, Any]) -> bool:
        distance = self._support_distance(observation)
        return distance is not None and 0.08 * self.diagonal <= distance <= 0.18 * self.diagonal

    def _support_distance(self, observation: dict[str, Any]) -> float | None:
        own = _assigned(observation, self.assigned_ids, living=True)
        supported = _assigned(observation, self.supported_ids, living=True)
        if not own or not supported:
            return None
        a, b = _centroid(own), _centroid(supported)
        return math.dist(a, b)

    def _meaningful_damage(self) -> float:
        return MEANINGFUL_DAMAGE_FRACTION * self.mean_enemy_max_health

    def _total_damage(self) -> float:
        return sum(self.target_damage.values())

    def _hhi(self) -> float:
        total = self._total_damage()
        return 0.0 if total <= 0 else sum((damage / total) ** 2 for damage in self.target_damage.values())

    def _entropy(self) -> float:
        total = self._total_damage()
        count = len(self.target_damage)
        if total <= 0 or count <= 1:
            return 0.0
        return -sum(
            (damage / total) * math.log(damage / total)
            for damage in self.target_damage.values()
            if damage > 0
        ) / math.log(count)

    def _metrics(self, snapshot: MissionSnapshot) -> dict[str, float]:
        return {
            "assignedLivingFraction": snapshot.living_fraction,
            "objectiveHealth": snapshot.objective_health,
            "targetDamage": self._total_damage(),
            "targetDamageHhi": self._hhi(),
            "targetDamageEntropy": self._entropy(),
            "holdFraction": self.inside_count / max(self.decision, 1),
            "consecutiveQualified": float(self.consecutive),
            "signedFlankSeparation": snapshot.signed_lateral,
            "supportDistanceFraction": (
                snapshot.support_distance / self.diagonal
                if snapshot.support_distance is not None
                else 0.0
            ),
        }


def _role_row(plan_observation: dict[str, Any], role: str) -> list[float]:
    values = plan_observation.get("planRoleState")
    if not isinstance(values, list) or len(values) != 60:
        raise ValueError("fixed option plan role state is invalid")
    start = ROLE_INDEX[role] * 20
    return [float(value) for value in values[start : start + 20]]


def _health_by_id(units: list[dict[str, Any]]) -> dict[int, float]:
    return {int(unit["id"]): float(unit["health"]) for unit in units}


def _team_max_health(units: list[dict[str, Any]]) -> float:
    return sum(float(unit["maxHealth"]) for unit in units)


def _assigned(
    observation: dict[str, Any], identifiers: tuple[int, ...], *, living: bool
) -> list[dict[str, Any]]:
    selected = set(identifiers)
    return [
        unit
        for unit in observation["allies"]
        if int(unit["id"]) in selected and (not living or bool(unit["alive"]))
    ]


def _centroid(units: list[dict[str, Any]]) -> tuple[float, float]:
    if not units:
        return (0.0, 0.0)
    return (
        sum(float(unit["x"]) for unit in units) / len(units),
        sum(float(unit["y"]) for unit in units) / len(units),
    )


def _fraction_within(
    units: list[dict[str, Any]], anchor: tuple[float, float], radius: float
) -> float:
    if not units:
        return 0.0
    return sum(math.dist((float(unit["x"]), float(unit["y"])), anchor) <= radius for unit in units) / len(units)
