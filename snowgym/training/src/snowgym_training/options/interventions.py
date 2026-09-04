"""Deterministic no-training causal interventions for failed Engage policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from ..executor import model_config
from ..ppo import HybridActorCritic
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import numpy_actions, tensor_dict
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from .environment import FixedPlanOptionBatchEnv
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol

FORMAT = "snowgym.engage.intervention.v0"
ARMS = ("learner", "teacher-move", "teacher-action", "teacher-action-move", "goal-anchor", "teacher")
PREFERRED_RANGE = {"close": 8.0, "medium": 9.0, "long": 10.5}


def compose_intervention_action(
    arm: str,
    learner: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    goal_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compose one hybrid action while changing only the named causal channel."""
    if arm not in ARMS:
        raise ValueError(f"unknown Engage intervention arm {arm!r}")
    action = {name: np.array(value, copy=True) for name, value in learner.items()}
    if arm == "teacher":
        return {name: np.array(value, copy=True) for name, value in teacher.items()}
    if arm in {"teacher-action", "teacher-action-move"}:
        action["action_type"] = np.array(teacher["action_type"], copy=True)
    selected = action["action_type"]
    if arm in {"teacher-move", "teacher-action-move"}:
        move = selected == ACTION_MOVE
        action["target"][move] = teacher["target"][move]
    elif arm == "goal-anchor":
        move = selected == ACTION_MOVE
        action["target"][move] = goal_target[move]
    return action


def grounded_goal_target(
    raw: dict[str, Any], plan_body: dict[str, Any], *, capacity: int
) -> tuple[np.ndarray, tuple[float, float], tuple[int, ...]]:
    assignments = plan_body.get("assignments")
    role_state = plan_body.get("planRoleState")
    if not isinstance(assignments, list) or not isinstance(role_state, list) or len(role_state) != 60:
        raise ValueError("Engage plan observation is incomplete")
    main = next((item for item in assignments if item.get("role") == "main"), None)
    if not isinstance(main, dict) or not isinstance(main.get("unitIds"), list):
        raise ValueError("Engage plan has no stable main assignment")
    assigned = tuple(int(value) for value in main["unitIds"])
    living = [
        unit for unit in raw["allies"]
        if int(unit["id"]) in assigned and bool(unit["alive"])
    ]
    center = centroid(living)
    width = float(raw["arena"]["width"])
    height = float(raw["arena"]["height"])
    anchor = (
        center[0] + float(role_state[8]) * width,
        center[1] + float(role_state[9]) * height,
    )
    target = np.zeros((capacity, 2), dtype=np.float32)
    target[:, 0] = np.clip(anchor[0] / (width / 2), -1, 1)
    target[:, 1] = np.clip(anchor[1] / (height / 2), -1, 1)
    return target, anchor, assigned


def evaluate_intervention_episode(
    model: HybridActorCritic,
    *,
    arm: str,
    seed: int,
    client: SnowGymBatchClient,
) -> dict[str, Any]:
    plan, spec = teacher_option_plan("engage")
    scenario = teacher_option_scenario("engage")
    base = SnowGymBatchEnv(1, client=client, observation_version=3)
    wrapped = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
    observation, _ = wrapped.reset(
        [seed], [scenario], [f"engage-intervention-{arm}-{seed}"], [plan], [spec]
    )
    initial_body = base.plan_observations()[1][0]
    raw = require_raw(base)
    _, initial_anchor, assigned = grounded_goal_target(
        raw, initial_body, capacity=base.max_team_units
    )
    initial_enemy_health = team_health(raw["enemies"])
    minimum_distance = objective_distance(raw, assigned, initial_anchor)
    first_contact: int | None = None
    first_throw: int | None = None
    first_hit: int | None = None
    rejected = 0
    counts = {"noop": 0, "move": 0, "throw": 0, "hold": 0}
    path: list[dict[str, Any]] = []
    final_option: dict[str, Any] | None = None
    for decision in range(spec.horizon):
        tensors = tensor_dict(observation)
        teacher = base.plan_teacher_tensor_actions()
        with torch.no_grad():
            learner_tensor, _, _ = model.act(tensors, deterministic=True)
        learner = numpy_actions(learner_tensor)
        body = base.plan_observations()[1][0]
        raw = require_raw(base)
        goal, anchor, assigned = grounded_goal_target(
            raw, body, capacity=base.max_team_units
        )
        action = compose_intervention_action(arm, learner, teacher, goal[None, ...])
        for value in action["action_type"][0, : len(raw["allies"])]:
            counts[("noop", "move", "throw", "hold")[int(value)]] += 1
        if first_throw is None and bool((action["action_type"] == ACTION_THROW).any()):
            first_throw = decision + 1
        move_mask = action["action_type"][0] == ACTION_MOVE
        move_target = action["target"][0][move_mask]
        move_center = (
            [float(move_target[:, 0].mean()), float(move_target[:, 1].mean())]
            if move_target.size else None
        )
        observation, _, terminated, truncated, infos = wrapped.step(action)
        info = infos[0]
        rejected += sum(
            item.get("accepted") is False
            for item in info.get("actionResults", [])
            if isinstance(item, dict)
        )
        raw = require_raw(base)
        next_body = base.plan_observations()[1][0]
        _, next_anchor, assigned = grounded_goal_target(
            raw, next_body, capacity=base.max_team_units
        )
        distance = objective_distance(raw, assigned, next_anchor)
        minimum_distance = min(minimum_distance, distance)
        contact = contact_distance(raw, assigned)
        if first_contact is None and contact <= PREFERRED_RANGE["medium"]:
            first_contact = decision + 1
        if first_hit is None and team_health(raw["enemies"]) < initial_enemy_health:
            first_hit = decision + 1
        done = bool(terminated[0] or truncated[0])
        if decision == 0 or (decision + 1) % 5 == 0 or done:
            path.append(
                {
                    "decision": decision + 1,
                    "blueCentroid": list(centroid(assigned_units(raw, assigned))),
                    "objectiveAnchor": list(next_anchor),
                    "objectiveDistance": distance,
                    "nearestEnemyDistance": contact,
                    "moveTargetMeanNormalized": move_center,
                }
            )
        if done:
            final_option = info["option"]
            break
    if final_option is None:
        raise RuntimeError("Engage intervention did not reach its option boundary")
    final_health = float(final_option["metrics"]["objectiveHealth"])
    return {
        "seed": seed,
        "arm": arm,
        "decisions": int(final_option["decision"]),
        "success": bool(final_option["success"]),
        "progress": float(final_option["progress"]),
        "firstContactDecision": first_contact,
        "minimumObjectiveDistance": minimum_distance,
        "firstThrowDecision": first_throw,
        "firstHitDecision": first_hit,
        "objectiveHealthChange": 1.0 - final_health,
        "rejectedActions": rejected,
        "actionCounts": counts,
        "path": path,
    }


def run_intervention_matrix(
    checkpoint: str | Path,
    *,
    output: str | Path,
    seed_count: int = 40,
) -> dict[str, Any]:
    if not 1 <= seed_count <= 40:
        raise ValueError("Engage intervention seed_count must be in [1,40]")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite intervention matrix {destination}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    config = model_config(metadata["architecture"])
    ppo_config = metadata["ppoConfig"]
    model = HybridActorCritic(
        config,
        initial_target_log_std=float(ppo_config["initial_target_log_std"]),
        initial_power_log_std=float(ppo_config["initial_power_log_std"]),
    ).eval()
    model.load_state_dict(state["model"])
    protocol = load_option_protocol()
    first_seed = int(protocol["seeds"]["development"][0])
    seeds = list(range(first_seed, first_seed + seed_count))
    records: dict[str, list[dict[str, Any]]] = {}
    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        for arm in ARMS:
            records[arm] = [
                evaluate_intervention_episode(model, arm=arm, seed=seed, client=client)
                for seed in seeds
            ]
    summary = {
        arm: {
            "successRate": sum(item["success"] for item in items) / len(items),
            "contactRate": sum(item["firstContactDecision"] is not None for item in items) / len(items),
            "hitRate": sum(item["firstHitDecision"] is not None for item in items) / len(items),
            "meanMinimumObjectiveDistance": float(
                np.mean([item["minimumObjectiveDistance"] for item in items])
            ),
            "rejectedActions": sum(item["rejectedActions"] for item in items),
        }
        for arm, items in records.items()
    }
    value = {
        "format": FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "implementationGitCommit": resolve_git_commit(),
        "simulationVersion": capabilities["simulationVersion"],
        "stateHashVersion": capabilities["stateHashVersion"],
        "upstreamBaseCommit": capabilities["upstreamBaseCommit"],
        "protocolDigest": json_digest(protocol),
        "seedPartition": "development",
        "seeds": seeds,
        "arms": list(ARMS),
        "summary": summary,
        "records": records,
    }
    value["artifactDigest"] = json_digest(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def require_capabilities(client: SnowGymBatchClient) -> dict[str, Any]:
    capabilities = client.capabilities.get("capabilities")
    required = ("simulationVersion", "stateHashVersion", "upstreamBaseCommit")
    if not isinstance(capabilities, dict) or any(
        not isinstance(capabilities.get(name), str) for name in required
    ):
        raise RuntimeError("SnowGym batch handshake lacks provenance capabilities")
    return capabilities


def require_raw(environment: SnowGymBatchEnv) -> dict[str, Any]:
    raw = environment.raw_observations[0]
    if raw is None:
        raise RuntimeError("Engage intervention has no raw observation")
    return raw


def assigned_units(raw: dict[str, Any], assigned: tuple[int, ...]) -> list[dict[str, Any]]:
    selected = set(assigned)
    return [
        unit for unit in raw["allies"]
        if int(unit["id"]) in selected and bool(unit["alive"])
    ]


def centroid(units: list[dict[str, Any]]) -> tuple[float, float]:
    if not units:
        return (0.0, 0.0)
    return (
        sum(float(unit["x"]) for unit in units) / len(units),
        sum(float(unit["y"]) for unit in units) / len(units),
    )


def objective_distance(
    raw: dict[str, Any], assigned: tuple[int, ...], anchor: tuple[float, float]
) -> float:
    return math.dist(centroid(assigned_units(raw, assigned)), anchor)


def contact_distance(raw: dict[str, Any], assigned: tuple[int, ...]) -> float:
    allies = assigned_units(raw, assigned)
    enemies = [unit for unit in raw["enemies"] if bool(unit["alive"])]
    if not allies or not enemies:
        return math.inf
    return min(
        math.dist((float(ally["x"]), float(ally["y"])), (float(enemy["x"]), float(enemy["y"])))
        for ally in allies
        for enemy in enemies
    )


def team_health(units: list[dict[str, Any]]) -> float:
    return sum(float(unit["health"]) for unit in units)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    result = run_intervention_matrix(
        args.checkpoint, output=args.output, seed_count=args.seed_count
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
