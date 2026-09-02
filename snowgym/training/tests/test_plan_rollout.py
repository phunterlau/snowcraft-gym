from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from snowgym_client.state_hash import hash_observation
from snowgym_training.data import TrajectoryDataset
from snowgym_training.plan_data import javascript_json_digest
from snowgym_training.plan_ablation import audit_plan_ablation, run_plan_ablation
from snowgym_training.plan_rollout import (
    audit_plan_rollouts,
    convert_plan_rollouts,
    load_plan_rollouts,
)
from snowgym_training.trajectory import audit_dataset


def test_javascript_digest_matches_json_stringify_number_forms() -> None:
    value = {"e": 1.0, "d": 1e21, "c": 1e20, "b": 1e-6, "a": 1e-7}
    canonical = (
        '{"a":1e-7,"b":0.000001,"c":100000000000000000000,'
        '"d":1e+21,"e":1}'
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    assert javascript_json_digest(value) == expected


def test_plan_rollout_conversion_is_audited_deterministic_and_model_ready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rollouts.json"
    source.write_text(json.dumps(rollout_value(), indent=2) + "\n", encoding="utf-8")
    assert load_plan_rollouts(source)["datasetDigest"].startswith("sha256:")

    first = convert_plan_rollouts(
        source=source, output=tmp_path / "first", max_team_units=3, shard_size=1
    )
    second = convert_plan_rollouts(
        source=source, output=tmp_path / "second", max_team_units=3, shard_size=1
    )
    assert first == second
    assert first["sourcePlanRolloutDigest"] == load_plan_rollouts(source)["datasetDigest"]
    assert audit_dataset(tmp_path / "first")["transitions"] == 1

    dataset = TrajectoryDataset(tmp_path / "first")
    assert dataset.observation_fields[-2:] == ("plan_groups", "plan_group_mask")
    observation, action = dataset.batch(np.asarray([0], dtype=np.int64))
    assert observation["plan_groups"].shape == (1, 3, 38)
    assert observation["plan_group_mask"].tolist() == [[1, 0, 0]]
    assert action["action_type"].shape == (1, 3)


def test_plan_rollout_audit_rejects_corruption_and_small_capacity(tmp_path: Path) -> None:
    value = rollout_value()
    value["episodes"][0]["transitions"][0]["planGroups"][0] = 2.0
    with pytest.raises(ValueError, match="plan tensor"):
        audit_plan_rollouts(value)

    valid = rollout_value()
    source = tmp_path / "rollouts.json"
    source.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="must cover"):
        convert_plan_rollouts(source=source, output=tmp_path / "out", max_team_units=1)


def test_matched_plan_ablation_trains_both_models_reproducibly(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.json"
    source.write_text(json.dumps(rollout_value()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    convert_plan_rollouts(source=source, output=dataset, max_team_units=3)
    config = ablation_config()

    first = run_plan_ablation(
        dataset_path=dataset, output=tmp_path / "first", config=config, git_commit="test"
    )
    second = run_plan_ablation(
        dataset_path=dataset, output=tmp_path / "second", config=config, git_commit="test"
    )
    assert first == second
    assert first["runs"]["noPlan"]["architecture"].get("plan_conditioned") is None
    assert first["runs"]["planConditioned"]["architecture"]["plan_conditioned"] is True
    assert audit_plan_ablation(tmp_path / "first") == first

    metadata_path = tmp_path / "first" / "plan-conditioned" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpointDigest"] = "sha256:" + "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint metadata digest mismatch"):
        audit_plan_ablation(tmp_path / "first")


def rollout_value() -> dict:
    observation = {
        "tick": 0,
        "selfTeam": "blue",
        "simulationHz": 60,
        "arena": {"width": 40, "height": 30},
        "allies": [unit(index + 1, "blue", -10, index * 2 - 2) for index in range(3)],
        "enemies": [unit(101, "red", 10, 0)],
        "projectiles": [],
        "obstacles": [],
        "match": {"blueAlive": 3, "redAlive": 1},
    }
    state_hash = hash_observation(observation)
    plan = {
        "schemaVersion": "snowgym.command-plan.v0",
        "intentSummary": "Hold the initial line.",
        "groups": [],
    }
    assignments: list[dict] = []
    sample = {
        "sourceSeed": 120,
        "planId": "synthetic-plan-120",
        "plan": plan,
        "assignments": assignments,
    }
    action = {
        "actions": [
            {"type": "hold", "unitId": index + 1} for index in range(3)
        ]
    }
    results = [{"action": item, "accepted": True} for item in action["actions"]]
    final_hash = "fnv1a64:1111111111111111"
    transition = {
        "decision": 0,
        "observation": observation,
        "action": action,
        "planGroups": [1.0] + [0.0] * 113,
        "planGroupMask": [1, 0, 0],
        "reward": 0,
        "terminated": False,
        "truncated": False,
        "preStateHash": state_hash,
        "postStateHash": final_hash,
        "nextTick": 6,
        "actionResults": results,
    }
    body = {
        "format": "snowgym.plan-rollout-dataset.v0",
        "apiVersion": "snowgym.v0",
        "simulationVersion": "test",
        "stateHashVersion": "snowgym.state.v1",
        "upstreamBaseCommit": "test",
        "scenario": "test",
        "environmentSeed": 42,
        "decisionHz": 10,
        "configuration": {
            "blueUnits": 3,
            "redUnits": 1,
            "arenaWidth": 40,
            "arenaHeight": 30,
            "maxTicks": 100,
            "redDifficulty": "easy",
            "redController": "scripted",
            "map": None,
        },
        "sourceStateHash": state_hash,
        "maxDecisions": 1,
        "curriculum": {
            "format": "snowgym.synthetic-plan-curriculum.v0",
            "baseSeed": 120,
            "sampleCount": 1,
            "source": {},
            "samples": [sample],
        },
        "episodes": [
            {
                **sample,
                "initialStateHash": state_hash,
                "transitions": [transition],
                "outcome": {
                    "decisions": 1,
                    "terminated": False,
                    "truncated": False,
                    "decisionLimited": True,
                    "winner": None,
                    "blueAlive": 3,
                    "redAlive": 1,
                    "finalTick": 6,
                    "finalStateHash": final_hash,
                },
            }
        ],
    }
    return {**body, "datasetDigest": javascript_json_digest(body)}


def unit(unit_id: int, team: str, x: float, y: float) -> dict:
    return {
        "id": unit_id,
        "team": team,
        "x": x,
        "y": y,
        "vx": 0,
        "vy": 0,
        "health": 100,
        "maxHealth": 100,
        "alive": True,
        "state": "idle",
        "throwCooldown": 0,
        "charge": 0,
    }


def ablation_config() -> dict:
    return {
        "format": "snowgym.plan-bc-ablation-config.v0",
        "name": "plan-smoke",
        "seed": 17,
        "steps": 1,
        "batchSize": 1,
        "learningRate": 0.001,
        "architecture": {
            "entity_hidden": 8,
            "entity_embedding": 4,
            "actor_hidden": 8,
        },
        "loss": {
            "action_weight": 1.0,
            "target_weight": 1.0,
            "power_weight": 1.0,
        },
        "evaluationSuite": "plan-smoke",
    }
