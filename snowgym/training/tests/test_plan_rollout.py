from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from snowgym_client.state_hash import hash_observation
from snowgym_training.data import TrajectoryDataset
from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.plan_data import javascript_json_digest
from snowgym_training.plan_ablation import audit_plan_ablation, run_plan_ablation
from snowgym_training.plan_evaluate import (
    audit_plan_evaluation,
    evaluate_plan_ablation,
)
from snowgym_training.plan_qualification import (
    qualify_plan_evaluation,
    validate_plan_qualification_spec,
)
from snowgym_training.plan_rollout import (
    audit_plan_rollouts,
    convert_plan_rollouts,
    load_plan_rollouts,
)
from snowgym_training.trajectory import audit_dataset, json_digest


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


def test_target_only_ablation_keeps_trained_action_parameters_identical(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.json"
    source.write_text(json.dumps(two_plan_rollout_value()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    convert_plan_rollouts(source=source, output=dataset, max_team_units=3)
    architecture = {
        **ablation_config()["architecture"],
        "action_conditioned_targets": True,
        "separate_target_actor": True,
        "plan_target_only": True,
    }
    run_plan_ablation(
        dataset_path=dataset,
        output=tmp_path / "ablation",
        config={**ablation_config(), "steps": 3, "architecture": architecture},
        git_commit="test",
    )
    _, no_plan = load_checkpoint(tmp_path / "ablation" / "no-plan")
    _, conditioned = load_checkpoint(tmp_path / "ablation" / "plan-conditioned")
    shared_prefixes = (
        "ally_encoder.",
        "enemy_encoder.",
        "projectile_encoder.",
        "obstacle_encoder.",
        "actor.",
        "action_head.",
    )
    for name, value in no_plan["model"].items():
        if name.startswith(shared_prefixes):
            assert name in conditioned["model"]
            assert np.array_equal(value.numpy(), conditioned["model"][name].numpy())


def test_counterfactual_evaluation_swaps_only_plan_input(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.json"
    source.write_text(json.dumps(two_plan_rollout_value()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    convert_plan_rollouts(source=source, output=dataset, max_team_units=3)
    ablation = tmp_path / "ablation"
    run_plan_ablation(
        dataset_path=dataset,
        output=ablation,
        config={**ablation_config(), "steps": 2},
        git_commit="test",
    )

    output = tmp_path / "evaluation.json"
    result = evaluate_plan_ablation(
        ablation_path=ablation, dataset_path=dataset, output=output
    )
    assert result["episodes"] == 2
    assert result["metrics"]["noPlan"]["counterfactualActionChangeRate"] == 0.0
    assert result["metrics"]["noPlan"]["counterfactualTargetMeanAbsoluteDelta"] == 0.0
    assert result["metrics"]["noPlan"]["counterfactualTargetMseDelta"] == 0.0
    assert (
        result["metrics"]["planConditioned"]["counterfactualTargetMeanAbsoluteDelta"]
        > 0.0
    )
    assert audit_plan_evaluation(output, ablation, dataset) == result

    corrupt = json.loads(output.read_text(encoding="utf-8"))
    corrupt["metrics"]["noPlan"]["actionAccuracy"] = 2.0
    output.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_plan_evaluation(output, ablation, dataset)


def test_predeclared_plan_qualification_gate_and_seed_separation(tmp_path: Path) -> None:
    config = ablation_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    spec = qualification_spec(config)
    validate_plan_qualification_spec(spec)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    evaluation_body = {
        "format": "snowgym.plan-counterfactual-evaluation.v0",
        "ablationResultDigest": "sha256:" + "1" * 64,
        "evaluationDatasetDigest": "sha256:" + "2" * 64,
        "episodes": 2,
        "transitions": 2,
        "metrics": qualification_metrics(),
    }
    evaluation = {**evaluation_body, "evaluationDigest": json_digest(evaluation_body)}
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    result = qualify_plan_evaluation(
        evaluation_path=evaluation_path,
        spec_path=spec_path,
        config_path=config_path,
        output=tmp_path / "qualification.json",
    )
    assert result["passed"] is True
    assert all(result["checks"].values())

    overlap = json.loads(json.dumps(spec))
    overlap["evaluationRollout"]["planSeed"] = 120
    with pytest.raises(ValueError, match="plan seed ranges must be disjoint"):
        validate_plan_qualification_spec(overlap)


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


def two_plan_rollout_value() -> dict:
    value = rollout_value()
    second_sample = json.loads(json.dumps(value["curriculum"]["samples"][0]))
    second_sample["sourceSeed"] = 121
    second_sample["planId"] = "synthetic-plan-121"
    second_episode = json.loads(json.dumps(value["episodes"][0]))
    second_episode["sourceSeed"] = 121
    second_episode["planId"] = "synthetic-plan-121"
    transition = second_episode["transitions"][0]
    transition["planGroups"][0] = 0.0
    transition["planGroups"][3] = 1.0
    transition["action"] = {
        "actions": [
            {"type": "move", "unitId": index + 1, "x": -5.0, "y": 2.0}
            for index in range(3)
        ]
    }
    transition["actionResults"] = [
        {"action": item, "accepted": True} for item in transition["action"]["actions"]
    ]
    transition["postStateHash"] = "fnv1a64:2222222222222222"
    second_episode["outcome"]["finalStateHash"] = transition["postStateHash"]
    value["curriculum"]["samples"].append(second_sample)
    value["curriculum"]["sampleCount"] = 2
    value["episodes"].append(second_episode)
    body = {name: item for name, item in value.items() if name != "datasetDigest"}
    value["datasetDigest"] = javascript_json_digest(body)
    return value


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


def qualification_spec(config: dict) -> dict:
    rollout = {
        "map": "arena6.json",
        "blueUnits": 6,
        "redUnits": 6,
        "samples": 6,
        "maxDecisions": 80,
        "redDifficulty": "easy",
    }
    return {
        "format": "snowgym.plan-qualification-spec.v0",
        "name": "test-qualification",
        "trainingRollout": {**rollout, "environmentSeed": 1, "planSeed": 120},
        "evaluationRollout": {**rollout, "environmentSeed": 2, "planSeed": 600},
        "ablationConfigDigest": json_digest(config),
        "thresholds": {
            "maxConditionedTargetMse": 0.1,
            "maxTargetMseRatio": 0.5,
            "minTargetMseSwapDelta": 0.1,
            "minTargetMeanAbsoluteDelta": 0.2,
            "maxActionAccuracyDeficit": 0.03,
            "noPlanMaxAbsoluteSensitivity": 1e-12,
        },
    }


def qualification_metrics() -> dict:
    common = {
        "firstDecisionActionAccuracy": 1.0,
        "correctPlanActionNll": 0.1,
        "shuffledPlanActionNll": 0.1,
        "counterfactualNllDelta": 0.0,
    }
    return {
        "noPlan": {
            **common,
            "actionAccuracy": 0.97,
            "correctPlanTargetMse": 0.3,
            "shuffledPlanTargetMse": 0.3,
            "counterfactualTargetMseDelta": 0.0,
            "counterfactualActionChangeRate": 0.0,
            "counterfactualTargetMeanAbsoluteDelta": 0.0,
        },
        "planConditioned": {
            **common,
            "actionAccuracy": 0.95,
            "correctPlanTargetMse": 0.05,
            "shuffledPlanTargetMse": 0.35,
            "counterfactualTargetMseDelta": 0.3,
            "counterfactualActionChangeRate": 0.0,
            "counterfactualTargetMeanAbsoluteDelta": 0.4,
        },
    }
