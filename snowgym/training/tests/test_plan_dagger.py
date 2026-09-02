from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_training.data import TrajectoryDataset
from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.export_plan_dagger import (
    export_plan_dagger_dataset,
    load_plan_dagger_spec,
)
from snowgym_training.merge_trajectory import merge_datasets
from snowgym_training.plan_action_adapter_qualification import (
    load_spec as load_action_adapter_qualification_spec,
    qualify_plan_action_adapter,
)
from snowgym_training.plan_counterfactual_evaluate import (
    audit_plan_counterfactual_evaluation,
    evaluate_plan_counterfactual,
)
from snowgym_training.plan_counterfactual_qualification import (
    load_spec as load_cf_gate,
    qualify as qualify_cf_gate,
)
from snowgym_training.plan_closed_loop import audit_closed_loop
from snowgym_training.plan_weight_sweep import summarize_sweep
from snowgym_training.policy import TorchPolicy
from snowgym_training.trainer import train_behavior_clone
from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import audit_dataset


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1" / "plan-conditioned"
PLAN_SUITE = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_closed_loop_behaviors_v1.json"
)
FROZEN_SPEC = (
    ROOT / "training" / "src" / "snowgym_training" / "configs" / "plan_dagger_v0.json"
)
CORRECTION_CONFIG = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_dagger_correction_v0.json"
)
ACTION_ADAPTER_CONFIG = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_action_adapter_v0.json"
)
ACTION_ADAPTER_V1_CONFIG = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_action_adapter_v1.json"
)
ACTION_ADAPTER_V2_DEV_CONFIG = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_action_adapter_v2_dev.json"
)
ACTION_ADAPTER_V1_QUALIFICATION = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_action_adapter_qualification_v1.json"
)
ACTION_ADAPTER_V1_CHECKPOINT = ROOT / "training" / "runs" / "plan_action_adapter_v1"
ACTION_ADAPTER_V1_EVALUATION = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_v1_evaluation.json"
)
ACTION_ADAPTER_V1_CLOSED_LOOP = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_v1_closed_loop.json"
)
ACTION_ADAPTER_V1_BEHAVIORS = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_v1_behaviors.json"
)
ACTION_ADAPTER_V2_DEV_EVALUATION = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_v2_dev_evaluation.json"
)
ACTION_ADAPTER_V2_DEV_BEHAVIORS = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_v2_dev_behaviors.json"
)
ACTION_ADAPTER_QUALIFICATION = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_action_adapter_qualification_v0.json"
)
PLAN_COUNTERFACTUAL_SPEC = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_counterfactual_dagger_v1.json"
)
PLAN_ROLE_SPEC = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_role_dagger_v2.json"
)
ACTION_ADAPTER_CHECKPOINT = ROOT / "training" / "runs" / "plan_action_adapter_v0"
ACTION_ADAPTER_OFFLINE = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_offline_v0.json"
)
ACTION_ADAPTER_CLOSED_LOOP = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_closed_loop_v0.json"
)
ACTION_ADAPTER_BEHAVIORS = (
    ROOT / "training" / "evaluations" / "plan_action_adapter_behaviors_v0.json"
)
ACTION_ADAPTER_BASELINE = (
    ROOT / "training" / "evaluations" / "plan_dagger_baseline_offline_v0.json"
)
ROLE_ADAPTER_CHECKPOINT = ROOT / "training" / "runs" / "plan_role_adapter_v0_dev"
ROLE_ADAPTER_VALIDATION_DATA = (
    ROOT / "training" / "artifacts" / "plan-role-dagger-v2-validation"
)
ROLE_ADAPTER_VALIDATION = (
    ROOT / "training" / "evaluations" / "plan_role_adapter_v0_dev_validation.json"
)
ROLE_ADAPTER_CLOSED_LOOP = (
    ROOT / "training" / "evaluations" / "plan_role_adapter_v0_dev_closed_loop.json"
)
ROLE_ADAPTER_BEHAVIORS = (
    ROOT / "training" / "evaluations" / "plan_role_adapter_v0_dev_behaviors.json"
)


def test_frozen_plan_dagger_spec_balances_missions_and_disjoins_seeds() -> None:
    spec = load_plan_dagger_spec(FROZEN_SPEC)
    assert set(spec["plans"]) == {"direct", "flank", "hold", "withdraw", "support"}
    assert len(spec["splits"]["train"]) == 10
    assert len(spec["splits"]["validation"]) == 5
    assert len(spec["splits"]["evaluation"]) == 5
    assert {episode["plan"] for episode in spec["splits"]["train"]} == set(spec["plans"])
    correction = load_training_config(CORRECTION_CONFIG)
    assert correction["trainable"] == "plan-target-path"
    assert correction["steps"] == 1500
    adapter = load_training_config(ACTION_ADAPTER_CONFIG)
    assert adapter["architecture"]["plan_action_adapter"] is True
    assert adapter["trainable"] == "plan-action-target-path"
    adapter_v1 = load_training_config(ACTION_ADAPTER_V1_CONFIG)
    assert adapter_v1["counterfactualLossWeight"] == 1.0
    assert adapter_v1["steps"] == 2000
    adapter_v2 = load_training_config(ACTION_ADAPTER_V2_DEV_CONFIG)
    assert adapter_v2["counterfactualChangedActionWeight"] == 5.0
    assert adapter_v2["evaluationSuite"].endswith("development")
    gate_v1 = load_cf_gate(ACTION_ADAPTER_V1_QUALIFICATION)
    assert gate_v1["paired"]["minimumChangedTeacherPredictionRecall"] == 0.5
    assert gate_v1["evaluationDatasetDigest"].endswith("c955efd22fb")
    sweep_configs = [
        load_training_config(
            ROOT / "training" / "src" / "snowgym_training" / "configs"
            / f"plan_action_adapter_weight{weight}_dev.json"
        )
        for weight in (1, 2, 3)
    ]
    assert [item["counterfactualChangedActionWeight"] for item in sweep_configs] == [
        1.0, 2.0, 3.0
    ]
    assert len({item["seed"] for item in sweep_configs}) == 1
    sweep = json.loads(
        (ROOT / "training" / "src" / "snowgym_training" / "configs"
         / "plan_action_adapter_weight_sweep_v0.json").read_text(encoding="utf-8")
    )
    assert sweep["format"] == "snowgym.plan-action-adapter-weight-sweep.v0"
    assert sweep["trainingConfigs"] == [
        "plan_action_adapter_weight1_dev.json",
        "plan_action_adapter_weight2_dev.json",
        "plan_action_adapter_weight3_dev.json",
    ]
    counterfactual = load_plan_dagger_spec(PLAN_COUNTERFACTUAL_SPEC)
    assert counterfactual["format"] == "snowgym.plan-dagger-export.v1"
    episodes = sum(counterfactual["splits"].values(), [])
    assert len({episode["seed"] for episode in episodes}) == 20
    assert all(
        episode["plan"] != episode["counterfactualPlan"] for episode in episodes
    )
    role_spec = load_plan_dagger_spec(PLAN_ROLE_SPEC)
    assert role_spec["format"] == "snowgym.plan-dagger-export.v2"
    role_episodes = sum(role_spec["splits"].values(), [])
    assert len({episode["seed"] for episode in role_episodes}) == 20
    assert min(episode["seed"] for episode in role_episodes) == 15201
    role_config = load_training_config(
        ROOT / "training" / "src" / "snowgym_training" / "configs"
        / "plan_role_adapter_v0_dev.json"
    )
    assert role_config["architecture"]["plan_role_conditioned"] is True
    assert role_config["counterfactualChangedActionWeight"] == 1.0


def test_frozen_plan_action_adapter_gate_is_audited_and_retains_failure(tmp_path: Path) -> None:
    spec = load_action_adapter_qualification_spec(ACTION_ADAPTER_QUALIFICATION)
    assert spec["offline"]["minimumCounterfactualActionChangeRate"] == 0.05
    result = qualify_plan_action_adapter(
        spec_path=ACTION_ADAPTER_QUALIFICATION,
        checkpoint=ACTION_ADAPTER_CHECKPOINT,
        baseline_path=ACTION_ADAPTER_BASELINE,
        offline_path=ACTION_ADAPTER_OFFLINE,
        closed_loop_path=ACTION_ADAPTER_CLOSED_LOOP,
        behaviors_path=ACTION_ADAPTER_BEHAVIORS,
        output=tmp_path / "qualification.json",
    )
    assert result["passed"] is False
    assert result["checks"]["actionAccuracy"] is True
    assert result["checks"]["counterfactualActionChangeRate"] is False
    assert result["checks"]["directBlueAlive"] is False
    assert result["checks"]["holdDecisions"] is True
    assert result["checks"]["withdrawDecisions"] is True
    assert result["checks"]["supportRedAlive"] is False

    tampered = json.loads(ACTION_ADAPTER_OFFLINE.read_text(encoding="utf-8"))
    tampered["metrics"]["actionAccuracy"] = 1.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        qualify_plan_action_adapter(
            spec_path=ACTION_ADAPTER_QUALIFICATION,
            checkpoint=ACTION_ADAPTER_CHECKPOINT,
            baseline_path=ACTION_ADAPTER_BASELINE,
            offline_path=tampered_path,
            closed_loop_path=ACTION_ADAPTER_CLOSED_LOOP,
            behaviors_path=ACTION_ADAPTER_BEHAVIORS,
            output=tmp_path / "tampered-result.json",
        )


def test_counterfactual_adapter_v1_gate_retains_partial_closed_loop_progress(
    tmp_path: Path,
) -> None:
    result = qualify_cf_gate(
        spec_path=ACTION_ADAPTER_V1_QUALIFICATION,
        checkpoint=ACTION_ADAPTER_V1_CHECKPOINT,
        paired_path=ACTION_ADAPTER_V1_EVALUATION,
        closed_loop_path=ACTION_ADAPTER_V1_CLOSED_LOOP,
        behaviors_path=ACTION_ADAPTER_V1_BEHAVIORS,
        output=tmp_path / "v1-qualification.json",
    )
    assert result["passed"] is False
    assert result["checks"]["predictedActionChangeRate"] is True
    assert result["checks"]["changedTeacherPredictionRecall"] is False
    assert result["checks"]["flankBlueAlive"] is True
    assert result["checks"]["supportRedAlive"] is True
    assert result["observed"]["closedLoop"]["supportRedAlive"] == 0


def test_changed_action_v2_development_retains_measured_tradeoff() -> None:
    paired = json.loads(ACTION_ADAPTER_V2_DEV_EVALUATION.read_text(encoding="utf-8"))
    behaviors = json.loads(ACTION_ADAPTER_V2_DEV_BEHAVIORS.read_text(encoding="utf-8"))
    metrics = paired["metrics"]
    assert metrics["changedTeacherPredictionRecall"] > 0.75
    assert metrics["changedTeacherPairAccuracy"] > 0.69
    assert metrics["predictedActionChangeRate"] > 0.39
    conditioned = {
        item["caseId"]: item
        for item in behaviors["results"] if item["policy"] == "planConditioned"
    }
    assert conditioned["withdraw-backfield"]["decisions"] == 600
    assert conditioned["withdraw-backfield"]["blueAlive"] == 2
    assert conditioned["main-with-reserve-support"]["winner"] == "red"


def test_changed_action_weight_sweep_applies_predeclared_selection(tmp_path: Path) -> None:
    result = summarize_sweep(
        spec_path=(
            ROOT / "training" / "src" / "snowgym_training" / "configs"
            / "plan_action_adapter_weight_sweep_v0.json"
        ),
        runs_dir=ROOT / "training" / "runs",
        evaluations_dir=ROOT / "training" / "evaluations",
        output=tmp_path / "weight-sweep.json",
    )
    assert result["selected"] == "weight3"
    assert [entry["checksPassed"] for entry in result["entries"]] == [7, 7, 7]
    assert result["entries"][0]["checks"]["maximumPredictedActionChangeRate"] is True
    assert result["entries"][2]["checks"]["maximumPredictedActionChangeRate"] is False


def test_role_adapter_v0_retains_failed_development_outcome() -> None:
    paired = audit_plan_counterfactual_evaluation(
        ROLE_ADAPTER_VALIDATION,
        ROLE_ADAPTER_CHECKPOINT,
        ROLE_ADAPTER_VALIDATION_DATA,
    )
    assert paired["metrics"]["primaryActionAccuracy"] > 0.86
    assert paired["metrics"]["predictedActionChangeRate"] > paired["metrics"][
        "teacherActionChangeRate"
    ]
    closed_loop = audit_closed_loop(
        ROLE_ADAPTER_CLOSED_LOOP,
        ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1",
        ROOT / "training" / "src" / "snowgym_training" / "configs"
        / "plan_closed_loop_v0.json",
        ROLE_ADAPTER_CHECKPOINT,
    )
    behaviors = audit_closed_loop(
        ROLE_ADAPTER_BEHAVIORS,
        ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1",
        PLAN_SUITE,
        ROLE_ADAPTER_CHECKPOINT,
    )
    approaches = {
        item["caseId"]: item
        for item in closed_loop["results"]
        if item["policy"] == "planConditioned"
    }
    assert approaches["direct-focus"]["blueAlive"] == 4
    assert approaches["left-flank-distributed"]["blueAlive"] == 4
    missions = {
        item["caseId"]: item
        for item in behaviors["results"]
        if item["policy"] == "planConditioned"
    }
    assert all(item["winner"] == "red" for item in missions.values())
    assert missions["main-with-reserve-support"]["redAlive"] == 6


def test_plan_dagger_labels_learner_visited_states_headlessly(tmp_path: Path) -> None:
    case = json.loads(PLAN_SUITE.read_text(encoding="utf-8"))["cases"][0]
    spec = {
        "format": "snowgym.plan-dagger-export.v0",
        "name": "plan-dagger-smoke",
        "teacher": "plan-teacher-action.v0",
        "maxTeamUnits": 10,
        "shardSize": 8,
        "plans": {"hold": case["plan"]},
        "splits": {
            name: [{
                "seed": seed,
                "scenario": case["scenario"],
                "plan": "hold",
            }]
            for name, seed in (("train", 12001), ("validation", 12002), ("evaluation", 12003))
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert load_plan_dagger_spec(spec_path)["plans"]["hold"] == case["plan"]
    output = tmp_path / "dataset"
    manifest = export_plan_dagger_dataset(
        output=output,
        checkpoint=CHECKPOINT,
        split="train",
        spec_path=spec_path,
        max_decisions=2,
    )

    assert manifest["teacher"] == "plan-teacher-action.v0"
    assert manifest["planConditioned"] is True
    assert manifest["transitions"] == 2
    assert manifest["episodes"][0]["decisionLimited"] is True
    assert audit_dataset(output)["datasetDigest"] == manifest["datasetDigest"]
    dataset = TrajectoryDataset(output)
    assert "plan_groups" in dataset.observation_fields
    assert dataset.arrays["observation__plan_groups"].shape == (2, 3, 38)
    with np.load(output / manifest["shards"][0]["path"], allow_pickle=False) as shard:
        assert np.all(shard["teacher_accepted"])
    merged = merge_datasets(output=tmp_path / "merged", inputs=[output, output])
    assert merged["planConditioned"] is True

    initial_metadata, initial_state = load_checkpoint(CHECKPOINT)
    config = {
        "format": "snowgym.bc-training-config.v0",
        "name": "plan-target-transfer-smoke",
        "seed": 44001,
        "steps": 1,
        "batchSize": 4,
        "learningRate": 0.001,
        "architecture": initial_metadata["architecture"],
        "loss": initial_metadata["loss"],
        "evaluationSuite": "plan-target-transfer-smoke",
        "trainable": "plan-target-path",
    }
    trained = train_behavior_clone(
        dataset_path=tmp_path / "merged",
        output=tmp_path / "trained",
        config=config,
        initialize=CHECKPOINT,
        git_commit="test",
    )
    _, trained_state = load_checkpoint(tmp_path / "trained")
    assert trained["initialization"]["checkpointDigest"] == initial_metadata["checkpointDigest"]
    assert trained_state["model"]["action_head.weight"].equal(
        initial_state["model"]["action_head.weight"]
    )
    assert not trained_state["model"]["plan_encoder.0.weight"].equal(
        initial_state["model"]["plan_encoder.0.weight"]
    )

    adapter_config = {
        **config,
        "name": "plan-action-adapter-transfer-smoke",
        "architecture": {**config["architecture"], "plan_action_adapter": True},
        "trainable": "plan-action-target-path",
    }
    train_behavior_clone(
        dataset_path=tmp_path / "merged",
        output=tmp_path / "adapted",
        config=adapter_config,
        initialize=CHECKPOINT,
        git_commit="test",
    )
    _, adapted_state = load_checkpoint(tmp_path / "adapted")
    assert adapted_state["model"]["action_head.weight"].equal(
        initial_state["model"]["action_head.weight"]
    )
    assert bool(torch.count_nonzero(adapted_state["model"]["plan_action_adapter.2.weight"]))


def test_plan_dagger_v2_retains_same_state_counterfactual_roles(tmp_path: Path) -> None:
    direct = json.loads(
        (ROOT / "training" / "src" / "snowgym_training" / "configs"
         / "plan_closed_loop_v0.json").read_text(encoding="utf-8")
    )["cases"][0]
    hold = json.loads(
        (ROOT / "training" / "src" / "snowgym_training" / "configs"
         / "plan_closed_loop_behaviors_v1.json").read_text(encoding="utf-8")
    )["cases"][0]
    spec = {
        "format": "snowgym.plan-dagger-export.v2",
        "name": "plan-counterfactual-smoke",
        "teacher": "plan-teacher-action.v0",
        "maxTeamUnits": 10,
        "shardSize": 8,
        "plans": {"direct": direct["plan"], "hold": hold["plan"]},
        "splits": {
            name: [{
                "seed": seed,
                "scenario": direct["scenario"],
                "plan": "direct" if name != "validation" else "hold",
                "counterfactualPlan": "hold" if name != "validation" else "direct",
            }]
            for name, seed in (("train", 12101), ("validation", 12102), ("evaluation", 12103))
        },
    }
    spec_path = tmp_path / "counterfactual-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "counterfactual"
    manifest = export_plan_dagger_dataset(
        output=output,
        checkpoint=CHECKPOINT,
        split="train",
        spec_path=spec_path,
        max_decisions=2,
    )
    assert manifest["counterfactualPlanLabels"] == {
        "teacher": "plan-teacher-action.v0",
        "pairing": "same-physical-state",
    }
    assert manifest["planUnitRoles"] is True
    assert manifest["episodes"][0]["counterfactualPlanName"] == "hold"
    assert manifest["episodes"][0]["counterfactualLimited"] is False
    assert audit_dataset(output)["datasetDigest"] == manifest["datasetDigest"]
    dataset = TrajectoryDataset(output)
    assert dataset.counterfactual_plan_labels is True
    observation, action = dataset.batch(np.asarray([0, 1]))
    assert observation["counterfactual_plan_groups"].shape == (2, 3, 38)
    assert observation["plan_unit_roles"].shape == (2, 10, 3)
    assert observation["counterfactual_plan_unit_roles"].shape == (2, 10, 3)
    assert action["counterfactual_action_type"].shape == (2, 10)
    assert not np.array_equal(
        dataset.arrays["observation__plan_groups"][0],
        dataset.arrays["observation__counterfactual_plan_groups"][0],
    )

    initial_metadata, _ = load_checkpoint(CHECKPOINT)
    config = {
        "format": "snowgym.bc-training-config.v0",
        "name": "plan-counterfactual-transfer-smoke",
        "seed": 44101,
        "steps": 1,
        "batchSize": 2,
        "learningRate": 0.001,
        "architecture": {
            **initial_metadata["architecture"],
            "plan_action_adapter": True,
            "plan_role_conditioned": True,
        },
        "loss": initial_metadata["loss"],
        "evaluationSuite": "plan-counterfactual-transfer-smoke",
        "trainable": "plan-action-target-path",
        "counterfactualLossWeight": 1.0,
        "counterfactualChangedActionWeight": 2.0,
    }
    trained = train_behavior_clone(
        dataset_path=output,
        output=tmp_path / "counterfactual-trained",
        config=config,
        initialize=CHECKPOINT,
        git_commit="test",
    )
    assert trained["trainingMetrics"]["final"]["counterfactual"] >= 0
    assert trained["trainingMetrics"]["final"]["counterfactualChangedAction"] >= 0
    policy_observation = {
        name: tensor[0].numpy()
        for name, tensor in observation.items()
        if not name.startswith("counterfactual_")
    }
    action_from_policy = TorchPolicy(tmp_path / "counterfactual-trained").act(
        policy_observation
    )
    assert action_from_policy["action_type"].shape == (10,)
    evaluation_path = tmp_path / "counterfactual-evaluation.json"
    evaluation = evaluate_plan_counterfactual(
        checkpoint=tmp_path / "counterfactual-trained",
        dataset_path=output,
        output=evaluation_path,
    )
    assert evaluation["metrics"]["presentUnitDecisions"] == 12
    assert 0 <= evaluation["metrics"]["teacherActionChangeRate"] <= 1
    assert audit_plan_counterfactual_evaluation(
        evaluation_path, tmp_path / "counterfactual-trained", output
    )["evaluationDigest"] == evaluation["evaluationDigest"]
    with pytest.raises(ValueError, match="requires same-state plan labels"):
        evaluate_plan_counterfactual(
            checkpoint=CHECKPOINT,
            dataset_path=ACTION_ADAPTER_OFFLINE.parent.parent / "artifacts"
            / "plan-dagger-v0-evaluation",
            output=tmp_path / "invalid-evaluation.json",
        )
