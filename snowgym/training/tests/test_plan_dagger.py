from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from snowgym_training.data import TrajectoryDataset
from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.export_plan_dagger import (
    export_plan_dagger_dataset,
    load_plan_dagger_spec,
)
from snowgym_training.merge_trajectory import merge_datasets
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
