from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from snowgym_training.data import TrajectoryDataset
from snowgym_training.export_plan_dagger import export_plan_dagger_dataset
from snowgym_training.trajectory import audit_dataset


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1" / "plan-conditioned"
PLAN_SUITE = (
    ROOT / "training" / "src" / "snowgym_training" / "configs"
    / "plan_closed_loop_behaviors_v1.json"
)


def test_plan_dagger_labels_learner_visited_states_headlessly(tmp_path: Path) -> None:
    case = json.loads(PLAN_SUITE.read_text(encoding="utf-8"))["cases"][0]
    spec = {
        "format": "snowgym.trajectory-export.v0",
        "name": "plan-dagger-smoke",
        "teacher": "plan-teacher-action.v0",
        "maxTeamUnits": 10,
        "shardSize": 8,
        "splits": {
            name: [{
                "seed": seed,
                "scenario": case["scenario"],
                "planId": f"{case['planId']}-{name}",
                "plan": case["plan"],
            }]
            for name, seed in (("train", 12001), ("validation", 12002), ("evaluation", 12003))
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
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
