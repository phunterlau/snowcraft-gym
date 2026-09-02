from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from snowgym_training.plan_data import PlanTensorDataset, audit_plan_tensor_dataset
from snowgym_training.trajectory import json_digest


def test_plan_tensor_loader_verifies_digest_shapes_and_transition_join(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "plans.json")
    dataset = PlanTensorDataset(path)

    assert len(dataset) == 2
    assert dataset.plan_groups.shape == (2, 3, 38)
    assert dataset.plan_groups.dtype == np.float32
    assert dataset.plan_group_mask.dtype == np.int8
    assert dataset.source_seeds.tolist() == [100, 101]
    assert dataset.plan_groups.flags.writeable is False
    joined = dataset.batch_for_transitions(np.asarray([1, 1, 0], dtype=np.int64))
    assert joined["plan_groups"].shape == (3, 3, 38)
    assert joined["plan_group_mask"].tolist() == [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    assert joined["plan_source_seed"].tolist() == [101, 101, 100]
    joined["plan_groups"][0, 0, 0] = 0.25
    assert dataset.plan_groups[1, 0, 0] == 1.0


def test_plan_tensor_loader_rejects_corruption_and_bad_indices(tmp_path: Path) -> None:
    value = dataset_value()
    value["tensors"][0]["groups"][0] = 2.0
    with pytest.raises(ValueError, match="outside"):
        audit_plan_tensor_dataset(value)

    dataset = PlanTensorDataset(write_dataset(tmp_path / "plans.json"))
    with pytest.raises(IndexError, match="outside"):
        dataset.batch_for_transitions(np.asarray([2], dtype=np.int64))
    with pytest.raises(ValueError, match="one-dimensional integer"):
        dataset.batch_for_transitions(np.asarray([0.0], dtype=np.float32))


def write_dataset(path: Path) -> Path:
    path.write_text(json.dumps(dataset_value(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def dataset_value() -> dict:
    samples = [
        {
            "sourceSeed": seed,
            "planId": f"synthetic-plan-{seed}",
            "plan": {},
            "assignments": [],
        }
        for seed in (100, 101)
    ]
    groups = [[0.0] * (3 * 38) for _ in samples]
    groups[0][0] = 1.0
    groups[1][0] = 1.0
    body = {
        "format": "snowgym.plan-tensor-dataset.v0",
        "apiVersion": "snowgym.v0",
        "simulationVersion": "test",
        "stateHashVersion": "test",
        "upstreamBaseCommit": "test",
        "scenario": "test",
        "environmentSeed": 42,
        "sourceStateHash": "fnv1a64:0000000000000000",
        "curriculum": {
            "format": "snowgym.synthetic-plan-curriculum.v0",
            "baseSeed": 100,
            "sampleCount": 2,
            "source": {},
            "samples": samples,
        },
        "tensors": [
            {"sourceSeed": 100, "groups": groups[0], "groupMask": [1, 0, 0]},
            {"sourceSeed": 101, "groups": groups[1], "groupMask": [1, 1, 0]},
        ],
    }
    return {**body, "datasetDigest": json_digest(body)}
