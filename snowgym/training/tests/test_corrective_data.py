import copy
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options import corrective_data as corrective
from snowgym_training.options import recovery_lineage as lineage
from snowgym_training.options.geometry_probe import load_probe

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/m7b_engage_r1i_geometry_probe_v0/absolute-epoch-020"
RESERVOIR = ROOT / "runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz"
AUDIT = ROOT / "runs/m7b_engage_r1k_opportunities_v0"


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    model, parent = load_probe(CHECKPOINT)
    with gzip.open(AUDIT / "teacher-states.jsonl.gz", "rt") as stream:
        teacher_rows = [json.loads(line) for line in itertools.islice(stream, 16)]
    with gzip.open(AUDIT / "learner-states.jsonl.gz", "rt") as stream:
        learner_rows = [json.loads(line) for line in itertools.islice(stream, 16)]
    # Unequal episode sizes make uniform-episode sampling testable.
    for rows in (teacher_rows, learner_rows):
        for i, row in enumerate(rows):
            row["seed"] = 100000 if i < 2 else 100001
    return model, parent, corrective.OpportunityDataset(teacher_rows), corrective.OpportunityDataset(learner_rows)


def test_reservations_fail_on_known_exposure_and_overlap():
    assert lineage.reserve_seeds([108002], lineage.RESERVED) == {"teacherRegression": [108002]}
    with pytest.raises(ValueError, match="overlap"):
        lineage.reserve_seeds([], {"one": [1, 2], "two": [2, 3]})
    with pytest.raises(ValueError):
        lineage.reserve_seeds([], {"one": [True, 3]})


def test_original_dataset_required_and_recovered_ancestry_passes(tmp_path):
    kwargs = dict(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, checkpoint_root=ROOT / "runs")
    missing = lineage.audit_ancestry(**kwargs, dataset_roots=[])
    assert not missing["passed"] and missing["missing"][0]["kind"] == "dataset"
    restored = lineage.audit_ancestry(**kwargs, dataset_roots=[ROOT / "artifacts"])
    assert restored["passed"] and not restored["collisions"]
    assert 101600 in restored["knownTrainingSeeds"] and 8200 in restored["knownTrainingSeeds"]
    assert 108000 not in restored["knownTrainingSeeds"]
    assert not restored["freshHoldoutsCollected"]
    result = lineage.run_preflight(**kwargs, output=tmp_path / "run", dataset_roots=[ROOT / "artifacts"])
    assert result["passed"]
    with pytest.raises(FileExistsError):
        lineage.run_preflight(**kwargs, output=tmp_path / "run")


def test_matched_sampling_streams_equal_counts_and_episode_balance(inputs):
    teacher, learner = inputs[2:]
    indices = teacher.sample(10000, np.random.default_rng(1))
    assert .47 < np.mean(teacher.seeds[indices] == 100000) < .53
    for left, right in (("A", "C"), ("B", "D")):
        x = corrective.sample_batch(teacher, learner, arm=left, count=256, rng=np.random.default_rng(93001))
        y = corrective.sample_batch(teacher, learner, arm=right, count=256, rng=np.random.default_rng(93001))
        for key in x[2]:
            np.testing.assert_array_equal(x[2][key], y[2][key])
        assert sum(len(v) for v in x[2].values()) == 256
        if left == "B":
            assert len(x[2]["teacher"]) == len(x[2]["learner"]) == 128
        else:
            assert len(x[2]["teacher"]) == 256 and "learner" not in x[2]
    with pytest.raises(ValueError):
        corrective.sample_batch(teacher, learner, arm="best", count=256, rng=np.random.default_rng(1))


def test_fixed_budget_frozen_inheritance_determinism_and_checkpoint(inputs, tmp_path):
    reference, parent, teacher, learner = inputs
    config = {**corrective.load_config(), "optimizerSteps": 2, "minibatchSize": 8}
    before = semantic_state_digest(reference.state_dict())
    model, optimizer, fit = corrective.fit_arm(reference, teacher, learner, arm="D", seed=93001, config=config)
    other, _, repeat = corrective.fit_arm(reference, teacher, learner, arm="D", seed=93001, config=config)
    assert fit == repeat and semantic_state_digest(model.state_dict()) == semantic_state_digest(other.state_dict())
    assert fit["optimizerSteps"] == 2 and fit["exposure"]["teacherStates"] == fit["exposure"]["learnerStates"] == 8
    assert fit["newParameterL2Change"] > 0 and fit["inheritedUnchanged"]
    assert semantic_state_digest(reference.state_dict()) == before
    corrective.save_corrective(tmp_path / "checkpoint", model, parent, arm="D", seed=93001, steps=2, optimizer=optimizer, config=config)
    restored, metadata = corrective.load_corrective(tmp_path / "checkpoint")
    assert metadata["optimizerSteps"] == 2 and not metadata["ppoCompatible"]
    assert semantic_state_digest(restored.state_dict()) == semantic_state_digest(model.state_dict())
    observation, _ = teacher.batch([0, 1], True)
    for key, value in model.act(observation, deterministic=True)[0].items():
        assert torch.equal(value, restored.act(observation, deterministic=True)[0][key])
    path = tmp_path / "checkpoint/checkpoint.json"
    changed = json.loads(path.read_text())
    changed["arm"] = "A"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance"):
        corrective.load_corrective(tmp_path / "checkpoint")


def episode(seed, success=False, rejected=0):
    return {"seed": seed, "success": success, "progress": .8 if success else .1,
            "physicalWin": False, "rejectedActions": rejected, "totalActions": 1000,
            "firstContactDecision": 1, "firstHitDecision": 2}


def test_replication_is_d_vs_a_not_best_arm_and_gates_are_autonomous():
    a, d = [episode(1)], [episode(1, True)]
    assert corrective.replication_allowed(a, d)
    assert not corrective.replication_allowed(a, a)
    assert not corrective.replication_allowed(a, [episode(1, True, 1)])
    assert corrective.bootstrap_gate(d, a, a)["passed"]
    assert not corrective.bootstrap_gate(d, d, a)["passed"]
    contrast = corrective.paired_effect(d, a, corrective.load_config())
    assert contrast["success"]["mean"] == 1
    with pytest.raises(ValueError, match="unpaired"):
        corrective.paired_effect(d, [episode(2)], corrective.load_config())


def test_config_and_failed_lineage_stop_before_collection(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**corrective.load_config(), "optimizerSteps": 1000}))
    with pytest.raises(ValueError, match="frozen R1l"):
        corrective.load_config(path)
    monkeypatch.setattr(corrective, "audit_ancestry", lambda **kwargs: {"passed": False})
    def unexpected(*args, **kwargs):
        raise AssertionError("lineage failure must stop collection")
    monkeypatch.setattr(corrective, "collect", unexpected)
    with pytest.raises(ValueError, match="ancestry unresolved"):
        corrective.run_factorial(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, audit_path=AUDIT, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()
