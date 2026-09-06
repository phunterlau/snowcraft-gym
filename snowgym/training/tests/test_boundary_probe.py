import copy
import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options import boundary_probe as b
from snowgym_training.trajectory import json_digest


def test_radial_null_direction_lateral_and_arrival_geometry():
    for arm in ("radial+1", "radial-1", "radial+5", "radial-5"):
        g = b.arm_geometry([0, 0], [10, 0], [1, 1], [50, 40], arm)
        assert g["radialUnclippedFar"] and not g["clipped"]
        assert g["desiredVelocityDelta"] == pytest.approx(0, abs=1e-7)
        assert g["effectiveShift"] == pytest.approx(int(arm[-1]), abs=1e-6)
    g = b.arm_geometry([0, 0], [10, 0], [1, 1], [50, 40], "lateral+1")
    assert g["desiredVelocityDelta"] > .5 and not g["radialUnclippedFar"]
    np.testing.assert_allclose(b.desired_velocity([0, 0], [1, 0]), [6/2.1, 0])
    np.testing.assert_allclose(b.desired_velocity([0, 0], [0, 0]), [0, 0])
    clipped = b.arm_geometry([0, 0], [49, 0], [0, 1], [50, 40], "radial+5")
    assert clipped["clipped"] and clipped["effective"][0] == 49.5
    assert clipped["effectiveShift"] == .5
    for target in ([0, 0], [np.nan, 0]):
        with pytest.raises(ValueError):
            b.arm_geometry([0, 0], target, [1, 1], [50, 40], "keep")
    with pytest.raises(ValueError):
        b.arm_geometry([0, 0], [1, 0], [1, 1], [50, 40], "other")


def test_substitution_changes_only_one_selected_target_and_preserves_source():
    first = {"action_type": np.array([[1, 2, 0]]), "target": np.zeros((1, 3, 2)), "power": np.ones((1, 3))}
    before = copy.deepcopy(first)
    geo = {"serialized": [.2, .3]}
    changed = b.substitute(first, 0, geo, "lateral+1")
    np.testing.assert_array_equal(changed["target"][0, 0], [.2, .3])
    np.testing.assert_array_equal(changed["target"][0, 1:], first["target"][0, 1:])
    for key in first:
        np.testing.assert_array_equal(first[key], before[key])
        np.testing.assert_array_equal(b.substitute(first, 0, geo, "keep")[key], first[key])
    with pytest.raises(ValueError):
        b.substitute(first, 1, geo, "teacher")


def test_selection_separates_recommendation_from_action_and_liveness():
    raw = {"arena": {"width": 100, "height": 80}, "allies": [
        {"id": 2, "alive": True, "x": 0, "y": 0}, {"id": 1, "alive": True, "x": 0, "y": 0}]}
    first = {"action_type": np.array([[1, 2]]), "target": np.ones((1, 2, 2))*.2}
    movement = {"valid": np.ones((1, 2), dtype=bool)}
    assert b.choose(raw, first, movement) == 0
    first["action_type"][0, 1] = 1
    assert b.choose(raw, first, movement) == 1
    movement["valid"][0, 1] = False
    raw["allies"][0]["alive"] = False
    assert b.choose(raw, first, movement) is None


def test_censored_motion_and_empty_intervals():
    assert b.motion({"checkpoints": {"5": None}}, {"checkpoints": {"5": None}}, 5) is None
    assert b.interval([]) == {"count": 0, "mean": None, "ci95": None}
    assert b.interval([0, 0, 0])["ci95"] == [0, 0]


def test_manifest_tampering_inventory_and_immutable_output(tmp_path):
    data = tmp_path/"data.json"
    data.write_text('{}')
    value = {"artifacts": {"data.json": b.file_digest(data)}}
    value["manifestDigest"] = json_digest(value)
    (tmp_path/"manifest.json").write_text(json.dumps(value))
    b.verified_manifest(tmp_path)
    bad = {**value, "unexpected": True}
    (tmp_path/"manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="self digest"):
        b.verified_manifest(tmp_path)
    (tmp_path/"manifest.json").write_text(json.dumps(value))
    with pytest.raises(FileExistsError):
        b.run(tmp_path)
    (tmp_path/"extra").write_text('x')
    with pytest.raises(ValueError, match="inventory"):
        b.verified_manifest(tmp_path)
    data.write_text('tamper')
    with pytest.raises(ValueError, match="digest mismatch"):
        b.verified_manifest(tmp_path)


def test_live_prefix_baseline_duplicate_single_decision_and_tamper():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source, _, frames, cfg, _ = b.frozen_source()
    before = semantic_state_digest(source.state_dict())
    with SnowGymBatchClient() as client:
        wrapper = b.make_wrapper(client, 1, cfg["config"]["gamma"])
        picks, exclusions, steps = b.selection(source, wrapper, frames)
        assert len(picks) == 24 and steps <= 57*200
        assert all(100000 <= p["seed"] <= 100063 for p in picks)
        assert all(p["actionMask"] is not None for p in picks)
        pick = picks[0]; frame = next(f for f in frames if f["seed"] == pick["seed"])
        baseline = b.branch(source, wrapper, frame, pick, "keep", cfg["config"]["gamma"])
        assert baseline["stateHashes"] == frame["suffixHashes"]
        altered = b.branch(source, wrapper, frame, pick, "lateral+5", cfg["config"]["gamma"])
        repeat = b.branch(source, wrapper, frame, pick, "lateral+5", cfg["config"]["gamma"])
        assert json_digest(altered) == json_digest(repeat)
        assert altered["trace"][0]["action"]["action_type"] == baseline["trace"][0]["action"]["action_type"]
        assert not altered["autonomousQualificationEligible"]
        report = b.summarize([{a: baseline if a == "keep" else altered for a in b.ARMS}])
        assert report["seeds"] == 1
        assert semantic_state_digest(source.state_dict()) == before
        bad = copy.deepcopy(pick); bad["identity"]["physical"] = 'bad'
        with pytest.raises(ValueError, match="identity mismatch"):
            b.branch(source, wrapper, frame, bad, "keep", cfg["config"]["gamma"])
        bad_frame = copy.deepcopy(frame); bad_frame["seed"] = 210000
        with pytest.raises(ValueError):
            b.recovery_train.validate_frames([bad_frame], training=True)
