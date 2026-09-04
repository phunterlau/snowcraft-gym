import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import control_channels as channels
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.options.recovery_report import audit_artifact_manifest
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from snowgym_training.trajectory import json_digest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020"


def unit(id, x, y, *, team="blue", alive=True, state="idle", cooldown=0):
    return {"id": id, "x": x, "y": y, "team": team, "alive": alive, "state": state, "throwCooldown": cooldown}


def test_movement_oracle_range_formation_casualties_and_bounds():
    raw = {"arena": {"width": 100, "height": 80},
           "allies": [unit(1, 0, 0), unit(2, 0, 2), unit(3, 0, 1, alive=False)],
           "enemies": [unit(8, 10, 0, team="red"), unit(9, 0, 0, alive=False)], "projectiles": []}
    before = copy.deepcopy(raw)
    result = channels.recommend_movement(raw, 4)
    left = np.array([1, 10]) / np.hypot(1, 10)
    np.testing.assert_allclose(result["target"][0, 0], (np.array([3.5, .12]) - left * .6) / [50, 40])
    assert result["valid"].tolist() == [[True, True, False, False]]
    assert result["distance"][0, 0] == 10
    assert result["ready"].tolist() == [[True, True, False, False]]
    assert raw == before
    raw["allies"][1]["alive"] = False
    np.testing.assert_allclose(channels.recommend_movement(raw, 4)["target"][0, 0], [3.5/50, 0])
    raw["enemies"][0]["x"] = 10000
    assert channels.recommend_movement(raw, 4)["target"][0, 0, 0] == pytest.approx(.99)
    raw["enemies"] = []
    assert not channels.recommend_movement(raw, 4)["valid"].any()
    with pytest.raises(ValueError, match="capacity"):
        channels.recommend_movement(raw, 1)


def test_movement_oracle_dodge_priority_and_readiness():
    raw = {"arena": {"width": 100, "height": 80}, "allies": [unit(2, 0, 0)],
           "enemies": [unit(8, 8, 0, team="red")],
           "projectiles": [{"team": "red", "x": 2, "y": 0, "vx": -1, "vy": 0}]}
    result = channels.recommend_movement(raw, 1)
    np.testing.assert_allclose(result["target"][0, 0], [0, -2.4/40])
    assert result["threat"][0, 0]
    raw["allies"][0]["state"] = "stunned"
    result = channels.recommend_movement(raw, 1)
    np.testing.assert_allclose(result["target"][0, 0], [1.5/50, 0])
    assert not result["ready"][0, 0]
    raw["allies"][0]["state"] = "recovering"
    np.testing.assert_allclose(channels.recommend_movement(raw, 1)["target"][0, 0], [0, -2.4/40])
    raw["projectiles"][0]["vx"] = 1
    assert not channels.recommend_movement(raw, 1)["threat"].any()
    raw["enemies"][0]["x"] = 0
    assert np.isfinite(channels.recommend_movement(raw, 1)["target"]).all()


@pytest.mark.parametrize("arm", channels.ARMS)
def test_conditional_heads_and_channel_isolation(arm):
    # Learner THROW becomes teacher MOVE in slot 0; inverse in slot 1.
    logits = torch.full((1, 4, 4), -5.)
    logits[0, range(4), [2, 1, 0, 3]] = 5
    targets = torch.zeros(1, 4, 4, 2)
    targets[:, :, 1] = .2
    targets[:, :, 2] = .9
    prediction = {"action_logits": logits, "target_raw_by_action": targets, "power_raw": torch.zeros(1, 4)}
    teacher = {"action_type": np.array([[1, 2, 3, 0]]), "target": np.full((1, 4, 2), .6), "power": np.full((1, 4), .8)}
    shot = {"target": np.full((1, 4, 2), .7), "power": np.full((1, 4), .8), "valid": np.ones((1, 4), bool)}
    movement = {"target": np.full((1, 4, 2), .6), "valid": np.ones((1, 4), bool)}
    before = {k: v.clone() for k, v in prediction.items()}
    result = channels.compose_action(arm, prediction, teacher, shot, movement)
    for k in prediction:
        assert torch.equal(prediction[k], before[k])
    if arm == "teacher":
        for k in teacher:
            np.testing.assert_array_equal(result[k], teacher[k])
        return
    types = [1, 2, 3, 0] if arm in {"teacher-choice", "teacher-choice-move"} else [2, 1, 0, 3]
    np.testing.assert_array_equal(result["action_type"], [types])
    for i, kind in enumerate(types):
        expected = .7 if kind == 2 else (.6 if "move" in arm else np.tanh(.2)) if kind == 1 else 0
        np.testing.assert_allclose(result["target"][0, i], expected, atol=1e-7)
        assert result["power"][0, i] == pytest.approx(.8 if kind == 2 else .5)


def test_invalid_oracles_and_unknown_arm_fail_closed():
    teacher = {"action_type": np.array([[1]]), "target": np.zeros((1, 1, 2))}
    oracle = {"target": np.ones((1, 1, 2)), "valid": np.ones((1, 1), bool)}
    with pytest.raises(ValueError, match="disagrees"):
        channels.validate_movement_agreement(teacher, oracle)
    with pytest.raises(ValueError, match="unknown"):
        channels.compose_action("bad", {}, {}, {}, {})


def test_live_matrix_parity_determinism_and_production_agreement():
    torch.set_num_threads(1)
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    with SnowGymBatchClient() as client:
        rows = {arm: channels.evaluate_episode(model, seed=200000, arm=arm, client=client) for arm in channels.ARMS}
        repeated = channels.evaluate_episode(model, seed=200000, arm="teacher-move", client=client)
    assert repeated == rows["teacher-move"]
    assert rows["teacher-choice-move"]["stateHashes"] == rows["teacher"]["stateHashes"]
    old = json.loads((ROOT / "runs/m7b_engage_r1g_throw_channels_v0/direction-power.json").read_text())[0]
    assert rows["shot-only"]["stateHashes"] == old["stateHashes"]
    assert rows["shot-only"]["actionsDigest"] == old["actionsDigest"]
    for arm, row in rows.items():
        assert row["teacherMoveAgreements"] > 0 and row["teacherThrowAgreements"] > 0
        assert row["rejectedActions"] == 0
        assert len(row["stateHashes"]) == row["decisions"] + 1
    assert rows["teacher"]["success"]


def fake_row(arm, seed):
    success = arm in {"teacher-choice-move", "teacher"}
    return {"seed": seed, "arm": arm, "success": success, "progress": float(success),
            "firstContactDecision": 1, "firstHitDecision": 1, "executedThrows": 1,
            "shotDistanceSum": 5., "choiceConfusion": [[0]*4]*4, "rejectedActions": 0,
            "totalActions": 1, "stateHashes": ["same", "next"]}


def test_paired_factorial_contrasts_and_validation():
    records = {arm: [fake_row(arm, i) for i in range(2)] for arm in channels.ARMS}
    result = channels.summarize(records)
    assert result["pairedContrasts"]["interaction"]["success"] == {"mean": 1., "bootstrap95": [1., 1.]}
    assert result["pairedContrasts"]["movementWithLearnerChoice"]["success"]["mean"] == 0
    assert result == channels.summarize(records)
    records["teacher"].reverse()
    with pytest.raises(ValueError, match="unique and paired"):
        channels.summarize(records)


def test_frozen_pipeline_artifact_audit_and_immutability(tmp_path, monkeypatch):
    config = channels.load_config()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**config, "seeds": [300000, 300039]}))
    with pytest.raises(ValueError, match="frozen R1h"):
        channels.load_config(bad)
    monkeypatch.setattr(channels, "evaluate_episode", lambda model, *, arm, seed, **kw: fake_row(arm, seed))
    report = channels.run_matrix(CHECKPOINT, output=tmp_path / "run")
    assert report["teacherTrajectoryParity"] and report["modelUnchanged"] and report["trainingUpdates"] == 0
    assert not report["qualificationEligible"]
    manifest = audit_artifact_manifest(tmp_path / "run", "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    with pytest.raises(FileExistsError):
        channels.run_matrix(CHECKPOINT, output=tmp_path / "run")


def test_pipeline_rejects_teacher_trajectory_mismatch(tmp_path, monkeypatch):
    def episode(model, *, arm, seed, **kw):
        row = fake_row(arm, seed)
        if arm == "teacher-choice-move":
            row["stateHashes"] = ["wrong"]
        return row
    monkeypatch.setattr(channels, "evaluate_episode", episode)
    with pytest.raises(RuntimeError, match="trajectory parity"):
        channels.run_matrix(CHECKPOINT, output=tmp_path / "bad-run")
    assert not (tmp_path / "bad-run").exists()
