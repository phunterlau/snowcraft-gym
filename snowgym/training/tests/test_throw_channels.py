import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import throw_channels as channels
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from snowgym_training.options.recovery_report import audit_artifact_manifest
from snowgym_training.trajectory import json_digest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020"


def unit(id, x, y, *, alive=True, vx=0.0, vy=0.0):
    return {"id": id, "x": x, "y": y, "alive": alive, "vx": vx, "vy": vy}


def test_oracle_targets_living_enemies_with_lead_power_and_stable_tie_break() -> None:
    raw = {"arena": {"width": 100, "height": 80}, "allies": [unit(1, 0, 0), unit(2, 1, 1, alive=False)],
           "enemies": [unit(5, 6, 0), unit(4, 0, 6, vx=2), unit(3, 0, 0, alive=False)]}
    shot = channels.recommend_shots(raw, 3)
    assert shot["enemyIds"].tolist() == [[4, -1, -1]]
    np.testing.assert_allclose(shot["target"][0, 0], [0.36 / 50, 6 / 40])
    assert shot["power"][0, 0] == pytest.approx(0.64)
    assert shot["valid"].tolist() == [[True, False, False]]
    raw["enemies"] = []
    assert not channels.recommend_shots(raw, 3)["valid"].any()


@pytest.mark.parametrize("arm", channels.ARMS)
def test_interventions_change_only_selected_throw_channels_without_mutation(arm) -> None:
    learner = {"action_type": np.array([[1, 2, 0, 3]]), "target": np.ones((1, 4, 2), dtype=np.float32) * 0.2,
               "power": np.ones((1, 4), dtype=np.float32) * 0.3}
    # Teacher moves at the learner's throw slot: its move target must not be used as aim.
    teacher = {"action_type": np.array([[2, 1, 0, 3]]), "target": np.zeros((1, 4, 2), dtype=np.float32),
               "power": np.zeros((1, 4), dtype=np.float32)}
    oracle = {"target": np.ones((1, 4, 2), dtype=np.float32) * 0.7, "power": np.ones((1, 4), dtype=np.float32) * 0.8,
              "valid": np.ones((1, 4), dtype=bool)}
    before = {name: value.copy() for name, value in learner.items()}
    result = channels.compose_action(arm, learner, teacher, oracle)
    for name in learner:
        np.testing.assert_array_equal(learner[name], before[name])
    if arm == "teacher":
        for name in teacher:
            np.testing.assert_array_equal(result[name], teacher[name])
        return
    np.testing.assert_array_equal(result["action_type"], learner["action_type"])
    np.testing.assert_array_equal(result["target"][:, [0, 2, 3]], learner["target"][:, [0, 2, 3]])
    np.testing.assert_array_equal(result["power"][:, [0, 2, 3]], learner["power"][:, [0, 2, 3]])
    np.testing.assert_allclose(result["target"][0, 1], 0.7 if arm in {"direction", "direction-power"} else 0.2)
    assert result["power"][0, 1] == pytest.approx(0.8 if arm in {"power", "direction-power"} else 0.3)


def test_oracle_disagreement_fails_closed() -> None:
    teacher = {"action_type": np.array([[2]]), "target": np.zeros((1, 1, 2)), "power": np.zeros((1, 1))}
    oracle = {"valid": np.ones((1, 1), dtype=bool), "target": np.ones((1, 1, 2)), "power": np.ones((1, 1))}
    with pytest.raises(ValueError, match="disagrees"):
        channels.validate_teacher_agreement(teacher, oracle)
    teacher["action_type"][:] = 1
    assert channels.validate_teacher_agreement(teacher, oracle) == 0


def test_live_teacher_oracle_agreement_and_deterministic_state_hashes() -> None:
    torch.set_num_threads(1)
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    with SnowGymBatchClient() as client:
        first = channels.evaluate_episode(model, arm="direction-power", seed=200000, client=client)
        repeated = channels.evaluate_episode(model, arm="direction-power", seed=200000, client=client)
        teacher = channels.evaluate_episode(model, arm="teacher", seed=200000, client=client)
    assert first == repeated
    assert first["directionReplacements"] == first["powerReplacements"] == first["learnerThrows"]
    assert first["teacherThrowAgreements"] > 0
    assert len(first["stateHashes"]) == first["decisions"] + 1
    assert teacher["success"]
    assert teacher["teacherThrowAgreements"] > 0
    assert first["rejectedActions"] == teacher["rejectedActions"] == 0


def test_paired_bootstrap_and_seed_validation() -> None:
    def row(seed, success):
        return {"seed": seed, "success": success, "progress": float(success), "firstHitDecision": 1 if success else None,
                "firstContactDecision": 1, "rejectedActions": 0, "totalActions": 1, "learnerThrows": 1,
                "teacherThrowAgreements": 1, "learnerThrowsWhileTeacherDoesNotThrow": 0,
                "directionReplacements": 0, "powerReplacements": 0}
    records = {"learner": [row(1, False), row(2, False)], "direction": [row(1, True), row(2, True)]}
    result = channels.summarize(records)
    assert result["direction"]["pairedDifferences"]["hit"] == {"mean": 1, "bootstrap95": [1, 1]}
    assert result == channels.summarize(records)
    records["direction"].reverse()
    with pytest.raises(ValueError, match="unique and paired"):
        channels.summarize(records)


def test_frozen_config_and_pipeline_artifacts(tmp_path, monkeypatch) -> None:
    config = channels.load_config()
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({**config, "seeds": [300000, 300039]}))
    with pytest.raises(ValueError, match="frozen R1g"):
        channels.load_config(changed)
    def episode(model, *, arm, seed, **kwargs):
        return {"seed": seed, "arm": arm, "success": arm == "teacher", "progress": float(arm == "teacher"),
                "firstContactDecision": 1, "firstHitDecision": 1, "rejectedActions": 0, "totalActions": 1,
                "learnerThrows": 1, "teacherThrowAgreements": 1, "learnerThrowsWhileTeacherDoesNotThrow": 0,
                "directionReplacements": 0, "powerReplacements": 0}
    monkeypatch.setattr(channels, "evaluate_episode", episode)
    output = tmp_path / "matrix"
    result = channels.run_matrix(CHECKPOINT, output=output)
    assert result["modelUnchanged"] and result["trainingUpdates"] == 0
    assert not result["qualificationEligible"]
    manifest = audit_artifact_manifest(output, "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    assert manifest["seeds"] == list(range(200000, 200040))
    with pytest.raises(FileExistsError):
        channels.run_matrix(CHECKPOINT, output=output)


def test_archived_matrix_integrity_pairing_and_baseline_parity() -> None:
    root = ROOT / "runs/m7b_engage_r1g_throw_channels_v0"
    manifest = audit_artifact_manifest(root, "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    assert manifest["config"] == channels.load_config()
    records = {arm: json.loads((root / f"{arm}.json").read_text()) for arm in channels.ARMS}
    report = json.loads((root / "report.json").read_text())
    assert channels.summarize(records) == report["summary"]
    assert report["modelUnchanged"] and report["trainingUpdates"] == 0
    assert not report["qualificationEligible"]
    for arm, rows in records.items():
        assert [row["seed"] for row in rows] == list(range(200000, 200040))
        assert sum(row["teacherThrowAgreements"] for row in rows) > 0
        for index, row in enumerate(rows):
            assert len(row["stateHashes"]) == row["decisions"] + 1
            assert row["stateHashes"][0] == records["learner"][index]["stateHashes"][0]
            assert row["rejectedActions"] == 0
            # Some learner trajectories never reach a state where the teacher
            # would fire. Agreement is checked only on actual teacher throws.
            assert row["teacherThrowAgreements"] >= 0
            if arm == "teacher":
                assert row["teacherThrowAgreements"] > 0
            assert row["directionReplacements"] == (row["learnerThrows"] if arm in {"direction", "direction-power"} else 0)
            assert row["powerReplacements"] == (row["learnerThrows"] if arm in {"power", "direction-power"} else 0)
    old = json.loads((ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/development-evaluation.json").read_text())
    for actual, expected in zip(records["learner"], old["missions"]["engage"]["correct"], strict=True):
        for name in ("seed", "success", "progress", "firstContactDecision", "firstHitDecision"):
            assert actual[name] == expected[name]
    assert sum(row["success"] for row in records["teacher"]) == 40
    assert sum(row["success"] for row in records["direction-power"]) == 10
