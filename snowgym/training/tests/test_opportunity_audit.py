import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.geometry_probe import geometry_loss
from snowgym_training.options import opportunity_audit as audit
from snowgym_training.options import opportunity_metrics as metrics
from snowgym_training.options.geometry_probe import load_probe, gate_indices
from snowgym_training.options.reservoir import load_teacher_bc_reservoir

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/m7b_engage_r1i_geometry_probe_v0/absolute-epoch-020"
RESERVOIR = ROOT / "runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz"


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    model, metadata = load_probe(CHECKPOINT)
    reservoir = load_teacher_bc_reservoir(RESERVOIR)
    observation, teacher = reservoir.batch(gate_indices(reservoir))
    return model, metadata, reservoir, observation, teacher


def test_config_and_identity_reject_changes(inputs, tmp_path):
    assert audit.load_config()["checkpointDigest"] == inputs[1]["checkpointDigest"]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps({**audit.load_config(), "trainingSeeds": [210000, 210039]}))
    with pytest.raises(ValueError, match="frozen R1k"):
        audit.load_config(path)
    with pytest.raises(FileExistsError):
        audit.run_audit(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path)


def old_labels(teacher):
    return {"move_target": teacher["target"], "shot_target": teacher["target"],
            "power": teacher["power"], "move_mask": teacher["action_type"] == 1,
            "shot_mask": teacher["action_type"] == 2}


def test_conditional_loss_matches_r1i_and_masks_independent_heads(inputs):
    model, _, _, observation, teacher = inputs
    expected = geometry_loss(model(observation), teacher, observation)
    labels = old_labels(teacher)
    actual = metrics.conditional_loss(model(observation), observation, labels)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    labels = {**labels, "move_mask": torch.zeros_like(labels["move_mask"])}
    probe = copy.deepcopy(model)
    metrics.conditional_loss(probe(observation), observation, labels)["total"].backward()
    assert all(p.grad is None or not p.grad.any() for p in probe.move.parameters())
    assert all(p.grad is None for p in probe.source.parameters())
    empty = {**labels, "shot_mask": torch.zeros_like(labels["shot_mask"])}
    assert metrics.conditional_loss(model(observation), observation, empty)["total"] == 0


def test_geometry_angles_boundary_and_degenerate_rays():
    result = metrics.physical_errors(np.zeros(2), [0, 0], [-1, 0], .5, [1, 1], [1, 0], 1.)
    assert result["angleDegrees"] == 180
    assert result["chordMissWorld"] == 100
    assert result["moveErrorWorld"] == pytest.approx(np.hypot(50, 40))
    zero = metrics.physical_errors(np.zeros(2), [0, 0], [0, 0], .5, [0, 0], [1, 0], .5)
    assert zero["degenerateRay"] and zero["angleDegrees"] is None


def opportunity(index, *, seed=100000, kind=1, teacher=1):
    return {"opportunityId": str(index), "seed": seed, "decision": index, "unitId": index,
            "teacherType": teacher, "learnerType": kind, "legal": [True]*4,
            "moveAvailable": True, "shotAvailable": True,
            "moveErrorWorld": float(index), "angleDegrees": float(index),
            "powerError": .1, "chordMissWorld": 1.}


def test_selection_balances_caps_and_legality_and_reports_mask_exclusion():
    rows = [opportunity(i, seed=100000+i//8, teacher=1 if i % 2 else 2) for i in range(48)]
    selected = metrics.select_opportunities(rows, "move", limit=16, per_episode=4)
    assert len(selected) == 16
    assert selected == metrics.select_opportunities(list(reversed(rows)), "move", limit=16, per_episode=4)
    assert sum(r["teacherType"] == 1 for r in selected) == 8
    assert max(sum(r["seed"] == s for r in selected) for s in {r["seed"] for r in selected}) <= 4
    rows[0]["legal"][1] = False
    assert rows[0] not in metrics.select_opportunities(rows, "move")
    assert metrics.cross_tabs(rows)["oldMaskExclusion"]["move"]["fraction"] == .5


def test_gradients_and_finite_difference_leave_model_unchanged(inputs):
    model, _, _, observation, teacher = inputs
    probe = copy.deepcopy(model)
    before = semantic_state_digest(probe.state_dict())
    gradient = metrics.gradient_audit(probe, observation, old_labels(teacher))
    assert gradient["targetEncoderReachable"]
    jacobian = metrics.physical_jacobian_audit(probe, observation, old_labels(teacher))
    assert all(r["passed"] for r in jacobian.values())
    assert semantic_state_digest(probe.state_dict()) == before
    assert all(p.grad is None for p in probe.source.parameters())


def test_gzip_is_deterministic_and_finite(tmp_path):
    for name in ("one.gz", "two.gz"):
        audit.write_jsonl(tmp_path / name, [{"a": np.array([1., np.inf]), "b": np.bool_(True)}])
    assert (tmp_path / "one.gz").read_bytes() == (tmp_path / "two.gz").read_bytes()
    with gzip.open(tmp_path / "one.gz", "rt") as stream:
        assert json.loads(stream.readline()) == {"a": [1., None], "b": True}


def test_branch_selection_is_one_unit_one_channel_and_no_aliasing():
    state = {"learner": {"action_type": np.array([[1, 2]]), "target": np.zeros((1, 2, 2)),
                         "power": np.zeros((1, 2))},
             "labels": {"move_target": np.ones((1, 2, 2)), "shot_target": np.full((1, 2, 2), .5),
                        "power": np.ones((1, 2))}}
    for channel, slot in (("move", 0), ("aim", 1), ("power", 1)):
        row = {**opportunity(1), "slot": slot}
        before = audit.plain(state)
        result = audit.substitute(state, row, channel)
        assert audit.plain(state) == before
        assert np.array_equal(result["action_type"], state["learner"]["action_type"])
        other = 1-slot
        assert np.array_equal(result["target"][0, other], state["learner"]["target"][0, other])
        assert result["power"][0, other] == 0
        if channel != "power":
            assert np.array_equal(result["power"], state["learner"]["power"])
        else:
            assert np.array_equal(result["target"], state["learner"]["target"])
    with pytest.raises(ValueError):
        audit.substitute(state, {**opportunity(1), "slot": 0}, "aim")


def test_paired_summary_clusters_episodes_and_fails_empty_channel():
    config = audit.load_config()
    base = {"damageDealt": 0., "damageReceived": 0., "progress": 0., "rangeError": 3.}
    rows = [{"channel": "move", "seed": seed, "base": base,
             "corrected": {**base, "damageDealt": 1., "rangeError": 1.}}
            for seed in (1, 1, 2)]
    report = audit.branch_summary(rows, config)
    assert report["move"]["useful"]
    assert report["move"]["effects"]["netDamage"]["episodes"] == 2
    assert not report["aim"]["useful"]


def test_live_labeling_reconstruction_branch_and_tamper(inputs):
    model, _, reservoir, _, _ = inputs
    before = semantic_state_digest(model.state_dict())
    with SnowGymBatchClient() as client:
        base, wrapped, observation = audit.reset_world(client, 100000, "teacher")
        state, _ = audit.label_state(model, base, wrapped, observation, seed=100000, decision=0, state_index=0, kind="teacher")
        for name, values in reservoir.observations.items():
            np.testing.assert_array_equal(observation[name][0], values[0].numpy())
        base, wrapped, observation = audit.reset_world(client, 100000, "learner")
        actions = []
        for decision in range(5):
            state, rows = audit.label_state(model, base, wrapped, observation, seed=100000, decision=decision, state_index=decision, kind="learner")
            actions.append(audit.plain(state["learner"]))
            observation, _, _, _, _ = wrapped.step(state["learner"])
        row = next(r for r in rows if r["learnerType"] == 1)
        episode = {"actions": actions}
        args = (model, client, state, row, episode, "move")
        a = audit.branch(*args, replace=False, horizon=3)
        b = audit.branch(*args, replace=False, horizon=3)
        assert a == b
        corrected = audit.branch(*args, replace=True, horizon=3)
        assert a["startIdentity"] == corrected["startIdentity"]
        state["identity"]["tracker"] = "tampered"
        with pytest.raises(RuntimeError, match="prefix replay"):
            audit.branch(*args, replace=False, horizon=3)
    assert semantic_state_digest(model.state_dict()) == before


def test_fit_selection_masks_only_invoked_unit():
    state = {"observation": {"allies": np.zeros((1, 2, 21))},
             "labels": {"move_mask": np.ones((1, 2), bool), "shot_mask": np.ones((1, 2), bool),
                        "move_target": np.zeros((1, 2, 2)), "shot_target": np.zeros((1, 2, 2)),
                        "power": np.zeros((1, 2))}}
    _, labels = metrics.fit_batch([state], [{"stateIndex": 0, "slot": 1, "learnerType": 2}])
    assert not labels["move_mask"].any()
    assert labels["shot_mask"].tolist() == [[False, True]]


def test_failed_gate_is_archived_and_manifest_detects_tampering(inputs, tmp_path, monkeypatch):
    from snowgym_training.options.recovery_report import audit_artifact_manifest
    monkeypatch.setattr(audit, "collect", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(audit, "hard_fit", lambda *args, **kwargs: {"passed": False})
    result = audit.run_audit(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")
    assert not result["r1lAllowed"] and not result["qualificationEligible"]
    assert result["productionUpdates"] == 0 and result["sourceUnchanged"]
    audit_artifact_manifest(tmp_path / "run", "manifest.json")
    (tmp_path / "run/hard-fit.json").write_text("{}")
    with pytest.raises(ValueError):
        audit_artifact_manifest(tmp_path / "run", "manifest.json")


def test_disposable_hard_fit_does_not_modify_source(inputs):
    model, _, _, observation, teacher = inputs
    states, rows = [], []
    # Synthetic labels based on real teacher states; this tests isolation and
    # split mechanics, never substitutes for the real hard-opportunity gate.
    for seed, start in ((100000, 0), (100032, 16)):
        for index in range(start, start+16):
            state_index = len(states)
            states.append({"observation": {k: v[index:index+1].numpy() for k, v in observation.items()},
                           "labels": {k: v[index:index+1].numpy() for k, v in old_labels(teacher).items()}})
            for slot in range(teacher["action_type"].shape[1]):
                kind = int(teacher["action_type"][index, slot])
                if kind in (1, 2):
                    rows.append({**opportunity(len(rows), seed=seed, kind=kind, teacher=kind),
                                 "stateIndex": state_index, "slot": slot})
    before = semantic_state_digest(model.state_dict())
    a = metrics.hard_fit(model, states, rows, steps=2)
    b = metrics.hard_fit(model, states, rows, steps=2)
    assert a == b
    assert a["sourceUnchanged"] and a["disposable"] and a["steps"] == 2
    assert semantic_state_digest(model.state_dict()) == before
