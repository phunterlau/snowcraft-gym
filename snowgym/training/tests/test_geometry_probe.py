import copy
import json
from pathlib import Path

import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.geometry_probe import geometry_loss
from snowgym_training.options import geometry_probe as probe
from snowgym_training.options.reservoir import load_teacher_bc_reservoir
from snowgym_training.options.recovery_report import audit_artifact_manifest
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from snowgym_training.trajectory import json_digest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020"
RESERVOIR = ROOT / "runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz"


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    reservoir = load_teacher_bc_reservoir(RESERVOIR)
    observation, teacher = reservoir.batch(probe.gate_indices(reservoir))
    return metadata, state, reservoir, observation, teacher


def model_for(inputs, relative=True):
    return probe.build_probe(inputs[0], inputs[1], relative=relative, seed=92001)


def test_zero_residual_identity_equal_capacity_and_frozen_classifier(inputs):
    absolute, relative = model_for(inputs, False), model_for(inputs)
    observation, teacher = inputs[3:]
    with torch.no_grad():
        expected, _, _ = relative.source.act(observation, deterministic=True)
        for model in (absolute, relative):
            actual, logp, _ = model.act(observation, deterministic=True)
            assert logp is None
            for k in expected:
                assert torch.equal(actual[k], expected[k])
            assert not any(p.requires_grad for p in model.source.parameters())
    assert sum(p.numel() for p in absolute.parameters()) == sum(p.numel() for p in relative.parameters())
    for name, value in absolute.state_dict().items():
        assert torch.equal(value, relative.state_dict()[name])
    before = relative.source(observation)["action_logits"].detach().clone()
    optimizer = probe.optimizer_for(relative, .001)
    for _ in range(3):
        probe.train_step(relative, optimizer, observation, teacher, .5)
    assert torch.equal(relative(observation)["action_logits"], before)
    assert semantic_state_digest(relative.source.state_dict()) == semantic_state_digest(inputs[1]["model"])
    with pytest.raises(ValueError, match="deterministic-only"):
        relative.act(observation)


def test_pair_coordinates_optional_targets_and_masked_pool_permutations(inputs):
    model = model_for(inputs)
    observation = {k: v[:2].clone() for k, v in inputs[3].items()}
    values = model.pair_features(observation, "enemies")
    expected = observation["enemies"][:, None, :, 2:4] - observation["allies"][:, :, None, 2:4]
    assert torch.equal(values[..., 2:4], expected)
    assert torch.equal(model_for(inputs, False).pair_features(observation, "enemies")[..., 2:4],
                       observation["enemies"][:, None, :, 2:4].expand_as(expected))
    observation["enemies"][..., 10] = 0
    assert (model.pair_features(observation, "enemies")[..., 11:15] == 0).all()
    baseline = model.features(observation)
    changed = {k: v.clone() for k, v in observation.items()}
    changed["enemies"] = changed["enemies"].flip(1)
    changed["enemy_mask"] = changed["enemy_mask"].flip(1)
    torch.testing.assert_close(model.features(changed), baseline)
    # Dead enemy slots must not influence the new representation.
    changed = {k: v.clone() for k, v in observation.items()}
    changed["enemy_mask"][:] = 0
    reference = model.features(changed)
    changed["enemies"][..., 2:] = 100
    assert torch.equal(model.features(changed), reference)
    changed["projectile_mask"][:] = 0
    changed["obstacle_mask"][:] = 0
    assert torch.isfinite(model.features(changed)).all()


def test_new_gradient_paths_and_unused_heads(inputs):
    model = model_for(inputs)
    observation, teacher = inputs[3:]
    optimizer = probe.optimizer_for(model, .001)
    # Zero-output initialization blocks upstream gradients for the first step;
    # one output-layer update opens the intended encoder path.
    probe.train_step(model, optimizer, observation, teacher, .5)
    for component, inactive in (("move", "shot"), ("direction", "move"), ("power", "move")):
        model.zero_grad(set_to_none=True)
        geometry_loss(model(observation), teacher, observation)[component].backward()
        assert sum(float(p.grad.square().sum()) for p in model.encoders.parameters() if p.grad is not None) > 0
        assert all(p.grad is None for p in model.source.parameters())
        assert all(p.grad is None or not p.grad.any() for p in getattr(model, inactive).parameters())


def test_zero_rays_no_active_labels_and_saturated_boundaries_are_finite(inputs):
    model = model_for(inputs)
    observation, teacher = inputs[3:]
    output = model(observation)
    target = observation["allies"][..., 2:4][:, :, None].expand(-1, -1, 4, -1).clone().requires_grad_()
    output = {**output, "target_by_action": target}
    losses = geometry_loss(output, teacher, observation)
    losses["total"].backward()
    assert torch.isfinite(target.grad).all()
    empty = {**teacher, "action_type": torch.zeros_like(teacher["action_type"])}
    assert geometry_loss(model(observation), empty, observation)["total"] == 0
    with torch.no_grad():
        model.move[-1].bias.fill_(1000)
        model.shot[-1].bias.fill_(-1000)
    result = model(observation)
    assert all(torch.isfinite(x) for x in geometry_loss(result, teacher, observation).values())
    assert result["target_by_action"].abs().max() <= 1


def test_small_batch_training_gate_disposable_and_deterministic(inputs):
    model = model_for(inputs)
    digest = semantic_state_digest(model.state_dict())
    result = probe.small_batch_gate(model, inputs[2], probe.load_config())
    assert result["passed"]
    assert result["reduction"] >= .5
    assert result == probe.small_batch_gate(model, inputs[2], probe.load_config())
    assert semantic_state_digest(model.state_dict()) == digest
    assert all(v["source"] == 0 for v in result["gradientNorms"].values())


def test_checkpoint_roundtrip_and_tampering(inputs, tmp_path):
    model = model_for(inputs)
    config = probe.load_config()
    metadata = probe.save_probe(tmp_path / "saved", model, inputs[0], epoch=0, config=config)
    restored, loaded = probe.load_probe(tmp_path / "saved")
    assert metadata == loaded
    assert not loaded["ppoCompatible"]
    assert semantic_state_digest(model.state_dict()) == semantic_state_digest(restored.state_dict())
    with pytest.raises(FileExistsError):
        probe.save_probe(tmp_path / "saved", model, inputs[0], epoch=0, config=config)
    path = tmp_path / "saved/checkpoint.json"
    value = json.loads(path.read_text())
    value["relative"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance"):
        probe.load_probe(tmp_path / "saved")


def test_config_validation_and_failed_gate_stops_before_training(inputs, tmp_path, monkeypatch):
    config = probe.load_config()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**config, "epochs": 200}))
    with pytest.raises(ValueError, match="frozen R1i"):
        probe.load_config(bad)
    monkeypatch.setattr(probe, "small_batch_gate", lambda *args: {"passed": False})
    def unexpected(*args, **kwargs):
        raise AssertionError("failed gate must not fit or evaluate")
    monkeypatch.setattr(probe, "train_step", unexpected)
    monkeypatch.setattr(probe, "evaluate_option_episode", unexpected)
    result = probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")
    assert not result["gatesPassed"] and not result["arms"]
    manifest = audit_artifact_manifest(tmp_path / "run", "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    with pytest.raises(FileExistsError):
        probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")


def test_successful_runner_retains_final_checkpoints_and_paired_evaluations(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "small_batch_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(probe, "teacher_agreement", lambda *args: {"all": {"test": True}})
    monkeypatch.setattr(probe, "train_step", lambda *args: {"total": 0.})
    def episode(model, initializer, *, seed, condition, **kwargs):
        return {"seed": seed, "success": False, "progress": 0., "physicalWin": False,
                "firstContactDecision": None, "firstHitDecision": None, "rejectedActions": 0, "totalActions": 1}
    monkeypatch.setattr(probe, "evaluate_option_episode", episode)
    result = probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")
    assert result["gatesPassed"] and set(result["arms"]) == {"absolute", "relative"}
    assert set(result["development"]) == {"source", "absolute", "relative"}
    for arm in result["arms"]:
        assert result["arms"][arm]["checkpointReloadExact"]
        assert len(result["arms"][arm]["epochs"]) == 20
        for condition in ("correct", "shuffled"):
            assert result["development"][arm][condition]["episodes"] == 40
    audit_artifact_manifest(tmp_path / "run", "manifest.json")


def test_archived_geometry_run_provenance_frozen_source_and_paired_baseline(inputs):
    root = ROOT / "runs/m7b_engage_r1i_geometry_probe_v0"
    manifest = audit_artifact_manifest(root, "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    assert manifest["config"] == probe.load_config()
    report = json.loads((root / "report.json").read_text())
    gates = json.loads((root / "small-batch-gates.json").read_text())
    assert report["gatesPassed"] and all(g["passed"] for g in gates.values())
    assert report["ppoUpdates"] == report["criticUpdates"] == 0
    assert not report["qualificationEligible"]
    assert report["arms"]["absolute"]["newParameterCount"] == report["arms"]["relative"]["newParameterCount"]
    assert report["arms"]["absolute"]["teacherAgreementBefore"] == report["arms"]["relative"]["teacherAgreementBefore"]
    old = json.loads((ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/development-evaluation.json").read_text())
    source_rows = json.loads((root / "source-development.json").read_text())
    for condition in ("correct", "shuffled"):
        assert source_rows[condition] == old["missions"]["engage"][condition]
    for arm in ("absolute", "relative"):
        result = report["arms"][arm]
        assert result["sourceUnchanged"] and result["checkpointReloadExact"]
        assert result["newParameterL2Change"] > 0
        assert len(result["epochs"]) == 20
        initial, _ = probe.load_probe(root / f"{arm}-epoch-000")
        model, _ = probe.load_probe(root / f"{arm}-epoch-020")
        assert semantic_state_digest(model.source.state_dict()) == semantic_state_digest(inputs[1]["model"])
        assert semantic_state_digest(initial.source.state_dict()) == semantic_state_digest(inputs[1]["model"])
        with torch.no_grad():
            assert torch.equal(model(inputs[3])["action_logits"], initial(inputs[3])["action_logits"])
        rows = json.loads((root / f"{arm}-development.json").read_text())
        for condition in ("correct", "shuffled"):
            assert [r["seed"] for r in rows[condition]] == list(range(200000, 200040))
            assert probe.summarize_rows(rows[condition]) == report["development"][arm][condition]
            assert all(r["rejectedActions"] == 0 for r in rows[condition])
