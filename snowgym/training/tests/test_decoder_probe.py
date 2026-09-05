import json
from pathlib import Path

import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.decoder_probe import ARMS, clip_along_ray, direction_target
from snowgym_training.executor.geometry_probe import geometry_loss
from snowgym_training.options import decoder_probe as probe
from snowgym_training.options import geometry_probe as geometry
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
    obs, labels = reservoir.batch(geometry.gate_indices(reservoir))
    return metadata, state, reservoir, obs, labels


def build(inputs, arm):
    return probe.build_probe(inputs[0], inputs[1], arm=arm, seed=92001)


@pytest.mark.parametrize("arm", ARMS)
def test_initializer_identity_and_correct_conditional_channels(inputs, arm):
    model = build(inputs, arm)
    old = geometry.build_probe(inputs[0], inputs[1], relative=False, seed=92001)
    for n, v in model.state_dict().items():
        assert torch.equal(v, old.state_dict()[n])
    with torch.no_grad():
        expected, _, _ = old.act(inputs[3], deterministic=True)
        actual, logp, _ = model.act(inputs[3], deterministic=True)
        assert logp is None
        for k in actual:
            assert torch.equal(actual[k], expected[k])
        model.move[-1].bias.fill_(.5)
        old.move[-1].bias.fill_(.5)
        model.shot[-1].bias[:] = torch.tensor([.3, -.2, .4])
        old.shot[-1].bias[:] = torch.tensor([.3, -.2, .4])
        updated, control = model(inputs[3]), old(inputs[3])
        assert torch.equal(updated["action_logits"], control["action_logits"])
        assert torch.equal(updated["power"], control["power"])
        for action in (0, 3):
            assert torch.equal(updated["target_by_action"][..., action, :], control["target_by_action"][..., action, :])
        if arm not in {"displacement", "both"}:
            assert torch.equal(updated["target_by_action"][..., 1, :], control["target_by_action"][..., 1, :])
        if arm not in {"direction", "both"}:
            assert torch.equal(updated["target_by_action"][..., 2, :], control["target_by_action"][..., 2, :])
    with pytest.raises(ValueError, match="deterministic-only"):
        model.act(inputs[3])
    if arm != "absolute":
        assert "target_raw_by_action" not in updated


def test_direction_geometry_zero_correction_and_non_square_boundary():
    origin = torch.tensor([[.2, -.3], [.98, .9], [0., 0.]])
    target = torch.tensor([[.4, -.3], [.9, .8], [0., 0.]])
    assert torch.equal(direction_target(origin, target, torch.zeros_like(origin)), target)
    # First ray points right; add [-1,1] to rotate it up. Length remains 10 world units.
    corrected = direction_target(origin[:1], target[:1], torch.tensor([[-1., 1.]]))
    torch.testing.assert_close(corrected, torch.tensor([[.2, -.05]]))
    clipped = clip_along_ray(torch.tensor([[.5, .5]]), torch.tensor([[2., 1.]]))
    torch.testing.assert_close(clipped, torch.tensor([[1., 2/3]]))
    assert torch.equal(clip_along_ray(origin, origin), origin)


def test_zero_rays_cancellation_and_boundaries_have_finite_gradients():
    origin = torch.tensor([[0., 0.], [.99, .99], [0., 0.]])
    target = torch.tensor([[0., 0.], [.8, .8], [.2, 0.]])
    residual = torch.tensor([[0., 0.], [1000., 1000.], [-1., 0.]], requires_grad=True)
    actual = direction_target(origin, target, residual)
    actual.sum().backward()
    assert torch.isfinite(actual).all() and torch.isfinite(residual.grad).all()
    assert actual.abs().max() <= 1


def test_move_displacement_world_scaling_and_bounds(inputs):
    model = build(inputs, "displacement")
    with torch.no_grad():
        model.move[-1].bias[:] = torch.tensor([.2, -.4])
        output = model(inputs[3])
        inherited = model.source(inputs[3])
    living = inputs[3]["ally_mask"].bool() & (inputs[3]["allies"][..., 1] > .5)
    expected = inherited["target_by_action"][..., 1, :].clone()
    expected[living] = (expected[living] + 10*torch.tanh(torch.tensor([.2, -.4])) / torch.tensor([50., 40.])).clamp(-1, 1)
    assert torch.equal(output["target_by_action"][..., 1, :], expected)
    with torch.no_grad():
        model.move[-1].bias.fill_(1000)
    assert torch.isfinite(geometry_loss(model(inputs[3]), inputs[4], inputs[3])["total"])


@pytest.mark.parametrize("arm", ("displacement", "direction", "both"))
def test_gate_gradient_reachability_and_frozen_state(inputs, arm):
    model = build(inputs, arm)
    initial = semantic_state_digest(model.state_dict())
    gate = geometry.small_batch_gate(model, inputs[2], probe.load_config())
    assert gate["passed"] and gate["reduction"] >= .5
    assert semantic_state_digest(model.state_dict()) == initial
    for component, unused in (("move", "shot"), ("direction", "move"), ("power", "move")):
        assert gate["gradientNorms"][component]["source"] == 0
        assert gate["gradientNorms"][component]["encoders"] > 0
        assert gate["gradientNorms"][component][unused] == 0


def test_checkpoint_reload_and_tamper_detection(inputs, tmp_path):
    model = build(inputs, "both")
    optimizer = geometry.optimizer_for(model, .001)
    geometry.train_step(model, optimizer, inputs[3], inputs[4], .5)
    probe.save_probe(tmp_path / "model", model, inputs[0], probe.load_config(), 1, optimizer)
    loaded, metadata = probe.load_probe(tmp_path / "model")
    with torch.no_grad():
        actual, _, _ = loaded.act(inputs[3], deterministic=True)
        expected, _, _ = model.act(inputs[3], deterministic=True)
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert not metadata["ppoCompatible"]
    with pytest.raises(FileExistsError):
        probe.save_probe(tmp_path / "model", model, inputs[0], probe.load_config(), 1)
    path = tmp_path / "model/checkpoint.json"
    metadata["arm"] = "absolute"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        probe.load_probe(tmp_path / "model")


def test_config_factorial_contrasts_and_gate_stop(inputs, tmp_path, monkeypatch):
    config = probe.load_config()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**config, "epochs": 100}))
    with pytest.raises(ValueError, match="frozen R1j"):
        probe.load_config(bad)
    rows = {a: {"correct": [{"seed": i, "success": a == "both", "progress": float(a == "both")} for i in range(3)]} for a in ARMS}
    result = probe.contrasts(rows, config)
    assert result["interaction"]["success"] == {"mean": 1., "bootstrap95": [1., 1.]}
    rows["both"]["correct"].reverse()
    with pytest.raises(ValueError, match="paired"):
        probe.contrasts(rows, config)
    monkeypatch.setattr(geometry, "small_batch_gate", lambda *args: {"passed": False})
    def forbidden(*args, **kwargs):
        raise AssertionError("failed gate cannot train or evaluate")
    monkeypatch.setattr(geometry, "train_step", forbidden)
    monkeypatch.setattr(probe, "evaluate_option_episode", forbidden)
    result = probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")
    assert not result["gatesPassed"] and not result["arms"]
    manifest = audit_artifact_manifest(tmp_path / "run", "manifest.json")
    assert manifest["manifestDigest"] == json_digest({k: v for k, v in manifest.items() if k != "manifestDigest"})
    with pytest.raises(FileExistsError):
        probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")


def test_live_zero_residual_control_reproduces_source_episode(inputs):
    from snowgym_client.batch import SnowGymBatchClient
    model = build(inputs, "both")
    with SnowGymBatchClient() as client:
        row = probe.evaluate_option_episode(model, model.source, option="engage", seed=200000,
                                           condition="correct", client=client)
    expected = json.loads((ROOT / "runs/m7b_engage_r1i_geometry_probe_v0/source-development.json").read_text())["correct"][0]
    assert row == expected


def test_passing_pipeline_preserves_arm_checkpoint_contracts(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "small_batch_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(geometry, "train_step", lambda *args: {"total": 0.})
    monkeypatch.setattr(probe, "teacher_agreement", lambda *args: {})
    def episode(model, source, *, seed, **kwargs):
        return {"seed": seed, "success": False, "progress": 0., "physicalWin": False,
                "firstContactDecision": None, "firstHitDecision": None, "rejectedActions": 0, "totalActions": 1}
    monkeypatch.setattr(probe, "evaluate_option_episode", episode)
    result = probe.run_probe(checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=tmp_path / "run")
    assert result["gatesPassed"] and len(result["arms"]) == 4 and len(result["development"]) == 5
    for arm in ARMS:
        assert len(result["arms"][arm]["epochs"]) == 20
        model, metadata = probe.load_probe(tmp_path / f"run/{arm}-epoch-020")
        assert model.arm == metadata["arm"] == arm
        assert metadata["ppoCompatible"] is False
    audit_artifact_manifest(tmp_path / "run", "manifest.json")
