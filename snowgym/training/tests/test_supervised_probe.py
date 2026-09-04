import json
from pathlib import Path

import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.loss import LossConfig, behavior_clone_loss
from snowgym_training.options import supervised_probe as probe
from snowgym_training.options import evaluate
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.options.probe_metrics import phase_masks, accumulate, empty_counts, finish_counts, teacher_agreement
from snowgym_training.options.reservoir import TeacherBcReservoir, load_teacher_bc_reservoir, file_digest
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from snowgym_training.trajectory import json_digest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = TRAINING_ROOT / "runs/m7b_engage_teacher_reservoir_r1e_continue200_v0/update-000200/checkpoint"
RESERVOIR = TRAINING_ROOT / "runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz"


def test_probe_phase_partition_uses_alive_and_position_columns() -> None:
    observation = {"allies": torch.zeros(1, 4, 21), "ally_mask": torch.ones(1, 4),
                   "enemies": torch.zeros(1, 1, 21), "enemy_mask": torch.ones(1, 1)}
    observation["allies"][..., 1] = torch.tensor([1, 1, 1, 0])
    observation["enemies"][..., 1] = 1
    observation["allies"][0, 0, 2] = -0.5  # 25 world units away
    observation["allies"][0, 1, 2] = -0.1  # 5 world units away
    phases = phase_masks(observation, torch.tensor([[1, 3, 2, 2]]), torch.tensor([50., 40.]))
    assert {name: int(mask.sum()) for name, mask in phases.items()} == {"all": 3, "approach": 1, "contact": 1, "fire": 1}
    assert torch.equal(phases["approach"] | phases["contact"] | phases["fire"], phases["all"])


def test_probe_ray_error_and_empty_phases_are_finite() -> None:
    observation = {"allies": torch.zeros(1, 1, 21)}
    teacher = {"action_type": torch.tensor([[2]]), "target": torch.tensor([[[1., 0.]]]), "power": torch.tensor([[0.5]])}
    prediction = {"action_logits": torch.tensor([[[0., 0., 3., 0.]]]),
                  "target_by_action": torch.zeros(1, 1, 4, 2), "power": torch.tensor([[0.5]])}
    prediction["target_by_action"][0, 0, 2, 1] = 1
    counts = empty_counts()
    accumulate(counts, prediction, teacher, observation, torch.ones(1, 1, dtype=torch.bool), torch.ones(2))
    result = finish_counts(counts)
    assert result["throwRayMeanDegrees"] == pytest.approx(90)
    assert result["throwTargetRmseWorld"] == pytest.approx(2 ** 0.5)
    assert result["throwPowerRmse"] == 0
    assert result["classAccuracy"] == 1
    assert finish_counts(empty_counts())["throwRayMeanDegrees"] is None
    prediction["target_by_action"].zero_()
    counts = empty_counts()
    accumulate(counts, prediction, teacher, observation, torch.ones(1, 1, dtype=torch.bool), torch.ones(2))
    assert finish_counts(counts)["undefinedThrowRays"] == 1


def test_frozen_probe_configuration_and_source_provenance(tmp_path) -> None:
    config = probe.load_probe_config()
    source, _ = load_ppo_checkpoint(CHECKPOINT)
    reservoir = load_teacher_bc_reservoir(RESERVOIR)
    assert probe.validate_inputs(config, source, reservoir) == list(range(100000, 100040))
    with pytest.raises(ValueError, match="checkpoint digest"):
        probe.validate_inputs(config, {**source, "checkpointDigest": "wrong"}, reservoir)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({**config, "epochs": 40}))
    with pytest.raises(ValueError, match="frozen R1f"):
        probe.load_probe_config(changed)


def test_supervised_epoch_is_deterministic_and_freezes_critic_and_inherited_actor() -> None:
    torch.set_num_threads(1)
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    reservoir = load_teacher_bc_reservoir(RESERVOIR)
    observation, actions = reservoir.batch(torch.arange(32))
    small = TeacherBcReservoir(observation, actions, {})
    final_digests = []
    all_metrics = []
    for _ in range(2):
        model = checkpoint_model(metadata)
        model.load_state_dict(state["model"])
        optimizer = probe.supervised_optimizer(model, learning_rate=0.0003)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        losses = LossConfig(**probe.load_probe_config()["bcLossConfig"])
        with torch.no_grad():
            before = float(behavior_clone_loss(model(observation), actions, observation, losses)["total"])
        metrics = [probe.supervised_epoch(model, optimizer, small, epoch=epoch, seed=91001,
            minibatch_size=16, loss_config=losses, max_grad_norm=0.5) for epoch in range(1, 5)]
        with torch.no_grad():
            after = float(behavior_clone_loss(model(observation), actions, observation, losses)["total"])
        assert after < before
        assert all(torch.equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen)
        assert all(p.grad is None for p in model.role_aware_critic.parameters())
        final_digests.append(semantic_state_digest(model.state_dict()))
        all_metrics.append(metrics)
    assert final_digests[0] == final_digests[1]
    assert all_metrics[0] == all_metrics[1]


def test_reservoir_metrics_reproduce_review_measurements() -> None:
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    result = teacher_agreement(model, load_teacher_bc_reservoir(RESERVOIR))
    assert result["all"]["count"] == 26815
    assert result["all"]["throwCount"] == 1162
    assert result["all"]["confusion"][2] == [0, 316, 846, 0]
    assert result["all"]["throwRayMeanDegrees"] == pytest.approx(31.80654, abs=1e-4)


def test_probe_pipeline_preserves_artifacts_and_cannot_promote(tmp_path, monkeypatch) -> None:
    # Exercise all checkpoint/report paths with stubbed training and episode work.
    monkeypatch.setattr(probe, "supervised_epoch", lambda *args, epoch, **kwargs: {"epoch": epoch, "samples": 0})
    monkeypatch.setattr(probe, "teacher_agreement", lambda *args: {"all": {"count": 0}})
    def episode(*args, seed, condition, **kwargs):
        return {"seed": seed, "success": condition == "correct", "progress": float(condition == "correct"),
                "physicalWin": False, "rejectedActions": 0, "totalActions": 1,
                "firstContactDecision": 1, "firstHitDecision": 2}
    monkeypatch.setattr(probe, "evaluate_option_episode", episode)
    monkeypatch.setattr(evaluate, "evaluate_option_episode", episode)
    output = tmp_path / "probe"
    result = probe.run_supervised_probe(source_checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=output)
    assert result["bootstrapDiagnosticPassed"]
    assert not result["qualificationEligible"]
    assert result["ppoUpdates"] == result["criticUpdates"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["manifestDigest"] == json_digest({key: value for key, value in manifest.items() if key != "manifestDigest"})
    for name, digest in manifest["artifacts"].items():
        assert file_digest(output / name) == digest
    metadata, _ = load_ppo_checkpoint(output / "epoch-020")
    assert metadata["collectorConfig"]["gateId"] == "m7b-engage-supervised-probe"
    assert metadata["environmentSteps"] == 0
    with pytest.raises(FileExistsError):
        probe.run_supervised_probe(source_checkpoint=CHECKPOINT, reservoir_path=RESERVOIR, output=output)
