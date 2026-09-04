from pathlib import Path

import pytest
import torch

from snowgym_training.checkpoint import load_checkpoint, semantic_state_digest
from snowgym_training.options import evaluate
from snowgym_training.options.identity import recover_initializer, optimizer_learning_rates, parameter_changes, checkpoint_model
from snowgym_training.options.train import DEFAULT_INITIALIZER
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


CHECKPOINT = Path(__file__).resolve().parents[1] / "runs/m7b_engage_teacher_reservoir_r1e_continue200_v0/update-000200/checkpoint"


def test_legacy_initializer_is_rng_independent_and_preserves_rng() -> None:
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    source_metadata, source_state = load_checkpoint(DEFAULT_INITIALIZER)
    torch.manual_seed(19)
    before = torch.get_rng_state().clone()
    first, identity = recover_initializer(metadata, state, source_metadata, source_state)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(29)
    second, repeated = recover_initializer(metadata, state, source_metadata, source_state)
    assert repeated == identity
    assert semantic_state_digest(first.state_dict()) == semantic_state_digest(second.state_dict())
    assert identity["sourceIdentityVerified"]
    assert parameter_changes(first, second)["actorTotal"] == 0
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    changes = parameter_changes(model, first)
    assert changes["actorTotal"] == pytest.approx(5.6855787707462495)
    assert changes["inheritedHeads"] == 0
    assert changes["otherInheritedActor"] == 0


def test_stored_initializer_and_source_identity_are_authoritative() -> None:
    metadata, state = load_ppo_checkpoint(CHECKPOINT)
    source_metadata, source_state = load_checkpoint(DEFAULT_INITIALIZER)
    initializer, _ = recover_initializer(metadata, state, source_metadata, source_state)
    with torch.no_grad():
        next(initializer.parameters()).add_(0.2)
    state["initializerModel"] = initializer.state_dict()
    metadata["initializerDigest"] = semantic_state_digest(state["initializerModel"])
    metadata["initializerSourceDigest"] = source_metadata["checkpointDigest"]
    restored, identity = recover_initializer(metadata, state, source_metadata, source_state)
    assert identity["method"] == "stored-expanded-state"
    assert semantic_state_digest(restored.state_dict()) == metadata["initializerDigest"]
    with pytest.raises(ValueError, match="source checkpoint mismatch"):
        recover_initializer(metadata, state, {**source_metadata, "checkpointDigest": "sha256:" + "0" * 64}, source_state)


def test_learning_rates_are_read_from_optimizer_state() -> None:
    state = {"optimizer": {"param_groups": [{"name": "new", "lr": 0.0007}, {"name": "heads", "lr": 0.00002}]}}
    assert optimizer_learning_rates(state) == {"new": 0.0007, "heads": 0.00002}
    with pytest.raises(ValueError, match="unique audited names"):
        optimizer_learning_rates({"optimizer": {"param_groups": [{"lr": 1}]}})


def test_repeated_evaluation_has_identical_digest(tmp_path, monkeypatch) -> None:
    # Isolate evaluator metadata from simulator cost; real episode parity is tested elsewhere.
    monkeypatch.setattr(evaluate, "evaluate_option_episode", lambda *args, seed, **kwargs: {
        "seed": seed, "success": False, "progress": 0.0, "physicalWin": False,
        "rejectedActions": 0, "totalActions": 1,
        "firstContactDecision": None, "firstHitDecision": None,
    })
    first = evaluate.evaluate_m7b_checkpoint(CHECKPOINT, output=tmp_path / "first.json", options=("engage",))
    torch.manual_seed(17)
    second = evaluate.evaluate_m7b_checkpoint(CHECKPOINT, output=tmp_path / "second.json", options=("engage",))
    assert first == second
    assert first["inheritedHeadLearningRate"] == 0
    assert first["newModuleLearningRate"] == 0.0003
