import copy
import gzip
import json

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.movement_ppo import movement_loss
from snowgym_training.options import recovery_audit as audit
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


def test_physical_scale_coverage_and_finite_boundary():
    row = audit.geometry([0, 0], [.02, 0], [.2, 0], [50, 40], .02)
    assert row["sampleWorldDisplacement"] == pytest.approx(50*np.tanh(.02))
    assert row["recommendationWorldGap"] == 10
    assert row["mahalanobis"] == pytest.approx(np.arctanh(.2)/.02)
    assert row["recommendationMeanShiftKl"] == pytest.approx(.5*row["mahalanobis"]**2)
    assert row["sampleImprovesRecommendationDistance"] and not row["withinThreeSigma"]
    doubled = audit.geometry([0, 0], [.02, 0], [.2, 0], [100, 80], .02)
    assert doubled["mahalanobis"] == row["mahalanobis"]
    assert doubled["recommendationWorldGap"] == 2*row["recommendationWorldGap"]
    assert doubled["sampleWorldDisplacement"] == 2*row["sampleWorldDisplacement"]
    boundary = audit.geometry([1000, -1000], [1000, -1000], [1, -1], [50, 40], .02)
    assert boundary["worldDirection"] == [0, 0]
    assert boundary["worldJacobian"] == [0, 0]
    assert np.isfinite(boundary["mahalanobis"])
    with pytest.raises(ValueError):
        audit.geometry([0, 0], [0, 0], [0, 0], [50, 40], 0)


def test_output_score_matches_frozen_ppo_gradient_and_masks():
    torch.manual_seed(43)
    means = torch.randn(5, 3, 2, requires_grad=True)
    latents = means.detach()+.02*torch.randn_like(means)
    living = torch.tensor([[1, 1, 0]]*5, dtype=torch.bool)
    moves = torch.tensor([[1, 0, 0]]*5, dtype=torch.bool)
    advantages = torch.tensor([1., -2., 3., 0., -1.])
    expected, _ = audit.score_directions(means.detach(), latents, advantages, living, moves,
                                       torch.arange(5), 5, .02)
    logp = Normal(means, .02).log_prob(latents).sum(-1)*moves
    losses = movement_loss(logp, logp.detach(), advantages, torch.zeros(5), torch.zeros(5),
                           {"living": living, "move_mask": moves})
    losses["policy"].backward()
    torch.testing.assert_close(expected, -means.grad, rtol=1e-5, atol=1e-5)
    assert not expected[:, 1:].any()
    # Count normalization cancels exact roster duplication in the sum of output gradients.
    repeated, _ = audit.score_directions(means.detach().repeat(1, 2, 1), latents.repeat(1, 2, 1), advantages,
        living.repeat(1, 2), moves.repeat(1, 2), torch.arange(5), 5, .02)
    torch.testing.assert_close(repeated.sum(1), expected.sum(1))


def test_empty_stats_and_existing_output(tmp_path):
    assert audit.statistics([])["count"] == 0
    assert audit.correlation([1, 1], [2, 3]) is None
    with pytest.raises(FileExistsError):
        audit.run(tmp_path)


def test_archived_update_reconstructs_exactly_and_rejects_tamper():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads((audit.ARCHIVE/"declaration.json").read_text())["config"]
    dataset = json.loads((audit.ARCHIVE/"snapshots.json").read_text())["datasets"]["training"]
    frames = {f["seed"]: f for f in dataset}
    metadata, state = load_ppo_checkpoint(audit.REFERENCE)
    source = checkpoint_model(metadata); source.load_state_dict(state["model"])
    with gzip.open(audit.ARCHIVE/"94101/events-011.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    with SnowGymBatchClient() as client:
        wrapper = audit.make_wrapper(client, 1, cfg["gamma"])
        model, _ = audit.behavior(source, audit.ARCHIVE, 94101, 11, cfg)
        before = semantic_state_digest(model.state_dict())
        result = audit.replay_update(model, wrapper, frames, rows, cfg)
        assert result["maxLogProbabilityError"] == 0
        assert len(result["trajectories"]) == 8 and result["opportunities"]
        original = json.loads((audit.ARCHIVE/"94101/training.json").read_text())["history"][10]["minibatches"][0]
        for k, v in result["firstMinibatchLoss"].items():
            assert v == pytest.approx(original[k], abs=1e-6)
        assert semantic_state_digest(model.state_dict()) == before
        model, _ = audit.behavior(source, audit.ARCHIVE, 94101, 11, cfg)
        again = audit.replay_update(model, wrapper, frames, rows, cfg)
        assert result == again
        bad = copy.deepcopy(rows)
        bad[0]["events"][0]["stateHash"] = "tampered"
        model, _ = audit.behavior(source, audit.ARCHIVE, 94101, 11, cfg)
        with pytest.raises(ValueError, match="transition mismatch"):
            audit.replay_update(model, wrapper, frames, bad, cfg)
