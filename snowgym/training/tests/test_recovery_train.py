import copy
import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.recovery_ppo import RecoveryPolicy
from snowgym_training.executor.movement_ppo import movement_loss
from snowgym_training.options import recovery_train as recovery, recovery_checkpoint, post_hit
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


def test_folded_tail_preserves_exact_discounted_return():
    for gamma in (.5, .9976921765, 1.):
        rewards = [.2, -.1, .7, -1., 1.]
        for learned in (1, 3, 5):
            folded, tail = recovery.fold_tail(rewards, learned, gamma)
            expected = sum(gamma**i*r for i, r in enumerate(rewards))
            actual = sum(gamma**i*float(r) for i, r in enumerate(folded))
            assert actual == pytest.approx(expected, abs=1e-6)
            if learned == 5:
                assert tail == 0
    with pytest.raises(ValueError):
        recovery.fold_tail([1.], 0, .99)


@pytest.fixture(scope="module")
def reference():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata, state = load_ppo_checkpoint(recovery.REFERENCE)
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    source.eval().requires_grad_(False)
    with SnowGymBatchClient() as client:
        wrapper = recovery.make_wrapper(client, 1, .9976921765)
        baseline = post_hit.continuation(source, wrapper, post_hit.reset(wrapper, 100000), 100000, baseline=True)
    frame = recovery.frame_from_baseline(baseline)
    assert frame is not None
    return source, metadata, frame


def test_snapshot_partition_and_tamper(reference, tmp_path):
    _, _, frame = reference
    recovery.validate_frames([frame], training=True)
    with pytest.raises(ValueError, match="evaluation seed"):
        recovery.validate_frames([frame], training=False)
    changed = copy.deepcopy(frame)
    changed["trigger"]["decision"] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        recovery.validate_frames([changed], training=True)
    changed = copy.deepcopy(frame)
    changed["seed"] = 200000
    changed["frameDigest"] = recovery.json_digest({k: v for k, v in changed.items() if k != "frameDigest"})
    with pytest.raises(ValueError, match="development snapshot"):
        recovery.validate_frames([changed], training=True)
    with pytest.raises(FileExistsError):
        recovery.run(tmp_path)


def test_zero_residual_handoff_likelihood_and_budget_paths(reference):
    source, _, frame = reference
    model = RecoveryPolicy(copy.deepcopy(source))
    config = recovery.configuration()
    with SnowGymBatchClient() as client:
        wrapper = recovery.make_wrapper(client, 1, config["gamma"])
        rollout, row, trajectory = recovery.episode(model, source, wrapper, frame, config, deterministic=True)
    assert row["stateHashes"] == frame["suffixHashes"]
    assert row["actionsDigest"] == frame["suffixActionsDigest"]
    assert row["learnedDecisions"] == 30
    assert row["tailDecisions"] > 0
    assert len(trajectory["behaviorLogProbabilities"]) == 30
    assert all(not e["learnerControlled"] for e in trajectory["events"][30:])
    obs = rollout["observation"]
    with torch.no_grad():
        logp, prediction = model.evaluate_latents(obs, rollout["action_type"], rollout["latent"])
    torch.testing.assert_close(logp, rollout["logp"], atol=1e-3, rtol=0)
    with torch.no_grad():
        sampled, latent, old, _ = model.act(obs)
    new, pred = model.evaluate_latents(obs, sampled["action_type"], latent)
    torch.testing.assert_close(new, old, atol=1e-6, rtol=0)
    loss = movement_loss(new, old, torch.linspace(-1, 1, 30), pred["value"], rollout["returns"], pred)
    loss["policy"].backward()
    assert model.budget_move.weight.grad is not None and model.budget_move.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.critic.parameters())
    assert all(p.grad is None for p in model.geometry.shot.parameters())
    model.zero_grad(set_to_none=True)
    for value in (None, torch.full((30, 1), float("nan")), torch.ones(30, 2), torch.ones(30, 1)*2):
        bad = {**obs, "recovery_remaining": value}
        with pytest.raises(ValueError, match="recovery_remaining"):
            model(bad)
    with torch.no_grad():
        model.budget_move.weight.fill_(.5)
    altered = {**obs, "recovery_remaining": torch.zeros(30, 1)}
    assert not torch.equal(model(obs)["mean"], model(altered)["mean"])
    assert torch.equal(model(obs)["action_type"], model(altered)["action_type"])
    new, pred = model.evaluate_latents(obs, rollout["action_type"], rollout["latent"])
    loss = movement_loss(new, rollout["logp"], rollout["advantage"], pred["value"], rollout["returns"], pred)
    loss["value"].backward()
    assert all(p.grad is None for p in model.actor_parameters())
    assert all(p.grad is None for p in model.geometry.source.parameters())
    assert all(p.grad is None for p in model.geometry.shot.parameters())
    with torch.no_grad():
        model.budget_move.weight.fill_(1000.)
        action, _, logp, _ = model.act(obs, deterministic=True)
    assert torch.isfinite(action["target"]).all() and torch.isfinite(logp).all()


def test_exact_update_boundary_resume_and_checkpoint_tamper(reference, tmp_path):
    source, metadata, frame = reference
    cfg = {**recovery.configuration(), "updates": 2, "batchSize": 1,
           "recoveryWindow": 3, "epochs": 1, "minibatchSize": 3}
    for name in ("full", "paused", "resumed", "bad"):
        (tmp_path/name).mkdir()
    with SnowGymBatchClient() as client:
        full, report = recovery.train(source, metadata, client, [frame], cfg, tmp_path/"full", 94101)
        recovery.train(source, metadata, client, [frame], cfg, tmp_path/"paused", 94101, pause_after=1)
        resumed, other = recovery.train(source, metadata, client, [frame], cfg, tmp_path/"resumed", 94101,
                                        resume=tmp_path/"paused/paused")
        assert report["history"] == other["history"]
        assert semantic_state_digest(full.state_dict()) == semantic_state_digest(resumed.state_dict())
        assert recovery_checkpoint.load(tmp_path/"full/final")[2]["checkpointDigest"] == recovery_checkpoint.load(tmp_path/"resumed/final")[2]["checkpointDigest"]
        with pytest.raises(ValueError, match="lineage mismatch"):
            recovery.train(source, metadata, client, [frame], {**cfg, "learningRate": .01}, tmp_path/"bad", 94101,
                           resume=tmp_path/"paused/paused")
    assert report["actorParameterL2Change"] > 0
    path = tmp_path/"full/final/checkpoint.json"
    body = json.loads(path.read_text())
    body["recoveryInput"] = "other"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identity mismatch"):
        recovery_checkpoint.load(path.parent)
