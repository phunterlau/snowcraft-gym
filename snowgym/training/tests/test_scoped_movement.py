import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.movement_ppo import AssistedMovementPolicy, movement_loss
from snowgym_training.options.definitions import FROZEN_OPTION_SPECS
from snowgym_training.options.engage_v1 import FrozenEngageTracker
from snowgym_training.options.engage_v1 import EngageOptionBatchV1
from snowgym_training.options.movement_collect import MovementCollector, corrected_shots, world_identities
from snowgym_training.ppo_collect import SeedSchedule
from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.trajectory import json_digest
from snowgym_training.options.opportunity_audit import plain
from snowgym_training.options.movement_checkpoint import save_movement, load_movement
from snowgym_training.options.geometry_probe import gate_indices
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.options.reservoir import load_teacher_bc_reservoir
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from test_options import observation as raw_observation, plan, plan_observation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    metadata, state = load_ppo_checkpoint(ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020")
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    dataset = load_teacher_bc_reservoir(ROOT / "runs/m7b_engage_teacher_reservoir_v0/teacher_states.npz")
    obs, _ = dataset.batch(gate_indices(dataset))
    obs["option_state"] = torch.tensor([[1., 1., 1.]]).expand(obs["allies"].shape[0], -1)
    with torch.random.fork_rng():
        torch.manual_seed(94001)
        model = AssistedMovementPolicy(source)
    return model, obs


def test_frozen_target_identity_budget_and_terminal_potential():
    raw = raw_observation()
    body = {**plan_observation(), "activationObjectives": [{"role": "main", "kind": "enemy_cluster", "enemyIds": [10]}]}
    tracker = FrozenEngageTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), raw, body)
    np.testing.assert_array_equal(tracker.option_state(raw), [1, 1, 1])
    # Another cluster remains at full health and the tactical role summary now
    # points to it. Scoring must retain the original ID and initial denominator.
    changed = copy.deepcopy(raw)
    changed["enemies"][0]["alive"] = False
    changed["enemies"][0]["health"] = 0
    fallback = {**body, "activationObjectives": [{"role": "main", "kind": "enemy_cluster", "enemyIds": [11]}]}
    result = tracker.update(changed, fallback, canonical_reward=0., gamma=.9976921765)
    assert result.success and result.done and result.progress == 1
    assert tracker.activated_target_ids == (10,)
    assert tracker.previous_progress == 0 and result.shaping_reward == 0
    assert tracker.option_state(changed)[0] == pytest.approx(199/200)
    raw["enemies"][0]["health"] = 40
    partial = FrozenEngageTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), raw, body)
    raw["enemies"][0]["health"] = 8
    assert partial.update(raw, body, canonical_reward=0., gamma=.9976921765).success


def test_engage_v1_missing_membership_and_timeout_are_explicit():
    raw, body = raw_observation(), plan_observation()
    with pytest.raises(ValueError, match="authoritative"):
        FrozenEngageTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), raw, body)
    body["activationObjectives"] = [{"role": "main", "kind": "enemy_cluster", "enemyIds": [10]}]
    tracker = FrozenEngageTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), raw, body)
    tracker.decision = 199
    result = tracker.update(raw, body, canonical_reward=0., gamma=.9976921765)
    assert result.done and result.timed_out and result.mission_reward == -1
    assert tracker.option_state(raw)[0] == 0 and tracker.previous_progress == 0


def test_zero_noise_exact_source_parity_and_latent_likelihood(inputs):
    model, obs = inputs
    with torch.no_grad():
        expected = model.geometry.source.act(obs, deterministic=True)[0]
        actual, _, _, _ = model.act(obs, deterministic=True)
        for key in actual:
            assert torch.equal(actual[key], expected[key])
        torch.manual_seed(94002)
        action, latent, stored, _ = model.act(obs)
        evaluated, prediction = model.evaluate_latents(obs, action["action_type"], latent)
        assert torch.equal(stored, evaluated)
        assert not latent[~prediction["move_mask"]].any()
        assert not stored[~prediction["move_mask"]].any()
        assert torch.isfinite(stored).all()
    changed = action["action_type"].clone()
    changed[0, 0] = (changed[0, 0]+1) % 4
    with pytest.raises(ValueError, match="frozen action"):
        model.evaluate_latents(obs, changed, latent)


def test_actor_and_critic_independent_gradients_and_unused_shots(inputs):
    reference, obs = inputs
    model = copy.deepcopy(reference)
    with torch.no_grad():
        action, latent, old, _ = model.act(obs)
    new, prediction = model.evaluate_latents(obs, action["action_type"], latent)
    advantage = torch.linspace(-1, 1, new.shape[0])
    losses = movement_loss(new, old, advantage, prediction["value"], torch.ones_like(advantage), prediction)
    losses["policy"].backward()
    assert all(p.grad is None or not p.grad.any() for p in model.critic.parameters())
    assert all(p.grad is None for p in model.geometry.source.parameters())
    assert all(p.grad is None for p in model.geometry.shot.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.actor_parameters())
    model.zero_grad(set_to_none=True)
    model(obs)["value"].sum().backward()
    assert all(p.grad is None for p in model.actor_parameters())


def test_roster_duplication_invariance_and_no_move_decisions():
    advantage = torch.tensor([-1., 1.])
    def loss(units):
        old = torch.zeros(2, units)
        new = torch.full_like(old, .01, requires_grad=True)
        prediction = {"living": torch.ones_like(old, dtype=torch.bool), "move_mask": torch.ones_like(old, dtype=torch.bool)}
        return movement_loss(new, old, advantage, advantage*0, advantage, prediction)
    baseline = loss(1)
    for units in (3, 5, 10):
        for key, value in loss(units).items():
            torch.testing.assert_close(value, baseline[key])
    empty = torch.zeros(2, 5)
    result = movement_loss(empty, empty, advantage, advantage*0, advantage,
                           {"living": empty.bool(), "move_mask": empty.bool()})
    assert result["policy"] == 0 and all(torch.isfinite(v) for v in result.values())


def test_option_observation_changes_new_paths_only_and_extremes_finite(inputs):
    source, obs = inputs
    model = copy.deepcopy(source)
    before = semantic_state_digest(model.geometry.source.state_dict())
    with torch.no_grad():
        model.option_move.weight.fill_(.5)
    changed = {**obs, "option_state": torch.zeros_like(obs["option_state"])}
    assert not torch.equal(model(obs)["mean"], model(changed)["mean"])
    assert torch.equal(model(obs)["action_type"], model(changed)["action_type"])
    with torch.no_grad():
        model.option_move.weight.fill_(1000.)
        action, latent, logp, _ = model.act(obs, deterministic=True)
        assert torch.isfinite(action["target"]).all() and torch.isfinite(logp).all()
        assert action["target"].abs().max() <= 1
    assert semantic_state_digest(model.geometry.source.state_dict()) == before
    with pytest.raises(ValueError, match="option_state"):
        model({k: v for k, v in obs.items() if k != "option_state"})


def test_live_prefix_resume_selective_reset_and_latents(inputs):
    model, _ = inputs
    with SnowGymBatchClient() as client:
        def collector():
            wrapper = EngageOptionBatchV1(SnowGymBatchEnv(2, client=client, observation_version=3), gamma=.9976921765)
            return MovementCollector(wrapper, model, SeedSchedule(100000, 107999))
        uninterrupted = collector()
        torch.manual_seed(94200)
        uninterrupted.start(6)
        uninterrupted.advance(2)
        # Different active prefix lengths exercise selected-world reconstruction.
        replacement = uninterrupted._reset([1], [100050])
        uninterrupted.seeds[1], uninterrupted.prefixes[1] = 100050, []
        for key in uninterrupted.observation:
            uninterrupted.observation[key][1] = replacement[key][0]
        snapshot = uninterrupted.snapshot()
        uninterrupted.advance()
        expected = uninterrupted.snapshot()
        restored = collector()
        restored.restore(snapshot)
        restored.advance()
        actual = restored.snapshot()
        assert json_digest(plain(actual)) == json_digest(plain(expected))
        rollout = restored.rollout(gamma=.9976921765, gae_lambda=.9885140204)
        with torch.no_grad():
            reevaluated, _ = model.evaluate_latents(rollout["observation"], rollout["action_type"], rollout["latent"])
        # Different GEMM batch sizes round frozen means slightly differently;
        # fixed std .02 magnifies that into at most 1e-4 latent log-density error.
        torch.testing.assert_close(reevaluated, rollout["logp"], rtol=0, atol=1e-4)
        assert all(torch.isfinite(rollout[k]).all() for k in ("advantage", "returns"))
        corrupted = copy.deepcopy(snapshot)
        corrupted["identities"][0]["physical"] = "changed"
        with pytest.raises(ValueError, match="identity mismatch"):
            collector().restore(corrupted)
        # Timeout is terminal, not the artificial rollout bootstrap cut.
        timeout = collector()
        timeout.start(1)
        timeout.wrapper.trackers[0].decision = 199
        before = world_identities(timeout.wrapper)[1]
        timeout.advance()
        assert timeout.records[0]["terminated"][0]
        assert not timeout.records[0]["truncated"][0]
        assert timeout.wrapper.trackers[0].decision == 0
        assert timeout.wrapper.trackers[1].decision == 1


def test_shot_assistance_cannot_change_move_or_action_choice(inputs):
    model, obs = inputs
    with torch.no_grad():
        action = model.act(obs, deterministic=True)[0]
    arrays = {k: v.numpy().copy() for k, v in action.items()}
    # Geometric recommendations remain independent from whether THROW is ready.
    raw = raw_observation()
    raw["allies"][0]["throwCooldown"] = 10
    arrays = {k: v[:1].copy() for k, v in arrays.items()}
    arrays["action_type"][:] = 0
    arrays["action_type"][0, 0] = 2
    result = corrected_shots(arrays, [raw])
    np.testing.assert_array_equal(result["action_type"], arrays["action_type"])
    np.testing.assert_array_equal(result["target"][0, 1:], arrays["target"][0, 1:])
    raw["enemies"] = []
    with pytest.raises(ValueError, match="recommendation"):
        corrected_shots(arrays, [raw])


def test_assisted_checkpoint_partial_collection_roundtrip_and_tamper(inputs, tmp_path):
    import json
    model, _ = inputs
    metadata, _ = load_ppo_checkpoint(ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    with SnowGymBatchClient() as client:
        wrapper = EngageOptionBatchV1(SnowGymBatchEnv(1, client=client, observation_version=3), gamma=.9976921765)
        collector = MovementCollector(wrapper, model, SeedSchedule(100000, 107999))
        collector.start(3)
        collector.advance(1)
        snapshot = collector.snapshot()
        path = tmp_path / "checkpoint"
        save_movement(path, model, optimizer, source=metadata, config={"latentStd": .02, "learningRate": 3e-4},
                      seed=94001, update=0, schedule=collector.schedule.state(), collection=snapshot)
        restored, _, manifest, saved = load_movement(path)
        assert manifest["autonomousQualificationEligible"] is False
        assert semantic_state_digest(restored.state_dict()) == semantic_state_digest(model.state_dict())
        assert json_digest(plain(saved)) == json_digest(plain(snapshot))
        with pytest.raises(FileExistsError):
            save_movement(path, model, optimizer, source=metadata, config={}, seed=1, update=0, schedule={})
        altered = json.loads((path / "checkpoint.json").read_text())
        altered["autonomousQualificationEligible"] = True
        (path / "checkpoint.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="identity"):
            load_movement(path)
