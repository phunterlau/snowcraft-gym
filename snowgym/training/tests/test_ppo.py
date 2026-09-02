from __future__ import annotations

import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.model import ModelConfig
from snowgym_training.curriculum import load_curriculum, validate_curriculum
from snowgym_training.ppo import (
    HybridActorCritic,
    PPOConfig,
    PPORollout,
    RolloutBuffer,
    generalized_advantage_estimate,
    health_potential,
    potential_shaped_reward,
    ppo_loss,
    ppo_update,
)
from snowgym_training.ppo_checkpoint import (
    restore_ppo_checkpoint,
    save_ppo_checkpoint,
)

def synthetic_observation(batch: int = 3, units: int = 2) -> dict[str, torch.Tensor]:
    action_mask = torch.zeros((batch, units, 4), dtype=torch.int8)
    action_mask[:, 0] = torch.tensor([1, 1, 0, 1], dtype=torch.int8)
    return {
        "allies": torch.zeros((batch, units, 10)),
        "ally_mask": torch.tensor([[1, 0]] * batch, dtype=torch.int8),
        "enemies": torch.zeros((batch, units, 10)),
        "enemy_mask": torch.tensor([[1, 0]] * batch, dtype=torch.int8),
        "projectiles": torch.zeros((batch, 64, 8)),
        "projectile_mask": torch.zeros((batch, 64), dtype=torch.int8),
        "unit_action_mask": action_mask,
        "tick": torch.zeros((batch, 1), dtype=torch.int64),
        "team_alive": torch.ones((batch, 2), dtype=torch.int32),
        "obstacles": torch.zeros((batch, 64, 9)),
        "obstacle_mask": torch.zeros((batch, 64), dtype=torch.int8),
    }


def synthetic_rollout(model: HybridActorCritic, config: PPOConfig) -> PPORollout:
    buffer = RolloutBuffer(steps=2, batch_size=2)
    for step in range(2):
        observation = synthetic_observation(batch=2)
        observation["allies"][:, 0, 0] = float(step)
        with torch.no_grad():
            action, log_probability, value = model.act(observation, deterministic=True)
        buffer.add(
            observation=observation,
            action=action,
            log_probability=log_probability,
            value=value,
            reward=torch.tensor([0.0, 1.0 if step == 1 else -0.25]),
            terminated=torch.tensor([False, step == 1]),
            truncated=torch.tensor([step == 1, False]),
            next_value=torch.tensor([0.1, 0.2]),
        )
    return buffer.finish(config)


def test_hybrid_policy_respects_masks_and_recomputes_log_probability() -> None:
    torch.manual_seed(4)
    observation = synthetic_observation()
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    action, sampled_log_probability, value = model.act(observation)
    recomputed, entropy = model.evaluate_actions(observation, action)

    assert all(value != 2 for value in action["action_type"][:, 0].tolist())
    assert action["action_type"][:, 1].tolist() == [0, 0, 0]
    assert torch.allclose(sampled_log_probability, recomputed, atol=2e-5)
    assert torch.isfinite(recomputed).all()
    assert torch.isfinite(entropy).all()
    assert value.shape == (3,)


def test_noop_joint_log_probability_ignores_continuous_values() -> None:
    observation = synthetic_observation(batch=1)
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    first = {
        "action_type": torch.zeros((1, 2), dtype=torch.int64),
        "target": torch.zeros((1, 2, 2)),
        "power": torch.full((1, 2), 0.5),
    }
    second = {
        **first,
        "target": torch.full((1, 2, 2), 0.9),
        "power": torch.full((1, 2), 0.9),
    }
    first_log_probability, _ = model.evaluate_actions(observation, first)
    second_log_probability, _ = model.evaluate_actions(observation, second)
    torch.testing.assert_close(first_log_probability, second_log_probability)


def test_gae_bootstraps_truncation_but_not_terminal_and_stops_recurrence() -> None:
    rewards = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[0.2], [0.3]])
    next_values = torch.tensor([[0.3], [0.7]])
    terminated = torch.tensor([[False], [True]])
    truncated = torch.tensor([[False], [False]])
    advantages, _ = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages[1], torch.tensor([0.7]))

    truncated[-1] = True
    terminated[-1] = False
    truncated_advantages, _ = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(truncated_advantages[1], torch.tensor([1.33]))
    torch.testing.assert_close(truncated_advantages[0], torch.tensor([1.267]))


def test_ppo_loss_and_gradients_are_finite() -> None:
    observation = synthetic_observation(batch=4)
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    with torch.no_grad():
        action, old_log_probability, _ = model.act(observation)
    prediction = model(observation)
    new_log_probability, entropy = model.evaluate_actions(
        observation, action, prediction=prediction
    )
    losses = ppo_loss(
        new_log_probability,
        old_log_probability,
        torch.tensor([1.0, 0.5, -0.5, -1.0]),
        prediction["value"],
        torch.zeros(4),
        entropy,
        PPOConfig(),
    )
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_curriculum_freezes_disjoint_training_and_evaluation_seeds() -> None:
    curriculum = load_curriculum()
    assert [gate["id"] for gate in curriculum["gates"]] == [
        "1v1-random",
        "1v1-easy-scripted",
        "3v3-random",
        "3v3-scripted",
        "3v3-terrain",
        "5v5-random-terrain",
        "10v10-random-terrain",
    ]
    broken = {**curriculum, "gates": [dict(gate) for gate in curriculum["gates"]]}
    broken["gates"][0]["evaluationSeeds"] = [10000]
    try:
        validate_curriculum(broken)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping curriculum seeds were accepted")


def test_potential_shaping_is_opt_in_and_zeroes_terminal_bootstrap() -> None:
    observation = synthetic_observation(batch=1)
    observation["allies"][0, 0, 6] = 0.8
    observation["enemies"][0, 0, 6] = 0.4
    potential = health_potential(observation)
    canonical = torch.tensor([1.0])
    shaped = potential_shaped_reward(
        canonical,
        potential,
        torch.tensor([0.9]),
        torch.tensor([True]),
        gamma=0.99,
    )
    torch.testing.assert_close(canonical, torch.tensor([1.0]))
    torch.testing.assert_close(shaped, torch.tensor([0.6]))


def test_rollout_buffer_snapshots_computes_gae_and_flattens() -> None:
    buffer = RolloutBuffer(steps=2, batch_size=2)
    observation = synthetic_observation(batch=2)
    action = {
        "action_type": torch.zeros((2, 2), dtype=torch.int64),
        "target": torch.zeros((2, 2, 2)),
        "power": torch.full((2, 2), 0.5),
    }
    buffer.add(
        observation=observation,
        action=action,
        log_probability=torch.tensor([-0.1, -0.2]),
        value=torch.tensor([0.2, 0.4]),
        reward=torch.tensor([0.0, 0.0]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        next_value=torch.tensor([0.3, 0.5]),
    )
    observation["allies"].fill_(99)
    buffer.add(
        observation=synthetic_observation(batch=2),
        action=action,
        log_probability=torch.tensor([-0.3, -0.4]),
        value=torch.tensor([0.3, 0.5]),
        reward=torch.tensor([1.0, -1.0]),
        terminated=torch.tensor([True, False]),
        truncated=torch.tensor([False, True]),
        next_value=torch.tensor([10.0, 0.7]),
    )

    rollout = buffer.finish(PPOConfig(gamma=0.9, gae_lambda=1.0))
    assert rollout.steps == 2
    assert rollout.batch_size == 2
    assert rollout.observations["allies"].shape == (2, 2, 2, 10)
    assert rollout.observations["allies"][0].max().item() == 0
    torch.testing.assert_close(rollout.advantages[1], torch.tensor([0.7, -0.87]))
    torch.testing.assert_close(rollout.advantages[0], torch.tensor([0.7, -0.733]))
    flattened = rollout.flatten()
    assert flattened["observations"]["allies"].shape == (4, 2, 10)
    assert flattened["actions"]["target"].shape == (4, 2, 2)
    assert flattened["advantages"].shape == (4,)


def test_rollout_buffer_rejects_incomplete_or_inconsistent_transitions() -> None:
    buffer = RolloutBuffer(steps=1, batch_size=2)
    try:
        buffer.finish(PPOConfig())
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete rollout was accepted")

    observation = synthetic_observation(batch=2)
    action = {
        "action_type": torch.zeros((2, 2), dtype=torch.int64),
        "target": torch.zeros((2, 2, 2)),
        "power": torch.zeros((2, 2)),
    }
    try:
        buffer.add(
            observation=observation,
            action={**action, "target": torch.zeros((2, 2, 3))},
            log_probability=torch.zeros(2),
            value=torch.zeros(2),
            reward=torch.zeros(2),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            next_value=torch.zeros(2),
        )
    except ValueError as error:
        assert "action.target" in str(error)
    else:
        raise AssertionError("invalid rollout action shape was accepted")


def test_ppo_update_is_deterministic_and_reports_clipping_diagnostics() -> None:
    torch.manual_seed(27)
    config = PPOConfig(update_epochs=2, minibatch_size=3, learning_rate=0.001)
    reference = HybridActorCritic(ModelConfig(16, 12, 24))
    rollout = synthetic_rollout(reference, config)
    first = HybridActorCritic(ModelConfig(16, 12, 24))
    second = HybridActorCritic(ModelConfig(16, 12, 24))
    first.load_state_dict(reference.state_dict())
    second.load_state_dict(reference.state_dict())
    first_optimizer = torch.optim.Adam(first.parameters(), lr=config.learning_rate)
    second_optimizer = torch.optim.Adam(second.parameters(), lr=config.learning_rate)

    first_metrics = ppo_update(
        first,
        first_optimizer,
        rollout,
        config,
        training_seed=91,
        update_index=0,
    )
    second_metrics = ppo_update(
        second,
        second_optimizer,
        rollout,
        config,
        training_seed=91,
        update_index=0,
    )

    assert first_metrics == second_metrics
    assert first_metrics["samples"] == 4
    assert first_metrics["minibatches"] == 4
    assert first_metrics["maximumGradientNormAfterClip"] <= config.max_grad_norm
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert any(
        not torch.equal(first.state_dict()[name], reference.state_dict()[name])
        for name in first.state_dict()
    )


def test_ppo_checkpoint_resume_matches_uninterrupted_updates(tmp_path) -> None:
    torch.manual_seed(41)
    config = PPOConfig(update_epochs=2, minibatch_size=3, learning_rate=0.001)
    initial = HybridActorCritic(ModelConfig(16, 12, 24))
    rollout = synthetic_rollout(initial, config)
    uninterrupted = HybridActorCritic(ModelConfig(16, 12, 24))
    interrupted = HybridActorCritic(ModelConfig(16, 12, 24))
    uninterrupted.load_state_dict(initial.state_dict())
    interrupted.load_state_dict(initial.state_dict())
    uninterrupted_optimizer = torch.optim.Adam(
        uninterrupted.parameters(), lr=config.learning_rate
    )
    interrupted_optimizer = torch.optim.Adam(
        interrupted.parameters(), lr=config.learning_rate
    )

    ppo_update(
        uninterrupted,
        uninterrupted_optimizer,
        rollout,
        config,
        training_seed=73,
        update_index=0,
    )
    ppo_update(
        interrupted,
        interrupted_optimizer,
        rollout,
        config,
        training_seed=73,
        update_index=0,
    )
    checkpoint = tmp_path / "ppo-checkpoint"
    metadata = save_ppo_checkpoint(
        checkpoint,
        model=interrupted,
        optimizer=interrupted_optimizer,
        config=config,
        curriculum_digest="sha256:test-curriculum",
        training_seed=73,
        update_index=1,
        environment_steps=4,
        git_commit="test",
        seed_schedule={"minimum": 10_000, "maximum": 10_099, "nextSeed": 10_004},
        collector_config={"gateId": "test", "worlds": 2, "rolloutSteps": 2, "rewardMode": "canonical"},
    )
    assert metadata["updateIndex"] == 1
    assert metadata["environmentSteps"] == 4
    assert metadata["seedSchedule"]["nextSeed"] == 10_004

    torch.manual_seed(999)
    resumed = HybridActorCritic(ModelConfig(16, 12, 24))
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=config.learning_rate)
    restored = restore_ppo_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        config=config,
        curriculum_digest="sha256:test-curriculum",
        training_seed=73,
        collector_config={"gateId": "test", "worlds": 2, "rolloutSteps": 2, "rewardMode": "canonical"},
    )
    assert restored["checkpointDigest"] == metadata["checkpointDigest"]
    uninterrupted_metrics = ppo_update(
        uninterrupted,
        uninterrupted_optimizer,
        rollout,
        config,
        training_seed=73,
        update_index=1,
    )
    resumed_metrics = ppo_update(
        resumed,
        resumed_optimizer,
        rollout,
        config,
        training_seed=73,
        update_index=1,
    )

    assert resumed_metrics == uninterrupted_metrics
    assert semantic_state_digest(
        {
            "model": resumed.state_dict(),
            "optimizer": resumed_optimizer.state_dict(),
        }
    ) == semantic_state_digest(
        {
            "model": uninterrupted.state_dict(),
            "optimizer": uninterrupted_optimizer.state_dict(),
        }
    )


def test_ppo_checkpoint_rejects_incompatible_resume(tmp_path) -> None:
    config = PPOConfig(update_epochs=1, minibatch_size=2)
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = tmp_path / "ppo-checkpoint"
    save_ppo_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        curriculum_digest="sha256:first",
        training_seed=5,
        update_index=0,
        environment_steps=0,
        git_commit="test",
        seed_schedule={"minimum": 10_000, "maximum": 10_099, "nextSeed": 10_000},
        collector_config={"gateId": "test", "worlds": 2, "rolloutSteps": 2, "rewardMode": "canonical"},
    )
    try:
        restore_ppo_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            curriculum_digest="sha256:other",
            training_seed=5,
            collector_config={"gateId": "test", "worlds": 2, "rolloutSteps": 2, "rewardMode": "canonical"},
        )
    except ValueError as error:
        assert "curriculumDigest" in str(error)
    else:
        raise AssertionError("incompatible PPO checkpoint was accepted")

    try:
        restore_ppo_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            curriculum_digest="sha256:first",
            training_seed=5,
            collector_config={"gateId": "test", "worlds": 3, "rolloutSteps": 2, "rewardMode": "canonical"},
        )
    except ValueError as error:
        assert "collectorConfig" in str(error)
    else:
        raise AssertionError("checkpoint accepted changed rollout geometry")
