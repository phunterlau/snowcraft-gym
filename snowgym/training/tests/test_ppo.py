from __future__ import annotations

import torch

from snowgym_training.model import ModelConfig
from snowgym_training.curriculum import load_curriculum, validate_curriculum
from snowgym_training.ppo import (
    HybridActorCritic,
    PPOConfig,
    generalized_advantage_estimate,
    health_potential,
    potential_shaped_reward,
    ppo_loss,
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
