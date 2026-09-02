from __future__ import annotations

import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.model import ModelConfig
from snowgym_training.ppo import HybridActorCritic, PPOConfig
from snowgym_training.ppo_collect import SeedSchedule, collect_rollout


def test_seed_schedule_is_monotonic_and_refuses_reuse() -> None:
    schedule = SeedSchedule(10, 12)
    assert schedule.take(2) == [10, 11]
    assert schedule.state()["nextSeed"] == 12
    assert schedule.take(1) == [12]
    try:
        schedule.take(1)
    except RuntimeError as error:
        assert "exhausted" in str(error)
    else:
        raise AssertionError("exhausted seed schedule reused a training seed")


def test_live_batch_collector_resets_done_worlds_and_marks_boundary() -> None:
    scenario = {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 12,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }
    torch.manual_seed(17)
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    schedule = SeedSchedule(10_000, 10_099)
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(2, client=client)
        collection = collect_rollout(
            environment,
            model,
            scenario=scenario,
            seed_schedule=schedule,
            rollout_steps=3,
            config=config,
        )

    assert collection.episode_seeds == (10_000, 10_001, 10_002, 10_003)
    assert collection.completed_episodes == 2
    assert collection.boundary_truncations == 2
    assert collection.rejected_actions == 0
    assert collection.canonical_reward_sum == float(collection.rollout.rewards.sum())
    assert collection.training_reward_sum == collection.canonical_reward_sum
    assert collection.seed_schedule["nextSeed"] == 10_004
    assert collection.rollout.rewards.shape == (3, 2)
    assert collection.rollout.observations["allies"].shape == (3, 2, 10, 10)
    assert collection.rollout.truncated[-1].tolist() == [True, True]
    assert collection.rollout.terminated[-1].tolist() == [False, False]
    assert torch.isfinite(collection.rollout.advantages).all()


def test_live_collector_health_shaping_is_explicit_and_preserves_canonical_sum() -> None:
    scenario = {
        "blueUnits": 1,
        "redUnits": 2,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 120,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }
    torch.manual_seed(23)
    model = HybridActorCritic(ModelConfig(16, 12, 24))
    config = PPOConfig(update_epochs=1, minibatch_size=1)
    with SnowGymBatchClient() as client:
        collection = collect_rollout(
            SnowGymBatchEnv(1, client=client),
            model,
            scenario=scenario,
            seed_schedule=SeedSchedule(20_000, 20_009),
            rollout_steps=1,
            config=config,
            reward_mode="health-potential",
        )
    assert collection.canonical_reward_sum == 0.0
    assert collection.training_reward_sum != collection.canonical_reward_sum
    assert float(collection.rollout.rewards.sum()) == collection.training_reward_sum
