from __future__ import annotations

import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.loss import LossConfig
from snowgym_training.model import ModelConfig
from snowgym_training.ppo import HybridActorCritic, PPOConfig
from snowgym_training.plan_ppo import (
    freeze_initializer,
    plan_ppo_parameter_groups,
    plan_ppo_update,
    target_only_plan_ppo_config,
)
from snowgym_training.ppo_collect import (
    PlanSchedule,
    SeedSchedule,
    collect_plan_rollout,
    collect_rollout,
)


def fixed_plan() -> dict[str, object]:
    return {
        "schemaVersion": "snowgym.command-plan.v0",
        "intentSummary": "Engage the nearest enemy as one main group.",
        "groups": [
            {
                "role": "main",
                "allocationWeight": 1,
                "selection": "balanced",
                "order": {
                    "mission": "engage",
                    "objective": {"kind": "enemy_cluster", "select": "nearest"},
                    "approach": "direct",
                    "engagement": {
                        "posture": "balanced",
                        "fire": "focus",
                        "preferredRange": "medium",
                        "cohesion": "normal",
                    },
                },
            }
        ],
    }


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


def test_plan_schedule_resume_binds_contents_and_cursor() -> None:
    plans = tuple(fixed_plan() for _ in range(4))
    schedule = PlanSchedule(plans, prefix="engage")
    assert schedule.take(2)[0] == ["engage-000000", "engage-000001"]
    restored = PlanSchedule.restore(plans, schedule.state())
    assert restored.take(2)[0] == ["engage-000002", "engage-000003"]
    changed = tuple([*plans[:-1], {**fixed_plan(), "intentSummary": "changed"}])
    try:
        PlanSchedule.restore(changed, schedule.state())
    except ValueError as error:
        assert "contents" in str(error)
    else:
        raise AssertionError("plan schedule restored against changed plans")


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
    assert collection.training_reward_sum == collection.canonical_reward_sum
    assert float(collection.rollout.rewards.sum()) == collection.training_reward_sum


def test_plan_collector_refreshes_v3_state_and_restores_plans_after_selective_reset() -> None:
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
    base = ModelConfig(
        16,
        12,
        24,
        action_conditioned_targets=True,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
    )
    torch.manual_seed(29)
    model = HybridActorCritic(target_only_plan_ppo_config(base))
    initializer = freeze_initializer(model)
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    schedule = PlanSchedule(tuple(fixed_plan() for _ in range(6)), prefix="fixed")
    with SnowGymBatchClient() as client:
        collection = collect_plan_rollout(
            SnowGymBatchEnv(2, client=client, observation_version=3),
            model,
            scenario=scenario,
            seed_schedule=SeedSchedule(30_000, 30_099),
            plan_schedule=schedule,
            rollout_steps=3,
            config=config,
        )
    assert collection.episode_plan_ids == (
        "fixed-000000", "fixed-000001", "fixed-000002", "fixed-000003"
    )
    assert collection.plan_schedule is not None
    assert collection.plan_schedule["nextIndex"] == 4
    assert collection.rollout.observations["allies"].shape == (3, 2, 10, 21)
    assert collection.rollout.observations["plan_role_state"].shape == (3, 2, 3, 20)
    assert collection.rollout.observations["mission_progress"].shape == (3, 2, 3)
    assert collection.teacher_actions is not None
    assert collection.teacher_actions["action_type"].shape == (3, 2, 10)
    assert collection.teacher_actions["target"].shape == (3, 2, 10, 2)
    assert collection.completed_episodes == 2
    assert collection.rejected_actions == 0
    inherited_before = model.policy.actor[0].weight.detach().clone()
    residual_before = model.policy.plan_ppo_residual[2].weight.detach().clone()
    optimizer = torch.optim.Adam(plan_ppo_parameter_groups(model, 1))
    assert collection.teacher_actions is not None
    metrics = plan_ppo_update(
        model,
        optimizer,
        collection.rollout,
        collection.teacher_actions,
        initializer,
        config,
        loss_config=LossConfig(),
        training_seed=29,
        update_index=0,
        total_updates=4,
    )
    assert metrics["anchorWeights"] == {"bc": 0.1, "initializerKl": 0.01}
    torch.testing.assert_close(model.policy.actor[0].weight, inherited_before)
    assert not torch.equal(model.policy.plan_ppo_residual[2].weight, residual_before)
