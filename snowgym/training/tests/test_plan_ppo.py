from __future__ import annotations

import torch

from snowgym_training.executor import EntityPolicy, ModelConfig, model_config
from snowgym_training.plan_ppo import (
    freeze_initializer,
    initialize_plan_ppo_policy,
    initializer_policy_kl,
    plan_ppo_anchor_weights,
    plan_ppo_parameter_groups,
    target_only_plan_ppo_config,
)
from snowgym_training.ppo import HybridActorCritic


def source_observation(batch: int = 2, units: int = 3) -> dict[str, torch.Tensor]:
    action_mask = torch.ones((batch, units, 4), dtype=torch.int8)
    observation = {
        "allies": torch.randn(batch, units, 10),
        "ally_mask": torch.ones((batch, units), dtype=torch.int8),
        "enemies": torch.randn(batch, units, 10),
        "enemy_mask": torch.ones((batch, units), dtype=torch.int8),
        "projectiles": torch.randn(batch, 4, 8),
        "projectile_mask": torch.ones((batch, 4), dtype=torch.int8),
        "obstacles": torch.randn(batch, 4, 9),
        "obstacle_mask": torch.ones((batch, 4), dtype=torch.int8),
        "unit_action_mask": action_mask,
        "team_alive": torch.full((batch, 2), units, dtype=torch.int32),
        "tick": torch.zeros((batch, 1), dtype=torch.int64),
        "plan_groups": torch.randn(batch, 3, 38).clamp(-1, 1),
        "plan_group_mask": torch.ones((batch, 3), dtype=torch.int8),
    }
    return observation


def target_observation(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(source)
    result["allies"] = torch.nn.functional.pad(source["allies"], (0, 11))
    result["enemies"] = torch.nn.functional.pad(source["enemies"], (0, 11))
    result["projectiles"] = torch.nn.functional.pad(source["projectiles"], (0, 1))
    batch, units = source["allies"].shape[:2]
    roles = torch.zeros((batch, units, 3), dtype=torch.int8)
    roles[..., 0] = 1
    result.update(
        {
            "plan_unit_roles": roles,
            "plan_role_state": torch.zeros((batch, 3, 20)),
            "mission_progress": torch.zeros((batch, 3)),
            "decision_hz": torch.full((batch, 1), 10, dtype=torch.int32),
            "decision_dt": torch.full((batch, 1), 0.1),
            "max_ticks": torch.full((batch, 1), 300, dtype=torch.int64),
            "remaining_fraction": torch.ones((batch, 1)),
        }
    )
    return result


def test_target_only_initializer_preserves_policy_outputs_with_zero_v3_fields() -> None:
    torch.manual_seed(101)
    source_config = ModelConfig(
        16,
        12,
        24,
        action_conditioned_targets=True,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
    )
    source = EntityPolicy(source_config).eval()
    target = HybridActorCritic(target_only_plan_ppo_config(source_config)).eval()
    report = initialize_plan_ppo_policy(target, source.state_dict())
    observation = source_observation()
    with torch.no_grad():
        expected = source(observation)
        actual = target.policy(target_observation(observation))
    for name in ("action_logits", "target_by_action", "power", "power_raw"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert report["expandedInputTensors"] == [
        "ally_encoder.0.weight",
        "enemy_encoder.0.weight",
        "projectile_encoder.0.weight",
    ]


def test_plan_ppo_architecture_validation_and_staged_unfreezing() -> None:
    source = ModelConfig(
        16,
        12,
        24,
        action_conditioned_targets=True,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
    )
    config = target_only_plan_ppo_config(source)
    assert model_config(config.as_dict()) == config
    model = HybridActorCritic(config)

    stage1 = plan_ppo_parameter_groups(model, 1)
    assert [group["name"] for group in stage1] == ["new"]
    assert all(
        parameter.requires_grad
        == name.startswith(("role_aware_critic.", "policy.plan_ppo_residual."))
        for name, parameter in model.named_parameters()
    )

    stage2 = plan_ppo_parameter_groups(model, 2)
    assert [group["name"] for group in stage2] == ["new", "heads"]
    assert stage2[1]["lr"] == 1e-5
    assert model.policy.action_head.weight.requires_grad
    assert not model.policy.ally_encoder[2].weight.requires_grad

    try:
        plan_ppo_parameter_groups(model, 3)
    except ValueError as error:
        assert "both physical and plan gates" in str(error)
    else:
        raise AssertionError("stage 3 bypassed its gates")
    stage3 = plan_ppo_parameter_groups(
        model, 3, physical_gate_passed=True, plan_gate_passed=True
    )
    assert [group["name"] for group in stage3] == [
        "new", "heads", "encoder-final"
    ]
    assert model.policy.ally_encoder[2].weight.requires_grad


def test_plan_ppo_anchor_decay_and_zero_initializer_kl() -> None:
    assert plan_ppo_anchor_weights(0, 20) == {"bc": 0.1, "initializerKl": 0.01}
    assert plan_ppo_anchor_weights(10, 20)["bc"] == 0
    assert plan_ppo_anchor_weights(15, 20)["initializerKl"] == 0
    source = ModelConfig(
        16,
        12,
        24,
        action_conditioned_targets=True,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
    )
    model = HybridActorCritic(target_only_plan_ppo_config(source))
    initializer = freeze_initializer(model)
    observation = target_observation(source_observation())
    prediction = model(observation)
    with torch.no_grad():
        initial = initializer(observation)
    torch.testing.assert_close(
        initializer_policy_kl(model, prediction, initial, observation),
        torch.zeros(()),
    )
