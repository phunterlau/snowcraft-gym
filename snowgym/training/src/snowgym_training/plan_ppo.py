"""Audited target-only initialization and staged unfreezing for plan PPO."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .executor import ModelConfig
from .ppo import HybridActorCritic

EXPANDABLE_V3_INPUTS = {
    "ally_encoder.0.weight",
    "enemy_encoder.0.weight",
    "projectile_encoder.0.weight",
}
INHERITED_HEAD_PREFIXES = (
    "policy.action_head.",
    "policy.move_target_head.",
    "policy.throw_target_head.",
    "policy.power_head.",
)
FINAL_ENTITY_LAYER_PREFIXES = (
    "policy.ally_encoder.2.",
    "policy.enemy_encoder.2.",
    "policy.projectile_encoder.2.",
    "policy.obstacle_encoder.2.",
)


def target_only_plan_ppo_config(source: ModelConfig) -> ModelConfig:
    """Construct the M7a plan-PPO architecture from an accepted target-only model."""
    if not (
        source.plan_conditioned
        and source.plan_target_only
        and source.separate_target_actor
        and source.action_conditioned_targets
        and not source.nearest_enemy_throw_target
    ):
        raise ValueError("plan PPO initializer must use a learned-head target-only policy")
    return ModelConfig(
        entity_hidden=source.entity_hidden,
        entity_embedding=source.entity_embedding,
        actor_hidden=source.actor_hidden,
        pairwise_enemy_attention=source.pairwise_enemy_attention,
        action_conditioned_targets=True,
        last_enemy_move_target=source.last_enemy_move_target,
        nearest_enemy_throw_target=False,
        nearest_enemy_features=source.nearest_enemy_features,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
        observation_version=3,
        plan_ppo_residuals=True,
    )


def initialize_plan_ppo_policy(
    model: HybridActorCritic, source_state: dict[str, Tensor]
) -> dict[str, Any]:
    """Copy every inherited policy tensor and zero-expand new v3 input columns."""
    if not model.policy.config.plan_ppo_residuals:
        raise ValueError("target model does not enable plan PPO residuals")
    target_state = model.policy.state_dict()
    inherited = set(target_state) - {
        name for name in target_state if name.startswith("plan_ppo_residual.")
    }
    if set(source_state) != inherited:
        raise ValueError(
            "initializer fields differ from inherited plan PPO policy: "
            f"missing={sorted(inherited - set(source_state))} "
            f"unexpected={sorted(set(source_state) - inherited)}"
        )
    loaded = dict(target_state)
    expanded: list[str] = []
    for name in sorted(inherited):
        source = source_state[name]
        target = target_state[name]
        if source.shape == target.shape:
            loaded[name] = source.detach().clone()
            continue
        if (
            name not in EXPANDABLE_V3_INPUTS
            or source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != target.shape[0]
            or source.shape[1] >= target.shape[1]
        ):
            raise ValueError(
                f"initializer tensor {name} has shape {tuple(source.shape)}, "
                f"expected {tuple(target.shape)}"
            )
        value = torch.zeros_like(target)
        value[:, : source.shape[1]] = source
        loaded[name] = value
        expanded.append(name)
    model.policy.load_state_dict(loaded)
    residual = {
        name: value
        for name, value in model.policy.state_dict().items()
        if name.startswith("plan_ppo_residual.")
    }
    final_names = [name for name in residual if name.endswith("2.weight") or name.endswith("2.bias")]
    if not final_names or any(bool(residual[name].count_nonzero()) for name in final_names):
        raise RuntimeError("plan PPO residual output layer is not zero initialized")
    return {
        "copiedTensors": len(inherited),
        "expandedInputTensors": expanded,
        "newResidualTensors": sorted(residual),
    }


def plan_ppo_parameter_groups(
    model: HybridActorCritic,
    stage: int,
    *,
    new_module_learning_rate: float = 1e-4,
    physical_gate_passed: bool = False,
    plan_gate_passed: bool = False,
) -> list[dict[str, Any]]:
    """Freeze parameters according to the three predeclared M7a stages."""
    if stage not in (1, 2, 3):
        raise ValueError("plan PPO unfreezing stage must be 1, 2, or 3")
    if new_module_learning_rate < 1e-4:
        raise ValueError("new-module learning rate must be at least 1e-4")
    if stage == 3 and not (physical_gate_passed and plan_gate_passed):
        raise ValueError("stage 3 requires both physical and plan gates")
    inherited_learning_rate = new_module_learning_rate / 10
    groups: dict[str, list[Tensor]] = {"new": [], "heads": [], "encoder-final": []}
    for name, parameter in model.named_parameters():
        group = None
        if name.startswith(("role_aware_critic.", "policy.plan_ppo_residual.")):
            group = "new"
        elif stage >= 2 and name.startswith(INHERITED_HEAD_PREFIXES):
            group = "heads"
        elif stage >= 3 and name.startswith(FINAL_ENTITY_LAYER_PREFIXES):
            group = "encoder-final"
        parameter.requires_grad_(group is not None)
        if group is not None:
            groups[group].append(parameter)
    if not groups["new"]:
        raise ValueError("model has no role-aware critic and plan PPO residual path")
    result = [
        {
            "name": "new",
            "params": groups["new"],
            "lr": new_module_learning_rate,
        }
    ]
    for name in ("heads", "encoder-final"):
        if groups[name]:
            result.append(
                {
                    "name": name,
                    "params": groups[name],
                    "lr": inherited_learning_rate,
                }
            )
    return result
