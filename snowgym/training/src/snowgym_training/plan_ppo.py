"""Audited target-only initialization and staged unfreezing for plan PPO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .executor import ModelConfig
from .loss import LossConfig, behavior_clone_loss
from .ppo import (
    HybridActorCritic,
    PPOConfig,
    PPORollout,
    living_unit_mask,
    ppo_loss,
    ratio_diagnostics,
)

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


def plan_ppo_anchor_weights(update_index: int, total_updates: int) -> dict[str, float]:
    """Predeclared linear BC and initializer-KL decay schedules."""
    if (
        not isinstance(update_index, int)
        or isinstance(update_index, bool)
        or not isinstance(total_updates, int)
        or isinstance(total_updates, bool)
        or total_updates <= 0
        or not 0 <= update_index < total_updates
    ):
        raise ValueError("plan PPO anchor schedule indices are invalid")
    progress = update_index / total_updates
    return {
        "bc": 0.1 * max(0.0, 1 - progress / 0.5),
        "initializerKl": 0.01 * max(0.0, 1 - progress / 0.75),
    }


def freeze_initializer(model: HybridActorCritic) -> HybridActorCritic:
    initializer = deepcopy(model).eval()
    for parameter in initializer.parameters():
        parameter.requires_grad_(False)
    return initializer


def initializer_policy_kl(
    model: HybridActorCritic,
    prediction: dict[str, Tensor],
    initializer_prediction: dict[str, Tensor],
    observation: dict[str, Tensor],
) -> Tensor:
    """KL of the masked categorical plus conditional Gaussian policy."""
    present = living_unit_mask(observation)
    current_log = F.log_softmax(prediction["action_logits"], dim=-1)
    initial_log = F.log_softmax(initializer_prediction["action_logits"], dim=-1)
    probabilities = current_log.exp()
    categorical = (probabilities * (current_log - initial_log)).sum(dim=-1)
    target_variance = model.target_log_std.exp().square().view(1, 1, 2)
    target_delta = (
        prediction["target_raw_by_action"] - initializer_prediction["target_raw_by_action"]
    ).square() / (2 * target_variance.unsqueeze(-2))
    target_kl = target_delta.sum(dim=-1)
    move_throw = (
        probabilities[..., 1] * target_kl[..., 1]
        + probabilities[..., 2] * target_kl[..., 2]
    )
    power_variance = model.power_log_std.exp().square()
    power_kl = (
        probabilities[..., 2]
        * (prediction["power_raw"] - initializer_prediction["power_raw"]).square()
        / (2 * power_variance)
    )
    per_unit = (categorical + move_throw + power_kl) * present
    return (
        per_unit.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
    ).mean()


def plan_ppo_update(
    model: HybridActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: PPORollout,
    teacher_actions: dict[str, Tensor],
    initializer: HybridActorCritic,
    config: PPOConfig,
    *,
    loss_config: LossConfig,
    training_seed: int,
    update_index: int,
    total_updates: int,
) -> dict[str, Any]:
    """Apply PPO with the frozen M7b BC and initializer-KL anchors."""
    weights = plan_ppo_anchor_weights(update_index, total_updates)
    flat = rollout.flatten()
    flattened_teacher = {
        name: value.flatten(0, 1) for name, value in teacher_actions.items()
    }
    if set(flattened_teacher) != {"action_type", "target", "power"}:
        raise ValueError("plan PPO teacher action fields are invalid")
    if any(value.shape[0] != flat["advantages"].shape[0] for value in flattened_teacher.values()):
        raise ValueError("plan PPO teacher actions do not align with rollout")
    advantages = flat["advantages"]
    normalized = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(training_seed + update_index * 1_000_003)
    sample_count = int(advantages.shape[0])
    totals = {
        "total": 0.0,
        "ppo": 0.0,
        "bc": 0.0,
        "bcAction": 0.0,
        "bcTarget": 0.0,
        "bcPower": 0.0,
        "initializerKl": 0.0,
    }
    seen = 0
    minibatches = 0
    maximum_gradient_norm = 0.0
    model.train()
    initializer.eval()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            observation = {
                name: value[indices] for name, value in flat["observations"].items()
            }
            actions = {name: value[indices] for name, value in flat["actions"].items()}
            teacher = {name: value[indices] for name, value in flattened_teacher.items()}
            prediction = model(observation)
            log_probability, entropy = model.evaluate_actions(
                observation, actions, prediction=prediction
            )
            base = ppo_loss(
                log_probability,
                flat["old_log_probability"][indices],
                normalized[indices],
                prediction["value"],
                flat["returns"][indices],
                entropy,
                config,
                normalize_advantages=False,
                active_mask=living_unit_mask(observation),
            )
            bc_losses = behavior_clone_loss(
                prediction, teacher, observation, loss_config
            )
            bc = bc_losses["total"]
            with torch.no_grad():
                initial_prediction = initializer(observation)
            initial_kl = initializer_policy_kl(
                model, prediction, initial_prediction, observation
            )
            total = base["total"] + weights["bc"] * bc + weights["initializerKl"] * initial_kl
            if not bool(torch.isfinite(total)):
                raise ValueError("non-finite plan PPO loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise ValueError("non-finite plan PPO gradient")
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("non-finite plan PPO gradient norm")
            optimizer.step()
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            count = int(indices.numel())
            for name, value in {
                "total": total,
                "ppo": base["total"],
                "bc": bc,
                "bcAction": bc_losses["action"],
                "bcTarget": bc_losses["target"],
                "bcPower": bc_losses["power"],
                "initializerKl": initial_kl,
            }.items():
                totals[name] += float(value.detach()) * count
            seen += count
            minibatches += 1
    with torch.no_grad():
        final_prediction = model(flat["observations"])
        final_log_probability, _ = model.evaluate_actions(
            flat["observations"], flat["actions"], prediction=final_prediction
        )
        diagnostics = ratio_diagnostics(
            final_log_probability,
            flat["old_log_probability"],
            flat["observations"],
        )
    return {
        "updateIndex": update_index,
        "samples": sample_count,
        "minibatches": minibatches,
        "anchorWeights": weights,
        **{name: value / seen for name, value in totals.items()},
        "targetStd": model.target_log_std.detach().exp().tolist(),
        "powerStd": float(model.power_log_std.detach().exp()),
        "maximumGradientNormBeforeClip": maximum_gradient_norm,
        **diagnostics,
    }


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
