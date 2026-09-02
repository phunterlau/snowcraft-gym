"""Centralized masked hybrid-action PPO building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from .model import EntityPolicy, ModelConfig

EPSILON = 1e-6


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PPORollout:
    observations: dict[str, Tensor]
    actions: dict[str, Tensor]
    old_log_probability: Tensor
    old_values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    next_values: Tensor
    advantages: Tensor
    returns: Tensor

    @property
    def steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.rewards.shape[1])

    def flatten(self) -> dict[str, Any]:
        """Flatten time and world dimensions without changing feature axes."""
        return {
            "observations": {
                name: value.flatten(0, 1) for name, value in self.observations.items()
            },
            "actions": {
                name: value.flatten(0, 1) for name, value in self.actions.items()
            },
            "old_log_probability": self.old_log_probability.flatten(0, 1),
            "old_values": self.old_values.flatten(0, 1),
            "rewards": self.rewards.flatten(0, 1),
            "terminated": self.terminated.flatten(0, 1),
            "truncated": self.truncated.flatten(0, 1),
            "next_values": self.next_values.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
        }


class RolloutBuffer:
    """Detached fixed-horizon storage for persistent vector SnowGym worlds."""

    def __init__(self, steps: int, batch_size: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("rollout steps must be a positive integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("rollout batch_size must be a positive integer")
        self.steps = steps
        self.batch_size = batch_size
        self._transitions: list[dict[str, Any]] = []
        self._observation_keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def full(self) -> bool:
        return len(self) == self.steps

    def add(
        self,
        *,
        observation: dict[str, Tensor],
        action: dict[str, Tensor],
        log_probability: Tensor,
        value: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        next_value: Tensor,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is full")
        keys = tuple(sorted(observation))
        if self._observation_keys is None:
            self._observation_keys = keys
        elif keys != self._observation_keys:
            raise ValueError("rollout observation keys changed")
        if set(action) != {"action_type", "target", "power"}:
            raise ValueError("rollout action must contain action_type, target, and power")
        for name, tensor in observation.items():
            self._batch_tensor(tensor, f"observation.{name}")
        unit_shape = action["action_type"].shape
        self._batch_tensor(action["action_type"], "action.action_type")
        self._batch_tensor(action["target"], "action.target")
        self._batch_tensor(action["power"], "action.power")
        if len(unit_shape) != 2:
            raise ValueError("action.action_type must have shape [batch, units]")
        if action["target"].shape != (*unit_shape, 2):
            raise ValueError("action.target must have shape [batch, units, 2]")
        if action["power"].shape != unit_shape:
            raise ValueError("action.power must have shape [batch, units]")
        scalars = {
            "log_probability": log_probability,
            "value": value,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_value": next_value,
        }
        for name, tensor in scalars.items():
            if tensor.shape != (self.batch_size,):
                raise ValueError(f"{name} must have shape [batch]")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite")
        self._transitions.append(
            {
                "observation": snapshot(observation),
                "action": snapshot(action),
                **{name: tensor.detach().clone() for name, tensor in scalars.items()},
            }
        )

    def finish(self, config: PPOConfig) -> PPORollout:
        if not self.full:
            raise RuntimeError(
                f"rollout buffer is incomplete: expected {self.steps}, got {len(self)}"
            )
        observations = stack_dict(self._transitions, "observation")
        actions = stack_dict(self._transitions, "action")
        values = stack_field(self._transitions, "value")
        rewards = stack_field(self._transitions, "reward")
        terminated = stack_field(self._transitions, "terminated").bool()
        truncated = stack_field(self._transitions, "truncated").bool()
        next_values = stack_field(self._transitions, "next_value")
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        return PPORollout(
            observations=observations,
            actions=actions,
            old_log_probability=stack_field(self._transitions, "log_probability"),
            old_values=values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            next_values=next_values,
            advantages=advantages,
            returns=returns,
        )

    def _batch_tensor(self, tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor) or tensor.ndim == 0:
            raise ValueError(f"{name} must be a batched tensor")
        if tensor.shape[0] != self.batch_size:
            raise ValueError(f"{name} leading dimension must equal batch_size")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")


def snapshot(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def stack_dict(transitions: list[dict[str, Any]], field: str) -> dict[str, Tensor]:
    first = transitions[0][field]
    return {
        name: torch.stack([transition[field][name] for transition in transitions])
        for name in first
    }


def stack_field(transitions: list[dict[str, Any]], field: str) -> Tensor:
    return torch.stack([transition[field] for transition in transitions])


class HybridActorCritic(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.policy = EntityPolicy(model_config)
        self.target_log_std = nn.Parameter(torch.full((2,), -1.0))
        self.power_log_std = nn.Parameter(torch.full((1,), -1.0))
        self.value_head = nn.Linear(model_config.actor_hidden, 1)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        prediction = self.policy(observation)
        ally_mask = observation["ally_mask"].bool()
        hidden = prediction["hidden"]
        weights = ally_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {**prediction, "value": self.value_head(pooled).squeeze(-1)}

    def act(
        self, observation: dict[str, Tensor], *, deterministic: bool = False
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        prediction = self(observation)
        categorical = Categorical(logits=prediction["action_logits"])
        action_type = (
            prediction["action_logits"].argmax(dim=-1)
            if deterministic
            else categorical.sample()
        )
        target_normal = Normal(
            prediction["target_raw"], self.target_log_std.exp().view(1, 1, 2)
        )
        power_normal = Normal(
            prediction["power_raw"], self.power_log_std.exp().view(1, 1)
        )
        target_raw = prediction["target_raw"] if deterministic else target_normal.sample()
        power_raw = prediction["power_raw"] if deterministic else power_normal.sample()
        action = {
            "action_type": action_type,
            "target": torch.tanh(target_raw),
            "power": torch.sigmoid(power_raw),
        }
        log_probability, entropy = self.evaluate_actions(
            observation, action, prediction=prediction
        )
        return action, log_probability, prediction["value"]

    def evaluate_actions(
        self,
        observation: dict[str, Tensor],
        action: dict[str, Tensor],
        *,
        prediction: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = prediction if prediction is not None else self(observation)
        present = observation["ally_mask"].bool()
        action_type = action["action_type"].long()
        categorical = Categorical(logits=output["action_logits"])
        type_log_prob = categorical.log_prob(action_type) * present
        type_entropy = categorical.entropy() * present

        target_value = action["target"].clamp(-1 + EPSILON, 1 - EPSILON)
        target_raw = torch.atanh(target_value)
        target_normal = Normal(
            output["target_raw"], self.target_log_std.exp().view(1, 1, 2)
        )
        target_log_prob = target_normal.log_prob(target_raw) - torch.log(
            1 - target_value.square() + EPSILON
        )
        target_log_prob = target_log_prob.sum(dim=-1)
        target_entropy = target_normal.entropy().sum(dim=-1)
        target_mask = present & (
            (action_type == ACTION_MOVE) | (action_type == ACTION_THROW)
        )

        power_value = action["power"].clamp(EPSILON, 1 - EPSILON)
        power_raw = torch.logit(power_value)
        power_normal = Normal(
            output["power_raw"], self.power_log_std.exp().view(1, 1)
        )
        power_log_prob = power_normal.log_prob(power_raw) - torch.log(
            power_value * (1 - power_value) + EPSILON
        )
        throw_mask = present & (action_type == ACTION_THROW)
        joint_log_prob = (
            type_log_prob
            + target_log_prob * target_mask
            + power_log_prob * throw_mask
        ).sum(dim=-1)
        entropy = (
            type_entropy
            + target_entropy * target_mask
            + power_normal.entropy() * throw_mask
        ).sum(dim=-1)
        return joint_log_prob, entropy


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    if not (
        rewards.shape
        == values.shape
        == next_values.shape
        == terminated.shape
        == truncated.shape
    ):
        raise ValueError("GAE tensors must have identical [time, batch] shapes")
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        bootstrap = (~terminated[step].bool()).to(rewards.dtype)
        continuation = (~(terminated[step].bool() | truncated[step].bool())).to(
            rewards.dtype
        )
        delta = rewards[step] + gamma * next_values[step] * bootstrap - values[step]
        following = delta + gamma * gae_lambda * continuation * following
        advantages[step] = following
    return advantages, advantages + values


def ppo_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantages: Tensor,
    new_values: Tensor,
    returns: Tensor,
    entropy: Tensor,
    config: PPOConfig,
) -> dict[str, Tensor]:
    normalized = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-8)
    ratio = (new_log_probability - old_log_probability).exp()
    policy = -torch.minimum(
        ratio * normalized,
        ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * normalized,
    ).mean()
    value = 0.5 * (new_values - returns).square().mean()
    entropy_loss = entropy.mean()
    total = policy + config.value_weight * value - config.entropy_weight * entropy_loss
    approximate_kl = (old_log_probability - new_log_probability).mean()
    clip_fraction = ((ratio - 1).abs() > config.clip_ratio).float().mean()
    return {
        "total": total,
        "policy": policy,
        "value": value,
        "entropy": entropy_loss,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
    }


def health_potential(observation: dict[str, Tensor]) -> Tensor:
    ally_health = observation["allies"][..., 6]
    enemy_health = observation["enemies"][..., 6]
    ally = (ally_health * observation["ally_mask"].to(ally_health.dtype)).sum(dim=-1)
    enemy = (enemy_health * observation["enemy_mask"].to(enemy_health.dtype)).sum(
        dim=-1
    )
    return ally - enemy


def potential_shaped_reward(
    canonical_reward: Tensor,
    potential: Tensor,
    next_potential: Tensor,
    terminated: Tensor,
    *,
    gamma: float,
) -> Tensor:
    terminal_next = torch.where(
        terminated.bool(), torch.zeros_like(next_potential), next_potential
    )
    return canonical_reward + gamma * terminal_next - potential
