"""Small masked entity policy for behavior cloning and later RL warm starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from snowgym_client.encoding import ACTION_TYPE_COUNT


@dataclass(frozen=True)
class ModelConfig:
    entity_hidden: int = 32
    entity_embedding: int = 24
    actor_hidden: int = 64
    pairwise_enemy_attention: bool = False
    action_conditioned_targets: bool = False
    last_enemy_move_target: bool = False
    nearest_enemy_throw_target: bool = False
    nearest_enemy_features: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        value: dict[str, int | bool] = {
            "entity_hidden": self.entity_hidden,
            "entity_embedding": self.entity_embedding,
            "actor_hidden": self.actor_hidden,
        }
        if self.pairwise_enemy_attention:
            value["pairwise_enemy_attention"] = True
        if self.action_conditioned_targets:
            value["action_conditioned_targets"] = True
        if self.last_enemy_move_target:
            value["last_enemy_move_target"] = True
        if self.nearest_enemy_throw_target:
            value["nearest_enemy_throw_target"] = True
        if self.nearest_enemy_features:
            value["nearest_enemy_features"] = True
        return value


class EntityPolicy(nn.Module):
    """Permutation-aware global context with a shared per-ally actor."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        embedding = config.entity_embedding
        self.ally_encoder = entity_encoder(10, config)
        self.enemy_encoder = entity_encoder(10, config)
        self.projectile_encoder = entity_encoder(8, config)
        self.obstacle_encoder = entity_encoder(9, config)
        global_features = embedding * 8 + 3
        self.pairwise_enemy_attention = config.pairwise_enemy_attention
        if self.pairwise_enemy_attention:
            self.enemy_query = nn.Linear(embedding, embedding, bias=False)
            self.enemy_key = nn.Linear(embedding, embedding, bias=False)
            self.enemy_value = nn.Linear(embedding, embedding, bias=False)
            global_features += embedding
        self.nearest_enemy_features = config.nearest_enemy_features
        if self.nearest_enemy_features:
            global_features += 5
        self.actor = nn.Sequential(
            nn.Linear(embedding + global_features, config.actor_hidden),
            nn.ReLU(),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(config.actor_hidden, ACTION_TYPE_COUNT)
        self.action_conditioned_targets = config.action_conditioned_targets
        if self.action_conditioned_targets:
            self.last_enemy_move_target = config.last_enemy_move_target
            self.move_target_head = nn.Linear(config.actor_hidden, 2)
            self.nearest_enemy_throw_target = config.nearest_enemy_throw_target
            if not self.nearest_enemy_throw_target:
                self.throw_target_head = nn.Linear(config.actor_hidden, 2)
        else:
            self.target_head = nn.Linear(config.actor_hidden, 2)
        self.power_head = nn.Linear(config.actor_hidden, 1)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden, action_mask, _ = self.features(observation)
        power_raw = self.power_head(hidden).squeeze(-1)
        logits = self.action_head(hidden).masked_fill(
            ~action_mask, torch.finfo(hidden.dtype).min
        )
        result = {
            "action_logits": logits,
            "power": torch.sigmoid(power_raw),
            "power_raw": power_raw,
            "hidden": hidden,
        }
        if self.action_conditioned_targets:
            zeros = torch.zeros(
                (*hidden.shape[:-1], 2), dtype=hidden.dtype, device=hidden.device
            )
            enemy_mask = living_enemy_mask(observation)
            nearest_target = nearest_enemy_target(
                observation["allies"][..., 2:4].float(),
                observation["enemies"][..., 2:4].float(),
                enemy_mask,
            )
            predicted_move_target = self.move_target_head(hidden)
            move_target = (
                torch.where(
                    (observation["team_alive"][:, 1] == 1)[:, None, None],
                    torch.atanh(nearest_target.clamp(-1 + 1e-6, 1 - 1e-6)),
                    predicted_move_target,
                )
                if self.last_enemy_move_target
                else predicted_move_target
            )
            throw_target = (
                torch.atanh(
                    nearest_target.clamp(-1 + 1e-6, 1 - 1e-6)
                )
                if self.nearest_enemy_throw_target
                else self.throw_target_head(hidden)
            )
            target_raw_by_action = torch.stack(
                [
                    zeros,
                    move_target,
                    throw_target,
                    zeros,
                ],
                dim=-2,
            )
            supervised_target_raw_by_action = torch.stack(
                [zeros, predicted_move_target, throw_target, zeros], dim=-2
            )
            selected = logits.argmax(dim=-1)
            target_raw = select_action_target(target_raw_by_action, selected)
            result.update(
                {
                    "target": torch.tanh(target_raw),
                    "target_raw": target_raw,
                    "target_by_action": torch.tanh(target_raw_by_action),
                    "target_raw_by_action": target_raw_by_action,
                    "supervised_target_by_action": torch.tanh(
                        supervised_target_raw_by_action
                    ),
                }
            )
        else:
            target_raw = self.target_head(hidden)
            result.update({"target": torch.tanh(target_raw), "target_raw": target_raw})
        return result

    def features(
        self, observation: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        allies = self.ally_encoder(observation["allies"].float())
        enemies = self.enemy_encoder(observation["enemies"].float())
        projectiles = self.projectile_encoder(observation["projectiles"].float())
        obstacles = self.obstacle_encoder(observation["obstacles"].float())
        ally_mask = observation["ally_mask"].bool()
        actor_inputs = [allies]
        enemy_mask = living_enemy_mask(observation)
        if self.pairwise_enemy_attention:
            actor_inputs.append(
                masked_enemy_attention(
                    self.enemy_query(allies),
                    self.enemy_key(enemies),
                    self.enemy_value(enemies),
                    enemy_mask,
                    observation["allies"][..., 2:4].float(),
                    observation["enemies"][..., 2:4].float(),
                )
            )
        if self.nearest_enemy_features:
            ally_position = observation["allies"][..., 2:4].float()
            nearest = nearest_enemy_target(
                ally_position,
                observation["enemies"][..., 2:4].float(),
                enemy_mask,
            )
            relative = nearest - ally_position
            relational = torch.cat(
                [nearest, relative, relative.square().sum(dim=-1, keepdim=True).sqrt()],
                dim=-1,
            )
            relational = relational * enemy_mask.any(dim=-1)[
                :, None, None
            ].to(relational.dtype)
            actor_inputs.append(relational)
        global_context = torch.cat(
            [
                *masked_mean_max(allies, ally_mask),
                *masked_mean_max(enemies, observation["enemy_mask"].bool()),
                *masked_mean_max(
                    projectiles, observation["projectile_mask"].bool()
                ),
                *masked_mean_max(obstacles, observation["obstacle_mask"].bool()),
                observation["team_alive"].float()
                / max(int(observation["allies"].shape[1]), 1),
                torch.log1p(observation["tick"].float()) / 10.0,
            ],
            dim=-1,
        )
        expanded = global_context[:, None, :].expand(-1, allies.shape[1], -1)
        hidden = self.actor(torch.cat([*actor_inputs, expanded], dim=-1))
        action_mask = observation["unit_action_mask"].bool().clone()
        action_mask[..., 0] |= ~ally_mask
        return hidden, action_mask, ally_mask


def entity_encoder(features: int, config: ModelConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(features, config.entity_hidden),
        nn.ReLU(),
        nn.Linear(config.entity_hidden, config.entity_embedding),
        nn.ReLU(),
    )


def masked_mean_max(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(values.dtype)
    mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    lowest = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~mask.unsqueeze(-1), lowest).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def living_enemy_mask(observation: dict[str, Tensor]) -> Tensor:
    """Exclude defeated roster slots from relational target selection."""
    return observation["enemy_mask"].bool() & (
        observation["enemies"][..., 1].float() > 0.5
    )


def masked_enemy_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    enemy_mask: Tensor,
    ally_position: Tensor,
    enemy_position: Tensor,
) -> Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    distance_squared = (
        ally_position[:, :, None, :] - enemy_position[:, None, :, :]
    ).square().sum(dim=-1)
    scores = scores - 8.0 * distance_squared
    scores = scores.masked_fill(
        ~enemy_mask[:, None, :], torch.finfo(scores.dtype).min
    )
    weights = torch.softmax(scores, dim=-1) * enemy_mask[:, None, :].to(scores.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.matmul(weights, value)


def select_action_target(targets: Tensor, action_type: Tensor) -> Tensor:
    index = action_type.long().unsqueeze(-1).unsqueeze(-1).expand(*action_type.shape, 1, 2)
    return targets.gather(-2, index).squeeze(-2)


def nearest_enemy_target(
    ally_position: Tensor, enemy_position: Tensor, enemy_mask: Tensor
) -> Tensor:
    distance = (
        ally_position[:, :, None, :] - enemy_position[:, None, :, :]
    ).square().sum(dim=-1)
    distance = distance.masked_fill(~enemy_mask[:, None, :], torch.inf)
    nearest = distance.argmin(dim=-1)
    index = nearest.unsqueeze(-1).unsqueeze(-1).expand(*nearest.shape, 1, 2)
    selected = enemy_position[:, None, :, :].expand(
        -1, ally_position.shape[1], -1, -1
    ).gather(-2, index).squeeze(-2)
    return torch.where(
        enemy_mask.any(dim=-1)[:, None, None], selected, torch.zeros_like(selected)
    )


def model_config(value: Any) -> ModelConfig:
    required = {
        "entity_hidden",
        "entity_embedding",
        "actor_hidden",
    }
    optional = {
        "pairwise_enemy_attention",
        "action_conditioned_targets",
        "last_enemy_move_target",
        "nearest_enemy_throw_target",
        "nearest_enemy_features",
    }
    if not isinstance(value, dict) or set(value) - optional != required:
        raise ValueError(
            "architecture must define entity_hidden, entity_embedding, actor_hidden "
            "and may enable relational or action-conditioned target features"
        )
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for name, item in value.items()
        if name in required
    ):
        raise ValueError("architecture dimensions must be positive integers")
    for name in optional:
        if not isinstance(value.get(name, False), bool):
            raise ValueError(f"architecture {name} must be boolean")
    target_priors = (
        value.get("last_enemy_move_target", False),
        value.get("nearest_enemy_throw_target", False),
    )
    if any(target_priors) and not value.get("action_conditioned_targets", False):
        raise ValueError(
            "architecture nearest-enemy target prior requires action_conditioned_targets"
        )
    return ModelConfig(**value)
