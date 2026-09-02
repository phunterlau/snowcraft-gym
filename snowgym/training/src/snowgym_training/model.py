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

    def as_dict(self) -> dict[str, int | bool]:
        value: dict[str, int | bool] = {
            "entity_hidden": self.entity_hidden,
            "entity_embedding": self.entity_embedding,
            "actor_hidden": self.actor_hidden,
        }
        if self.pairwise_enemy_attention:
            value["pairwise_enemy_attention"] = True
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
        self.actor = nn.Sequential(
            nn.Linear(embedding + global_features, config.actor_hidden),
            nn.ReLU(),
            nn.Linear(config.actor_hidden, config.actor_hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(config.actor_hidden, ACTION_TYPE_COUNT)
        self.target_head = nn.Linear(config.actor_hidden, 2)
        self.power_head = nn.Linear(config.actor_hidden, 1)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden, action_mask, _ = self.features(observation)
        target_raw = self.target_head(hidden)
        power_raw = self.power_head(hidden).squeeze(-1)
        logits = self.action_head(hidden).masked_fill(
            ~action_mask, torch.finfo(hidden.dtype).min
        )
        return {
            "action_logits": logits,
            "target": torch.tanh(target_raw),
            "power": torch.sigmoid(power_raw),
            "target_raw": target_raw,
            "power_raw": power_raw,
            "hidden": hidden,
        }

    def features(
        self, observation: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        allies = self.ally_encoder(observation["allies"].float())
        enemies = self.enemy_encoder(observation["enemies"].float())
        projectiles = self.projectile_encoder(observation["projectiles"].float())
        obstacles = self.obstacle_encoder(observation["obstacles"].float())
        ally_mask = observation["ally_mask"].bool()
        actor_inputs = [allies]
        if self.pairwise_enemy_attention:
            actor_inputs.append(
                masked_enemy_attention(
                    self.enemy_query(allies),
                    self.enemy_key(enemies),
                    self.enemy_value(enemies),
                    observation["enemy_mask"].bool(),
                )
            )
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


def masked_enemy_attention(
    query: Tensor, key: Tensor, value: Tensor, enemy_mask: Tensor
) -> Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(
        ~enemy_mask[:, None, :], torch.finfo(scores.dtype).min
    )
    weights = torch.softmax(scores, dim=-1) * enemy_mask[:, None, :].to(scores.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.matmul(weights, value)


def model_config(value: Any) -> ModelConfig:
    required = {
        "entity_hidden",
        "entity_embedding",
        "actor_hidden",
    }
    optional = {"pairwise_enemy_attention"}
    if not isinstance(value, dict) or set(value) - optional != required:
        raise ValueError(
            "architecture must define entity_hidden, entity_embedding, actor_hidden "
            "and may enable pairwise_enemy_attention"
        )
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for name, item in value.items()
        if name in required
    ):
        raise ValueError("architecture dimensions must be positive integers")
    attention = value.get("pairwise_enemy_attention", False)
    if not isinstance(attention, bool):
        raise ValueError("architecture pairwise_enemy_attention must be boolean")
    return ModelConfig(**value)
