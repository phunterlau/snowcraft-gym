"""Small masked entity policy for behavior cloning and later RL warm starts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from snowgym_client.encoding import ACTION_TYPE_COUNT


@dataclass(frozen=True)
class ModelConfig:
    entity_hidden: int = 32
    entity_embedding: int = 24
    actor_hidden: int = 64

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


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
        allies = self.ally_encoder(observation["allies"].float())
        enemies = self.enemy_encoder(observation["enemies"].float())
        projectiles = self.projectile_encoder(observation["projectiles"].float())
        obstacles = self.obstacle_encoder(observation["obstacles"].float())
        ally_mask = observation["ally_mask"].bool()
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
        hidden = self.actor(torch.cat([allies, expanded], dim=-1))
        logits = self.action_head(hidden)
        action_mask = observation["unit_action_mask"].bool().clone()
        action_mask[..., 0] |= ~ally_mask
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return {
            "action_logits": logits,
            "target": torch.tanh(self.target_head(hidden)),
            "power": torch.sigmoid(self.power_head(hidden).squeeze(-1)),
        }


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


def model_config(value: Any) -> ModelConfig:
    if not isinstance(value, dict) or set(value) != {
        "entity_hidden",
        "entity_embedding",
        "actor_hidden",
    }:
        raise ValueError("architecture must define entity_hidden, entity_embedding, actor_hidden")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in value.values()
    ):
        raise ValueError("architecture dimensions must be positive integers")
    return ModelConfig(**value)
