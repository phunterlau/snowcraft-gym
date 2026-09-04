"""Deterministic-only, matched absolute/relative feature probe for R1i.

This experiment keeps the existing action decoder and has no PPO interface.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..ppo import HybridActorCritic, living_unit_mask
from .model import masked_mean_max, select_action_target


class GeometryProbe(nn.Module):
    def __init__(self, source: HybridActorCritic, *, relative: bool):
        super().__init__()
        self.source = source.requires_grad_(False).eval()
        self.relative = relative
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            for name, size in (("allies", 21), ("enemies", 21), ("projectiles", 9), ("obstacles", 9))
        })
        # Own v3 state + four mean/max pools + own directive + own/support role state.
        features = 21 + 4 * 32 + 38 + 40
        self.move = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 2))
        self.shot = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 3))
        for head in (self.move, self.shot):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def pair_features(self, observation: dict[str, Tensor], name: str) -> Tensor:
        own = observation["allies"].float()
        values = observation[name].float()[:, None].expand(-1, own.shape[1], -1, -1).clone()
        if self.relative:
            origin = own[..., None, 2:4]
            coordinate = 1 if name == "obstacles" else 2
            values[..., coordinate:coordinate+2] -= origin
            if name in {"allies", "enemies"}:
                # Optional controller targets remain zero when absent.
                present = values[..., 10:11]
                values[..., 11:13] = (values[..., 11:13] - origin) * present
                values[..., 13:15] = (values[..., 13:15] - origin) * present
        return values

    def features(self, observation: dict[str, Tensor]) -> Tensor:
        own = observation["allies"].float()
        batch, units = own.shape[:2]
        pools = []
        for name in self.encoders:
            values = self.pair_features(observation, name)
            mask_name = {"allies": "ally_mask", "enemies": "enemy_mask",
                         "projectiles": "projectile_mask", "obstacles": "obstacle_mask"}[name]
            mask = observation[mask_name].bool()
            if name in {"allies", "enemies"}:
                mask = mask & (observation[name][..., 1] > .5)
            encoded = self.encoders[name](values)
            mask = mask[:, None].expand(-1, units, -1).reshape(batch * units, -1)
            pooled = masked_mean_max(encoded.reshape(batch * units, -1, 16), mask)
            pools.extend(value.reshape(batch, units, 16) for value in pooled)
        roles = observation["plan_unit_roles"].float() * observation["plan_group_mask"][:, None].float()
        groups = observation["plan_groups"].float() * observation["plan_group_mask"][..., None].float()
        directives = torch.einsum("bur,brf->buf", roles, groups)
        state = observation["plan_role_state"].float() * observation["plan_group_mask"][..., None].float()
        role = torch.cat([torch.einsum("bur,brf->buf", roles, state),
                          torch.einsum("bur,brf->buf", directives[..., 34:37], state)], -1)
        return torch.cat([own, *pools, directives, role], -1)

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            inherited = self.source(observation)
        features = self.features(observation)
        live = living_unit_mask(observation)[..., None]
        move, shot = self.move(features) * live, self.shot(features) * live
        base = inherited["target_raw_by_action"]
        raw = torch.stack([base[..., 0, :], base[..., 1, :] + move,
                           base[..., 2, :] + shot[..., :2], base[..., 3, :]], -2)
        power_raw = inherited["power_raw"] + shot[..., 2]
        selected = inherited["action_logits"].argmax(-1)
        return {**inherited, "target_raw_by_action": raw, "target_by_action": torch.tanh(raw),
                "supervised_target_by_action": torch.tanh(raw),
                "target_raw": select_action_target(raw, selected),
                "target": torch.tanh(select_action_target(raw, selected)),
                "power_raw": power_raw, "power": torch.sigmoid(power_raw)}

    def act(self, observation: dict[str, Tensor], *, deterministic: bool = False):
        if not deterministic:
            raise ValueError("geometry probe is deterministic-only; no sampled/PPO likelihood contract")
        output = self(observation)
        action = {"action_type": output["action_logits"].argmax(-1),
                  "target": output["target"], "power": output["power"]}
        return action, None, output["value"]


def geometry_loss(prediction: dict[str, Tensor], teacher: dict[str, Tensor], observation: dict[str, Tensor]) -> dict[str, Tensor]:
    alive = living_unit_mask(observation)
    moves = alive & (teacher["action_type"] == 1)
    throws = alive & (teacher["action_type"] == 2)
    targets = prediction["target_by_action"]
    def mse(actual, expected, mask):
        return (actual[mask] - expected[mask]).square().mean() if mask.any() else actual.sum() * 0
    move = mse(targets[..., 1, :], teacher["target"], moves)
    endpoint = mse(targets[..., 2, :], teacher["target"], throws)
    power = mse(prediction["power"], teacher["power"], throws)
    # Frozen 100 x 80 arena. Compute direction in world units, not anisotropic normalized space.
    scale = targets.new_tensor([50., 40.])
    actual_ray = (targets[..., 2, :] - observation["allies"][..., 2:4]) * scale
    expected_ray = (teacher["target"] - observation["allies"][..., 2:4]) * scale
    valid = throws & (expected_ray.norm(dim=-1) > 1e-6)
    if valid.any():
        cosine = (F.normalize(actual_ray[valid], dim=-1, eps=1e-6)
                  * F.normalize(expected_ray[valid], dim=-1, eps=1e-6)).sum(-1)
        direction = (1 - cosine.clamp(-1, 1)).mean()
    else:
        direction = actual_ray.sum() * 0
    return {"total": move + direction + .1 * endpoint + .5 * power,
            "move": move, "direction": direction, "endpoint": endpoint, "power": power}
