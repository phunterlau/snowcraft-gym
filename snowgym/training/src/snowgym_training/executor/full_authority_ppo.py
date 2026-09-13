"""R1n (reviewer E3): from-scratch full-authority Engage policy.

No frozen source, no runtime assistance. The egocentric feature extractor is
exactly the relative-position transform that won R1m-S12 (`GeometryProbe`'s
`relative=True` path), copied here rather than shared, since `geometry_probe.py`
is production code other experiments audit by file digest. Latents are stored
explicitly and decoded separately from their log-probability (the convention
`movement_ppo.py`/`recovery_ppo.py` already use), so the two movement-geometry
arms differ only in `decode_move`; `evaluate_latents` is identical between them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..ppo import RoleAwareCentralCritic, living_unit_mask
from .model import ModelConfig, masked_mean_max

ARENA_HALF_EXTENT = (50., 40.)  # Frozen 100x80 Engage-derived arenas.
DESTINATIONS = ("local", "global")


def calibrated_target_log_std(destination: str, local_radius: float, *, target_world_sigma: float = 2.0):
    """Physics-derived initial log-std: a world-sigma-2 median immediate heading
    perturbation for both arms, computed from each decoder's own sensitivity at
    the origin (local: the isotropic radius; global: the anisotropic arena scale).
    Never computed from returns."""
    if destination == "local":
        return [math.log(target_world_sigma / local_radius)] * 2
    return [math.log(target_world_sigma / extent) for extent in ARENA_HALF_EXTENT]


class FullAuthorityPolicy(nn.Module):
    def __init__(self, *, destination: str, local_radius: float = 8.0,
                 initial_target_log_std: float | list[float] | None = None,
                 initial_power_log_std: float = -1.0,
                 critic_config: ModelConfig | None = None):
        super().__init__()
        if destination not in DESTINATIONS:
            raise ValueError("unknown full-authority destination geometry")
        if local_radius <= 0:
            raise ValueError("local destination radius must be positive")
        self.destination, self.local_radius = destination, float(local_radius)
        if initial_target_log_std is None:
            initial_target_log_std = calibrated_target_log_std(destination, local_radius)
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            for name, size in (("allies", 21), ("enemies", 21), ("projectiles", 9), ("obstacles", 9))
        })
        features = 21 + 4 * 32 + 38 + 40
        self.action_head = nn.Linear(features, 4)
        self.move_head = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 2))
        self.throw_head = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 2))
        self.power_head = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 1))
        target_log_std_init = torch.as_tensor(initial_target_log_std, dtype=torch.float32)
        if target_log_std_init.ndim == 0:
            target_log_std_init = target_log_std_init.expand(2)
        if target_log_std_init.shape != (2,):
            raise ValueError("initial_target_log_std must be a scalar or a 2-vector")
        self.target_log_std = nn.Parameter(target_log_std_init.clone())
        self.power_log_std = nn.Parameter(torch.full((1,), float(initial_power_log_std)))
        self.critic = RoleAwareCentralCritic(critic_config or ModelConfig(observation_version=3))

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad and not name.startswith("critic.")]

    # -- Egocentric features: GeometryProbe(relative=True)'s pair_features/features, copied. --

    def pair_features(self, observation: dict[str, Tensor], name: str) -> Tensor:
        own = observation["allies"].float()
        values = observation[name].float()[:, None].expand(-1, own.shape[1], -1, -1).clone()
        origin = own[..., None, 2:4]
        coordinate = 1 if name == "obstacles" else 2
        values[..., coordinate:coordinate + 2] -= origin
        if name in {"allies", "enemies"}:
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

    # -- Forward, decode, act, likelihood. --

    def forward(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        features = self.features(observation)
        live = living_unit_mask(observation)
        action_logits = self.action_head(features).masked_fill(~live[..., None], -1e9)
        action_logits = action_logits.masked_fill(~observation["unit_action_mask"].bool(), -1e9)
        return {"action_logits": action_logits, "living": live,
                "move_raw": self.move_head(features), "throw_raw": self.throw_head(features),
                "power_raw": self.power_head(features).squeeze(-1), "value": self.critic(observation)}

    def decode_move(self, observation: dict[str, Tensor], latent: Tensor) -> Tensor:
        own = observation["allies"][..., 2:4].float()
        if self.destination == "global":
            return torch.tanh(latent)
        # An isotropic radius: squash the latent's norm (not each axis independently,
        # which would reach R*sqrt(2) on the diagonal), then scale back to world units.
        norm = latent.norm(dim=-1, keepdim=True)
        direction = latent / norm.clamp_min(1e-6)
        world_offset = self.local_radius * torch.tanh(norm) * direction
        scale = latent.new_tensor(ARENA_HALF_EXTENT)
        return (own + world_offset / scale).clamp(-1., 1.)

    def act(self, observation: dict[str, Tensor], *, deterministic: bool = False):
        prediction = self(observation)
        move_mean, throw_mean, power_mean = prediction["move_raw"], prediction["throw_raw"], prediction["power_raw"]
        move_std, power_std = self.target_log_std.exp(), self.power_log_std.exp()
        action_type = (prediction["action_logits"].argmax(-1) if deterministic
                       else Categorical(logits=prediction["action_logits"]).sample())
        move_latent = move_mean if deterministic else Normal(move_mean, move_std).sample()
        throw_latent = throw_mean if deterministic else Normal(throw_mean, move_std).sample()
        power_latent = power_mean if deterministic else Normal(power_mean, power_std).sample()
        moves, throws = action_type == ACTION_MOVE, action_type == ACTION_THROW
        move_latent = torch.where(moves[..., None], move_latent, torch.zeros_like(move_latent))
        throw_latent = torch.where(throws[..., None], throw_latent, torch.zeros_like(throw_latent))
        power_latent = torch.where(throws, power_latent, torch.zeros_like(power_latent))
        move_target = self.decode_move(observation, move_latent)
        throw_target = torch.tanh(throw_latent)  # Throw destination stays global for both arms.
        target = torch.where(moves[..., None], move_target, torch.where(throws[..., None], throw_target,
            torch.zeros_like(move_target)))
        action = {"action_type": action_type, "target": target, "power": torch.sigmoid(power_latent)}
        latent = {"move": move_latent, "throw": throw_latent, "power": power_latent}
        logp, _ = self.evaluate_latents(observation, action_type, latent, prediction=prediction)
        return action, latent, logp, prediction["value"]

    def evaluate_latents(self, observation, action_type, latent, *, prediction=None):
        prediction = prediction if prediction is not None else self(observation)
        live = prediction["living"]
        moves, throws = live & (action_type == ACTION_MOVE), live & (action_type == ACTION_THROW)
        move_std, power_std = self.target_log_std.exp(), self.power_log_std.exp()
        move_normal = Normal(prediction["move_raw"], move_std)
        throw_normal = Normal(prediction["throw_raw"], move_std)
        power_normal = Normal(prediction["power_raw"], power_std)
        categorical = Categorical(logits=prediction["action_logits"])
        type_logp, type_entropy = categorical.log_prob(action_type) * live, categorical.entropy() * live
        move_logp = move_normal.log_prob(latent["move"]).sum(-1) * moves
        throw_logp = throw_normal.log_prob(latent["throw"]).sum(-1) * throws
        power_logp = power_normal.log_prob(latent["power"]) * throws
        per_unit_logp = type_logp + move_logp + throw_logp + power_logp
        probabilities = categorical.probs
        move_probability, throw_probability = probabilities[..., ACTION_MOVE], probabilities[..., ACTION_THROW]
        per_unit_entropy = (type_entropy
            + move_probability * move_normal.entropy().sum(-1) * live
            + throw_probability * (throw_normal.entropy().sum(-1) + power_normal.entropy()) * live)
        return per_unit_logp, {"entropy": per_unit_entropy, **prediction}
