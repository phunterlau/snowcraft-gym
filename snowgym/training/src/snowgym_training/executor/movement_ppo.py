"""Scoped, teacher-assisted movement policy with exact latent action density."""

from __future__ import annotations

import torch
import math
from torch import nn
from torch.distributions import Normal

from ..ppo import RoleAwareCentralCritic, living_unit_mask
from .geometry_probe import GeometryProbe


def require_option_state(observation):
    value = observation.get("option_state")
    batch = observation["allies"].shape[0]
    if value is None or value.shape != (batch, 3) or not torch.isfinite(value).all():
        raise ValueError("movement PPO requires finite option_state [batch,3]")
    if (value < 0).any() or (value > 1).any():
        raise ValueError("option state fractions must lie in [0,1]")
    return value.float()


class OptionCentralCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pools = RoleAwareCentralCritic(config)
        width = self.pools.value[0].in_features
        # The existing critic supplies its independent global/role pools; the
        # new head conditions their joint value on explicit option state.
        self.pools.value = nn.Identity()
        self.value = nn.Sequential(nn.Linear(width+3, config.actor_hidden), nn.ReLU(), nn.Linear(config.actor_hidden, 1))

    def forward(self, observation):
        return self.value(torch.cat([self.pools(observation), require_option_state(observation)], -1)).squeeze(-1)


class AssistedMovementPolicy(nn.Module):
    def __init__(self, source, *, standard_deviation=.02):
        super().__init__()
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError("movement standard deviation must be positive")
        self.geometry = GeometryProbe(source, relative=False)
        self.geometry.shot.requires_grad_(False)
        self.option_move = nn.Linear(3, 2, bias=False)
        nn.init.zeros_(self.option_move.weight)
        self.critic = OptionCentralCritic(source.policy.config)
        self.register_buffer("standard_deviation", torch.tensor(float(standard_deviation)))

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad and not name.startswith("critic.")]

    def forward(self, observation):
        option = require_option_state(observation)
        inherited = self.geometry(observation)
        mean = inherited["target_raw_by_action"][..., 1, :] + self.option_move(option)[:, None]
        kind = inherited["action_logits"].argmax(-1)
        live = living_unit_mask(observation)
        return {"mean": mean, "action_type": kind, "living": live,
                "move_mask": live & (kind == 1), "inherited": inherited,
                "value": self.critic(observation)}

    def act(self, observation, *, deterministic=False):
        prediction = self(observation)
        mean = prediction["mean"]
        latent = mean if deterministic else Normal(mean, self.standard_deviation).sample()
        # Unused latents are explicitly zero and carry no density/entropy.
        latent = torch.where(prediction["move_mask"][..., None], latent, torch.zeros_like(latent))
        logp = Normal(mean, self.standard_deviation).log_prob(latent).sum(-1)
        logp = torch.where(prediction["move_mask"], logp, torch.zeros_like(logp))
        targets = prediction["inherited"]["target_by_action"]
        selected = targets.gather(-2, prediction["action_type"][..., None, None].expand(-1, -1, 1, 2)).squeeze(-2)
        selected = torch.where(prediction["move_mask"][..., None], torch.tanh(latent), selected)
        action = {"action_type": prediction["action_type"], "target": selected,
                  "power": prediction["inherited"]["power"]}
        return action, latent, logp, prediction["value"]

    def evaluate_latents(self, observation, action_type, latent):
        prediction = self(observation)
        if not torch.equal(action_type, prediction["action_type"]):
            raise ValueError("frozen action choice changed on stored observations")
        distribution = Normal(prediction["mean"], self.standard_deviation)
        logp = distribution.log_prob(latent).sum(-1)
        logp = torch.where(prediction["move_mask"], logp, torch.zeros_like(logp))
        return logp, prediction


def movement_loss(new_logp, old_logp, advantages, values, returns, prediction, *, clip_ratio=.2):
    live, moves = prediction["living"].float(), prediction["move_mask"].float()
    counts = live.sum(-1).clamp_min(1)
    log_ratio = (new_logp-old_logp)*moves
    ratio = torch.exp(log_ratio.clamp(-30, 30))
    advantage = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    surrogate = torch.minimum(ratio*advantage[:, None], ratio.clamp(1-clip_ratio, 1+clip_ratio)*advantage[:, None])
    policy = -((surrogate*moves).sum(-1)/counts).mean()
    value = (values-returns).square().mean()
    approx_kl = ((ratio-1)-log_ratio)*moves
    return {"total": policy+.5*value, "policy": policy, "value": value,
            "meanMovementKl": approx_kl.sum()/moves.sum().clamp_min(1),
            "maxMovementKl": approx_kl.max(),
            "clipFraction": (((ratio-1).abs() > clip_ratio)*moves).sum()/moves.sum().clamp_min(1),
            "ratioMean": (ratio*moves).sum()/moves.sum().clamp_min(1)}
