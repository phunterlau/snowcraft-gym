"""R1n-b: full-authority Engage policy with the E3 contract repairs.

Differs from `full_authority_ppo.FullAuthorityPolicy` (E3, digest-pinned and
unmodified) only where `reviews/m7b_r1n_b_declaration.md` section 2 says:

- B1: throw sampling/likelihood use a separate `throw_log_std`, initialized to
  the global (arena-normalized) calibration in both arms, so the local and
  global arms differ only in `decode_move` and `move_log_std`.
- B5: the critic is `OptionCentralCritic`, which reads `option_state`.

The egocentric features and movement decoders are E3's own methods, bound
here unchanged rather than copied, so the representation cannot drift from
the archived E3 run.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..ppo import living_unit_mask
from .full_authority_ppo import DESTINATIONS, FullAuthorityPolicy, calibrated_target_log_std
from .model import ModelConfig
from .movement_ppo import OptionCentralCritic

FEATURES = 21 + 4 * 32 + 38 + 40


def throw_log_std_init(target_world_sigma: float = 2.0) -> list[float]:
    """Throw aim decodes as tanh(latent) in arena-normalized space in both arms,
    so its physically consistent initialization is the global calibration."""
    return calibrated_target_log_std("global", 1.0, target_world_sigma=target_world_sigma)


class FullAuthorityPolicyV1(nn.Module):
    # E3's exact feature transform and movement decoders (declaration F5, B1).
    pair_features = FullAuthorityPolicy.pair_features
    features = FullAuthorityPolicy.features
    decode_move = FullAuthorityPolicy.decode_move

    def __init__(self, *, destination: str, local_radius: float = 8.0,
                 target_world_sigma: float = 2.0, initial_power_log_std: float = -1.0,
                 critic_config: ModelConfig | None = None):
        super().__init__()
        if destination not in DESTINATIONS:
            raise ValueError("unknown full-authority destination geometry")
        if local_radius <= 0:
            raise ValueError("local destination radius must be positive")
        self.destination, self.local_radius = destination, float(local_radius)
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            for name, size in (("allies", 21), ("enemies", 21), ("projectiles", 9), ("obstacles", 9))
        })
        self.action_head = nn.Linear(FEATURES, 4)
        self.move_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 2))
        self.throw_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 2))
        self.power_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 1))
        self.move_log_std = nn.Parameter(torch.tensor(
            calibrated_target_log_std(destination, local_radius, target_world_sigma=target_world_sigma)))
        self.throw_log_std = nn.Parameter(torch.tensor(throw_log_std_init(target_world_sigma)))
        self.power_log_std = nn.Parameter(torch.full((1,), float(initial_power_log_std)))
        self.critic = OptionCentralCritic(critic_config or ModelConfig(observation_version=3))

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad and not name.startswith("critic.")]

    def forward(self, observation: dict[str, Tensor], *, with_value: bool = True) -> dict[str, Tensor]:
        features = self.features(observation)
        live = living_unit_mask(observation)
        action_logits = self.action_head(features).masked_fill(~live[..., None], -1e9)
        action_logits = action_logits.masked_fill(~observation["unit_action_mask"].bool(), -1e9)
        prediction = {"action_logits": action_logits, "living": live,
                      "move_raw": self.move_head(features), "throw_raw": self.throw_head(features),
                      "power_raw": self.power_head(features).squeeze(-1)}
        if with_value:
            prediction["value"] = self.critic(observation)
        return prediction

    def act(self, observation: dict[str, Tensor], *, deterministic: bool = False):
        prediction = self(observation)
        move_mean, throw_mean, power_mean = prediction["move_raw"], prediction["throw_raw"], prediction["power_raw"]
        move_std, throw_std, power_std = self.move_log_std.exp(), self.throw_log_std.exp(), self.power_log_std.exp()
        action_type = (prediction["action_logits"].argmax(-1) if deterministic
                       else Categorical(logits=prediction["action_logits"]).sample())
        move_latent = move_mean if deterministic else Normal(move_mean, move_std).sample()
        throw_latent = throw_mean if deterministic else Normal(throw_mean, throw_std).sample()
        power_latent = power_mean if deterministic else Normal(power_mean, power_std).sample()
        moves, throws = action_type == ACTION_MOVE, action_type == ACTION_THROW
        move_latent = torch.where(moves[..., None], move_latent, torch.zeros_like(move_latent))
        throw_latent = torch.where(throws[..., None], throw_latent, torch.zeros_like(throw_latent))
        power_latent = torch.where(throws, power_latent, torch.zeros_like(power_latent))
        move_target = self.decode_move(observation, move_latent)
        throw_target = torch.tanh(throw_latent)
        target = torch.where(moves[..., None], move_target, torch.where(throws[..., None], throw_target,
            torch.zeros_like(move_target)))
        action = {"action_type": action_type, "target": target, "power": torch.sigmoid(power_latent)}
        latent = {"move": move_latent, "throw": throw_latent, "power": power_latent}
        logp, _ = self.evaluate_latents(observation, action_type, latent, prediction=prediction)
        return action, latent, logp, prediction["value"]

    def evaluate_latents(self, observation, action_type, latent, *, prediction=None, with_value=True):
        prediction = prediction if prediction is not None else self(observation, with_value=with_value)
        live = prediction["living"]
        moves, throws = live & (action_type == ACTION_MOVE), live & (action_type == ACTION_THROW)
        move_normal = Normal(prediction["move_raw"], self.move_log_std.exp())
        throw_normal = Normal(prediction["throw_raw"], self.throw_log_std.exp())
        power_normal = Normal(prediction["power_raw"], self.power_log_std.exp())
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
