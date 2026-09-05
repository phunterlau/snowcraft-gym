"""Versioned, time-limited movement actor with a continuation-aware critic."""

import torch
from torch import nn

from .movement_ppo import AssistedMovementPolicy, OptionCentralCritic, require_option_state

RECOVERY_INPUT = "snowgym.recovery-time.v0"


def recovery_time(observation):
    value = observation.get("recovery_remaining")
    if value is None or value.shape != (observation["allies"].shape[0], 1):
        raise ValueError("recovery_remaining must have shape [batch,1]")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("recovery_remaining must be a finite fraction")
    return value.float()


class RecoveryCritic(OptionCentralCritic):
    def __init__(self, config):
        super().__init__(config)
        width = self.value[0].in_features + 1
        self.value = nn.Sequential(nn.Linear(width, config.actor_hidden), nn.ReLU(), nn.Linear(config.actor_hidden, 1))

    def forward(self, observation):
        return self.value(torch.cat([self.pools(observation), require_option_state(observation),
                                     recovery_time(observation)], -1)).squeeze(-1)


class RecoveryPolicy(AssistedMovementPolicy):
    def __init__(self, source, *, standard_deviation=.02):
        super().__init__(source, standard_deviation=standard_deviation)
        self.critic = RecoveryCritic(source.policy.config)
        self.budget_move = nn.Linear(1, 2, bias=False)
        nn.init.zeros_(self.budget_move.weight)

    def forward(self, observation):
        prediction = super().forward(observation)
        prediction["mean"] = prediction["mean"] + self.budget_move(recovery_time(observation))[:, None]
        return prediction
