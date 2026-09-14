"""R1n-d: a critic that reads the actor's egocentric feature transform.

`OptionCentralCritic` pools world-frame features. This critic has its own
encoders with the actor's shapes and E3's `features` transform, bound
unchanged (as in `FullAuthorityPolicyV1`), so it sees the same egocentric
227-wide features the actor does. See `reviews/m7b_r1n_d_declaration.md` §4.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..ppo import living_unit_mask
from .full_authority_ppo import FullAuthorityPolicy
from .full_authority_ppo_v1 import FEATURES
from .movement_ppo import require_option_state


class EgocentricCritic(nn.Module):
    pair_features = FullAuthorityPolicy.pair_features
    features = FullAuthorityPolicy.features

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            for name, size in (("allies", 21), ("enemies", 21), ("projectiles", 9), ("obstacles", 9))
        })
        self.value = nn.Sequential(nn.Linear(FEATURES + 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        features = self.features(observation)
        live = living_unit_mask(observation).float()
        pooled = (features * live[..., None]).sum(1) / live.sum(1, keepdim=True).clamp(min=1.)
        return self.value(torch.cat([pooled, require_option_state(observation).float()], -1)).squeeze(-1)
