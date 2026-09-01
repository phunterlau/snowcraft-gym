"""Detached NumPy bridge from SnowGym observations to a Torch checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from snowgym_client.encoding import GymAction, GymObservation, make_action_space

from .checkpoint import load_checkpoint
from .data import OBSERVATION_FIELDS
from .model import EntityPolicy, model_config


class TorchPolicy:
    def __init__(self, checkpoint: str | Path) -> None:
        self.metadata, state = load_checkpoint(checkpoint)
        self.model = EntityPolicy(model_config(self.metadata["architecture"])).cpu()
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def act(self, observation: GymObservation) -> GymAction:
        missing = set(OBSERVATION_FIELDS) - set(observation)
        if missing:
            raise ValueError(f"policy observation missing fields: {sorted(missing)}")
        tensors = {
            name: torch.from_numpy(np.array(observation[name], copy=True))[None, ...]
            for name in OBSERVATION_FIELDS
        }
        with torch.no_grad():
            prediction = self.model(tensors)
        action_type = prediction["action_logits"].argmax(dim=-1)[0].cpu().numpy()
        ally_mask = np.asarray(observation["ally_mask"], dtype=np.int8)
        action_type = action_type.astype(np.int64, copy=True)
        action_type[ally_mask == 0] = 0
        action = {
            "action_type": action_type,
            "target": prediction["target"][0].cpu().numpy().astype(np.float32, copy=True),
            "power": prediction["power"][0].cpu().numpy().astype(np.float32, copy=True),
        }
        space = make_action_space(len(ally_mask))
        if not space.contains(action):
            raise ValueError("checkpoint policy emitted an invalid SnowGym action")
        mask = np.asarray(observation["unit_action_mask"], dtype=np.int8)
        selected = mask[np.arange(len(action_type)), action_type]
        if np.any(selected[ally_mask.astype(bool)] != 1):
            raise ValueError("checkpoint policy emitted a masked action")
        return action

    __call__ = act


class LearnedOpponent:
    """Opponent adapter matching the existing callable policy convention."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.policy = TorchPolicy(checkpoint)

    def act(self, observation: GymObservation) -> GymAction:
        return self.policy.act(observation)

    __call__ = act
