"""Detached NumPy bridge from SnowGym observations to a Torch checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from snowgym_client.encoding import GymAction, GymObservation, make_action_space

from .checkpoint import load_checkpoint
from .data import OBSERVATION_FIELDS
from .model import EntityPolicy, model_config
from .plan_data import PLAN_FEATURE_VECTOR_SIZE, PLAN_GROUP_SLOTS


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
        if self.model.plan_conditioned:
            required_plan = {"plan_groups", "plan_group_mask"}
            missing_plan = required_plan - set(observation)
            if missing_plan:
                raise ValueError(
                    f"policy observation missing plan fields: {sorted(missing_plan)}"
                )
            plan_groups = np.asarray(observation["plan_groups"])
            plan_mask = np.asarray(observation["plan_group_mask"])
            if plan_groups.shape != (PLAN_GROUP_SLOTS, PLAN_FEATURE_VECTOR_SIZE):
                raise ValueError("policy plan_groups has an invalid shape")
            if plan_mask.shape != (PLAN_GROUP_SLOTS,):
                raise ValueError("policy plan_group_mask has an invalid shape")
            tensors.update(
                {
                    "plan_groups": torch.from_numpy(np.array(plan_groups, copy=True))[
                        None, ...
                    ],
                    "plan_group_mask": torch.from_numpy(np.array(plan_mask, copy=True))[
                        None, ...
                    ],
                }
            )
        if self.model.plan_role_conditioned:
            if "plan_unit_roles" not in observation:
                raise ValueError("policy observation missing plan field: plan_unit_roles")
            plan_unit_roles = np.asarray(observation["plan_unit_roles"])
            expected = (len(np.asarray(observation["ally_mask"])), PLAN_GROUP_SLOTS)
            if plan_unit_roles.shape != expected:
                raise ValueError(
                    f"policy plan_unit_roles must have shape {expected}"
                )
            tensors["plan_unit_roles"] = torch.from_numpy(
                np.array(plan_unit_roles, copy=True)
            )[None, ...]
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
