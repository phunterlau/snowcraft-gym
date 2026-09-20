"""M8-S3: roster-N helpers over the existing batch path (`reviews/m8_s3_declaration.md`).

Nothing here changes an existing module. The 3v3 scenario reuses the R1n arena and
horizon from `full_authority_train_v1.scenario()`; Red runs natively behind the team
action (`redController`), never through a joint step."""

from __future__ import annotations

import torch

from . import full_authority_train_v1 as v1
from .mixture_imitation import EVAL_ARMS

SLOT_INDEXED = ("allies", "ally_mask", "unit_action_mask", "plan_unit_roles")


def roster_scenario(units: int, arm: str = "normal") -> dict:
    if type(units) is not int or not 1 <= units <= 10:
        raise ValueError("roster must be an integer in 1..10")
    return {**v1.scenario(), "blueUnits": units, "redUnits": units, **EVAL_ARMS[arm]}


def permute_ally_slots(observation: dict[str, torch.Tensor], permutation) -> dict[str, torch.Tensor]:
    """Reorder the first len(permutation) ally slots in every slot-indexed tensor together.
    Unpermuted tensors are shared with the input, not copied."""
    order = [int(i) for i in permutation]
    if sorted(order) != list(range(len(order))):
        raise ValueError("permutation must reorder 0..n-1")
    capacity = observation["allies"].shape[1]
    if len(order) > capacity:
        raise ValueError("permutation is larger than the roster capacity")
    index = torch.tensor(order + list(range(len(order), capacity)))
    return {name: value[:, index] if name in SLOT_INDEXED else value for name, value in observation.items()}


def equivariance_gap(model, observation: dict[str, torch.Tensor], permutation) -> dict[str, float]:
    """Max absolute difference between permuting the model's outputs and running it on
    permuted inputs. Zero (to float tolerance) means each unit's row depends on that unit
    and the shared team tensor, not on its slot number."""
    order = [int(i) for i in permutation]
    capacity = observation["allies"].shape[1]
    index = torch.tensor(order + list(range(len(order), capacity)))
    with torch.no_grad():
        base = model(observation)
        moved = model(permute_ally_slots(observation, order))
    gaps = {name: float((base[name][:, index] - moved[name]).abs().max())
            for name in ("action_logits", "move_raw", "throw_raw", "power_raw")}
    gaps["value"] = float((base["value"] - moved["value"]).abs().max())
    return gaps
