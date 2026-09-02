"""Mask-aware hybrid behavior-cloning objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from .model import select_action_target


@dataclass(frozen=True)
class LossConfig:
    action_weight: float = 1.0
    target_weight: float = 1.0
    power_weight: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def behavior_clone_loss(
    prediction: dict[str, Tensor],
    action: dict[str, Tensor],
    observation: dict[str, Tensor],
    config: LossConfig,
) -> dict[str, Tensor]:
    present = observation["ally_mask"].bool()
    labels = action["action_type"].long()
    if not bool(present.any()):
        raise ValueError("batch has no present ally labels")
    action_loss = F.cross_entropy(prediction["action_logits"][present], labels[present])
    target_mask = present & ((labels == ACTION_MOVE) | (labels == ACTION_THROW))
    throw_mask = present & (labels == ACTION_THROW)
    predicted_target = (
        select_action_target(prediction["target_by_action"], labels)
        if "target_by_action" in prediction
        else prediction["target"]
    )
    target_loss = masked_mse(predicted_target, action["target"].float(), target_mask)
    power_loss = masked_mse(prediction["power"], action["power"].float(), throw_mask)
    total = (
        config.action_weight * action_loss
        + config.target_weight * target_loss
        + config.power_weight * power_loss
    )
    return {
        "total": total,
        "action": action_loss,
        "target": target_loss,
        "power": power_loss,
    }


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[mask], target[mask])


def loss_config(value: Any) -> LossConfig:
    keys = {"action_weight", "target_weight", "power_weight"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"loss must define {', '.join(sorted(keys))}")
    if not all(
        isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(item)
        and item >= 0
        for item in value.values()
    ):
        raise ValueError("loss weights must be finite non-negative numbers")
    config = LossConfig(**{key: float(value[key]) for key in keys})
    if sum(config.as_dict().values()) <= 0:
        raise ValueError("at least one loss weight must be positive")
    return config
