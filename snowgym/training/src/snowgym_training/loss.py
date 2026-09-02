"""Mask-aware hybrid behavior-cloning objective."""

from __future__ import annotations

from dataclasses import dataclass
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
    throw_action_weight: float = 1.0

    def as_dict(self) -> dict[str, float]:
        value = {
            "action_weight": self.action_weight,
            "target_weight": self.target_weight,
            "power_weight": self.power_weight,
        }
        if self.throw_action_weight != 1.0:
            value["throw_action_weight"] = self.throw_action_weight
        return value


def behavior_clone_loss(
    prediction: dict[str, Tensor],
    action: dict[str, Tensor],
    observation: dict[str, Tensor],
    config: LossConfig,
    unit_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    present = observation["ally_mask"].bool()
    labels = action["action_type"].long()
    if not bool(present.any()):
        raise ValueError("batch has no present ally labels")
    action_class_weights = torch.ones(
        prediction["action_logits"].shape[-1],
        dtype=prediction["action_logits"].dtype,
        device=prediction["action_logits"].device,
    )
    action_class_weights[ACTION_THROW] = config.throw_action_weight
    if unit_weights is not None and unit_weights.shape != present.shape:
        raise ValueError("unit_weights must match ally_mask shape")
    if unit_weights is None:
        action_loss = F.cross_entropy(
            prediction["action_logits"][present],
            labels[present],
            weight=action_class_weights,
        )
    else:
        sample_weights = unit_weights[present].float()
        raw_action = F.cross_entropy(
            prediction["action_logits"][present],
            labels[present],
            weight=action_class_weights,
            reduction="none",
        )
        denominator = (
            action_class_weights[labels[present]] * sample_weights
        ).sum().clamp_min(torch.finfo(raw_action.dtype).eps)
        action_loss = (raw_action * sample_weights).sum() / denominator
    target_mask = present & ((labels == ACTION_MOVE) | (labels == ACTION_THROW))
    throw_mask = present & (labels == ACTION_THROW)
    predicted_target = (
        select_action_target(
            prediction.get("supervised_target_by_action", prediction["target_by_action"]),
            labels,
        )
        if "target_by_action" in prediction
        else prediction["target"]
    )
    target_loss = masked_mse(
        predicted_target, action["target"].float(), target_mask, unit_weights
    )
    power_loss = masked_mse(
        prediction["power"], action["power"].float(), throw_mask, unit_weights
    )
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


def masked_mse(
    prediction: Tensor, target: Tensor, mask: Tensor, weights: Tensor | None = None
) -> Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    if weights is None:
        return F.mse_loss(prediction[mask], target[mask])
    error = (prediction[mask] - target[mask]).square()
    selected = weights[mask].float()
    if error.ndim > selected.ndim:
        selected = selected.unsqueeze(-1).expand_as(error)
    return (error * selected).sum() / selected.sum().clamp_min(
        torch.finfo(error.dtype).eps
    )


def loss_config(value: Any) -> LossConfig:
    keys = {"action_weight", "target_weight", "power_weight"}
    optional = {"throw_action_weight"}
    if not isinstance(value, dict) or set(value) - optional != keys:
        raise ValueError(
            f"loss must define {', '.join(sorted(keys))} and may set throw_action_weight"
        )
    if not all(
        isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(item)
        and item >= 0
        for item in value.values()
    ):
        raise ValueError("loss weights must be finite non-negative numbers")
    config = LossConfig(**{key: float(item) for key, item in value.items()})
    if sum(getattr(config, key) for key in keys) <= 0:
        raise ValueError("at least one loss weight must be positive")
    if config.throw_action_weight <= 0:
        raise ValueError("throw_action_weight must be positive")
    return config
