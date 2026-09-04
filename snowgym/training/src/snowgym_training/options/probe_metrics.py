"""Living-unit, phase-conditioned teacher agreement in physical coordinates."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ..ppo import HybridActorCritic, living_unit_mask
from .reservoir import TeacherBcReservoir


def phase_masks(observation: dict[str, Tensor], labels: Tensor, scale: Tensor) -> dict[str, Tensor]:
    alive = living_unit_mask(observation)
    own = observation["allies"][..., 2:4] * scale
    enemy = observation["enemies"][..., 2:4] * scale
    enemy_alive = observation["enemy_mask"].bool() & (observation["enemies"][..., 1] > 0.5)
    distance = torch.linalg.vector_norm(own.unsqueeze(-2) - enemy.unsqueeze(-3), dim=-1)
    nearest = distance.masked_fill(~enemy_alive.unsqueeze(-2), torch.inf).min(-1).values
    fire = alive & (labels == 2)
    contact = alive & ~fire & (nearest <= 9)
    return {"all": alive, "approach": alive & ~fire & ~contact, "contact": contact, "fire": fire}


def empty_counts() -> dict[str, Any]:
    return {"count": 0, "classCorrect": 0, "moveCount": 0, "throwCount": 0,
            "moveSquaredError": 0.0, "throwSquaredError": 0.0, "powerSquaredError": 0.0,
            "rayCount": 0, "rayDegreesSum": 0.0,
            "confusion": torch.zeros(4, 4, dtype=torch.int64)}


def accumulate(
    counts: dict[str, Any], prediction: dict[str, Tensor], teacher: dict[str, Tensor],
    observation: dict[str, Tensor], mask: Tensor, scale: Tensor,
) -> None:
    labels = teacher["action_type"].long()
    predicted = prediction["action_logits"].argmax(-1)
    counts["count"] += int(mask.sum())
    counts["classCorrect"] += int(((predicted == labels) & mask).sum())
    counts["confusion"] += torch.bincount((labels[mask] * 4 + predicted[mask]), minlength=16).reshape(4, 4)
    for action, name in ((1, "move"), (2, "throw")):
        selected = mask & (labels == action)
        counts[f"{name}Count"] += int(selected.sum())
        error = (prediction["target_by_action"][..., action, :][selected] - teacher["target"][selected]) * scale
        counts[f"{name}SquaredError"] += float(error.square().sum())
        if action == 2:
            counts["powerSquaredError"] += float((prediction["power"][selected] - teacher["power"][selected]).square().sum())
            origin = observation["allies"][..., 2:4][selected] * scale
            expected_ray = teacher["target"][selected] * scale - origin
            actual_ray = prediction["target_by_action"][..., 2, :][selected] * scale - origin
            norm = expected_ray.norm(dim=-1) * actual_ray.norm(dim=-1)
            valid = (expected_ray.norm(dim=-1) > 1e-6) & (actual_ray.norm(dim=-1) > 1e-6)
            cosine = (expected_ray * actual_ray).sum(-1)[valid] / norm[valid]
            counts["rayCount"] += int(valid.sum())
            counts["rayDegreesSum"] += float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).sum())


def finish_counts(counts: dict[str, Any]) -> dict[str, Any]:
    result = {**counts, "confusion": counts["confusion"].tolist()}
    result["classAccuracy"] = counts["classCorrect"] / counts["count"] if counts["count"] else None
    for name in ("move", "throw"):
        count = counts[f"{name}Count"]
        result[f"{name}TargetRmseWorld"] = math.sqrt(counts[f"{name}SquaredError"] / count) if count else None
    result["throwPowerRmse"] = math.sqrt(counts["powerSquaredError"] / counts["throwCount"]) if counts["throwCount"] else None
    result["throwRayMeanDegrees"] = counts["rayDegreesSum"] / counts["rayCount"] if counts["rayCount"] else None
    result["undefinedThrowRays"] = counts["throwCount"] - counts["rayCount"]
    return result


def teacher_agreement(model: HybridActorCritic, reservoir: TeacherBcReservoir, *, width: float = 100, height: float = 80) -> dict[str, Any]:
    model.eval()
    scale = torch.tensor([width / 2, height / 2])
    counts = {name: empty_counts() for name in ("all", "approach", "contact", "fire")}
    with torch.no_grad():
        for start in range(0, reservoir.size, 256):
            observation, teacher = reservoir.batch(torch.arange(start, min(start + 256, reservoir.size)))
            prediction = model(observation)
            for name, mask in phase_masks(observation, teacher["action_type"], scale).items():
                accumulate(counts[name], prediction, teacher, observation, mask, scale)
    return {name: finish_counts(value) for name, value in counts.items()}
