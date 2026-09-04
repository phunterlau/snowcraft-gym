"""Immutable successful-teacher samples for the M7b-R1 BC auxiliary loss."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TeacherBcReservoir:
    observations: dict[str, Tensor]
    actions: dict[str, Tensor]
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return int(next(iter(self.actions.values())).shape[0])

    def batch(self, indices: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("teacher reservoir indices must be a rank-one long tensor")
        if bool((indices < 0).any()) or bool((indices >= self.size).any()):
            raise ValueError("teacher reservoir index is out of range")
        return (
            {name: value[indices] for name, value in self.observations.items()},
            {name: value[indices] for name, value in self.actions.items()},
        )


def load_teacher_bc_reservoir(
    path: str | Path, *, manifest_path: str | Path | None = None
) -> TeacherBcReservoir:
    source = Path(path)
    manifest_source = (
        Path(manifest_path) if manifest_path is not None else source.parent / "manifest.json"
    )
    try:
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load teacher reservoir manifest: {error}") from error
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    expected = artifacts.get(source.name) if isinstance(artifacts, dict) else None
    digest = file_digest(source)
    if expected != digest:
        raise ValueError("teacher reservoir digest does not match its manifest")
    try:
        with np.load(source, allow_pickle=False) as arrays:
            successes = arrays["episode_success"]
            observations = {
                name.removeprefix("observation__"): torch.from_numpy(
                    arrays[name].copy()
                )
                for name in arrays.files
                if name.startswith("observation__")
            }
            actions = {
                "action_type": torch.from_numpy(arrays["teacher_action_type"].copy()),
                "target": torch.from_numpy(arrays["teacher_target"].copy()),
                "power": torch.from_numpy(arrays["teacher_power"].copy()),
            }
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"cannot load teacher reservoir: {error}") from error
    if successes.ndim != 1 or successes.size == 0 or not bool(successes.all()):
        raise ValueError("teacher reservoir must contain only successful episodes")
    sizes = {int(value.shape[0]) for value in (*observations.values(), *actions.values())}
    if len(sizes) != 1 or next(iter(sizes), 0) <= 0:
        raise ValueError("teacher reservoir tensors have inconsistent leading dimensions")
    required_observations = {
        "allies", "ally_mask", "enemies", "enemy_mask", "projectiles",
        "projectile_mask", "obstacles", "obstacle_mask", "team_alive", "tick",
        "unit_action_mask", "plan_groups", "plan_group_mask", "plan_unit_roles",
        "plan_role_state", "mission_progress", "decision_hz", "decision_dt",
        "max_ticks", "remaining_fraction",
    }
    if set(observations) != required_observations:
        raise ValueError("teacher reservoir observation fields do not match v3 plan PPO")
    metadata = {
        "format": "snowgym.teacher-bc-reservoir.v0",
        "path": str(source),
        "digest": digest,
        "manifestPath": str(manifest_source),
        "manifestDigest": manifest.get("manifestDigest"),
        "sourceCheckpointDigest": manifest.get("checkpointDigest"),
        "implementationGitCommit": manifest.get("implementationGitCommit"),
        "simulationVersion": manifest.get("simulationVersion"),
        "stateHashVersion": manifest.get("stateHashVersion"),
        "episodes": int(successes.size),
        "samples": next(iter(sizes)),
    }
    return TeacherBcReservoir(observations, actions, metadata)


def file_digest(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read teacher reservoir {path}: {error}") from error
