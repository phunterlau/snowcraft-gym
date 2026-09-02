"""Audited, deterministic trajectory loading for SnowGym training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .trajectory import audit_dataset

OBSERVATION_FIELDS = (
    "allies",
    "ally_mask",
    "enemies",
    "enemy_mask",
    "obstacles",
    "obstacle_mask",
    "projectiles",
    "projectile_mask",
    "team_alive",
    "tick",
    "unit_action_mask",
)
ACTION_FIELDS = ("action_type", "target", "power")
PLAN_OBSERVATION_FIELDS = ("plan_groups", "plan_group_mask")
PLAN_ROLE_FIELD = ("plan_unit_roles",)
COUNTERFACTUAL_OBSERVATION_FIELDS = (
    "counterfactual_plan_groups", "counterfactual_plan_group_mask"
)
COUNTERFACTUAL_ACTION_FIELDS = (
    "counterfactual_action_type", "counterfactual_target", "counterfactual_power"
)


class TrajectoryDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        audit_dataset(self.path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        self.observation_fields = OBSERVATION_FIELDS + (
            PLAN_OBSERVATION_FIELDS if self.manifest.get("planConditioned") is True else ()
        ) + (PLAN_ROLE_FIELD if self.manifest.get("planUnitRoles") is True else ())
        self.counterfactual_plan_labels = self.manifest.get("counterfactualPlanLabels") is not None
        self.loaded_observation_fields = self.observation_fields + (
            COUNTERFACTUAL_OBSERVATION_FIELDS if self.counterfactual_plan_labels else ()
        ) + (
            ("counterfactual_plan_unit_roles",)
            if self.manifest.get("planUnitRoles") is True else ()
        )
        self.loaded_action_fields = ACTION_FIELDS + (
            COUNTERFACTUAL_ACTION_FIELDS if self.counterfactual_plan_labels else ()
        )
        chunks: dict[str, list[np.ndarray]] = {
            **{f"observation__{name}": [] for name in self.loaded_observation_fields},
            **{f"action__{name}": [] for name in self.loaded_action_fields},
        }
        for shard in self.manifest["shards"]:
            with np.load(self.path / shard["path"], allow_pickle=False) as archive:
                for name in chunks:
                    if name not in archive.files:
                        raise ValueError(f"training field missing from shard: {name}")
                    chunks[name].append(np.array(archive[name], copy=True))
        self.arrays = {
            name: np.concatenate(parts, axis=0) for name, parts in chunks.items()
        }

    def __len__(self) -> int:
        return int(self.manifest["transitions"])

    def batch(
        self, indices: np.ndarray
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        observation = {
            name: torch.from_numpy(self.arrays[f"observation__{name}"][indices])
            for name in self.loaded_observation_fields
        }
        action = {
            name: torch.from_numpy(self.arrays[f"action__{name}"][indices])
            for name in self.loaded_action_fields
        }
        return observation, action

    def plan_mission_batch_indices(
        self, batch_size: int, seed: int, step: int
    ) -> np.ndarray:
        """Sample missions uniformly, then sample a transition within each mission."""
        if batch_size <= 0 or step < 0:
            raise ValueError("batch_size must be positive and step non-negative")
        episodes = self.manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("plan-mission sampling requires episode metadata")
        by_mission: dict[str, list[np.ndarray]] = {}
        for episode in episodes:
            if not isinstance(episode, dict) or not isinstance(episode.get("planName"), str):
                raise ValueError("plan-mission sampling requires episode planName")
            start = episode.get("startTransition")
            count = episode.get("transitions")
            if not isinstance(start, int) or not isinstance(count, int) or count <= 0:
                raise ValueError("plan-mission sampling requires valid transition ranges")
            by_mission.setdefault(episode["planName"], []).append(
                np.arange(start, start + count, dtype=np.int64)
            )
        missions = sorted(by_mission)
        pools = {name: np.concatenate(by_mission[name]) for name in missions}
        generator = np.random.default_rng(np.random.SeedSequence([seed, step]))
        offset = (step * batch_size) % len(missions)
        return np.asarray(
            [
                generator.choice(pools[missions[(offset + slot) % len(missions)]])
                for slot in range(batch_size)
            ],
            dtype=np.int64,
        )

    def inverse_plan_role_weights(self) -> torch.Tensor:
        """Return mean-one inverse-frequency weights for assigned unit roles."""
        if "plan_unit_roles" not in self.observation_fields:
            raise ValueError("role-balanced loss requires plan unit roles")
        roles = self.arrays["observation__plan_unit_roles"]
        counts = roles.sum(axis=(0, 1), dtype=np.float64)
        observed = counts > 0
        if not np.any(observed):
            raise ValueError("role-balanced loss requires at least one assigned role")
        inverse = np.zeros_like(counts)
        inverse[observed] = 1.0 / counts[observed]
        normalized = inverse / np.mean(inverse[observed])
        return torch.from_numpy(normalized.astype(np.float32))


def deterministic_batch_indices(
    size: int, batch_size: int, seed: int, step: int
) -> np.ndarray:
    if size <= 0 or batch_size <= 0 or step < 0:
        raise ValueError("size/batch_size must be positive and step non-negative")
    batches_per_epoch = max((size + batch_size - 1) // batch_size, 1)
    epoch, batch = divmod(step, batches_per_epoch)
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    permutation = generator.permutation(size)
    start = batch * batch_size
    return permutation[start : min(start + batch_size, size)]


def manifest_versions(dataset: TrajectoryDataset) -> dict[str, str]:
    versions = dataset.manifest.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("dataset manifest has no versions")
    return {str(key): str(value) for key, value in versions.items()}
