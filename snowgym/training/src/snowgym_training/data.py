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


class TrajectoryDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        audit_dataset(self.path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        chunks: dict[str, list[np.ndarray]] = {
            **{f"observation__{name}": [] for name in OBSERVATION_FIELDS},
            **{f"action__{name}": [] for name in ACTION_FIELDS},
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
            for name in OBSERVATION_FIELDS
        }
        action = {
            name: torch.from_numpy(self.arrays[f"action__{name}"][indices])
            for name in ACTION_FIELDS
        }
        return observation, action


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
