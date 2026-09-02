"""Versioned exact-resume checkpoints for SnowGym PPO updates."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import semantic_state_digest
from .model import model_config
from .ppo import HybridActorCritic, PPOConfig
from .trajectory import json_digest

PPO_CHECKPOINT_FORMAT = "snowgym.ppo-checkpoint.v0"


def save_ppo_checkpoint(
    path: str | Path,
    *,
    model: HybridActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    curriculum_digest: str,
    training_seed: int,
    update_index: int,
    environment_steps: int,
    git_commit: str,
    seed_schedule: dict[str, int],
    collector_config: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite PPO checkpoint {destination}")
    validate_counters(training_seed, update_index, environment_steps)
    for name, value in {
        "curriculum_digest": curriculum_digest,
        "git_commit": git_commit,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    validate_seed_schedule(seed_schedule)
    validate_collector_config(collector_config)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torchRngState": torch.get_rng_state(),
    }
    metadata: dict[str, Any] = {
        "format": PPO_CHECKPOINT_FORMAT,
        "gitCommit": git_commit,
        "curriculumDigest": curriculum_digest,
        "architecture": model.policy.config.as_dict(),
        "ppoConfig": asdict(config),
        "trainingSeed": training_seed,
        "updateIndex": update_index,
        "environmentSteps": environment_steps,
        "seedSchedule": dict(seed_schedule),
        "collectorConfig": dict(collector_config),
        "stateDigest": semantic_state_digest(state),
    }
    metadata["checkpointDigest"] = json_digest(metadata)
    destination.mkdir(parents=True)
    torch.save(state, destination / "state.pt")
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_ppo_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load PPO checkpoint metadata: {error}") from error
    validate_ppo_checkpoint_metadata(metadata)
    try:
        state = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot load PPO checkpoint state: {error}") from error
    if not isinstance(state, dict) or set(state) != {
        "model",
        "optimizer",
        "torchRngState",
    }:
        raise ValueError("PPO checkpoint state fields are invalid")
    if semantic_state_digest(state) != metadata["stateDigest"]:
        raise ValueError("PPO checkpoint state digest mismatch")
    return metadata, state


def restore_ppo_checkpoint(
    path: str | Path,
    *,
    model: HybridActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    curriculum_digest: str,
    training_seed: int,
    collector_config: dict[str, Any],
) -> dict[str, Any]:
    metadata, state = load_ppo_checkpoint(path)
    expected = {
        "architecture": model.policy.config.as_dict(),
        "ppoConfig": asdict(config),
        "curriculumDigest": curriculum_digest,
        "trainingSeed": training_seed,
        "collectorConfig": collector_config,
    }
    for name, value in expected.items():
        if metadata[name] != value:
            raise ValueError(f"PPO checkpoint {name} does not match training run")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng_state = state["torchRngState"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8:
        raise ValueError("PPO checkpoint torch RNG state is invalid")
    torch.set_rng_state(rng_state)
    return metadata


def validate_ppo_checkpoint_metadata(value: Any) -> None:
    required = {
        "format",
        "gitCommit",
        "curriculumDigest",
        "architecture",
        "ppoConfig",
        "trainingSeed",
        "updateIndex",
        "environmentSteps",
        "seedSchedule",
        "collectorConfig",
        "stateDigest",
        "checkpointDigest",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"PPO checkpoint metadata must contain exactly {sorted(required)}")
    if value["format"] != PPO_CHECKPOINT_FORMAT:
        raise ValueError(f"PPO checkpoint format must be {PPO_CHECKPOINT_FORMAT}")
    for name in ("gitCommit", "curriculumDigest", "stateDigest"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"PPO checkpoint {name} must be a non-empty string")
    model_config(value["architecture"])
    ppo_value = value["ppoConfig"]
    if not isinstance(ppo_value, dict) or set(ppo_value) != set(asdict(PPOConfig())):
        raise ValueError("PPO checkpoint ppoConfig fields are invalid")
    PPOConfig(**ppo_value)
    validate_counters(
        value["trainingSeed"], value["updateIndex"], value["environmentSteps"]
    )
    validate_seed_schedule(value["seedSchedule"])
    validate_collector_config(value["collectorConfig"])
    source = {name: item for name, item in value.items() if name != "checkpointDigest"}
    if value["checkpointDigest"] != json_digest(source):
        raise ValueError("PPO checkpoint metadata digest mismatch")


def validate_counters(training_seed: Any, update_index: Any, environment_steps: Any) -> None:
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise ValueError("PPO training_seed must be an integer")
    for name, value in {
        "update_index": update_index,
        "environment_steps": environment_steps,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"PPO {name} must be a non-negative integer")


def validate_seed_schedule(value: Any) -> None:
    required = {"minimum", "maximum", "nextSeed"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"PPO seed schedule must contain exactly {sorted(required)}")
    if not all(
        isinstance(value[name], int) and not isinstance(value[name], bool)
        for name in required
    ):
        raise ValueError("PPO seed schedule values must be integers")
    if value["minimum"] > value["maximum"]:
        raise ValueError("PPO seed schedule minimum must not exceed maximum")
    if not value["minimum"] <= value["nextSeed"] <= value["maximum"] + 1:
        raise ValueError("PPO seed schedule cursor is outside its range")


def validate_collector_config(value: Any) -> None:
    required = {"gateId", "worlds", "rolloutSteps"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"PPO collector config must contain exactly {sorted(required)}")
    if not isinstance(value["gateId"], str) or not value["gateId"]:
        raise ValueError("PPO collector gateId must be non-empty")
    for name in ("worlds", "rolloutSteps"):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"PPO collector {name} must be a positive integer")
