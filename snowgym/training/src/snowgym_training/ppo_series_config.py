"""Versioned configuration and CLI for reproducible PPO checkpoint series."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .model import model_config
from .ppo import PPOConfig
from .ppo_checkpoint import load_ppo_checkpoint
from .ppo_series import run_ppo_series
from .trajectory import json_digest

PPO_SERIES_CONFIG_FORMAT = "snowgym.ppo-series-config.v0"


def default_series_config_path() -> Path:
    return Path(str(files("snowgym_training").joinpath("configs/ppo_1v1_bc_v0.json")))


def load_series_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_series_config_path()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load PPO series config {source}: {error}") from error
    validate_series_config(value)
    return value


def validate_series_config(value: Any) -> None:
    required = {
        "format", "name", "gateId", "worlds", "rolloutSteps", "trainingSeed",
        "rewardMode", "checkpointUpdates", "maxEvaluationDecisions", "architecture",
        "ppoConfig",
    }
    initialization_keys = {"warmStart", "ppoWarmStart"}
    if (
        not isinstance(value, dict)
        or set(value) - initialization_keys != required
        or len(set(value) & initialization_keys) != 1
    ):
        raise ValueError(
            "PPO series config must contain the required fields and exactly one of "
            "warmStart or ppoWarmStart"
        )
    if value["format"] != PPO_SERIES_CONFIG_FORMAT:
        raise ValueError(f"PPO series config format must be {PPO_SERIES_CONFIG_FORMAT}")
    if not all(isinstance(value[name], str) and value[name] for name in ("name", "gateId")):
        raise ValueError("PPO series config name and gateId must be non-empty")
    for name in ("worlds", "rolloutSteps", "maxEvaluationDecisions"):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"PPO series config {name} must be a positive integer")
    if not isinstance(value["trainingSeed"], int) or isinstance(value["trainingSeed"], bool):
        raise ValueError("PPO series config trainingSeed must be an integer")
    if value["rewardMode"] not in {"canonical", "health-potential"}:
        raise ValueError("PPO series config rewardMode is invalid")
    checkpoints = value["checkpointUpdates"]
    if not isinstance(checkpoints, list) or not checkpoints or checkpoints != sorted(set(checkpoints)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in checkpoints
    ):
        raise ValueError("PPO series config checkpointUpdates are invalid")
    model_config(value["architecture"])
    ppo = value["ppoConfig"]
    if not isinstance(ppo, dict):
        raise ValueError("PPO series config ppoConfig must be an object")
    PPOConfig(**ppo)
    initialization_key = next(iter(set(value) & initialization_keys))
    warm = value[initialization_key]
    if not isinstance(warm, dict) or set(warm) != {"path", "checkpointDigest"} or not all(
        isinstance(warm[name], str) and warm[name] for name in warm
    ):
        raise ValueError(f"PPO series config {initialization_key} is invalid")


def run_configured_series(
    *, output: str | Path, config_path: str | Path | None = None, qualifying: bool = False
) -> dict[str, Any]:
    config = load_series_config(config_path)
    training_root = Path(__file__).resolve().parents[2]
    warm_start = ppo_warm_start = None
    if "warmStart" in config:
        warm_start = training_root / config["warmStart"]["path"]
        warm_metadata, _ = load_checkpoint(warm_start)
        expected_digest = config["warmStart"]["checkpointDigest"]
    else:
        ppo_warm_start = training_root / config["ppoWarmStart"]["path"]
        warm_metadata, _ = load_ppo_checkpoint(ppo_warm_start)
        expected_digest = config["ppoWarmStart"]["checkpointDigest"]
    if warm_metadata["checkpointDigest"] != expected_digest:
        raise ValueError("PPO series config initialization checkpoint digest mismatch")
    return run_ppo_series(
        output=output,
        checkpoints=config["checkpointUpdates"],
        gate_id=config["gateId"],
        worlds=config["worlds"],
        rollout_steps=config["rolloutSteps"],
        training_seed=config["trainingSeed"],
        reward_mode=config["rewardMode"],
        max_decisions=config["maxEvaluationDecisions"],
        qualifying=qualifying,
        model_config=model_config(config["architecture"]),
        ppo_config=PPOConfig(**config["ppoConfig"]),
        warm_start=warm_start,
        ppo_warm_start=ppo_warm_start,
        series_config_digest=json_digest(config),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen SnowGym PPO series config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--qualifying", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_configured_series(
            output=args.output, config_path=args.config, qualifying=args.qualifying
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "mode": result["mode"],
        "gate": result["gateId"],
        "finalThresholdPassed": result["finalThresholdPassed"],
        "seriesConfigDigest": result["seriesConfigDigest"],
        "seriesDigest": result["seriesDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
