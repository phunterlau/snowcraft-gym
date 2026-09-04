"""Continue the frozen M7b-R1d trajectory to update 200."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

from ..loss import LossConfig
from ..ppo import PPOConfig
from .followup_run import run_engage_followup

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "m7b_engage_r1e_continue200_v0.json"
)


def load_r1e_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Engage R1e configuration: {error}") from error
    required = {
        "format", "option", "intervention", "singleChange",
        "sourceCheckpointDigest", "sourceUpdate", "bcAnchorFloor",
        "retainedUpdates", "finalUpdate", "selectionPolicy",
        "reservoirBcFraction", "reservoirSeedPartition", "reservoirSeedCount",
        "worlds", "rolloutSteps", "anchorTotalUpdates", "trainingSeed",
        "ppoConfig", "bcLossConfig", "bootstrapGates",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Engage R1e configuration fields are invalid")
    if (
        value["format"] != "snowgym.engage-r1e-config.v0"
        or value["option"] != "engage"
        or value["intervention"] != "successful-teacher-bc-reservoir"
        or value["singleChange"]
        != "continue-r1d-training-from-100-to-200-updates"
        or value["sourceUpdate"] != 100
        or value["bcAnchorFloor"] != 0.05
        or value["reservoirBcFraction"] != 0.9
        or value["selectionPolicy"] != "final-update-only"
        or value["reservoirSeedPartition"] != "training"
        or value["retainedUpdates"] != [100, 150, 200]
        or value["finalUpdate"] != 200
        or value["anchorTotalUpdates"] != 200
    ):
        raise ValueError("Engage R1e configuration identity is invalid")
    if set(value["ppoConfig"]) != {field.name for field in fields(PPOConfig)}:
        raise ValueError("Engage R1e PPO configuration is incomplete")
    if set(value["bcLossConfig"]) != {field.name for field in fields(LossConfig)}:
        raise ValueError("Engage R1e BC configuration is incomplete")
    PPOConfig(**value["ppoConfig"])
    LossConfig(**value["bcLossConfig"])
    return value


def run_engage_r1e(
    *,
    source_checkpoint: str | Path,
    reservoir_path: str | Path,
    output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    return run_engage_followup(
        load_r1e_config(config_path),
        source_checkpoint=source_checkpoint,
        reservoir_path=reservoir_path,
        output=output,
        run_format="snowgym.engage-r1e-run.v0",
        trajectory_format="snowgym.engage-r1e-trajectory.v0",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_engage_r1e(
        source_checkpoint=args.source_checkpoint,
        reservoir_path=args.reservoir,
        output=args.output,
        config_path=args.config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
