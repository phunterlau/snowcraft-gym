"""Run the frozen M7b-R1d 90-percent teacher-reservoir experiment."""

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
    / "m7b_engage_r1d_reservoir90_v0.json"
)


def load_r1d_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Engage R1d configuration: {error}") from error
    required = {
        "format", "option", "intervention", "singleChange", "bcAnchorFloor",
        "retainedUpdates", "finalUpdate", "selectionPolicy",
        "reservoirBcFraction", "reservoirSeedPartition", "reservoirSeedCount",
        "worlds", "rolloutSteps", "anchorTotalUpdates", "trainingSeed",
        "ppoConfig", "bcLossConfig", "bootstrapGates",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Engage R1d configuration fields are invalid")
    if (
        value["format"] != "snowgym.engage-r1d-config.v0"
        or value["option"] != "engage"
        or value["intervention"] != "successful-teacher-bc-reservoir"
        or value["singleChange"]
        != "increase-reservoir-bc-fraction-from-0.5-to-0.9"
        or value["bcAnchorFloor"] != 0.05
        or value["reservoirBcFraction"] != 0.9
        or value["selectionPolicy"] != "final-update-only"
        or value["reservoirSeedPartition"] != "training"
        or value["retainedUpdates"] != [50, 75, 100]
        or value["finalUpdate"] != 100
    ):
        raise ValueError("Engage R1d configuration identity is invalid")
    if set(value["ppoConfig"]) != {field.name for field in fields(PPOConfig)}:
        raise ValueError("Engage R1d PPO configuration is incomplete")
    if set(value["bcLossConfig"]) != {field.name for field in fields(LossConfig)}:
        raise ValueError("Engage R1d BC configuration is incomplete")
    PPOConfig(**value["ppoConfig"])
    LossConfig(**value["bcLossConfig"])
    return value


def run_engage_r1d(
    *,
    reservoir_path: str | Path,
    output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    return run_engage_followup(
        load_r1d_config(config_path),
        reservoir_path=reservoir_path,
        output=output,
        run_format="snowgym.engage-r1d-run.v0",
        trajectory_format="snowgym.engage-r1d-trajectory.v0",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_engage_r1d(
        reservoir_path=args.reservoir,
        output=args.output,
        config_path=args.config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
