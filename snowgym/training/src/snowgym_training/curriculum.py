"""Validation for frozen PPO training/evaluation gates."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

CURRICULUM_FORMAT = "snowgym.ppo-curriculum.v0"


def default_curriculum_path() -> Path:
    return Path(
        str(files("snowgym_training").joinpath("configs/ppo_curriculum_v0.json"))
    )


def load_curriculum(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_curriculum_path()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load PPO curriculum {source}: {error}") from error
    validate_curriculum(value)
    return value


def validate_curriculum(value: Any) -> None:
    if not isinstance(value, dict) or value.get("format") != CURRICULUM_FORMAT:
        raise ValueError(f"curriculum format must be {CURRICULUM_FORMAT}")
    ranges = value.get("trainingSeedRanges")
    gates = value.get("gates")
    if not isinstance(ranges, dict) or not isinstance(gates, list) or not gates:
        raise ValueError("curriculum requires trainingSeedRanges and gates")
    training_seeds: set[int] = set()
    for gate_id, bounds in ranges.items():
        if (
            not isinstance(gate_id, str)
            or not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(item, int) for item in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("invalid training seed range")
        training_seeds.update(range(bounds[0], bounds[1] + 1))
    evaluation_seeds: set[int] = set()
    ids: set[str] = set()
    for gate in gates:
        if not isinstance(gate, dict) or not isinstance(gate.get("id"), str):
            raise ValueError("invalid curriculum gate")
        if gate["id"] in ids or gate["id"] not in ranges:
            raise ValueError("curriculum gate ids must be unique and have training ranges")
        ids.add(gate["id"])
        seeds = gate.get("evaluationSeeds")
        if not isinstance(seeds, list) or not seeds or not all(
            isinstance(seed, int) for seed in seeds
        ):
            raise ValueError("curriculum evaluation seeds are invalid")
        if evaluation_seeds.intersection(seeds) or training_seeds.intersection(seeds):
            raise ValueError("curriculum training/evaluation seeds overlap")
        evaluation_seeds.update(seeds)
        if not isinstance(gate.get("scenario"), dict):
            raise ValueError("curriculum gate scenario is invalid")
        for key in ("minimumWinRate", "minimumImprovementOverMaskedRandom"):
            threshold = gate.get(key)
            if not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
                raise ValueError(f"curriculum {key} must be in [0, 1]")
