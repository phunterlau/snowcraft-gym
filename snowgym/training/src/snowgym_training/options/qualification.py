"""Strict paired M7b qualification over frozen per-mission result records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .definitions import FROZEN_OPTION_SPECS
from ..trajectory import json_digest


def paired_bootstrap_lower_bound(
    differences: np.ndarray, *, seed: int, samples: int = 10_000
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap differences must be a finite vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025, method="linear"))


def qualify_m7b(value: dict[str, Any]) -> dict[str, Any]:
    """Require every frozen mission to pass; no aggregate substitution is allowed."""
    if not isinstance(value, dict) or value.get("format") != "snowgym.m7b-evaluation.v0":
        raise ValueError("M7b evaluation format is invalid")
    claimed_digest = value.get("evaluationDigest")
    source = {name: item for name, item in value.items() if name != "evaluationDigest"}
    if claimed_digest != json_digest(source):
        raise ValueError("M7b evaluation digest mismatch")
    for name in ("checkpointDigest", "sourceDigest", "protocolDigest"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
        ):
            raise ValueError(f"M7b evaluation {name} is invalid")
    missions = value.get("missions")
    if not isinstance(missions, dict) or set(missions) != set(FROZEN_OPTION_SPECS):
        raise ValueError("M7b evaluation must contain every frozen mission")
    inherited_lr = value.get("inheritedHeadLearningRate")
    new_lr = value.get("newModuleLearningRate")
    parameter_change = value.get("parameterL2Change")
    for name, number in {
        "inheritedHeadLearningRate": inherited_lr,
        "newModuleLearningRate": new_lr,
        "parameterL2Change": parameter_change,
    }.items():
        if not isinstance(number, int | float) or isinstance(number, bool) or not np.isfinite(number):
            raise ValueError(f"M7b evaluation {name} is invalid")
    global_gates = {
        "inheritedLearningRate": inherited_lr >= 1e-5,
        "newModuleLearningRate": new_lr >= 1e-4,
        "parameterChange": parameter_change > 0,
    }
    reports: dict[str, Any] = {}
    for mission_index, name in enumerate(FROZEN_OPTION_SPECS):
        conditions = missions[name]
        if not isinstance(conditions, dict) or set(conditions) != {
            "correct", "shuffled", "initializer"
        }:
            raise ValueError(f"M7b {name} conditions are invalid")
        records = {
            condition: _records(rows, name, condition)
            for condition, rows in conditions.items()
        }
        seeds = [row["seed"] for row in records["correct"]]
        if any([row["seed"] for row in records[key]] != seeds for key in records):
            raise ValueError(f"M7b {name} paired seeds are misaligned")
        correct_success = np.asarray(
            [row["success"] for row in records["correct"]], dtype=np.float64
        )
        shuffled_success = np.asarray(
            [row["success"] for row in records["shuffled"]], dtype=np.float64
        )
        initializer_success = np.asarray(
            [row["success"] for row in records["initializer"]], dtype=np.float64
        )
        progress_wins = np.mean(
            [
                left["progress"] > right["progress"]
                for left, right in zip(
                    records["correct"], records["shuffled"], strict=True
                )
            ]
        )
        correct_win = np.mean([row["physicalWin"] for row in records["correct"]])
        initializer_win = np.mean(
            [row["physicalWin"] for row in records["initializer"]]
        )
        total_actions = sum(row["totalActions"] for row in records["correct"])
        rejected = sum(row["rejectedActions"] for row in records["correct"])
        success_rate = float(correct_success.mean())
        improvement = float((correct_success - shuffled_success).mean())
        initializer_rate = float(initializer_success.mean())
        lower = paired_bootstrap_lower_bound(
            correct_success - shuffled_success,
            seed=730_000 + mission_index,
        )
        gates = {
            "success": success_rate >= 0.75,
            "shuffledImprovement": improvement >= 0.2,
            "pairedBootstrap": lower > 0,
            "pairedProgress": float(progress_wins) >= 0.7,
            "physicalRetention": float(correct_win) >= float(initializer_win) - 0.1,
            "rejectedActions": total_actions > 0 and rejected / total_actions < 0.001,
            "initializerImprovement": success_rate > initializer_rate,
        }
        reports[name] = {
            "successRate": success_rate,
            "shuffledSuccessRate": float(shuffled_success.mean()),
            "initializerSuccessRate": initializer_rate,
            "shuffledImprovement": improvement,
            "pairedBootstrapLower95": lower,
            "pairedProgressWinFraction": float(progress_wins),
            "physicalWinRate": float(correct_win),
            "initializerPhysicalWinRate": float(initializer_win),
            "rejectedActionRate": rejected / total_actions if total_actions else 1.0,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "format": "snowgym.m7b-qualification.v0",
        "checkpointDigest": value.get("checkpointDigest"),
        "sourceDigest": value.get("sourceDigest"),
        "protocolDigest": value.get("protocolDigest"),
        "evaluationDigest": claimed_digest,
        "globalGates": global_gates,
        "missions": reports,
        "passed": all(global_gates.values()) and all(
            report["passed"] for report in reports.values()
        ),
    }


def _records(value: Any, mission: str, condition: str) -> list[dict[str, Any]]:
    required = {
        "seed", "success", "progress", "physicalWin", "rejectedActions", "totalActions"
    }
    if not isinstance(value, list) or len(value) != 100:
        raise ValueError(f"M7b {mission}/{condition} must contain 100 results")
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"M7b {mission}/{condition} result fields are invalid")
        if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
            raise ValueError(f"M7b {mission}/{condition} seed is invalid")
        if not isinstance(row["success"], bool) or not isinstance(row["physicalWin"], bool):
            raise ValueError(f"M7b {mission}/{condition} booleans are invalid")
        if (
            not isinstance(row["progress"], int | float)
            or isinstance(row["progress"], bool)
            or not 0 <= row["progress"] <= 1
        ):
            raise ValueError(f"M7b {mission}/{condition} progress is invalid")
        for key in ("rejectedActions", "totalActions"):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(f"M7b {mission}/{condition} action count is invalid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite M7b qualification {destination}")
    try:
        value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load M7b evaluation: {error}") from error
    report = qualify_m7b(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
