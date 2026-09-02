"""Matched behavior-cloning runs with and without aligned commander-plan input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .data import TrajectoryDataset
from .trainer import train_behavior_clone, validate_training_config
from .trajectory import json_digest

PLAN_ABLATION_CONFIG_FORMAT = "snowgym.plan-bc-ablation-config.v0"
PLAN_ABLATION_RESULT_FORMAT = "snowgym.plan-bc-ablation-result.v0"


def load_plan_ablation_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan ablation config {source}: {error}") from error
    validate_plan_ablation_config(value)
    return value


def validate_plan_ablation_config(value: Any) -> None:
    required = {
        "format", "name", "seed", "steps", "batchSize", "learningRate",
        "architecture", "loss", "evaluationSuite",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"plan ablation config must contain exactly {sorted(required)}")
    if value["format"] != PLAN_ABLATION_CONFIG_FORMAT:
        raise ValueError(f"plan ablation format must be {PLAN_ABLATION_CONFIG_FORMAT}")
    architecture = value.get("architecture")
    if not isinstance(architecture, dict) or "plan_conditioned" in architecture:
        raise ValueError("paired architecture must omit plan_conditioned")
    for conditioned in (False, True):
        validate_training_config(_training_config(value, conditioned))


def run_plan_ablation(
    *, dataset_path: str | Path, output: str | Path, config: dict[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    validate_plan_ablation_config(config)
    dataset = TrajectoryDataset(dataset_path)
    if "plan_groups" not in dataset.observation_fields:
        raise ValueError("plan ablation requires an aligned plan dataset")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plan ablation {destination}")
    destination.mkdir(parents=True)
    no_plan = train_behavior_clone(
        dataset_path=dataset_path,
        output=destination / "no-plan",
        config=_training_config(config, False),
        git_commit=git_commit,
    )
    conditioned = train_behavior_clone(
        dataset_path=dataset_path,
        output=destination / "plan-conditioned",
        config=_training_config(config, True),
        git_commit=git_commit,
    )
    body = {
        "format": PLAN_ABLATION_RESULT_FORMAT,
        "name": config["name"],
        "sourceConfigDigest": json_digest(config),
        "datasetManifestHash": dataset.manifest["datasetDigest"],
        "matchedBudget": {
            key: config[key]
            for key in ("seed", "steps", "batchSize", "learningRate", "loss")
        },
        "runs": {
            "noPlan": _run_record(no_plan, "no-plan"),
            "planConditioned": _run_record(conditioned, "plan-conditioned"),
        },
    }
    result = {**body, "resultDigest": json_digest(body)}
    (destination / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_plan_ablation(destination)
    return result


def audit_plan_ablation(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    try:
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan ablation result: {error}") from error
    if not isinstance(value, dict) or value.get("format") != PLAN_ABLATION_RESULT_FORMAT:
        raise ValueError(f"plan ablation result format must be {PLAN_ABLATION_RESULT_FORMAT}")
    claimed = value.get("resultDigest")
    body = {name: item for name, item in value.items() if name != "resultDigest"}
    if claimed != json_digest(body):
        raise ValueError("plan ablation result digest mismatch")
    runs = value.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"noPlan", "planConditioned"}:
        raise ValueError("plan ablation result runs are invalid")
    checkpoints = {
        name: load_checkpoint(root / record["path"])[0]
        for name, record in runs.items()
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if set(checkpoints) != set(runs):
        raise ValueError("plan ablation checkpoint records are invalid")
    no_plan = checkpoints["noPlan"]
    conditioned = checkpoints["planConditioned"]
    if no_plan["architecture"].get("plan_conditioned", False):
        raise ValueError("no-plan checkpoint unexpectedly uses plan input")
    if conditioned["architecture"].get("plan_conditioned") is not True:
        raise ValueError("conditioned checkpoint does not use plan input")
    for name in ("datasetManifestHash", "optimizer", "loss", "trainingSeed", "step"):
        if no_plan[name] != conditioned[name]:
            raise ValueError(f"plan ablation checkpoints differ in matched field {name}")
    if no_plan["datasetManifestHash"] != value.get("datasetManifestHash"):
        raise ValueError("plan ablation dataset provenance differs")
    for name, metadata in checkpoints.items():
        if runs[name].get("checkpointDigest") != metadata["checkpointDigest"]:
            raise ValueError(f"plan ablation {name} checkpoint digest differs")
    return value


def _training_config(value: dict[str, Any], conditioned: bool) -> dict[str, Any]:
    label = "plan-conditioned" if conditioned else "no-plan"
    return {
        "format": "snowgym.bc-training-config.v0",
        "name": f"{value['name']}-{label}",
        "seed": value["seed"],
        "steps": value["steps"],
        "batchSize": value["batchSize"],
        "learningRate": value["learningRate"],
        "architecture": {
            **value["architecture"],
            **({"plan_conditioned": True} if conditioned else {}),
        },
        "loss": value["loss"],
        "evaluationSuite": value["evaluationSuite"],
    }


def _run_record(metadata: dict[str, Any], path: str) -> dict[str, Any]:
    return {
        "path": path,
        "checkpointDigest": metadata["checkpointDigest"],
        "stateDigest": metadata["stateDigest"],
        "architecture": metadata["architecture"],
        "trainingMetrics": metadata["trainingMetrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matched SnowGym plan-input ablation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_plan_ablation(
            dataset_path=args.dataset,
            output=args.output,
            config=load_plan_ablation_config(args.config),
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "resultDigest": result["resultDigest"],
        "runs": {
            name: record["checkpointDigest"] for name, record in result["runs"].items()
        },
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
