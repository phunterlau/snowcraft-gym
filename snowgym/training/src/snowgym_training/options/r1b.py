"""Run the frozen M7b-R1b Stage-1-hold Engage experiment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from ..loss import LossConfig
from ..ppo import PPOConfig
from ..ppo_checkpoint import load_ppo_checkpoint, normalized_ppo_config
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .bootstrap import engage_bootstrap_report
from .evaluate import CONDITIONS, evaluate_m7b_checkpoint
from .reservoir import file_digest, load_teacher_bc_reservoir
from .train import train_option_ppo

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "m7b_engage_r1b_stage1_hold_v0.json"
)


def load_r1b_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Engage R1b configuration: {error}") from error
    required = {
        "format", "option", "intervention", "singleChange",
        "sourceCheckpointDigest", "sourceUpdate", "retainedUpdates",
        "finalUpdate", "selectionPolicy", "reservoirBcFraction",
        "reservoirSeedPartition", "reservoirSeedCount", "worlds",
        "rolloutSteps", "anchorTotalUpdates", "trainingSeed", "ppoConfig",
        "bcLossConfig", "bootstrapGates",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Engage R1b configuration fields are invalid")
    if (
        value["format"] != "snowgym.engage-r1b-config.v0"
        or value["option"] != "engage"
        or value["intervention"] != "successful-teacher-bc-reservoir"
        or value["singleChange"] != "retain-stage-1-freeze-through-update-100"
        or value["selectionPolicy"] != "final-update-only"
        or value["reservoirSeedPartition"] != "training"
    ):
        raise ValueError("Engage R1b configuration identity is invalid")
    retained = value["retainedUpdates"]
    if (
        retained != [value["sourceUpdate"], 75, value["finalUpdate"]]
        or value["sourceUpdate"] != 50
        or value["finalUpdate"] != 100
        or retained != sorted(set(retained))
    ):
        raise ValueError("Engage R1b retained update schedule is invalid")
    ppo_keys = {field.name for field in fields(PPOConfig)}
    loss_keys = {field.name for field in fields(LossConfig)}
    if set(value["ppoConfig"]) != ppo_keys or set(value["bcLossConfig"]) != loss_keys:
        raise ValueError("Engage R1b optimizer configuration is incomplete")
    PPOConfig(**value["ppoConfig"])
    LossConfig(**value["bcLossConfig"])
    return value


def validate_r1b_inputs(
    config: dict[str, Any],
    source_checkpoint: str | Path,
    reservoir_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    source, _ = load_ppo_checkpoint(source_checkpoint)
    if source["checkpointDigest"] != config["sourceCheckpointDigest"]:
        raise ValueError("Engage R1b source checkpoint digest changed")
    if source["updateIndex"] != config["sourceUpdate"]:
        raise ValueError("Engage R1b source checkpoint update changed")
    if normalized_ppo_config(source["ppoConfig"]) != config["ppoConfig"]:
        raise ValueError("Engage R1b source PPO configuration changed")
    collector = source["collectorConfig"]
    expected_collector = {
        "worlds": config["worlds"],
        "rolloutSteps": config["rolloutSteps"],
        "option": "engage",
        "stage": 1,
        "anchorTotalUpdates": config["anchorTotalUpdates"],
        "reservoirBcFraction": config["reservoirBcFraction"],
    }
    for name, expected in expected_collector.items():
        if collector.get(name) != expected:
            raise ValueError(f"Engage R1b source collector {name} changed")
    reservoir_metadata = collector.get("teacherReservoir")
    if not isinstance(reservoir_metadata, dict):
        raise ValueError("Engage R1b source has no teacher reservoir")
    canonical_path = Path(str(reservoir_metadata.get("path", "")))
    if canonical_path.resolve() != Path(reservoir_path).resolve():
        raise ValueError("Engage R1b reservoir path does not match the source checkpoint")
    reservoir = load_teacher_bc_reservoir(canonical_path)
    if reservoir.metadata != reservoir_metadata:
        raise ValueError("Engage R1b reservoir metadata changed")
    if (
        reservoir.metadata["episodes"] != config["reservoirSeedCount"]
        or reservoir.metadata["samples"] <= 0
    ):
        raise ValueError("Engage R1b reservoir does not match the frozen configuration")
    reservoir_manifest = json.loads(
        Path(reservoir.metadata["manifestPath"]).read_text(encoding="utf-8")
    )
    if reservoir_manifest.get("seedPartition") != config["reservoirSeedPartition"]:
        raise ValueError("Engage R1b reservoir uses the wrong seed partition")
    return source, canonical_path


def evaluation_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = value["missions"]["engage"][condition]
        total_actions = sum(int(row["totalActions"]) for row in rows)
        summary[condition] = {
            "successRate": float(np.mean([row["success"] for row in rows])),
            "meanProgress": float(np.mean([row["progress"] for row in rows])),
            "contactRate": float(np.mean([
                row["firstContactDecision"] is not None for row in rows
            ])),
            "hitRate": float(np.mean([
                row["firstHitDecision"] is not None for row in rows
            ])),
            "physicalWinRate": float(np.mean([row["physicalWin"] for row in rows])),
            "rejectedActionRate": (
                sum(int(row["rejectedActions"]) for row in rows) / total_actions
                if total_actions else 1.0
            ),
        }
    return summary


def run_engage_r1b(
    *,
    source_checkpoint: str | Path,
    reservoir_path: str | Path,
    output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Engage R1b run {destination}")
    config = load_r1b_config(config_path)
    source, canonical_reservoir = validate_r1b_inputs(
        config, source_checkpoint, reservoir_path
    )
    commit = resolve_git_commit()
    ppo = PPOConfig(**config["ppoConfig"])
    bc = LossConfig(**config["bcLossConfig"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        checkpoints: dict[int, Path] = {
            config["sourceUpdate"]: Path(source_checkpoint)
        }
        training_runs: dict[str, Any] = {}
        previous = Path(source_checkpoint)
        for update in config["retainedUpdates"][1:]:
            run = train_option_ppo(
                output=root / f"update-{update:06d}",
                option="engage",
                worlds=config["worlds"],
                rollout_steps=config["rolloutSteps"],
                target_updates=update,
                anchor_total_updates=config["anchorTotalUpdates"],
                stage=1,
                training_seed=config["trainingSeed"],
                resume=previous,
                ppo_config=ppo,
                loss_config=bc,
                git_commit=commit,
                teacher_reservoir_path=canonical_reservoir,
                reservoir_bc_fraction=config["reservoirBcFraction"],
            )
            checkpoint = root / f"update-{update:06d}" / "checkpoint"
            checkpoints[update] = checkpoint
            previous = checkpoint
            training_runs[str(update)] = {
                "startUpdate": run["startUpdate"],
                "targetUpdates": run["targetUpdates"],
                "environmentSteps": run["environmentSteps"],
                "checkpointDigest": run["checkpoint"]["checkpointDigest"],
                "stateDigest": run["checkpoint"]["stateDigest"],
            }
        evaluations: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        for update in config["retainedUpdates"]:
            name = f"development-evaluation-update-{update:06d}.json"
            evaluation = evaluate_m7b_checkpoint(
                checkpoints[update],
                output=root / name,
                split="development",
                options=("engage",),
            )
            evaluations[str(update)] = {
                "checkpointDigest": evaluation["checkpointDigest"],
                "evaluationDigest": evaluation["evaluationDigest"],
            }
            summaries[str(update)] = evaluation_summary(evaluation)
        trajectory = {
            "format": "snowgym.engage-r1b-trajectory.v0",
            "selectionPolicy": config["selectionPolicy"],
            "retainedUpdates": config["retainedUpdates"],
            "summaries": summaries,
        }
        trajectory["trajectoryDigest"] = json_digest(trajectory)
        (root / "evaluation-trajectory.json").write_text(
            json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final_evaluation = json.loads(
            (root / f"development-evaluation-update-{config['finalUpdate']:06d}.json")
            .read_text(encoding="utf-8")
        )
        bootstrap = engage_bootstrap_report(final_evaluation)
        (root / "bootstrap-report.json").write_text(
            json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_names = [
            *(f"development-evaluation-update-{update:06d}.json"
              for update in config["retainedUpdates"]),
            "evaluation-trajectory.json",
            "bootstrap-report.json",
        ]
        manifest = {
            "format": "snowgym.engage-r1b-run.v0",
            "gitCommit": commit,
            "config": config,
            "configDigest": json_digest(config),
            "sourceCheckpoint": {
                "checkpointDigest": source["checkpointDigest"],
                "stateDigest": source["stateDigest"],
                "updateIndex": source["updateIndex"],
            },
            "reservoir": source["collectorConfig"]["teacherReservoir"],
            "trainingRuns": training_runs,
            "evaluations": evaluations,
            "trajectoryDigest": trajectory["trajectoryDigest"],
            "bootstrapReportDigest": bootstrap["reportDigest"],
            "passed": bootstrap["passed"],
            "artifacts": {
                name: file_digest(root / name) for name in artifact_names
            },
        }
        manifest["manifestDigest"] = json_digest(manifest)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        root.replace(destination)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_engage_r1b(
        source_checkpoint=args.source_checkpoint,
        reservoir_path=args.reservoir,
        output=args.output,
        config_path=args.config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
