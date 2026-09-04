"""Run the frozen M7b-R1c BC-anchor-floor Engage experiment."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ..loss import LossConfig
from ..ppo import PPOConfig
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .bootstrap import engage_bootstrap_report
from .evaluate import evaluate_m7b_checkpoint
from .r1b import evaluation_summary
from .reservoir import file_digest, load_teacher_bc_reservoir
from .train import train_option_ppo

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "m7b_engage_r1c_bc_floor_v0.json"
)


def load_r1c_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Engage R1c configuration: {error}") from error
    required = {
        "format", "option", "intervention", "singleChange", "bcAnchorFloor",
        "referenceUpdate50StateDigest", "retainedUpdates", "finalUpdate",
        "selectionPolicy", "reservoirBcFraction", "reservoirSeedPartition",
        "reservoirSeedCount", "worlds", "rolloutSteps", "anchorTotalUpdates",
        "trainingSeed", "ppoConfig", "bcLossConfig", "bootstrapGates",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Engage R1c configuration fields are invalid")
    if (
        value["format"] != "snowgym.engage-r1c-config.v0"
        or value["option"] != "engage"
        or value["intervention"] != "successful-teacher-bc-reservoir"
        or value["singleChange"] != "bc-anchor-floor-at-update-50-weight"
        or value["bcAnchorFloor"] != 0.05
        or value["selectionPolicy"] != "final-update-only"
        or value["reservoirSeedPartition"] != "training"
        or value["retainedUpdates"] != [50, 75, 100]
        or value["finalUpdate"] != 100
    ):
        raise ValueError("Engage R1c configuration identity is invalid")
    ppo_keys = {field.name for field in fields(PPOConfig)}
    loss_keys = {field.name for field in fields(LossConfig)}
    if set(value["ppoConfig"]) != ppo_keys or set(value["bcLossConfig"]) != loss_keys:
        raise ValueError("Engage R1c optimizer configuration is incomplete")
    PPOConfig(**value["ppoConfig"])
    LossConfig(**value["bcLossConfig"])
    return value


def run_engage_r1c(
    *,
    reservoir_path: str | Path,
    output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Engage R1c run {destination}")
    config = load_r1c_config(config_path)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    if (
        reservoir.metadata["episodes"] != config["reservoirSeedCount"]
        or reservoir.metadata["samples"] <= 0
    ):
        raise ValueError("Engage R1c reservoir does not match the frozen configuration")
    reservoir_manifest = json.loads(
        Path(reservoir.metadata["manifestPath"]).read_text(encoding="utf-8")
    )
    if reservoir_manifest.get("seedPartition") != config["reservoirSeedPartition"]:
        raise ValueError("Engage R1c reservoir uses the wrong seed partition")
    commit = resolve_git_commit()
    ppo = PPOConfig(**config["ppoConfig"])
    bc = LossConfig(**config["bcLossConfig"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        checkpoints: dict[int, Path] = {}
        training_runs: dict[str, Any] = {}
        previous: Path | None = None
        for update in config["retainedUpdates"]:
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
                teacher_reservoir_path=reservoir_path,
                reservoir_bc_fraction=config["reservoirBcFraction"],
                bc_anchor_floor=config["bcAnchorFloor"],
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
            if (
                update == 50
                and run["checkpoint"]["stateDigest"]
                != config["referenceUpdate50StateDigest"]
            ):
                raise RuntimeError(
                    "R1c diverged before the BC floor became active at update 50"
                )
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
            "format": "snowgym.engage-r1c-trajectory.v0",
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
            "format": "snowgym.engage-r1c-run.v0",
            "gitCommit": commit,
            "config": config,
            "configDigest": json_digest(config),
            "reservoir": reservoir.metadata,
            "trainingRuns": training_runs,
            "evaluations": evaluations,
            "trajectoryDigest": trajectory["trajectoryDigest"],
            "bootstrapReportDigest": bootstrap["reportDigest"],
            "passed": bootstrap["passed"],
            "artifacts": {name: file_digest(root / name) for name in artifact_names},
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
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_engage_r1c(
        reservoir_path=args.reservoir,
        output=args.output,
        config_path=args.config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
