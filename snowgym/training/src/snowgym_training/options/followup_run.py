"""Shared immutable runner for predeclared Engage R1 follow-up experiments."""

from __future__ import annotations

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


def run_engage_followup(
    config: dict[str, Any],
    *,
    reservoir_path: str | Path,
    output: str | Path,
    run_format: str,
    trajectory_format: str,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Engage follow-up {destination}")
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    if (
        reservoir.metadata["episodes"] != config["reservoirSeedCount"]
        or reservoir.metadata["samples"] <= 0
    ):
        raise ValueError("Engage follow-up reservoir does not match its configuration")
    reservoir_manifest = json.loads(
        Path(reservoir.metadata["manifestPath"]).read_text(encoding="utf-8")
    )
    if reservoir_manifest.get("seedPartition") != config["reservoirSeedPartition"]:
        raise ValueError("Engage follow-up reservoir uses the wrong seed partition")
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
            "format": trajectory_format,
            "selectionPolicy": config["selectionPolicy"],
            "retainedUpdates": config["retainedUpdates"],
            "summaries": summaries,
        }
        trajectory["trajectoryDigest"] = json_digest(trajectory)
        (root / "evaluation-trajectory.json").write_text(
            json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final_path = root / (
            f"development-evaluation-update-{config['finalUpdate']:06d}.json"
        )
        bootstrap = engage_bootstrap_report(
            json.loads(final_path.read_text(encoding="utf-8"))
        )
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
            "format": run_format,
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
