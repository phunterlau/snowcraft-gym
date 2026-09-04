"""Run the frozen M7b-R1 Engage teacher-reservoir experiment."""

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
from .reservoir import file_digest, load_teacher_bc_reservoir
from .train import train_option_ppo

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m7b_engage_r1_v0.json"


def load_r1_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load Engage R1 configuration: {error}") from error
    required = {
        "format", "option", "intervention", "reservoirBcFraction",
        "reservoirSeedPartition", "reservoirSeedCount", "worlds", "rolloutSteps",
        "anchorTotalUpdates", "stage1TargetUpdates", "stage2TargetUpdates",
        "trainingSeed", "ppoConfig", "bcLossConfig", "bootstrapGates",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Engage R1 configuration fields are invalid")
    if (
        value["format"] != "snowgym.engage-r1-config.v0"
        or value["option"] != "engage"
        or value["intervention"] != "successful-teacher-bc-reservoir"
        or value["reservoirSeedPartition"] != "training"
    ):
        raise ValueError("Engage R1 configuration identity is invalid")
    ppo_keys = {field.name for field in fields(PPOConfig)}
    loss_keys = {field.name for field in fields(LossConfig)}
    if set(value["ppoConfig"]) != ppo_keys or set(value["bcLossConfig"]) != loss_keys:
        raise ValueError("Engage R1 optimizer configuration is incomplete")
    PPOConfig(**value["ppoConfig"])
    LossConfig(**value["bcLossConfig"])
    return value


def run_engage_r1(
    *,
    reservoir_path: str | Path,
    output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Engage R1 run {destination}")
    config = load_r1_config(config_path)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    if (
        reservoir.metadata["episodes"] != config["reservoirSeedCount"]
        or reservoir.metadata["samples"] <= 0
    ):
        raise ValueError("Engage R1 reservoir does not match the frozen configuration")
    manifest_path = Path(reservoir_path).parent / "manifest.json"
    reservoir_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reservoir_manifest.get("seedPartition") != config["reservoirSeedPartition"]:
        raise ValueError("Engage R1 reservoir uses the wrong seed partition")
    commit = resolve_git_commit()
    ppo = PPOConfig(**config["ppoConfig"])
    bc = LossConfig(**config["bcLossConfig"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        common = {
            "option": "engage",
            "worlds": config["worlds"],
            "rollout_steps": config["rolloutSteps"],
            "anchor_total_updates": config["anchorTotalUpdates"],
            "training_seed": config["trainingSeed"],
            "ppo_config": ppo,
            "loss_config": bc,
            "git_commit": commit,
            "teacher_reservoir_path": reservoir_path,
            "reservoir_bc_fraction": config["reservoirBcFraction"],
        }
        stage1 = train_option_ppo(
            output=root / "stage1",
            target_updates=config["stage1TargetUpdates"],
            stage=1,
            **common,
        )
        stage2 = train_option_ppo(
            output=root / "stage2",
            target_updates=config["stage2TargetUpdates"],
            stage=2,
            ppo_warm_start=root / "stage1" / "checkpoint",
            **common,
        )
        evaluation = evaluate_m7b_checkpoint(
            root / "stage2" / "checkpoint",
            output=root / "development-evaluation.json",
            split="development",
            options=("engage",),
        )
        bootstrap = engage_bootstrap_report(evaluation)
        (root / "bootstrap-report.json").write_text(
            json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "format": "snowgym.engage-r1-run.v0",
            "gitCommit": commit,
            "config": config,
            "configDigest": json_digest(config),
            "reservoir": reservoir.metadata,
            "stage1CheckpointDigest": stage1["checkpoint"]["checkpointDigest"],
            "stage2CheckpointDigest": stage2["checkpoint"]["checkpointDigest"],
            "developmentEvaluationDigest": evaluation["evaluationDigest"],
            "bootstrapReportDigest": bootstrap["reportDigest"],
            "passed": bootstrap["passed"],
            "artifacts": {
                name: file_digest(root / name)
                for name in ("development-evaluation.json", "bootstrap-report.json")
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
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_engage_r1(
        reservoir_path=args.reservoir,
        output=args.output,
        config_path=args.config,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
