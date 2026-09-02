"""Deterministic CPU behavior-cloning trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import TrajectoryDataset, deterministic_batch_indices, manifest_versions
from .loss import behavior_clone_loss, loss_config
from .model import EntityPolicy, model_config

TRAINING_CONFIG_FORMAT = "snowgym.bc-training-config.v0"


def default_training_config_path() -> Path:
    return Path(str(files("snowgym_training").joinpath("configs/bc_1v1_v0.json")))


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_training_config_path()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load training config {source}: {error}") from error
    validate_training_config(value)
    return value


def validate_training_config(value: Any) -> None:
    required = {
        "format",
        "name",
        "seed",
        "steps",
        "batchSize",
        "learningRate",
        "architecture",
        "loss",
        "evaluationSuite",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"training config must contain exactly {sorted(required)}")
    if value["format"] != TRAINING_CONFIG_FORMAT:
        raise ValueError(f"training config format must be {TRAINING_CONFIG_FORMAT}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("training config name must be non-empty")
    for key in ("seed", "steps", "batchSize"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0:
            raise ValueError(f"training config {key} must be a positive integer")
    rate = value["learningRate"]
    if not isinstance(rate, int | float) or isinstance(rate, bool) or not 0 < rate < 1:
        raise ValueError("learningRate must be in (0, 1)")
    model_config(value["architecture"])
    loss_config(value["loss"])
    if not isinstance(value["evaluationSuite"], str) or not value["evaluationSuite"]:
        raise ValueError("evaluationSuite must be non-empty")


def train_behavior_clone(
    *,
    dataset_path: str | Path,
    output: str | Path,
    config: dict[str, Any],
    resume: str | Path | None = None,
    target_step: int | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    validate_training_config(config)
    dataset = TrajectoryDataset(dataset_path)
    target = int(target_step if target_step is not None else config["steps"])
    if target <= 0:
        raise ValueError("target step must be positive")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(int(config["seed"]))
    architecture = model_config(config["architecture"])
    if architecture.plan_conditioned and "plan_groups" not in dataset.observation_fields:
        raise ValueError("plan-conditioned training requires an aligned plan dataset")
    losses = loss_config(config["loss"])
    model = EntityPolicy(architecture).cpu()
    optimizer_config = {
        "name": "Adam",
        "learningRate": float(config["learningRate"]),
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weightDecay": 0.0,
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learningRate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weightDecay"],
    )
    start_step = 0
    if resume is not None:
        previous, state = load_checkpoint(resume)
        assert_resume_compatible(previous, dataset, config)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(previous["step"])
    if start_step >= target:
        raise ValueError("target step must be greater than resumed checkpoint step")

    model.train()
    first_loss: float | None = None
    final_components: dict[str, float] = {}
    for step in range(start_step, target):
        indices = deterministic_batch_indices(
            len(dataset), int(config["batchSize"]), int(config["seed"]), step
        )
        observation, action = dataset.batch(indices)
        optimizer.zero_grad(set_to_none=True)
        components = behavior_clone_loss(model(observation), action, observation, losses)
        if not torch.isfinite(components["total"]):
            raise ValueError(f"non-finite training loss at step {step}")
        components["total"].backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError(f"non-finite gradient at step {step}")
        optimizer.step()
        scalar = float(components["total"].detach())
        if first_loss is None:
            first_loss = scalar
        final_components = {
            name: float(value.detach()) for name, value in components.items()
        }

    metadata = save_checkpoint(
        output,
        model=model,
        optimizer=optimizer,
        metadata={
            "gitCommit": git_commit or resolve_git_commit(),
            "datasetManifestHash": dataset.manifest["datasetDigest"],
            "versions": manifest_versions(dataset),
            "architecture": architecture.as_dict(),
            "optimizer": optimizer_config,
            "loss": losses.as_dict(),
            "trainingSeed": int(config["seed"]),
            "step": target,
            "evaluationSuite": config["evaluationSuite"],
            "trainingConfig": config,
            "trainingMetrics": {
                "startStep": start_step,
                "firstLoss": first_loss,
                "final": final_components,
            },
        },
    )
    return metadata


def assert_resume_compatible(
    checkpoint: dict[str, Any], dataset: TrajectoryDataset, config: dict[str, Any]
) -> None:
    expected = {
        "datasetManifestHash": dataset.manifest["datasetDigest"],
        "architecture": config["architecture"],
        "loss": {key: float(value) for key, value in config["loss"].items()},
        "trainingSeed": config["seed"],
        "evaluationSuite": config["evaluationSuite"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint {key} does not match training run")
    optimizer = checkpoint.get("optimizer")
    expected_optimizer = {
        "name": "Adam",
        "learningRate": float(config["learningRate"]),
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weightDecay": 0.0,
    }
    if optimizer != expected_optimizer:
        raise ValueError("resume checkpoint optimizer does not match training run")


def resolve_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot resolve git commit for checkpoint provenance") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deterministic SnowGym BC policy")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = train_behavior_clone(
            dataset_path=args.dataset,
            output=args.output,
            config=load_training_config(args.config),
            resume=args.resume,
            target_step=args.target_step,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "step": result["step"],
        "checkpointDigest": result["checkpointDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
