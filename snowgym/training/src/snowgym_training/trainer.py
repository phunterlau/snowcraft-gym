"""Deterministic CPU behavior-cloning trainer."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from snowgym_client.encoding import ACTION_THROW

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
    optional = {
        "trainable", "counterfactualLossWeight", "counterfactualChangedActionWeight",
        "sampling", "roleBalancedLoss",
    }
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        raise ValueError(f"training config must contain {sorted(required)} and optional trainable")
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
    if value.get("trainable", "all") not in {"all", "plan-target-path", "plan-action-target-path"}:
        raise ValueError("trainable must be all, plan-target-path, or plan-action-target-path")
    counterfactual_weight = value.get("counterfactualLossWeight", 0)
    if (
        not isinstance(counterfactual_weight, int | float)
        or isinstance(counterfactual_weight, bool)
        or not 0 <= counterfactual_weight <= 10
    ):
        raise ValueError("counterfactualLossWeight must be in [0, 10]")
    changed_weight = value.get("counterfactualChangedActionWeight", 0)
    if (
        not isinstance(changed_weight, int | float)
        or isinstance(changed_weight, bool)
        or not 0 <= changed_weight <= 100
    ):
        raise ValueError("counterfactualChangedActionWeight must be in [0, 100]")
    if value.get("sampling", "transition-uniform") not in {
        "transition-uniform", "plan-mission-uniform"
    }:
        raise ValueError("sampling must be transition-uniform or plan-mission-uniform")
    if not isinstance(value.get("roleBalancedLoss", False), bool):
        raise ValueError("roleBalancedLoss must be boolean")


def train_behavior_clone(
    *,
    dataset_path: str | Path,
    output: str | Path,
    config: dict[str, Any],
    resume: str | Path | None = None,
    initialize: str | Path | None = None,
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
    if architecture.plan_role_conditioned and "plan_unit_roles" not in dataset.observation_fields:
        raise ValueError("plan role-conditioned training requires aligned unit roles")
    role_class_weights = (
        dataset.inverse_plan_role_weights()
        if config.get("roleBalancedLoss", False) else None
    )
    counterfactual_weight = float(config.get("counterfactualLossWeight", 0))
    changed_action_weight = float(config.get("counterfactualChangedActionWeight", 0))
    if (
        counterfactual_weight > 0 or changed_action_weight > 0
    ) and not dataset.counterfactual_plan_labels:
        raise ValueError("counterfactual training requires same-state plan labels")
    losses = loss_config(config["loss"])
    model = EntityPolicy(architecture).cpu()
    if resume is not None and initialize is not None:
        raise ValueError("resume and initialize are mutually exclusive")
    initialization: dict[str, Any] | None = None
    if initialize is not None:
        initial_metadata, initial_state = load_checkpoint(initialize)
        initial_architecture = initial_metadata.get("architecture")
        compatible_architecture = architecture.as_dict()
        compatible_architecture.pop("plan_action_adapter", None)
        compatible_architecture.pop("plan_role_conditioned", None)
        compatible_architecture.pop("plan_unit_directive_conditioned", None)
        compatible_architecture.pop("plan_directive_experts", None)
        if initial_architecture not in (architecture.as_dict(), compatible_architecture):
            raise ValueError("initializer architecture does not match training config")
        missing, unexpected = model.load_state_dict(initial_state["model"], strict=False)
        if unexpected or any(
            not name.startswith((
                "plan_action_adapter.", "plan_role_target_adapter.",
                "plan_action_experts.", "plan_target_experts.",
            ))
            for name in missing
        ):
            raise ValueError("initializer state is incompatible with training architecture")
        initialization = {
            "checkpointDigest": initial_metadata["checkpointDigest"],
            "stateDigest": initial_metadata["stateDigest"],
        }
    trainable_mode = config.get("trainable", "all")
    if trainable_mode in {"plan-target-path", "plan-action-target-path"}:
        if not (architecture.plan_conditioned and architecture.plan_target_only and architecture.separate_target_actor):
            raise ValueError("plan-target-path requires plan target-only separate-target architecture")
        prefixes = ("plan_encoder.", "target_actor.", "move_target_head.", "throw_target_head.", "power_head.")
        if trainable_mode == "plan-action-target-path":
            if not architecture.plan_action_adapter:
                raise ValueError("plan-action-target-path requires plan_action_adapter")
            prefixes += ("plan_action_adapter.",)
            if architecture.plan_role_conditioned:
                prefixes += ("plan_role_target_adapter.",)
            if architecture.plan_directive_experts:
                prefixes += ("plan_action_experts.", "plan_target_experts.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    optimizer_config = {
        "name": "Adam",
        "learningRate": float(config["learningRate"]),
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weightDecay": 0.0,
    }
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
        indices = (
            dataset.plan_mission_batch_indices(
                int(config["batchSize"]), int(config["seed"]), step
            )
            if config.get("sampling") == "plan-mission-uniform"
            else deterministic_batch_indices(
                len(dataset), int(config["batchSize"]), int(config["seed"]), step
            )
        )
        observation, action = dataset.batch(indices)
        primary_unit_weights = (
            observation["plan_unit_roles"].float() @ role_class_weights
            if role_class_weights is not None else None
        )
        optimizer.zero_grad(set_to_none=True)
        primary_prediction = model(observation)
        components = behavior_clone_loss(
            primary_prediction, action, observation, losses, primary_unit_weights
        )
        if counterfactual_weight > 0 or changed_action_weight > 0:
            counterfactual_observation = {
                **observation,
                "plan_groups": observation["counterfactual_plan_groups"],
                "plan_group_mask": observation["counterfactual_plan_group_mask"],
                **(
                    {
                        "plan_unit_roles": observation[
                            "counterfactual_plan_unit_roles"
                        ]
                    }
                    if architecture.plan_role_conditioned else {}
                ),
            }
            counterfactual_action = {
                field: action[f"counterfactual_{field}"]
                for field in ("action_type", "target", "power")
            }
            counterfactual_prediction = model(counterfactual_observation)
            counterfactual_unit_weights = (
                counterfactual_observation["plan_unit_roles"].float()
                @ role_class_weights
                if role_class_weights is not None else None
            )
            counterfactual = behavior_clone_loss(
                counterfactual_prediction,
                counterfactual_action,
                counterfactual_observation,
                losses,
                counterfactual_unit_weights,
            )
            components["total"] = components["total"] + (
                counterfactual_weight * counterfactual["total"]
            )
            components["counterfactual"] = counterfactual["total"]
            if changed_action_weight > 0:
                present = observation["ally_mask"].bool()
                primary_labels = action["action_type"].long()
                counterfactual_labels = counterfactual_action["action_type"].long()
                changed = present & (primary_labels != counterfactual_labels)
                if bool(changed.any()):
                    class_weights = torch.ones(
                        primary_prediction["action_logits"].shape[-1],
                        dtype=primary_prediction["action_logits"].dtype,
                    )
                    class_weights[ACTION_THROW] = losses.throw_action_weight
                    def changed_ce(
                        logits: torch.Tensor, labels: torch.Tensor,
                        unit_weights: torch.Tensor | None,
                    ) -> torch.Tensor:
                        if unit_weights is None:
                            return F.cross_entropy(
                                logits[changed], labels[changed], weight=class_weights
                            )
                        selected = unit_weights[changed]
                        raw = F.cross_entropy(
                            logits[changed], labels[changed], weight=class_weights,
                            reduction="none",
                        )
                        denominator = (
                            class_weights[labels[changed]] * selected
                        ).sum().clamp_min(torch.finfo(raw.dtype).eps)
                        return (raw * selected).sum() / denominator

                    changed_loss = 0.5 * (
                        changed_ce(
                            primary_prediction["action_logits"], primary_labels,
                            primary_unit_weights,
                        )
                        + changed_ce(
                            counterfactual_prediction["action_logits"],
                            counterfactual_labels, counterfactual_unit_weights,
                        )
                    )
                else:
                    changed_loss = components["total"].new_zeros(())
                components["total"] = (
                    components["total"] + changed_action_weight * changed_loss
                )
                components["counterfactualChangedAction"] = changed_loss
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
            "trainable": trainable_mode,
            **({"initialization": initialization} if initialization is not None else {}),
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
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = train_behavior_clone(
            dataset_path=args.dataset,
            output=args.output,
            config=load_training_config(args.config),
            resume=args.resume,
            initialize=args.initialize,
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
