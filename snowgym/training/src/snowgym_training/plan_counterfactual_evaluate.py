"""Evaluate same-state primary/counterfactual plan labels across all transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from .checkpoint import load_checkpoint
from .data import TrajectoryDataset
from .model import EntityPolicy, model_config, select_action_target
from .trajectory import audit_dataset, json_digest

FORMAT = "snowgym.plan-counterfactual-evaluation.v0"


def evaluate_plan_counterfactual(
    *, checkpoint: str | Path, dataset_path: str | Path, output: str | Path
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual evaluation {destination}")
    dataset_summary = audit_dataset(dataset_path)
    dataset = TrajectoryDataset(dataset_path)
    if not dataset.counterfactual_plan_labels:
        raise ValueError("counterfactual evaluation requires same-state plan labels")
    metadata, state = load_checkpoint(checkpoint)
    model = EntityPolicy(model_config(metadata["architecture"])).cpu()
    model.load_state_dict(state["model"])
    model.eval()
    indices = torch.arange(len(dataset)).numpy()
    observation, action = dataset.batch(indices)
    alternate_observation = {
        **observation,
        "plan_groups": observation["counterfactual_plan_groups"],
        "plan_group_mask": observation["counterfactual_plan_group_mask"],
        **(
            {"plan_unit_roles": observation["counterfactual_plan_unit_roles"]}
            if "counterfactual_plan_unit_roles" in observation else {}
        ),
    }
    alternate_action = {
        field: action[f"counterfactual_{field}"]
        for field in ("action_type", "target", "power")
    }
    with torch.no_grad():
        primary = model(observation)
        alternate = model(alternate_observation)
    present = observation["ally_mask"].bool()
    primary_label = action["action_type"].long()
    alternate_label = alternate_action["action_type"].long()
    primary_prediction = primary["action_logits"].argmax(dim=-1)
    alternate_prediction = alternate["action_logits"].argmax(dim=-1)
    teacher_changed = present & (primary_label != alternate_label)
    predicted_changed = present & (primary_prediction != alternate_prediction)
    primary_target = _selected_target(primary, primary_label)
    alternate_target = _selected_target(alternate, alternate_label)
    primary_target_mask = present & (
        (primary_label == ACTION_MOVE) | (primary_label == ACTION_THROW)
    )
    alternate_target_mask = present & (
        (alternate_label == ACTION_MOVE) | (alternate_label == ACTION_THROW)
    )
    metrics = {
        "teacherActionChangeRate": _rate(teacher_changed, present),
        "predictedActionChangeRate": _rate(predicted_changed, present),
        "primaryActionAccuracy": _accuracy(primary_prediction, primary_label, present),
        "counterfactualActionAccuracy": _accuracy(
            alternate_prediction, alternate_label, present
        ),
        "primaryActionNll": _nll(primary["action_logits"], primary_label, present),
        "counterfactualActionNll": _nll(
            alternate["action_logits"], alternate_label, present
        ),
        "changedTeacherPredictionRecall": _rate(predicted_changed & teacher_changed, teacher_changed),
        "changedTeacherPairAccuracy": _rate(
            teacher_changed
            & (primary_prediction == primary_label)
            & (alternate_prediction == alternate_label),
            teacher_changed,
        ),
        "primaryTargetMse": _target_mse(
            primary_target, action["target"].float(), primary_target_mask
        ),
        "counterfactualTargetMse": _target_mse(
            alternate_target, alternate_action["target"].float(), alternate_target_mask
        ),
        "predictionTargetMeanAbsoluteDelta": float(
            torch.abs(primary_target - alternate_target)[present].mean()
        ),
        "teacherChangedUnitDecisions": int(teacher_changed.sum()),
        "presentUnitDecisions": int(present.sum()),
    }
    body = {
        "format": FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "checkpointStateDigest": metadata["stateDigest"],
        "datasetDigest": dataset_summary["datasetDigest"],
        "transitions": len(dataset),
        "metrics": metrics,
    }
    result = {**body, "evaluationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_plan_counterfactual_evaluation(destination, checkpoint, dataset_path)
    return result


def audit_plan_counterfactual_evaluation(
    path: str | Path, checkpoint: str | Path, dataset_path: str | Path
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load counterfactual evaluation: {error}") from error
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"counterfactual evaluation format must be {FORMAT}")
    body = {key: item for key, item in value.items() if key != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError("counterfactual evaluation digest mismatch")
    metadata, _ = load_checkpoint(checkpoint)
    if value.get("checkpointDigest") != metadata["checkpointDigest"]:
        raise ValueError("counterfactual evaluation checkpoint differs")
    if value.get("datasetDigest") != audit_dataset(dataset_path)["datasetDigest"]:
        raise ValueError("counterfactual evaluation dataset differs")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("presentUnitDecisions", 0) <= 0:
        raise ValueError("counterfactual evaluation metrics are invalid")
    return value


def _accuracy(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float((prediction[mask] == target[mask]).float().mean())


def _nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(F.cross_entropy(logits[mask], target[mask]))


def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.sum())
    return float(numerator.sum()) / count if count else 0.0


def _selected_target(prediction: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return (
        select_action_target(
            prediction.get("supervised_target_by_action", prediction["target_by_action"]),
            labels,
        )
        if "target_by_action" in prediction else prediction["target"]
    )


def _target_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(F.mse_loss(prediction[mask], target[mask])) if bool(mask.any()) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired same-state plan labels")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_plan_counterfactual(
            checkpoint=args.checkpoint, dataset_path=args.dataset, output=args.output
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result["metrics"])


if __name__ == "__main__":
    main()
