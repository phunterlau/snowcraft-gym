"""Held-out same-state counterfactual metrics for matched plan-input ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data import TrajectoryDataset
from .model import EntityPolicy, model_config
from .plan_ablation import audit_plan_ablation
from .trajectory import audit_dataset, json_digest

PLAN_EVALUATION_FORMAT = "snowgym.plan-counterfactual-evaluation.v0"


def evaluate_plan_ablation(
    *, ablation_path: str | Path, dataset_path: str | Path, output: str | Path,
) -> dict[str, Any]:
    ablation_root = Path(ablation_path)
    ablation = audit_plan_ablation(ablation_root)
    audit_dataset(dataset_path)
    dataset = TrajectoryDataset(dataset_path)
    if "plan_groups" not in dataset.observation_fields:
        raise ValueError("plan evaluation requires an aligned plan dataset")
    episode_index = _load_array(dataset.path, dataset.manifest, "episode_index")
    first_indices = _first_transition_indices(episode_index)
    if len(first_indices) < 2:
        raise ValueError("counterfactual evaluation requires at least two plan episodes")

    metrics: dict[str, Any] = {}
    for name, record in ablation["runs"].items():
        metadata, state = load_checkpoint(ablation_root / record["path"])
        model = EntityPolicy(model_config(metadata["architecture"])).cpu()
        model.load_state_dict(state["model"])
        model.eval()
        metrics[name] = _metrics(model, dataset, first_indices)

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plan evaluation {destination}")
    body = {
        "format": PLAN_EVALUATION_FORMAT,
        "ablationResultDigest": ablation["resultDigest"],
        "evaluationDatasetDigest": dataset.manifest["datasetDigest"],
        "episodes": len(first_indices),
        "transitions": len(dataset),
        "metrics": metrics,
    }
    result = {**body, "evaluationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_plan_evaluation(destination, ablation_root, dataset.path)
    return result


def audit_plan_evaluation(
    path: str | Path, ablation_path: str | Path, dataset_path: str | Path
) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan evaluation: {error}") from error
    if not isinstance(value, dict) or value.get("format") != PLAN_EVALUATION_FORMAT:
        raise ValueError(f"plan evaluation format must be {PLAN_EVALUATION_FORMAT}")
    body = {name: item for name, item in value.items() if name != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError("plan evaluation digest mismatch")
    ablation = audit_plan_ablation(ablation_path)
    dataset_summary = audit_dataset(dataset_path)
    if value.get("ablationResultDigest") != ablation["resultDigest"]:
        raise ValueError("plan evaluation ablation provenance differs")
    if value.get("evaluationDatasetDigest") != dataset_summary["datasetDigest"]:
        raise ValueError("plan evaluation dataset provenance differs")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"noPlan", "planConditioned"}:
        raise ValueError("plan evaluation metrics are invalid")
    for run_metrics in metrics.values():
        if not isinstance(run_metrics, dict) or not all(
            isinstance(item, int | float) and not isinstance(item, bool) and np.isfinite(item)
            for item in run_metrics.values()
        ):
            raise ValueError("plan evaluation metric values are invalid")
    return value


def _metrics(
    model: EntityPolicy, dataset: TrajectoryDataset, first_indices: np.ndarray
) -> dict[str, float]:
    all_observation, all_action = dataset.batch(np.arange(len(dataset)))
    with torch.no_grad():
        all_prediction = model(all_observation)
    ally_mask = all_observation["ally_mask"].bool()
    predicted = all_prediction["action_logits"].argmax(dim=-1)
    action_accuracy = _masked_accuracy(predicted, all_action["action_type"], ally_mask)

    observation, action = dataset.batch(first_indices)
    shuffled = dict(observation)
    shuffled["plan_groups"] = observation["plan_groups"].roll(1, dims=0)
    shuffled["plan_group_mask"] = observation["plan_group_mask"].roll(1, dims=0)
    with torch.no_grad():
        correct = model(observation)
        counterfactual = model(shuffled)
    first_mask = observation["ally_mask"].bool()
    correct_nll = _masked_nll(correct["action_logits"], action["action_type"], first_mask)
    shuffled_nll = _masked_nll(
        counterfactual["action_logits"], action["action_type"], first_mask
    )
    target_mask = first_mask & (
        (action["action_type"] == 1) | (action["action_type"] == 2)
    )
    correct_target_mse = _masked_target_mse(
        correct["target"], action["target"], target_mask
    )
    shuffled_target_mse = _masked_target_mse(
        counterfactual["target"], action["target"], target_mask
    )
    changed = (correct["action_logits"].argmax(dim=-1) != counterfactual["action_logits"].argmax(dim=-1))
    sensitivity = float(changed[first_mask].float().mean())
    target_sensitivity = float(
        torch.abs(correct["target"] - counterfactual["target"])[first_mask].mean()
    )
    return {
        "actionAccuracy": action_accuracy,
        "firstDecisionActionAccuracy": _masked_accuracy(
            correct["action_logits"].argmax(dim=-1), action["action_type"], first_mask
        ),
        "correctPlanActionNll": correct_nll,
        "shuffledPlanActionNll": shuffled_nll,
        "counterfactualNllDelta": shuffled_nll - correct_nll,
        "correctPlanTargetMse": correct_target_mse,
        "shuffledPlanTargetMse": shuffled_target_mse,
        "counterfactualTargetMseDelta": shuffled_target_mse - correct_target_mse,
        "counterfactualActionChangeRate": sensitivity,
        "counterfactualTargetMeanAbsoluteDelta": target_sensitivity,
    }


def _masked_accuracy(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float((predicted[mask] == target[mask]).float().mean())


def _masked_nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = -torch.log_softmax(logits.float(), dim=-1).gather(
        -1, target.long().unsqueeze(-1)
    ).squeeze(-1)
    return float(selected[mask].mean())


def _masked_target_mse(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(torch.square(predicted.float() - target.float())[mask].mean())


def _first_transition_indices(episode_index: np.ndarray) -> np.ndarray:
    if episode_index.ndim != 1:
        raise ValueError("episode_index must be one-dimensional")
    _, indices = np.unique(episode_index, return_index=True)
    return np.sort(indices).astype(np.int64)


def _load_array(path: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    chunks = []
    for shard in manifest["shards"]:
        with np.load(path / shard["path"], allow_pickle=False) as archive:
            if name not in archive.files:
                raise ValueError(f"plan evaluation field missing from shard: {name}")
            chunks.append(np.array(archive[name], copy=True))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SnowGym plan-input ablation")
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_plan_ablation(
            ablation_path=args.ablation, dataset_path=args.dataset, output=args.output
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
