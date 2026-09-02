"""Audited offline plan metrics for one checkpoint and one aligned dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .data import TrajectoryDataset
from .model import EntityPolicy, model_config
from .plan_evaluate import _first_transition_indices, _load_array, _metrics
from .trajectory import audit_dataset, json_digest

FORMAT = "snowgym.plan-checkpoint-evaluation.v0"


def evaluate_plan_checkpoint(
    *, checkpoint: str | Path, dataset_path: str | Path, output: str | Path
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plan checkpoint evaluation {destination}")
    dataset_summary = audit_dataset(dataset_path)
    dataset = TrajectoryDataset(dataset_path)
    if "plan_groups" not in dataset.observation_fields:
        raise ValueError("plan checkpoint evaluation requires aligned plan tensors")
    metadata, state = load_checkpoint(checkpoint)
    model = EntityPolicy(model_config(metadata["architecture"])).cpu()
    model.load_state_dict(state["model"])
    model.eval()
    episode_index = _load_array(dataset.path, dataset.manifest, "episode_index")
    first_indices = _first_transition_indices(episode_index)
    if len(first_indices) < 2:
        raise ValueError("plan checkpoint evaluation requires at least two episodes")
    body = {
        "format": FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "checkpointStateDigest": metadata["stateDigest"],
        "datasetDigest": dataset_summary["datasetDigest"],
        "episodes": len(first_indices),
        "transitions": len(dataset),
        "metrics": _metrics(model, dataset, first_indices),
    }
    result = {**body, "evaluationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_plan_checkpoint_evaluation(destination, checkpoint, dataset_path)
    return result


def audit_plan_checkpoint_evaluation(
    path: str | Path, checkpoint: str | Path, dataset_path: str | Path
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan checkpoint evaluation: {error}") from error
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError(f"plan checkpoint evaluation format must be {FORMAT}")
    body = {key: item for key, item in value.items() if key != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError("plan checkpoint evaluation digest mismatch")
    metadata, _ = load_checkpoint(checkpoint)
    if value.get("checkpointDigest") != metadata["checkpointDigest"]:
        raise ValueError("plan checkpoint evaluation checkpoint differs")
    if value.get("datasetDigest") != audit_dataset(dataset_path)["datasetDigest"]:
        raise ValueError("plan checkpoint evaluation dataset differs")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("plan checkpoint evaluation metrics are invalid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one plan-conditioned checkpoint offline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_plan_checkpoint(
            checkpoint=args.checkpoint, dataset_path=args.dataset, output=args.output
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result["metrics"])


if __name__ == "__main__":
    main()
