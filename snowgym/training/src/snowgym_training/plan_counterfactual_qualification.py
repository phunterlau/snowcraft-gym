"""Frozen gate for same-state plan-action supervision and closed-loop behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .plan_closed_loop import RESULT_FORMAT as CLOSED_LOOP_FORMAT
from .plan_counterfactual_evaluate import FORMAT as PAIRED_FORMAT
from .trajectory import json_digest

SPEC_FORMAT = "snowgym.plan-action-adapter-counterfactual-qualification.v1"
RESULT_FORMAT = "snowgym.plan-action-adapter-counterfactual-qualification-result.v1"


def load_spec(path: str | Path) -> dict[str, Any]:
    value = _load_json(path, "qualification spec")
    required = {
        "format", "name", "trainingConfig", "trainingDatasetDigest",
        "validationDatasetDigest", "evaluationDatasetDigest",
        "initializationCheckpointDigest", "trainingBaselineEvaluationDigest",
        "paired", "closedLoop",
    }
    if set(value) != required or value.get("format") != SPEC_FORMAT:
        raise ValueError("counterfactual qualification spec fields or format are invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("counterfactual qualification name is invalid")
    if not isinstance(value["trainingConfig"], str) or not value["trainingConfig"]:
        raise ValueError("counterfactual qualification trainingConfig is invalid")
    for field in (
        "trainingDatasetDigest", "validationDatasetDigest", "evaluationDatasetDigest",
        "initializationCheckpointDigest", "trainingBaselineEvaluationDigest",
    ):
        if not isinstance(value[field], str) or not value[field].startswith("sha256:"):
            raise ValueError(f"counterfactual qualification {field} is invalid")
    _thresholds(value["paired"], {
        "minimumTeacherActionChangeRate", "minimumPrimaryActionAccuracy",
        "minimumCounterfactualActionAccuracy", "minimumPredictedActionChangeRate",
        "minimumChangedTeacherPredictionRecall", "minimumChangedTeacherPairAccuracy",
        "maximumPrimaryTargetMse", "maximumCounterfactualTargetMse",
    }, "paired")
    _thresholds(value["closedLoop"], {
        "maximumRejectedActions", "minimumDirectBlueAlive", "minimumFlankBlueAlive",
        "minimumHoldDecisions", "minimumWithdrawDecisions", "maximumSupportRedAlive",
    }, "closedLoop")
    return value


def qualify(
    *, spec_path: str | Path, checkpoint: str | Path, paired_path: str | Path,
    closed_loop_path: str | Path, behaviors_path: str | Path, output: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite qualification result {destination}")
    spec = load_spec(spec_path)
    metadata, _ = load_checkpoint(checkpoint)
    config = _load_json(Path(spec_path).parent / spec["trainingConfig"], "training config")
    if metadata.get("trainingConfig") != config:
        raise ValueError("checkpoint training config differs from qualification spec")
    if metadata.get("datasetManifestHash") != spec["trainingDatasetDigest"]:
        raise ValueError("checkpoint training dataset differs from qualification spec")
    initialization = metadata.get("initialization")
    if (
        not isinstance(initialization, dict)
        or initialization.get("checkpointDigest") != spec["initializationCheckpointDigest"]
    ):
        raise ValueError("checkpoint initializer differs from qualification spec")
    paired = _audited(paired_path, PAIRED_FORMAT, "paired evaluation")
    if paired.get("datasetDigest") != spec["evaluationDatasetDigest"]:
        raise ValueError("paired evaluation dataset differs from qualification spec")
    if paired.get("checkpointDigest") != metadata["checkpointDigest"]:
        raise ValueError("paired evaluation checkpoint differs")
    direct = _audited(closed_loop_path, CLOSED_LOOP_FORMAT, "closed-loop evaluation")
    behaviors = _audited(behaviors_path, CLOSED_LOOP_FORMAT, "behavior evaluation")
    expected_checkpoint = {
        "checkpointDigest": metadata["checkpointDigest"],
        "stateDigest": metadata["stateDigest"],
    }
    for evaluation in (direct, behaviors):
        if evaluation.get("policyCheckpoints", {}).get("planConditioned") != expected_checkpoint:
            raise ValueError("closed-loop evaluation checkpoint differs")
    metrics = paired.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("paired evaluation metrics are missing")
    cases = {
        item.get("caseId"): item
        for evaluation in (direct, behaviors)
        for item in evaluation.get("results", [])
        if isinstance(item, dict) and item.get("policy") == "planConditioned"
    }
    expected_cases = {
        "direct-focus", "left-flank-distributed", "hold-current",
        "withdraw-backfield", "main-with-reserve-support",
    }
    if set(cases) != expected_cases:
        raise ValueError("closed-loop conditioned cases are incomplete")
    paired_gate = spec["paired"]
    closed_gate = spec["closedLoop"]
    rejected = sum(_number(item, "rejectedActions") for item in cases.values())
    observed = {
        "paired": {key: _number(metrics, key) for key in (
            "teacherActionChangeRate", "primaryActionAccuracy",
            "counterfactualActionAccuracy", "predictedActionChangeRate",
            "changedTeacherPredictionRecall", "changedTeacherPairAccuracy",
            "primaryTargetMse", "counterfactualTargetMse",
        )},
        "closedLoop": {
            "rejectedActions": rejected,
            "directBlueAlive": _number(cases["direct-focus"], "blueAlive"),
            "flankBlueAlive": _number(cases["left-flank-distributed"], "blueAlive"),
            "holdDecisions": _number(cases["hold-current"], "decisions"),
            "withdrawDecisions": _number(cases["withdraw-backfield"], "decisions"),
            "supportRedAlive": _number(cases["main-with-reserve-support"], "redAlive"),
        },
    }
    p = observed["paired"]
    c = observed["closedLoop"]
    checks = {
        "teacherActionChangeRate": p["teacherActionChangeRate"]
        >= paired_gate["minimumTeacherActionChangeRate"],
        "primaryActionAccuracy": p["primaryActionAccuracy"]
        >= paired_gate["minimumPrimaryActionAccuracy"],
        "counterfactualActionAccuracy": p["counterfactualActionAccuracy"]
        >= paired_gate["minimumCounterfactualActionAccuracy"],
        "predictedActionChangeRate": p["predictedActionChangeRate"]
        >= paired_gate["minimumPredictedActionChangeRate"],
        "changedTeacherPredictionRecall": p["changedTeacherPredictionRecall"]
        >= paired_gate["minimumChangedTeacherPredictionRecall"],
        "changedTeacherPairAccuracy": p["changedTeacherPairAccuracy"]
        >= paired_gate["minimumChangedTeacherPairAccuracy"],
        "primaryTargetMse": p["primaryTargetMse"] <= paired_gate["maximumPrimaryTargetMse"],
        "counterfactualTargetMse": p["counterfactualTargetMse"]
        <= paired_gate["maximumCounterfactualTargetMse"],
        "rejectedActions": rejected <= closed_gate["maximumRejectedActions"],
        "directBlueAlive": c["directBlueAlive"] >= closed_gate["minimumDirectBlueAlive"],
        "flankBlueAlive": c["flankBlueAlive"] >= closed_gate["minimumFlankBlueAlive"],
        "holdDecisions": c["holdDecisions"] >= closed_gate["minimumHoldDecisions"],
        "withdrawDecisions": c["withdrawDecisions"]
        >= closed_gate["minimumWithdrawDecisions"],
        "supportRedAlive": c["supportRedAlive"] <= closed_gate["maximumSupportRedAlive"],
    }
    body = {
        "format": RESULT_FORMAT,
        "specDigest": json_digest(spec),
        "checkpointDigest": metadata["checkpointDigest"],
        "checkpointStateDigest": metadata["stateDigest"],
        "inputEvaluationDigests": {
            "paired": paired["evaluationDigest"],
            "closedLoop": direct["evaluationDigest"],
            "behaviors": behaviors["evaluationDigest"],
        },
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result = {**body, "qualificationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load_json(path: str | Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _audited(path: str | Path, expected_format: str, name: str) -> dict[str, Any]:
    value = _load_json(path, name)
    if value.get("format") != expected_format:
        raise ValueError(f"{name} format is invalid")
    body = {key: item for key, item in value.items() if key != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError(f"{name} digest mismatch")
    return value


def _thresholds(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"counterfactual qualification {name} thresholds are invalid")
    if any(
        not isinstance(item, int | float) or isinstance(item, bool) or item < 0
        for item in value.values()
    ):
        raise ValueError(f"counterfactual qualification {name} thresholds are invalid")


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, int | float) or isinstance(item, bool):
        raise ValueError(f"qualification metric {key} is invalid")
    return float(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen counterfactual plan gate")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--closed-loop", type=Path, required=True)
    parser.add_argument("--behaviors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = qualify(
            spec_path=args.spec, checkpoint=args.checkpoint, paired_path=args.paired,
            closed_loop_path=args.closed_loop, behaviors_path=args.behaviors,
            output=args.output,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
