"""Audited acceptance gate for the frozen plan action-adapter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .plan_checkpoint_evaluate import FORMAT as OFFLINE_FORMAT
from .plan_closed_loop import RESULT_FORMAT as CLOSED_LOOP_FORMAT
from .trajectory import json_digest

SPEC_FORMAT = "snowgym.plan-action-adapter-qualification.v0"
RESULT_FORMAT = "snowgym.plan-action-adapter-qualification-result.v0"


def load_spec(path: str | Path) -> dict[str, Any]:
    value = _load_json(path, "qualification spec")
    required = {
        "format", "name", "trainingConfig", "trainingDatasetDigest",
        "evaluationDatasetDigest", "baselineEvaluationDigest", "offline", "closedLoop",
    }
    if set(value) != required:
        raise ValueError(f"qualification spec must contain exactly {sorted(required)}")
    if value["format"] != SPEC_FORMAT:
        raise ValueError(f"qualification spec format must be {SPEC_FORMAT}")
    for field in ("name", "trainingConfig"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"qualification {field} must be non-empty")
    for field in (
        "trainingDatasetDigest", "evaluationDatasetDigest", "baselineEvaluationDigest"
    ):
        if not isinstance(value[field], str) or not value[field].startswith("sha256:"):
            raise ValueError(f"qualification {field} is invalid")
    _thresholds(value["offline"], {
        "minimumActionAccuracy", "minimumFirstDecisionActionAccuracy",
        "maximumCorrectPlanTargetMse", "minimumCounterfactualActionChangeRate",
        "minimumCounterfactualTargetMeanAbsoluteDelta",
    }, "offline")
    _thresholds(value["closedLoop"], {
        "maximumRejectedActions", "minimumDirectBlueAlive", "minimumFlankBlueAlive",
        "minimumHoldDecisions", "minimumWithdrawDecisions", "maximumSupportRedAlive",
    }, "closedLoop")
    return value


def qualify_plan_action_adapter(
    *, spec_path: str | Path, checkpoint: str | Path, offline_path: str | Path,
    baseline_path: str | Path, closed_loop_path: str | Path, behaviors_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite qualification result {destination}")
    spec = load_spec(spec_path)
    metadata, _ = load_checkpoint(checkpoint)
    config_path = Path(spec_path).parent / spec["trainingConfig"]
    training_config = _load_json(config_path, "training config")
    if metadata.get("trainingConfig") != training_config:
        raise ValueError("checkpoint training config differs from qualification spec")
    if metadata.get("datasetManifestHash") != spec["trainingDatasetDigest"]:
        raise ValueError("checkpoint training dataset differs from qualification spec")

    baseline = _audited( baseline_path, OFFLINE_FORMAT, "evaluationDigest", "baseline evaluation")
    offline = _audited(offline_path, OFFLINE_FORMAT, "evaluationDigest", "offline evaluation")
    direct = _audited(
        closed_loop_path, CLOSED_LOOP_FORMAT, "evaluationDigest", "closed-loop evaluation"
    )
    behaviors = _audited(
        behaviors_path, CLOSED_LOOP_FORMAT, "evaluationDigest", "behavior evaluation"
    )
    if baseline["evaluationDigest"] != spec["baselineEvaluationDigest"]:
        raise ValueError("baseline evaluation differs from qualification spec")
    if offline.get("datasetDigest") != spec["evaluationDatasetDigest"]:
        raise ValueError("offline evaluation dataset differs from qualification spec")
    if offline.get("checkpointDigest") != metadata["checkpointDigest"]:
        raise ValueError("offline evaluation checkpoint differs")
    expected_checkpoint = {
        "checkpointDigest": metadata["checkpointDigest"],
        "stateDigest": metadata["stateDigest"],
    }
    for evaluation in (direct, behaviors):
        if evaluation.get("policyCheckpoints", {}).get("planConditioned") != expected_checkpoint:
            raise ValueError("closed-loop evaluation checkpoint differs")

    metrics = offline.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("offline evaluation metrics are missing")
    cases = {
        item.get("caseId"): item
        for evaluation in (direct, behaviors)
        for item in evaluation.get("results", [])
        if isinstance(item, dict) and item.get("policy") == "planConditioned"
    }
    required_cases = {
        "direct-focus", "left-flank-distributed", "hold-current",
        "withdraw-backfield", "main-with-reserve-support",
    }
    if set(cases) != required_cases:
        raise ValueError(f"conditioned closed-loop cases must be exactly {sorted(required_cases)}")
    rejected = sum(_number(item, "rejectedActions") for item in cases.values())
    offline_gate = spec["offline"]
    closed_gate = spec["closedLoop"]
    observed = {
        "offline": {
            "actionAccuracy": _number(metrics, "actionAccuracy"),
            "firstDecisionActionAccuracy": _number(metrics, "firstDecisionActionAccuracy"),
            "correctPlanTargetMse": _number(metrics, "correctPlanTargetMse"),
            "counterfactualActionChangeRate": _number(
                metrics, "counterfactualActionChangeRate"
            ),
            "counterfactualTargetMeanAbsoluteDelta": _number(
                metrics, "counterfactualTargetMeanAbsoluteDelta"
            ),
        },
        "closedLoop": {
            "rejectedActions": rejected,
            "directBlueAlive": _number(cases["direct-focus"], "blueAlive"),
            "flankBlueAlive": _number(cases["left-flank-distributed"], "blueAlive"),
            "holdDecisions": _number(cases["hold-current"], "decisions"),
            "withdrawDecisions": _number(cases["withdraw-backfield"], "decisions"),
            "supportRedAlive": _number(cases["main-with-reserve-support"], "redAlive"),
        },
    }
    checks = {
        "actionAccuracy": observed["offline"]["actionAccuracy"]
        >= offline_gate["minimumActionAccuracy"],
        "firstDecisionActionAccuracy": observed["offline"]["firstDecisionActionAccuracy"]
        >= offline_gate["minimumFirstDecisionActionAccuracy"],
        "correctPlanTargetMse": observed["offline"]["correctPlanTargetMse"]
        <= offline_gate["maximumCorrectPlanTargetMse"],
        "counterfactualActionChangeRate": observed["offline"]["counterfactualActionChangeRate"]
        >= offline_gate["minimumCounterfactualActionChangeRate"],
        "counterfactualTargetMeanAbsoluteDelta": observed["offline"][
            "counterfactualTargetMeanAbsoluteDelta"
        ] >= offline_gate["minimumCounterfactualTargetMeanAbsoluteDelta"],
        "rejectedActions": rejected <= closed_gate["maximumRejectedActions"],
        "directBlueAlive": observed["closedLoop"]["directBlueAlive"]
        >= closed_gate["minimumDirectBlueAlive"],
        "flankBlueAlive": observed["closedLoop"]["flankBlueAlive"]
        >= closed_gate["minimumFlankBlueAlive"],
        "holdDecisions": observed["closedLoop"]["holdDecisions"]
        >= closed_gate["minimumHoldDecisions"],
        "withdrawDecisions": observed["closedLoop"]["withdrawDecisions"]
        >= closed_gate["minimumWithdrawDecisions"],
        "supportRedAlive": observed["closedLoop"]["supportRedAlive"]
        <= closed_gate["maximumSupportRedAlive"],
    }
    body = {
        "format": RESULT_FORMAT,
        "specDigest": json_digest(spec),
        "checkpointDigest": metadata["checkpointDigest"],
        "checkpointStateDigest": metadata["stateDigest"],
        "inputEvaluationDigests": {
            "baseline": baseline["evaluationDigest"],
            "offline": offline["evaluationDigest"],
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


def _audited(path: str | Path, format_name: str, digest_key: str, name: str) -> dict[str, Any]:
    value = _load_json(path, name)
    if value.get("format") != format_name:
        raise ValueError(f"{name} format must be {format_name}")
    body = {key: item for key, item in value.items() if key != digest_key}
    if value.get(digest_key) != json_digest(body):
        raise ValueError(f"{name} digest mismatch")
    return value


def _thresholds(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"qualification {name} thresholds must be exactly {sorted(expected)}")
    if any(
        not isinstance(item, int | float) or isinstance(item, bool) or item < 0
        for item in value.values()
    ):
        raise ValueError(f"qualification {name} thresholds must be non-negative numbers")


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, int | float) or isinstance(item, bool):
        raise ValueError(f"qualification metric {key} is invalid")
    return float(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen plan action-adapter gate")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--closed-loop", type=Path, required=True)
    parser.add_argument("--behaviors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = qualify_plan_action_adapter(
            spec_path=args.spec, checkpoint=args.checkpoint, baseline_path=args.baseline,
            offline_path=args.offline, closed_loop_path=args.closed_loop,
            behaviors_path=args.behaviors, output=args.output,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
