"""Audit and select a counterfactual changed-action development sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .plan_closed_loop import RESULT_FORMAT as CLOSED_FORMAT
from .plan_counterfactual_evaluate import FORMAT as PAIRED_FORMAT
from .trainer import load_training_config
from .trajectory import json_digest

SPEC_FORMAT = "snowgym.plan-action-adapter-weight-sweep.v0"
RESULT_FORMAT = "snowgym.plan-action-adapter-weight-sweep-result.v0"


def summarize_sweep(
    *, spec_path: str | Path, runs_dir: str | Path, evaluations_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite sweep result {destination}")
    spec = _load_json(spec_path)
    _validate_spec(spec)
    config_root = Path(spec_path).parent
    entries = []
    reference: dict[str, Any] | None = None
    for config_name in spec["trainingConfigs"]:
        config = load_training_config(config_root / config_name)
        comparable = {
            key: value for key, value in config.items()
            if key not in {"name", "counterfactualChangedActionWeight"}
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("sweep training configs differ beyond name and changed-action weight")
        weight = float(config["counterfactualChangedActionWeight"])
        label = f"weight{int(weight)}"
        checkpoint_path = Path(runs_dir) / f"plan_action_adapter_{label}_dev"
        metadata, _ = load_checkpoint(checkpoint_path)
        if metadata.get("trainingConfig") != config:
            raise ValueError(f"{label} checkpoint training config differs")
        if metadata.get("datasetManifestHash") != spec["trainingDatasetDigest"]:
            raise ValueError(f"{label} checkpoint dataset differs")
        if metadata.get("initialization", {}).get("checkpointDigest") != spec[
            "initializationCheckpointDigest"
        ]:
            raise ValueError(f"{label} checkpoint initializer differs")
        paired = _audited(
            Path(evaluations_dir) / f"plan_action_adapter_{label}_dev_validation.json",
            PAIRED_FORMAT,
        )
        closed = _audited(
            Path(evaluations_dir) / f"plan_action_adapter_{label}_dev_closed_loop.json",
            CLOSED_FORMAT,
        )
        behaviors = _audited(
            Path(evaluations_dir) / f"plan_action_adapter_{label}_dev_behaviors.json",
            CLOSED_FORMAT,
        )
        if paired.get("datasetDigest") != spec["validationDatasetDigest"]:
            raise ValueError(f"{label} paired dataset differs")
        expected_checkpoint = {
            "checkpointDigest": metadata["checkpointDigest"],
            "stateDigest": metadata["stateDigest"],
        }
        if paired.get("checkpointDigest") != metadata["checkpointDigest"]:
            raise ValueError(f"{label} paired checkpoint differs")
        for evaluation in (closed, behaviors):
            if evaluation.get("policyCheckpoints", {}).get("planConditioned") != expected_checkpoint:
                raise ValueError(f"{label} closed-loop checkpoint differs")
        metrics = paired["metrics"]
        cases = {
            item["caseId"]: item
            for evaluation in (closed, behaviors)
            for item in evaluation["results"]
            if item.get("policy") == "planConditioned"
        }
        observed = {
            "primaryActionAccuracy": metrics["primaryActionAccuracy"],
            "counterfactualActionAccuracy": metrics["counterfactualActionAccuracy"],
            "changedTeacherPredictionRecall": metrics["changedTeacherPredictionRecall"],
            "changedTeacherPairAccuracy": metrics["changedTeacherPairAccuracy"],
            "predictedActionChangeRate": metrics["predictedActionChangeRate"],
            "directBlueAlive": cases["direct-focus"]["blueAlive"],
            "flankBlueAlive": cases["left-flank-distributed"]["blueAlive"],
            "holdDecisions": cases["hold-current"]["decisions"],
            "withdrawDecisions": cases["withdraw-backfield"]["decisions"],
            "supportRedAlive": cases["main-with-reserve-support"]["redAlive"],
        }
        checks = _checks(observed, spec["developmentChecks"])
        entries.append({
            "label": label,
            "weight": weight,
            "checkpointDigest": metadata["checkpointDigest"],
            "evaluationDigests": {
                "paired": paired["evaluationDigest"],
                "closedLoop": closed["evaluationDigest"],
                "behaviors": behaviors["evaluationDigest"],
            },
            "observed": observed,
            "checks": checks,
            "checksPassed": sum(checks.values()),
        })
    selected = max(
        entries,
        key=lambda item: (
            item["checksPassed"],
            item["observed"]["changedTeacherPairAccuracy"],
            -item["weight"],
        ),
    )
    body = {
        "format": RESULT_FORMAT,
        "specDigest": json_digest(spec),
        "selection": spec["selection"],
        "entries": entries,
        "selected": selected["label"],
        "selectedCheckpointDigest": selected["checkpointDigest"],
    }
    result = {**body, "resultDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _checks(observed: dict[str, Any], gate: dict[str, Any]) -> dict[str, bool]:
    return {
        "primaryActionAccuracy": observed["primaryActionAccuracy"]
        >= gate["minimumPrimaryActionAccuracy"],
        "counterfactualActionAccuracy": observed["counterfactualActionAccuracy"]
        >= gate["minimumCounterfactualActionAccuracy"],
        "changedTeacherPredictionRecall": observed["changedTeacherPredictionRecall"]
        >= gate["minimumChangedTeacherPredictionRecall"],
        "changedTeacherPairAccuracy": observed["changedTeacherPairAccuracy"]
        >= gate["minimumChangedTeacherPairAccuracy"],
        "minimumPredictedActionChangeRate": observed["predictedActionChangeRate"]
        >= gate["minimumPredictedActionChangeRate"],
        "maximumPredictedActionChangeRate": observed["predictedActionChangeRate"]
        <= gate["maximumPredictedActionChangeRate"],
        "directBlueAlive": observed["directBlueAlive"] >= gate["minimumDirectBlueAlive"],
        "flankBlueAlive": observed["flankBlueAlive"] >= gate["minimumFlankBlueAlive"],
        "holdDecisions": observed["holdDecisions"] >= gate["minimumHoldDecisions"],
        "withdrawDecisions": observed["withdrawDecisions"]
        >= gate["minimumWithdrawDecisions"],
        "supportRedAlive": observed["supportRedAlive"] <= gate["maximumSupportRedAlive"],
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load sweep JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("sweep JSON must be an object")
    return value


def _audited(path: str | Path, expected_format: str) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("format") != expected_format:
        raise ValueError(f"evaluation format differs: {path}")
    body = {key: item for key, item in value.items() if key != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError(f"evaluation digest mismatch: {path}")
    return value


def _validate_spec(value: dict[str, Any]) -> None:
    required = {
        "format", "name", "trainingDatasetDigest", "validationDatasetDigest",
        "initializationCheckpointDigest", "trainingConfigs", "developmentChecks",
        "selection",
    }
    if set(value) != required or value.get("format") != SPEC_FORMAT:
        raise ValueError("weight-sweep spec fields or format are invalid")
    if value.get("selection") != "most-checks-then-pair-accuracy-then-lower-weight":
        raise ValueError("weight-sweep selection rule is invalid")
    if not isinstance(value.get("trainingConfigs"), list) or len(value["trainingConfigs"]) < 2:
        raise ValueError("weight-sweep requires at least two training configs")
    if not isinstance(value.get("developmentChecks"), dict):
        raise ValueError("weight-sweep development checks are invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a changed-action weight sweep")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = summarize_sweep(
            spec_path=args.spec, runs_dir=args.runs_dir,
            evaluations_dir=args.evaluations_dir, output=args.output,
        )
    except (FileExistsError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
