"""Predeclared acceptance gate for plan-conditioned behavior cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .plan_ablation import load_plan_ablation_config
from .plan_evaluate import PLAN_EVALUATION_FORMAT
from .trajectory import json_digest

PLAN_QUALIFICATION_SPEC_FORMAT = "snowgym.plan-qualification-spec.v0"
PLAN_QUALIFICATION_RESULT_FORMAT = "snowgym.plan-qualification-result.v0"


def load_plan_qualification_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan qualification spec {source}: {error}") from error
    validate_plan_qualification_spec(value)
    return value


def validate_plan_qualification_spec(value: Any) -> None:
    required = {
        "format", "name", "trainingRollout", "evaluationRollout",
        "ablationConfigDigest", "thresholds",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"plan qualification spec must contain exactly {sorted(required)}")
    if value["format"] != PLAN_QUALIFICATION_SPEC_FORMAT:
        raise ValueError(f"plan qualification spec format must be {PLAN_QUALIFICATION_SPEC_FORMAT}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("plan qualification name must be non-empty")
    training = _rollout(value["trainingRollout"], "trainingRollout")
    evaluation = _rollout(value["evaluationRollout"], "evaluationRollout")
    for key in ("map", "blueUnits", "redUnits", "samples", "maxDecisions", "redDifficulty"):
        if key != "samples" and training[key] != evaluation[key]:
            raise ValueError(f"qualification rollout {key} must match")
    if training["environmentSeed"] == evaluation["environmentSeed"]:
        raise ValueError("qualification environment seeds must be disjoint")
    training_plans = set(range(training["planSeed"], training["planSeed"] + training["samples"]))
    evaluation_plans = set(
        range(evaluation["planSeed"], evaluation["planSeed"] + evaluation["samples"])
    )
    if training_plans & evaluation_plans:
        raise ValueError("qualification plan seed ranges must be disjoint")
    digest = value["ablationConfigDigest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("qualification ablationConfigDigest is invalid")
    thresholds = value["thresholds"]
    expected = {
        "maxConditionedTargetMse", "maxTargetMseRatio", "minTargetMseSwapDelta",
        "minTargetMeanAbsoluteDelta", "maxActionAccuracyDeficit",
        "noPlanMaxAbsoluteSensitivity",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected:
        raise ValueError(f"qualification thresholds must contain exactly {sorted(expected)}")
    if not all(
        isinstance(item, int | float) and not isinstance(item, bool) and item >= 0
        for item in thresholds.values()
    ):
        raise ValueError("qualification thresholds must be non-negative numbers")


def qualify_plan_evaluation(
    *, evaluation_path: str | Path, spec_path: str | Path, config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    spec = load_plan_qualification_spec(spec_path)
    config = load_plan_ablation_config(config_path)
    if json_digest(config) != spec["ablationConfigDigest"]:
        raise ValueError("qualification ablation config digest differs")
    try:
        evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan evaluation {evaluation_path}: {error}") from error
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    evaluation_body = {
        name: item for name, item in evaluation.items() if name != "evaluationDigest"
    } if isinstance(evaluation, dict) else {}
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("format") != PLAN_EVALUATION_FORMAT
        or evaluation.get("evaluationDigest") != json_digest(evaluation_body)
    ):
        raise ValueError("qualification evaluation digest is invalid")
    if not isinstance(metrics, dict):
        raise ValueError("qualification evaluation metrics are missing")
    no_plan = metrics.get("noPlan")
    conditioned = metrics.get("planConditioned")
    if not isinstance(no_plan, dict) or not isinstance(conditioned, dict):
        raise ValueError("qualification paired metrics are missing")
    threshold = spec["thresholds"]
    no_plan_mse = _metric(no_plan, "correctPlanTargetMse")
    conditioned_mse = _metric(conditioned, "correctPlanTargetMse")
    checks = {
        "conditionedTargetMse": conditioned_mse <= threshold["maxConditionedTargetMse"],
        "targetMseRatio": conditioned_mse <= no_plan_mse * threshold["maxTargetMseRatio"],
        "targetMseSwapDelta": _metric(conditioned, "counterfactualTargetMseDelta")
        >= threshold["minTargetMseSwapDelta"],
        "targetMeanAbsoluteDelta": _metric(
            conditioned, "counterfactualTargetMeanAbsoluteDelta"
        ) >= threshold["minTargetMeanAbsoluteDelta"],
        "actionAccuracyDeficit": _metric(no_plan, "actionAccuracy")
        - _metric(conditioned, "actionAccuracy")
        <= threshold["maxActionAccuracyDeficit"],
        "noPlanActionSensitivity": abs(_metric(no_plan, "counterfactualActionChangeRate"))
        <= threshold["noPlanMaxAbsoluteSensitivity"],
        "noPlanTargetSensitivity": abs(_metric(no_plan, "counterfactualTargetMeanAbsoluteDelta"))
        <= threshold["noPlanMaxAbsoluteSensitivity"],
        "noPlanTargetMseDelta": abs(_metric(no_plan, "counterfactualTargetMseDelta"))
        <= threshold["noPlanMaxAbsoluteSensitivity"],
    }
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite plan qualification {destination}")
    body = {
        "format": PLAN_QUALIFICATION_RESULT_FORMAT,
        "specDigest": json_digest(spec),
        "ablationConfigDigest": json_digest(config),
        "evaluationDigest": evaluation.get("evaluationDigest"),
        "checks": checks,
        "passed": all(checks.values()),
    }
    result = {**body, "qualificationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _rollout(value: Any, name: str) -> dict[str, Any]:
    required = {
        "map", "blueUnits", "redUnits", "environmentSeed", "planSeed",
        "samples", "maxDecisions", "redDifficulty",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{name} fields are invalid")
    if not isinstance(value["map"], str) or not value["map"]:
        raise ValueError(f"{name}.map must be non-empty")
    for key in ("blueUnits", "redUnits", "samples", "maxDecisions"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0:
            raise ValueError(f"{name}.{key} must be positive")
    for key in ("environmentSeed", "planSeed"):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise ValueError(f"{name}.{key} must be an integer")
    if value["redDifficulty"] not in {"easy", "normal", "hard"}:
        raise ValueError(f"{name}.redDifficulty is invalid")
    return value


def _metric(value: dict[str, Any], name: str) -> float:
    item = value.get(name)
    if not isinstance(item, int | float) or isinstance(item, bool):
        raise ValueError(f"qualification metric {name} is invalid")
    return float(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen SnowGym plan qualification gate")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = qualify_plan_evaluation(
            evaluation_path=args.evaluation,
            spec_path=args.spec,
            config_path=args.config,
            output=args.output,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
