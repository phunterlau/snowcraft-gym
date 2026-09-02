"""Matched closed-loop evaluation for plan-conditioned executor checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from .plan_ablation import audit_plan_ablation
from .policy import TorchPolicy
from .trajectory import json_digest

SUITE_FORMAT = "snowgym.plan-closed-loop-suite.v0"
RESULT_FORMAT = "snowgym.plan-closed-loop-evaluation.v0"
POLICIES = ("noPlan", "planConditioned")


def load_suite(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load closed-loop suite {source}: {error}") from error
    if not isinstance(value, dict) or value.get("format") != SUITE_FORMAT:
        raise ValueError(f"closed-loop suite format must be {SUITE_FORMAT}")
    if set(value) != {"format", "name", "maxTeamUnits", "cases"}:
        raise ValueError("closed-loop suite fields are invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("closed-loop suite name is invalid")
    capacity = value["maxTeamUnits"]
    if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= 10:
        raise ValueError("maxTeamUnits must be an integer in [1, 10]")
    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("closed-loop suite requires at least two cases")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {
            "id", "seed", "scenario", "planId", "plan", "maxDecisions"
        }:
            raise ValueError(f"closed-loop case {index} fields are invalid")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError(f"closed-loop case {index} id is invalid or duplicate")
        ids.add(case["id"])
        for field in ("seed", "maxDecisions"):
            item = case[field]
            if not isinstance(item, int) or isinstance(item, bool) or item < (1 if field == "maxDecisions" else 0):
                raise ValueError(f"closed-loop case {index} {field} is invalid")
        if not isinstance(case["scenario"], dict) or not isinstance(case["plan"], dict):
            raise ValueError(f"closed-loop case {index} scenario/plan is invalid")
        if not isinstance(case["planId"], str) or not case["planId"]:
            raise ValueError(f"closed-loop case {index} planId is invalid")
    return value


def evaluate_closed_loop(
    *, ablation_path: str | Path, suite_path: str | Path, output: str | Path
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite closed-loop evaluation {destination}")
    ablation_root = Path(ablation_path)
    ablation = audit_plan_ablation(ablation_root)
    suite = load_suite(suite_path)
    policies = {
        name: TorchPolicy(ablation_root / ablation["runs"][name]["path"])
        for name in POLICIES
    }
    results: list[dict[str, Any]] = []
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(
            1, max_team_units=suite["maxTeamUnits"], client=client
        )
        for case in suite["cases"]:
            for policy_name in POLICIES:
                results.append(_run_case(environment, policies[policy_name], policy_name, case))
    comparisons = [_compare_case(results, case["id"]) for case in suite["cases"]]
    body = {
        "format": RESULT_FORMAT,
        "ablationResultDigest": ablation["resultDigest"],
        "suiteDigest": json_digest(suite),
        "suite": suite,
        "results": results,
        "comparisons": comparisons,
        "summary": {
            name: _summarize([item for item in results if item["policy"] == name])
            for name in POLICIES
        },
    }
    result = {**body, "evaluationDigest": json_digest(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_closed_loop(destination, ablation_root, suite_path)
    return result


def audit_closed_loop(
    path: str | Path, ablation_path: str | Path, suite_path: str | Path
) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load closed-loop evaluation {source}: {error}") from error
    if not isinstance(value, dict) or value.get("format") != RESULT_FORMAT:
        raise ValueError(f"closed-loop evaluation format must be {RESULT_FORMAT}")
    body = {key: item for key, item in value.items() if key != "evaluationDigest"}
    if value.get("evaluationDigest") != json_digest(body):
        raise ValueError("closed-loop evaluation digest mismatch")
    if value.get("ablationResultDigest") != audit_plan_ablation(ablation_path)["resultDigest"]:
        raise ValueError("closed-loop evaluation ablation provenance differs")
    if value.get("suiteDigest") != json_digest(load_suite(suite_path)):
        raise ValueError("closed-loop evaluation suite provenance differs")
    results = value.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("closed-loop evaluation results are invalid")
    if any(not isinstance(item, dict) or item.get("policy") not in POLICIES for item in results):
        raise ValueError("closed-loop evaluation policy is invalid")
    return value


def _run_case(
    environment: SnowGymBatchEnv,
    policy: TorchPolicy,
    policy_name: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    observation, infos = environment.reset([case["seed"]], [case["scenario"]])
    environment.activate_plans([case["planId"]], [case["plan"]])
    distances: list[float] = []
    first_target: list[list[float]] | None = None
    final_group_position: list[float] | None = None
    decisions = rejected = 0
    terminated = truncated = False
    info = infos[0]
    reward = 0.0
    while not (terminated or truncated) and decisions < case["maxDecisions"]:
        plan_tensors, _ = environment.plan_observations()
        plan_groups = plan_tensors["plan_groups"][0]
        plan_mask = plan_tensors["plan_group_mask"][0].astype(bool)
        active = plan_groups[plan_mask]
        distances.append(float(np.linalg.norm(active[:, 28:30] - active[:, 30:32], axis=1).mean()))
        final_group_position = active[:, 30:32].mean(axis=0).astype(float).tolist()
        single = {name: values[0] for name, values in observation.items()}
        if policy_name == "planConditioned":
            single.update({name: values[0] for name, values in plan_tensors.items()})
        action = policy.act(single)
        if first_target is None:
            first_target = np.asarray(action["target"], dtype=float).tolist()
        observation, rewards, terms, truncs, infos = environment.step(
            {name: values[None, ...] for name, values in action.items()}
        )
        reward = float(rewards[0])
        terminated, truncated = bool(terms[0]), bool(truncs[0])
        info = infos[0]
        rejected += sum(item.get("accepted") is False for item in info.get("actionResults", []))
        decisions += 1
    return {
        "caseId": case["id"],
        "policy": policy_name,
        "seed": case["seed"],
        "decisions": decisions,
        "winner": info.get("winner"),
        "canonicalReturn": reward,
        "terminated": terminated,
        "truncated": truncated,
        "decisionLimited": not (terminated or truncated),
        "blueAlive": info.get("blueAlive"),
        "redAlive": info.get("redAlive"),
        "rejectedActions": rejected,
        "initialObjectiveDistance": distances[0],
        "finalObjectiveDistance": distances[-1],
        "objectiveProgress": distances[0] - distances[-1],
        "meanObjectiveDistance": float(np.mean(distances)),
        "firstActionTarget": first_target,
        "finalMeanGroupPosition": final_group_position,
    }


def _compare_case(results: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    records = {item["policy"]: item for item in results if item["caseId"] == case_id}
    conditioned = records["planConditioned"]
    baseline = records["noPlan"]
    return {
        "caseId": case_id,
        "firstTargetMeanAbsoluteDelta": float(np.mean(np.abs(
            np.asarray(conditioned["firstActionTarget"]) - np.asarray(baseline["firstActionTarget"])
        ))),
        "finalGroupPositionDistance": float(np.linalg.norm(
            np.asarray(conditioned["finalMeanGroupPosition"])
            - np.asarray(baseline["finalMeanGroupPosition"])
        )),
        "objectiveProgressDelta": conditioned["objectiveProgress"] - baseline["objectiveProgress"],
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(results),
        "blueWins": sum(item["winner"] == "blue" for item in results),
        "winRate": sum(item["winner"] == "blue" for item in results) / len(results),
        "meanObjectiveProgress": float(np.mean([item["objectiveProgress"] for item in results])),
        "rejectedActions": sum(item["rejectedActions"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a matched plan executor closed loop")
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_closed_loop(
            ablation_path=args.ablation, suite_path=args.suite, output=args.output
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {key: result[key] for key in ("format", "evaluationDigest", "summary")}
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
