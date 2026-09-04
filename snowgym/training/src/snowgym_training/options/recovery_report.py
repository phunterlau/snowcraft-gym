"""Summarize M7b-R0 evidence and select one Engage bootstrap intervention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ..trajectory import json_digest
from ..trainer import resolve_git_commit

FORMAT = "snowgym.engage.recovery-report.v0"


def audit_artifact_manifest(directory: str | Path, manifest_name: str) -> dict[str, Any]:
    root = Path(directory)
    try:
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read artifact manifest: {error}") from error
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifact manifest has no artifacts")
    for relative, expected in artifacts.items():
        path = root / relative
        try:
            actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"cannot read archived artifact {relative}: {error}") from error
        if actual != expected:
            raise ValueError(f"archived artifact digest mismatch: {relative}")
    return manifest


def build_recovery_report(
    matrix_path: str | Path,
    diagnostics_directory: str | Path,
    gradients_directory: str | Path,
    *,
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite recovery report {destination}")
    audit_artifact_manifest(diagnostics_directory, "manifest.json")
    audit_artifact_manifest(gradients_directory, "manifest.json")
    matrix = load_digested_json(Path(matrix_path), "artifactDigest")
    distribution = load_digested_json(
        Path(diagnostics_directory) / "distribution_report.json", "reportDigest"
    )
    components = load_digested_json(
        Path(gradients_directory) / "gradient_components.json", "artifactDigest"
    )
    with (Path(gradients_directory) / "gradient_cosines.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        cosine_rows = list(csv.DictReader(stream))
    value = select_recovery_intervention(matrix, distribution, components, cosine_rows)
    value.update(
        {
            "format": FORMAT,
            "matrixDigest": matrix["artifactDigest"],
            "distributionDigest": distribution["reportDigest"],
            "gradientDigest": components["artifactDigest"],
            "implementationGitCommit": resolve_git_commit(),
        }
    )
    value["reportDigest"] = json_digest(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def select_recovery_intervention(
    matrix: dict[str, Any],
    distribution: dict[str, Any],
    components: dict[str, Any],
    cosine_rows: list[dict[str, str]],
) -> dict[str, Any]:
    summary = matrix["summary"]
    learner = summary["learner"]
    teacher = summary["teacher"]
    teacher_move = summary["teacher-move"]
    teacher_action = summary["teacher-action"]
    interaction = summary["teacher-action-move"]
    goal = summary["goal-anchor"]
    if teacher["successRate"] < 0.75:
        return {
            "selectedIntervention": None,
            "attribution": "inconclusive-teacher-achievability",
            "reason": "The production teacher did not establish the frozen Engage task.",
        }
    interaction_hit_gain = interaction["hitRate"] - max(
        teacher_move["hitRate"], teacher_action["hitRate"]
    )
    coverage = distribution["coverage"]
    successful_support_gap = coverage.get("d1OutsideD2P95Fraction", 0.0)
    deterministic_support_gap = coverage["d3OutsideD2P95Fraction"]
    relevant = {
        "ppo-actor", "action-bc", "move-target-bc", "throw-target-bc", "power-bc",
        "categorical-initializer-kl", "move-initializer-anchor",
        "throw-initializer-anchor", "power-initializer-anchor",
    }
    conflicts = sorted(
        (
            {
                "parameterGroup": row["parameterGroup"],
                "firstComponent": row["firstComponent"],
                "secondComponent": row["secondComponent"],
                "cosine": float(row["cosine"]),
            }
            for row in cosine_rows
            if row.get("cosine")
            and row["firstComponent"] in relevant
            and row["secondComponent"] in relevant
            and float(row["cosine"]) <= -0.25
        ),
        key=lambda row: row["cosine"],
    )
    evidence = {
        "learnerContactRate": learner["contactRate"],
        "learnerHitRate": learner["hitRate"],
        "teacherMoveContactRate": teacher_move["contactRate"],
        "teacherActionHitRate": teacher_action["hitRate"],
        "teacherActionMoveHitRate": interaction["hitRate"],
        "teacherSuccessRate": teacher["successRate"],
        "goalAnchorContactRate": goal["contactRate"],
        "interactionHitGain": interaction_hit_gain,
        "successfulTeacherOutsideStochasticP95Fraction": successful_support_gap,
        "deterministicOutsideStochasticP95Fraction": deterministic_support_gap,
        "currentGlobalClipScale": components["currentGlobalClipScale"],
        "strongGradientConflicts": conflicts,
    }
    if interaction_hit_gain >= 0.2 and successful_support_gap >= 0.5:
        selection = "successful-teacher-bc-reservoir"
        attribution = "action-move-throw-interaction-with-successful-state-coverage-gap"
        reason = (
            "Teacher action plus movement restores hits, only the full teacher completes "
            "Engage, and successful teacher states lie outside the learner-state support."
        )
    elif goal["contactRate"] - learner["contactRate"] >= 0.2:
        selection = "objective-anchored-move-residual"
        attribution = "move-target-representation"
        reason = "The grounded objective anchor causally restores contact."
    elif deterministic_support_gap >= 0.5:
        selection = "persistent-low-scale-move-exploration"
        attribution = "stochastic-deterministic-state-distribution-gap"
        reason = "The deterministic policy leaves the stochastic learner-state support."
    elif conflicts:
        selection = "component-specific-anchor-redesign"
        attribution = "loss-gradient-conflict"
        reason = "BC/PPO and initializer anchors have strongly opposed gradients."
    else:
        selection = None
        attribution = "inconclusive"
        reason = "No predeclared intervention-selection condition passed."
    return {
        "selectedIntervention": selection,
        "attribution": attribution,
        "reason": reason,
        "evidence": evidence,
        "constraints": {
            "ppoRolloutsRemainOnPolicy": True,
            "teacherReservoirUsedOnlyForBc": True,
            "benchmarkUnchanged": True,
            "qualificationSeedsUsed": False,
            "otherLearningInterventionsDeferred": True,
        },
    }


def load_digested_json(path: Path, digest_field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get(digest_field), str):
        raise ValueError(f"{path} has no {digest_field}")
    source = {name: item for name, item in value.items() if name != digest_field}
    if value[digest_field] != json_digest(source):
        raise ValueError(f"{path} digest mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--gradients", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_recovery_report(
        args.matrix, args.diagnostics, args.gradients, output=args.output
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
