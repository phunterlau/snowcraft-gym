from __future__ import annotations

import json
from pathlib import Path

import pytest

from snowgym_training.plan_closed_loop import (
    audit_closed_loop,
    evaluate_closed_loop,
    load_suite,
)


ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1"
SUITE = ROOT / "training" / "src" / "snowgym_training" / "configs" / "plan_closed_loop_v0.json"


def test_frozen_closed_loop_suite_is_valid() -> None:
    suite = load_suite(SUITE)
    assert suite["name"] == "plan-target-only-6v6-approach-v0"
    assert [case["id"] for case in suite["cases"]] == [
        "direct-focus", "left-flank-distributed"
    ]
    assert suite["cases"][0]["seed"] == suite["cases"][1]["seed"]


def test_closed_loop_suite_rejects_duplicate_cases(tmp_path: Path) -> None:
    value = json.loads(SUITE.read_text(encoding="utf-8"))
    value["cases"][1]["id"] = value["cases"][0]["id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(path)


def test_closed_loop_evaluator_runs_matched_real_worlds(tmp_path: Path) -> None:
    value = json.loads(SUITE.read_text(encoding="utf-8"))
    for case in value["cases"]:
        case["maxDecisions"] = 1
    suite = tmp_path / "smoke.json"
    suite.write_text(json.dumps(value), encoding="utf-8")
    result = evaluate_closed_loop(
        ablation_path=ABLATION, suite_path=suite, output=tmp_path / "result.json"
    )
    assert result["format"] == "snowgym.plan-closed-loop-evaluation.v0"
    assert len(result["results"]) == 4
    assert all(item["decisions"] == 1 for item in result["results"])
    assert all(item["rejectedActions"] == 0 for item in result["results"])
    assert all(item["firstTargetMeanAbsoluteDelta"] > 0 for item in result["comparisons"])
    assert audit_closed_loop(tmp_path / "result.json", ABLATION, suite) == result

    result["summary"]["noPlan"]["blueWins"] = 99
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_closed_loop(tmp_path / "result.json", ABLATION, suite)
