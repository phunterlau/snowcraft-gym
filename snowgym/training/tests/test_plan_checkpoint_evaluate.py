from __future__ import annotations

import json
from pathlib import Path

import pytest

from snowgym_training.plan_checkpoint_evaluate import (
    audit_plan_checkpoint_evaluation,
    evaluate_plan_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "training" / "runs" / "plan_bc_ablation_qual_v1" / "plan-conditioned"
DATASET = ROOT / "training" / "artifacts" / "plan-dagger-v0-evaluation"


def test_plan_checkpoint_evaluation_is_audited(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    result = evaluate_plan_checkpoint(checkpoint=CHECKPOINT, dataset_path=DATASET, output=path)
    assert result["episodes"] == 5
    assert result["transitions"] == 1210
    assert 0 <= result["metrics"]["actionAccuracy"] <= 1
    assert audit_plan_checkpoint_evaluation(path, CHECKPOINT, DATASET) == result

    result["metrics"]["actionAccuracy"] = 2
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_plan_checkpoint_evaluation(path, CHECKPOINT, DATASET)
