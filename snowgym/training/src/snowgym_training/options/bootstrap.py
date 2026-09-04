"""Apply the frozen 40-seed M7b-R1 Engage bootstrap gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..trajectory import json_digest


def engage_bootstrap_report(value: dict[str, Any]) -> dict[str, Any]:
    claimed = value.get("evaluationDigest") if isinstance(value, dict) else None
    source = {name: item for name, item in value.items() if name != "evaluationDigest"}
    if value.get("format") != "snowgym.m7b-development-evaluation.v0" or claimed != json_digest(source):
        raise ValueError("Engage bootstrap evaluation contract is invalid")
    if value.get("evaluatedOptions") != ["engage"]:
        raise ValueError("Engage bootstrap gate requires only the Engage mission")
    conditions = value.get("missions", {}).get("engage")
    if not isinstance(conditions, dict) or set(conditions) != {
        "correct", "shuffled", "initializer"
    }:
        raise ValueError("Engage bootstrap conditions are invalid")
    if any(not isinstance(rows, list) or len(rows) != 40 for rows in conditions.values()):
        raise ValueError("Engage bootstrap requires 40 paired development seeds")
    seeds = [row.get("seed") for row in conditions["correct"]]
    if any([row.get("seed") for row in conditions[name]] != seeds for name in conditions):
        raise ValueError("Engage bootstrap seeds are not paired")
    correct = conditions["correct"]
    success = float(np.mean([row["success"] for row in correct]))
    shuffled = float(np.mean([row["success"] for row in conditions["shuffled"]]))
    initializer = float(np.mean([row["success"] for row in conditions["initializer"]]))
    contact = float(np.mean([row.get("firstContactDecision") is not None for row in correct]))
    hit = float(np.mean([row.get("firstHitDecision") is not None for row in correct]))
    total_actions = sum(int(row["totalActions"]) for row in correct)
    rejected = sum(int(row["rejectedActions"]) for row in correct)
    rejection_rate = rejected / total_actions if total_actions else 1.0
    gates = {
        "success": success >= 0.5,
        "shuffledImprovement": success - shuffled >= 0.2,
        "initializerImprovement": success - initializer >= 0.2,
        "firstContact": contact >= 0.8,
        "rejectedActions": rejection_rate < 0.001,
    }
    report = {
        "format": "snowgym.engage-bootstrap.v0",
        "evaluationDigest": claimed,
        "checkpointDigest": value.get("checkpointDigest"),
        "seeds": seeds,
        "metrics": {
            "successRate": success,
            "shuffledSuccessRate": shuffled,
            "initializerSuccessRate": initializer,
            "shuffledImprovement": success - shuffled,
            "initializerImprovement": success - initializer,
            "firstContactRate": contact,
            "firstHitRate": hit,
            "rejectedActionRate": rejection_rate,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    report["reportDigest"] = json_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap report {destination}")
    value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = engage_bootstrap_report(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
