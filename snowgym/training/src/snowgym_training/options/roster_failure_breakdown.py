"""M8-S6: archive-only failure breakdown of the S5 3v3 initializers (`reviews/m8_s6_declaration.md`).

Reads sealed S4/S5 episode rows only; no simulator, no training. Outcome categories are
descriptive, never causal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from .full_authority_diagnostics import TRAINING
from .reservoir import file_digest
from .supervised_probe import write_json

S4_RUN, S5_RUN = "m8_s4_roster_baseline_v0", "m8_s5_roster_imitation_v0"
ARMS = ("random", "easy", "normal")
SEEDS = ("97101", "97102", "97103")
BANDS = ("none", "lt100", "100-199", "200-239", "ge240")
OUTCOMES = ("success", "wipe", "timeout-ge100", "timeout-lt100", "other")
SUCCESS_DAMAGE = 240.0


def damage_band(damage):
    if damage <= 0:
        return "none"
    if damage < 100:
        return "lt100"
    if damage < 200:
        return "100-199"
    return "200-239" if damage < SUCCESS_DAMAGE else "ge240"


def outcome(row):
    if row["success"]:
        return "success"
    if row["blueAliveCount"] == 0:
        return "wipe"
    if row["timedOut"]:
        return "timeout-ge100" if row["finalTargetDamage"] >= 100 else "timeout-lt100"
    return "other"


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summarize_cell(rows):
    total = len(rows)
    if total == 0:
        raise ValueError("no episodes")
    failures = [r for r in rows if not r["success"]]
    damage = np.asarray([r["finalTargetDamage"] for r in rows], dtype=float)
    lost = sum(r["assignedUnits"] - r["blueAliveCount"] for r in rows)
    return {"episodes": total,
        "contact": sum(r["finalTargetDamage"] > 0 for r in rows) / total,
        "outcomes": {name: sum(outcome(r) == name for r in rows) / total for name in OUTCOMES},
        "failureDamageBands": {band: (sum(damage_band(r["finalTargetDamage"]) == band for r in failures)
                                      / len(failures)) if failures else None for band in BANDS},
        "failures": len(failures),
        "damageQuantiles": {q: float(np.percentile(damage, q)) for q in (25, 50, 75)},
        "medianMinDistance": _median([r["minDistance"] for r in rows]),
        "medianDistanceAtFirstHit": _median([r["distanceAtFirstHit"] for r in rows]),
        "medianFirstHitDecision": _median([r["firstHitDecision"] for r in rows]),
        "medianFinalDecision": _median([r["finalDecision"] for r in rows]),
        "damagePerUnitLost": (float(damage.sum()) / lost) if lost else None, "unitsLost": lost}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_paired(arm, seed):
    learner = read_rows(TRAINING / "runs" / S5_RUN / "paired-eval" / f"seed-{seed}" / arm / "episodes.jsonl")
    teacher = read_rows(TRAINING / "runs" / S4_RUN / "teacher" / arm / "episodes.jsonl")
    if [r["seed"] for r in learner] != [r["seed"] for r in teacher]:
        raise ValueError("learner and teacher worlds are not aligned by seed")
    return learner, teacher


def breakdown():
    result = {"teacher": {}, "learners": {}}
    for arm in ARMS:
        for seed in SEEDS:
            learner, teacher = load_paired(arm, seed)
            result["learners"].setdefault(seed, {})[arm] = summarize_cell(learner)
            result["teacher"][arm] = summarize_cell(teacher)
    return result


def declare(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"format": "snowgym.m8-s6-breakdown-declaration.v0",
        "gitCommit": resolve_git_commit(), "s4Manifest": dr.verify_sealed(TRAINING / "runs" / S4_RUN),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / S5_RUN),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s6_declaration.md"),
        "implementationDigest": file_digest(here), "simulatorDecisions": 0,
        "autonomousQualificationEligible": False})


def run(output):
    root = Path(output)
    declare(root)
    report = {"format": "snowgym.m8-s6-breakdown-report.v0", **breakdown(), "simulatorDecisions": 0,
              "autonomousQualificationEligible": False}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s6-manifest.v0")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
