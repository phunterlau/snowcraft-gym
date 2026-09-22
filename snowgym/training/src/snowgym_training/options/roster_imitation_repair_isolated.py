"""M8-S11: the pre-declared S10 contingency, isolating the loss weight from the data draw
(`reviews/m8_s11_declaration.md`).

Reuses `roster_imitation_repair`'s training and measurement machinery unchanged, on the SAME fresh seed bands
as S10, with `aimWeight` reverted to 1. That module is not edited; only the configuration passed to its functions
differs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import roster_baseline as rb
from . import roster_imitation_repair as rp
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .reservoir import file_digest
from .supervised_probe import write_json

S10_RUN = "m8_s10_throw_aim_repair_v0"
AIM_SEED, LOW, HIGH = 97101, 0.20, 0.35


def configuration():
    return {**rp.configuration(), "format": "snowgym.m8-s11-throw-aim-isolation-config.v0", "aimWeight": 1.0}


def budget_bound(cfg):
    return rp.budget_bound(cfg)


def s10_rows(seed, arm, worlds):
    path = TRAINING / "runs" / S10_RUN / "condition-M" / f"seed-{seed}" / "paired-eval" / arm / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:worlds]


def s10_success(seed, arm, cfg):
    return rb.summarize(s10_rows(seed, arm, cfg["pairedEvalWorlds"]), cfg)["success"]


def s5_success(seed, arm, cfg):
    return rb.summarize(rp.s5_rows(arm, seed, cfg["pairedEvalWorlds"]), cfg)["success"]


def classify(gain_s11, s10_value):
    if gain_s11 <= LOW:
        return "loss-weight-driven"
    if HIGH <= gain_s11 <= s10_value + 1e-9:
        return "data-draw-driven"
    return "mixed-or-inconsistent"


def score(seeds_out, cfg):
    result = {}
    for arm in cfg["pairedEvalArms"]:
        result[arm] = {}
        for seed_str, entry in seeds_out.items():
            seed = int(seed_str)
            s11_value = entry["pairedEval"][arm]["success"]
            s10_value = s10_success(seed, arm, cfg)
            s5_value = s5_success(seed, arm, cfg)
            result[arm][seed_str] = {"s11": s11_value, "s10": s10_value, "s5": s5_value,
                "s11MinusS5_dataEffect": s11_value - s5_value, "s10MinusS11_lossEffect": s10_value - s11_value,
                "s10MinusS5_combined": s10_value - s5_value}
    aim_result = result["normal"][str(AIM_SEED)]
    classification = classify(aim_result["s11"], aim_result["s10"])
    other_seeds = {s: result["normal"][s] for s in result["normal"] if int(s) != AIM_SEED}
    critic_ok = {s: e["critic"]["gatePassed"] for s, e in seeds_out.items()}
    easy = {s: {"s11": result["easy"][s]["s11"], "s10": result["easy"][s]["s10"], "s5": result["easy"][s]["s5"],
                "restoredVsS10": result["easy"][s]["s11"] >= result["easy"][s]["s10"] + cfg["easyRegressionMargin"] / 3,
                "backNearS5": result["easy"][s]["s11"] >= result["easy"][s]["s5"] - cfg["easyRegressionMargin"]}
            for s in result["easy"]}
    return {"byArm": result, "seed97101Classification": classification, "otherSeedsNormal": other_seeds,
            "criticGatePassed": critic_ok, "easyComparedToS10AndS5": easy,
            "decompositionCheck": {"lossEffect": aim_result["s10MinusS11_lossEffect"],
                "dataEffect": aim_result["s11MinusS5_dataEffect"], "combined": aim_result["s10MinusS5_combined"],
                "sumOfParts": aim_result["s10MinusS11_lossEffect"] + aim_result["s11MinusS5_dataEffect"]}}


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "s10Manifest": dr.verify_sealed(TRAINING / "runs" / S10_RUN),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / "m8_s5_roster_imitation_v0"),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s11_declaration.md"),
        "implementationDigest": file_digest(here), "repairModuleDigest": file_digest(Path(rp.__file__)),
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, seeds_out, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    report = {"format": "snowgym.m8-s11-throw-aim-isolation-report.v0", "learners": seeds_out,
        "score": score(seeds_out, cfg), "simulatorDecisions": decisions,
        "withinBudgetBound": decisions <= bound["total"], "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s11-manifest.v0")
    return report


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("M8-S11 budget exceeded")

    with rp.SnowGymBatchClient() as client:
        rp.require_capabilities(client)
        with rp.scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            seeds_out = rp.train_and_measure(client, cfg, root, account)
    return aggregate(root, cfg, seeds_out, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
