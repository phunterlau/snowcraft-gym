"""M8-S13: does the new throw decoder's gain over the old one hold against random Red?
(`reviews/m8_s13_declaration.md`)

Loads the 3 cohorts' already-trained `old`/`new` checkpoints from S12 (`m8_s12_enemy_relative_throw_v0`) and runs
one additional paired-eval arm, `random`, that S12 did not test. No training happens here; `rb.collect_cell` and
`enemy_relative_throw.build_model`/`paired_eval_seeds` are reused unchanged. No existing module is edited."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from snowgym_client.batch import SnowGymBatchClient
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import enemy_relative_throw as ert
from . import full_authority_imitation as fi
from . import roster_baseline as rb
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .interventions import require_capabilities
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

ARM = "random"


def configuration():
    return {**ert.configuration(), "format": "snowgym.m8-s13-random-red-eval-config.v0",
        "sourceRun": "m8_s12_enemy_relative_throw_v0", "evalArm": ARM,
        "assistType": "none; evaluation of already-trained checkpoints, no training",
        "assistVersion": "snowgym.m8-s13.v0", "budgetCap": 200_000, "autonomousQualificationEligible": False}


def budget_bound(cfg):
    per_run = cfg["pairedEvalWorlds"] * cfg["optionHorizon"]
    return {"perRun": per_run, "total": per_run * len(cfg["cohorts"]) * len(ert.DECODERS)}


def load_checkpoint(cohort, decoder, cfg):
    path = TRAINING / "runs" / cfg["sourceRun"] / f"cohort-{cohort}" / decoder / f"fit-{cfg['fits'] - 1}.pt"
    model = ert.build_model(decoder, cfg, seed=0)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def paired_summary(new_rows, old_rows, cfg):
    new_by_seed = {r["seed"]: r for r in new_rows}
    old_by_seed = {r["seed"]: r for r in old_rows}
    seeds = sorted(set(new_by_seed) & set(old_by_seed))
    if len(seeds) != len(new_rows) or len(seeds) != len(old_rows):
        raise ValueError("new/old rows must cover the same paired-eval seeds exactly once each")
    new_success = [new_by_seed[s]["success"] for s in seeds]
    old_success = [old_by_seed[s]["success"] for s in seeds]
    new_lost = [new_by_seed[s]["unitsLostFraction"] for s in seeds]
    old_lost = [old_by_seed[s]["unitsLostFraction"] for s in seeds]
    return {
        "deltaSuccess": fi.paired_difference(new_success, old_success, samples=cfg["bootstrapSamples"],
            seed=cfg["bootstrapSeed"]),
        "deltaUnitsLostFraction": fi.paired_difference(new_lost, old_lost, samples=cfg["bootstrapSamples"],
            seed=cfg["bootstrapSeed"]),
    }


def evaluate_one(client, cfg, decoder, cohort, root, account):
    model = load_checkpoint(cohort, decoder, cfg)
    seeds = ert.paired_eval_seeds(cfg)
    rows = rb.collect_cell(client, seeds, cfg, cfg["evalArm"], model=model, mode="deterministic", account=account)
    directory = Path(root) / f"cohort-{cohort}" / decoder / cfg["evalArm"]
    rb.write_rows(directory, rows)
    summary = rb.summarize(rows, cfg)
    del model
    return rows, summary


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
        "budgetBound": budget_bound(cfg),
        "s12Manifest": dr.verify_sealed(TRAINING / "runs" / cfg["sourceRun"]),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s13_declaration.md"),
        "implementationDigest": file_digest(here), "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, results, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    report = {"format": "snowgym.m8-s13-random-red-eval-report.v0", "cohorts": results,
        "simulatorDecisions": decisions, "withinBudgetBound": decisions <= bound["total"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s13-manifest.v0")
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
            raise ValueError("M8-S13 budget exceeded")

    results = {}
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            for cohort in cfg["cohorts"]:
                rows = {}
                for decoder in ert.DECODERS:
                    rows[decoder], summary = evaluate_one(client, cfg, decoder, cohort, root, account)
                    results.setdefault(str(cohort), {})[decoder] = summary
                results[str(cohort)]["paired"] = paired_summary(rows["new"], rows["old"], cfg)
    return aggregate(root, cfg, results, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
