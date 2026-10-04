"""M8-S15: does S14's normal-only PPO continuation retain the initializer's random/easy Red performance?
(`reviews/m8_s15_declaration.md`)

Loads S14's three frozen `update-200.pt` checkpoints and S14's own sigma-scaled initializer `reference`, replays
S14's first archived evaluation block exactly (deterministic and stochastic) as a fidelity gate, then evaluates both
policies on a fresh world split against random and scripted-easy Red. No training, no checkpoint selection.
`enemy_relative_throw_ppo.prepare_policy` and `roster_baseline.collect_cell` are reused unchanged; no existing module
is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import enemy_relative_throw_ppo as s14
from . import full_authority_imitation as fi
from . import roster_baseline as rb
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .interventions import require_capabilities
from .reservoir import file_digest
from .supervised_probe import write_json

POLICIES = ("initializer", "final")
ARMS = ("random", "easy")
MODES = ("deterministic", "stochastic")
METRICS = ("success", "unitsLostFraction", "timedOut", "teamWipe")
S13_RANDOM_DETERMINISTIC_SUCCESS = {1: .77, 2: .87, 3: .79}  # m8_s13_results.md, new decoder, 100 worlds


def configuration():
    return {**s14.configuration(), "format": "snowgym.m8-s15-ppo-retention-eval-config.v0",
        "finalRun": "m8_s14_ppo_continuation_v0", "finalCheckpoint": "update-200.pt",
        "probeRun": "m8_s14_probe_v0", "evalArms": ARMS, "modes": MODES, "policies": POLICIES,
        # S14's own evaluation band and stochastic torch seeds, used ONLY by the fidelity replay (declaration §2).
        "fidelitySeedBase": 2600000, "fidelityWorlds": 64, "fidelityTorchSeedBase": 984500, "fidelityArm": "normal",
        # The fresh retention split and its stochastic torch seeds (declaration §3); these replace S14's values
        # so no inherited helper can silently reuse 2600000/984500 as result data.
        "evaluationSeedBase": 2700000, "evaluationWorlds": 400, "stochasticEvalSeedBase": 985000,
        "bootstrapSeed": 985001, "margin": .05, "rejectionRateCeiling": .001, "crossCheckTolerance": .12,
        "budgetCap": 2_100_000,
        "assistType": "none; evaluation of frozen checkpoints, no training",
        "assistVersion": "snowgym.m8-s15.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    fidelity = len(cfg["cohorts"]) * len(POLICIES) * len(MODES) * cfg["fidelityWorlds"] * horizon
    evaluation = len(cfg["cohorts"]) * len(POLICIES) * len(ARMS) * len(MODES) * cfg["evaluationWorlds"] * horizon
    return {"fidelity": fidelity, "evaluation": evaluation, "total": fidelity + evaluation}


def evaluation_seeds(cfg):
    return list(range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + cfg["evaluationWorlds"]))


def fidelity_seeds(cfg):
    return list(range(cfg["fidelitySeedBase"], cfg["fidelitySeedBase"] + cfg["fidelityWorlds"]))


def fidelity_torch_seed(cfg, cohort, policy):
    """S14's own formula: `stochasticEvalSeedBase + 10 * cohort + m`, m = 0 initializer, 1 final."""
    return cfg["fidelityTorchSeedBase"] + 10 * cohort + POLICIES.index(policy)


def stochastic_torch_seed(cfg, cohort, arm, policy):
    """985100-985311 for cohorts 1-3 (declaration §3)."""
    return cfg["stochasticEvalSeedBase"] + 100 * cohort + 10 * ARMS.index(arm) + POLICIES.index(policy)


# -- Policies (declaration §1) --------------------------------------------------------------


def final_checkpoint_path(cfg, cohort):
    return TRAINING / "runs" / cfg["finalRun"] / f"cohort-{cohort}" / cfg["finalCheckpoint"]


def initializer_checkpoint_path(cfg, cohort):
    return TRAINING / "runs" / cfg["sourceRun"] / f"cohort-{cohort}" / "new" / cfg["policyCheckpoint"]


def load_policies(cfg, cohort):
    """S14's `reference` (the sigma-scaled initializer) and the update-200 policy, both in eval mode."""
    final, initializer = s14.prepare_policy(cfg, cohort, sigma_scale=cfg["sigmaScaleByCohort"][cohort])
    state = torch.load(final_checkpoint_path(cfg, cohort), map_location="cpu", weights_only=True)
    if state["update"] != cfg["updates"]:
        raise RuntimeError(f"expected the update-{cfg['updates']} checkpoint, got update {state['update']}")
    final.load_state_dict(state["model"])
    final.eval()
    initializer.eval()
    for name in (*s14.LOG_STDS, "throw_offset_log_std"):
        # frozen throughout S14 training, so the final policy must sample with exactly the initializer's noise
        if not torch.equal(getattr(final, name), getattr(initializer, name)):
            raise RuntimeError(f"cohort {cohort}: {name} differs between the final checkpoint and the initializer")
    if dr.parameter_distance(final, initializer) <= 0:
        raise RuntimeError(f"cohort {cohort}: the final checkpoint equals its initializer")
    return {"initializer": initializer, "final": final}


# -- Fidelity gate (declaration §2) -----------------------------------------------------------


def archived_rows(cfg, cohort, policy, mode):
    path = (TRAINING / "runs" / cfg["finalRun"] / f"cohort-{cohort}" / "evaluation" / f"{policy}-{mode}"
            / "episodes.jsonl")
    return {row["seed"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def fidelity_check(client, cfg, cohort, policies, root, account):
    seeds = fidelity_seeds(cfg)
    results = {}
    for policy in POLICIES:
        for mode in MODES:
            if mode == "stochastic":
                torch.manual_seed(fidelity_torch_seed(cfg, cohort, policy))
            rows = rb.collect_cell(client, seeds, cfg, cfg["fidelityArm"], model=policies[policy], mode=mode,
                                   account=account)
            rb.write_rows(Path(root) / "fidelity" / f"cohort-{cohort}" / f"{policy}-{mode}", rows)
            archived = archived_rows(cfg, cohort, policy, mode)
            mismatched = [row["seed"] for row in rows if archived.get(row["seed"]) != row]
            results[f"{policy}-{mode}"] = {"worlds": len(rows), "mismatchedSeeds": mismatched,
                                           "exact": not mismatched and len(rows) == len(seeds)}
    return results


# -- Evaluation (declaration §3) --------------------------------------------------------------


def cell_directory(root, cohort, arm, policy, mode):
    return Path(root) / f"cohort-{cohort}" / arm / f"{policy}-{mode}"


def evaluate_cohort(client, cfg, cohort, policies, root, account):
    seeds = evaluation_seeds(cfg)
    summaries = {}
    for arm in ARMS:
        for policy in POLICIES:
            for mode in MODES:
                if mode == "stochastic":
                    torch.manual_seed(stochastic_torch_seed(cfg, cohort, arm, policy))
                rows = rb.collect_cell(client, seeds, cfg, arm, model=policies[policy], mode=mode, account=account)
                if any(row["assignedUnits"] != cfg["roster"] for row in rows):
                    raise RuntimeError(f"expected roster {cfg['roster']} assigned units")
                rb.write_rows(cell_directory(root, cohort, arm, policy, mode), rows)
                summaries[f"{arm}/{policy}-{mode}"] = rb.summarize(rows, cfg)
    return summaries


# -- Analysis and decision rules (declaration §4, §5, §6) ---------------------------------------


def metric_value(row, name):
    if name == "teamWipe":
        return float(row["blueAliveCount"] == 0)
    return float(row[name])


def load_cell(root, cohort, arm, policy, mode):
    path = cell_directory(root, cohort, arm, policy, mode) / "episodes.jsonl"
    return {row["seed"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def paired_analysis(root, cfg):
    """Every (arm, mode) cell x every metric, per cohort and cohort-averaged (per-world mean over the cohorts'
    policies, then world-paired), final - initializer."""
    expected = evaluation_seeds(cfg)
    cells = {}
    for arm in ARMS:
        for mode in MODES:
            finals, initials, per_cohort = [], [], {}
            for cohort in cfg["cohorts"]:
                after = load_cell(root, cohort, arm, "final", mode)
                before = load_cell(root, cohort, arm, "initializer", mode)
                if sorted(after) != expected or sorted(before) != expected:
                    raise RuntimeError(f"cohort {cohort} {arm}/{mode}: rows do not cover the declared worlds")
                per_cohort[str(cohort)] = {name: fi.paired_difference(
                    [metric_value(after[w], name) for w in expected], [metric_value(before[w], name) for w in expected],
                    samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]) for name in METRICS}
                finals.append(after)
                initials.append(before)
            averaged = {name: fi.paired_difference(
                [np.mean([metric_value(f[w], name) for f in finals]) for w in expected],
                [np.mean([metric_value(i[w], name) for i in initials]) for w in expected],
                samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]) for name in METRICS}
            cells[f"{arm}/{mode}"] = {"perCohort": per_cohort, "cohortAveraged": averaged, "worlds": len(expected)}
    return cells


def harm_interval(difference, name):
    """The 95% interval of the harmful direction: success harms by falling, the other metrics by rising."""
    low, high = difference["interval95"]
    return (-high, -low) if name == "success" else (low, high)


def cell_status(differences, margin):
    harm = {name: harm_interval(differences[name], name) for name in METRICS}
    non_inferior = all(high < margin for _, high in harm.values())
    confident_harm = [name for name, (low, _) in harm.items() if low > 0]
    status = "non-inferior" if non_inferior else ("regressed" if confident_harm else "inconclusive")
    return {"status": status, "confidentHarm": confident_harm}


def decision_rules(cells, summaries, cfg):
    margin = cfg["margin"]
    statuses = {key: cell_status(cell["cohortAveraged"], margin) for key, cell in cells.items()}
    cohort_flags = [{"cell": key, "cohort": cohort, "metric": name}
                    for key, cell in cells.items() for cohort, differences in cell["perCohort"].items()
                    for name in METRICS if harm_interval(differences[name], name)[0] > margin]
    rejection = {f"{cohort}/{arm}/{mode}": summaries[str(cohort)][f"{arm}/final-{mode}"]["rejectedActionRate"]
                 for cohort in cfg["cohorts"] for arm in ARMS for mode in MODES}
    rejection_ok = all(rate is not None and rate < cfg["rejectionRateCeiling"] for rate in rejection.values())
    if not rejection_ok or any(s["status"] == "regressed" for s in statuses.values()):
        outcome = "regressed"
    elif all(s["status"] == "non-inferior" for s in statuses.values()):
        outcome = "retained-with-cohort-regression" if cohort_flags else "retained"
    else:
        outcome = "inconclusive"
    recommendation = {
        "retained": "use the S14 final/initializer pairs as the frozen executors for the command-control audit",
        "retained-with-cohort-regression": "same, reporting the flagged cohort separately in every later result",
        "regressed": "keep random/easy in any further fighter training's rollout or anchor mixture; report final "
                     "and initializer side by side in later audits",
        "inconclusive": "report final and initializer side by side in later audits"}[outcome]
    return {"outcome": outcome, "cells": statuses, "cohortFlags": cohort_flags,
            "rejectionGate": {"passed": rejection_ok, "finalRejectedActionRate": rejection},
            "recommendation": recommendation, "authorizes": "nothing; a follow-up needs its own declaration"}


def prediction_scores(cells, summaries, rules, cfg):
    averaged = {key: cell["cohortAveraged"] for key, cell in cells.items()}
    random_det, random_sto = averaged["random/deterministic"]["success"], averaged["random/stochastic"]["success"]
    cross_check = {str(c): {"initializerSuccess": summaries[str(c)]["random/initializer-deterministic"]["success"],
                            "s13Success": S13_RANDOM_DETERMINISTIC_SUCCESS[c]} for c in cfg["cohorts"]}
    for entry in cross_check.values():
        entry["withinTolerance"] = abs(entry["initializerSuccess"] - entry["s13Success"]) <= cfg["crossCheckTolerance"]
    return {"1-retained": rules["outcome"] == "retained",
            "2-randomDeterministicSuccessGain": random_det["interval95"][0] > 0,
            "3-randomStochasticGainSmaller": random_sto["mean"] < random_det["mean"],
            "4-easyDeterministicUnitsLostNotWorse": averaged["easy/deterministic"]["unitsLostFraction"]["mean"] <= 0,
            "crossCheckS13": cross_check}


# -- Declaration, run, archive -------------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    runs = TRAINING / "runs"
    manifests = {"s14Training": dr.verify_sealed(runs / cfg["finalRun"]),
                 "s14Probe": dr.verify_sealed(runs / cfg["probeRun"]),
                 "s12": dr.verify_sealed(runs / cfg["sourceRun"])}
    checkpoints = {str(c): {"final": file_digest(final_checkpoint_path(cfg, c)),
                            "initializer": file_digest(initializer_checkpoint_path(cfg, c))} for c in cfg["cohorts"]}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "sourceManifests": manifests, "checkpointDigests": checkpoints,
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s15_declaration.md"),
        "implementationDigest": file_digest(Path(__file__).resolve()), "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, fidelity, summaries, fidelity_decisions, evaluation_decisions):
    """`torchThreads` is recorded because exact replay depends on it, as it does on block size (declaration §2)."""
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    passed = fidelity_passed(fidelity)
    cells = paired_analysis(root, cfg) if passed else None
    rules = decision_rules(cells, summaries, cfg) if passed else {
        "outcome": "fidelity-failed", "recommendation": "diagnose the mismatch before any new world is evaluated",
        "authorizes": "nothing; a follow-up needs its own declaration"}
    total = fidelity_decisions + evaluation_decisions
    report = {"format": "snowgym.m8-s15-ppo-retention-eval-report.v0", "fidelity": fidelity, "fidelityPassed": passed,
        "torchThreads": torch.get_num_threads(), "blockWorlds": cfg["blockWorlds"],
        "summaries": summaries, "pairedDifferences": cells, "decisionRules": rules,
        "predictions": prediction_scores(cells, summaries, rules, cfg) if passed else None,
        "fidelitySimulatorDecisions": fidelity_decisions, "evaluationSimulatorDecisions": evaluation_decisions,
        "simulatorDecisions": total, "budgetBound": budget_bound(cfg), "withinBudgetCap": total <= cfg["budgetCap"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s15-manifest.v0")
    return report


def fidelity_passed(fidelity):
    return bool(fidelity) and all(cell["exact"] for entry in fidelity.values() for cell in entry.values())


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    spent = {"fidelity": 0, "evaluation": 0}

    def accountant(stage):
        def account(count):
            spent[stage] += count
            if sum(spent.values()) > cfg["budgetCap"]:
                raise ValueError("M8-S15 budget exceeded")
        return account

    fidelity, summaries = {}, {}
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        policies = {cohort: load_policies(cfg, cohort) for cohort in cfg["cohorts"]}
        for cohort in cfg["cohorts"]:  # every cohort's gate passes before any new world is played
            fidelity[str(cohort)] = fidelity_check(client, cfg, cohort, policies[cohort], root, accountant("fidelity"))
            print(json.dumps({"stage": "fidelity", "cohort": cohort, "result": fidelity[str(cohort)]}), flush=True)
        if fidelity_passed(fidelity):
            for cohort in cfg["cohorts"]:
                summaries[str(cohort)] = evaluate_cohort(client, cfg, cohort, policies[cohort], root,
                                                         accountant("evaluation"))
                print(json.dumps({"stage": "evaluation", "cohort": cohort, "decisions": spent}), flush=True)
    return aggregate(root, cfg, fidelity, summaries, spent["fidelity"], spent["evaluation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps({"outcome": report["decisionRules"]["outcome"], "decisions": report["simulatorDecisions"]}))
