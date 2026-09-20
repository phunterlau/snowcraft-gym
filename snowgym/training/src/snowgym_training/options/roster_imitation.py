"""M8-S5: R1n-h's mixture-imitation recipe at 3v3, evaluated world-paired against the S4 teacher
(`reviews/m8_s5_declaration.md`).

Training is `mixture_imitation.train_condition` for condition M, unchanged, under an outer
roster-3 scenario override. The measurement is new: each seed's final policy is run
deterministically on S4's paired worlds and compared with S4's archived teacher episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import mixture_imitation as mi
from . import roster_baseline as rb
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

CONDITION = "M"
TEACHER_RUN = "m8_s4_roster_baseline_v0"


def configuration():
    base = mi.configuration()
    return {**base, "format": "snowgym.m8-s5-roster-imitation-config.v0", "roster": rb.ROSTER,
        "roundSeedBaseByCondition": {"M": 3300000, "C": 3330000}, "roundSeedStride": 1000,
        "evalSeedBase": {"random": 3320000, "easy": 3321000, "normal": 3322000},
        "stochasticEvalSeedBase": 3323000, "floorSeedBase": 3324000,
        "criticTrainSeedBaseByCondition": {"M": 3310000, "C": 3340000},
        "criticHeldSeedBaseByCondition": {"M": 3315000, "C": 3345000},
        "pairedEvalSeedBase": rb.configuration()["evaluationSeedBase"], "pairedEvalWorlds": 400,
        "budgetCap": 2_000_000, "pairedBlockWorlds": rb.configuration()["blockWorlds"], "viabilitySuccessNormal": 0.25,
        "assistType": "teacher-imitation training; none at runtime",
        "assistVersion": "snowgym.m8-s5-roster-imitation.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    seeds = len(cfg["optimizerSeeds"])
    round_zero = cfg["roundEpisodes"] * horizon
    learner_rounds = seeds * (cfg["fits"] - 1) * cfg["roundEpisodes"] * horizon
    development = seeds * (3 + 1) * cfg["evaluationEpisodes"] * horizon
    critic = seeds * (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    paired = seeds * len(rb.ARMS) * cfg["pairedEvalWorlds"] * horizon
    return {"roundZero": round_zero, "learnerRounds": learner_rounds, "development": development,
            "critic": critic, "pairedEval": paired,
            "total": round_zero + learner_rounds + development + critic + paired}


def seed_bands(cfg):
    """Every world-seed range this run touches, as (name, low, high) inclusive."""
    bands = []
    for round_index in range(cfg["fits"]):
        low = cfg["roundSeedBaseByCondition"][CONDITION] + cfg["roundSeedStride"] * round_index
        bands.append((f"round-{round_index}", low, low + cfg["roundEpisodes"] - 1))
    for index in range(len(cfg["optimizerSeeds"])):
        stride = cfg["criticSeedBandStride"] * index
        low = cfg["criticTrainSeedBaseByCondition"][CONDITION] + stride
        bands.append((f"critic-train-{index}", low, low + cfg["warmStartTrainEpisodes"] - 1))
        low = cfg["criticHeldSeedBaseByCondition"][CONDITION] + stride
        bands.append((f"critic-held-{index}", low, low + cfg["warmStartHeldOutEpisodes"] - 1))
    for name, base in cfg["evalSeedBase"].items():
        bands.append((f"development-{name}", base, base + cfg["evaluationEpisodes"] - 1))
    bands.append(("development-stochastic", cfg["stochasticEvalSeedBase"],
                  cfg["stochasticEvalSeedBase"] + cfg["evaluationEpisodes"] - 1))
    bands.append(("paired-eval", cfg["pairedEvalSeedBase"], cfg["pairedEvalSeedBase"] + cfg["pairedEvalWorlds"] - 1))
    return bands


def training_bands(cfg):
    return [band for band in seed_bands(cfg) if band[0] != "paired-eval"]


# -- Measurement --------------------------------------------------------------------------


def load_final(cfg, root, seed):
    model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
    state = torch.load(Path(root) / f"condition-{CONDITION}" / f"seed-{seed}" / f"fit-{cfg['fits'] - 1}.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def paired_seeds(cfg):
    return list(range(cfg["pairedEvalSeedBase"], cfg["pairedEvalSeedBase"] + cfg["pairedEvalWorlds"]))


def teacher_rows(arm, worlds):
    path = TRAINING / "runs" / TEACHER_RUN / "teacher" / arm / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) < worlds:
        raise ValueError("teacher archive has fewer worlds than requested")
    return rows[:worlds]


def paired_difference(learner, teacher, metric, cfg):
    """Learner minus teacher, world by world (seeds must match exactly), with a world bootstrap."""
    if [r["seed"] for r in learner] != [r["seed"] for r in teacher]:
        raise ValueError("learner and teacher worlds are not the same")
    values = np.asarray([float(metric(l)) - float(metric(t)) for l, t in zip(learner, teacher)])
    return {"mean": float(values.mean()),
            "interval": rb.bootstrap_mean(values, samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])}


def measure_seed(client, cfg, root, seed, account):
    model = load_final(cfg, root, seed)
    seeds = paired_seeds(cfg)
    result = {}
    for arm in rb.ARMS:
        rows = rb.collect_cell(client, seeds, {**cfg, "blockWorlds": cfg["pairedBlockWorlds"]}, arm, model=model,
                               mode="deterministic", account=account)
        rb.write_rows(Path(root) / "paired-eval" / f"seed-{seed}" / arm, rows)
        teacher = teacher_rows(arm, cfg["pairedEvalWorlds"])
        result[arm] = {**rb.summarize(rows, cfg),
            "successMinusTeacher": paired_difference(rows, teacher, lambda r: r["success"], cfg),
            "unitsLostMinusTeacher": paired_difference(rows, teacher, lambda r: r["unitsLostFraction"], cfg)}
    return result


def readings(paired, cfg):
    normal = {seed: cells["normal"]["success"] for seed, cells in paired.items()}
    mean_normal = float(np.mean(list(normal.values())))
    threshold = cfg["viabilitySuccessNormal"]
    viable = mean_normal >= threshold
    reliable = viable and all(value >= threshold for value in normal.values())
    return {"normalSuccessBySeed": normal, "seedMeanNormalSuccess": mean_normal, "viable": viable,
            "reliableAcrossSeeds": reliable,
            "outcome": ("viable-and-reliable" if reliable else "viable-but-unreliable" if viable else "not-viable")}


# -- Declaration, run, archive ---------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    here = Path(__file__).resolve()
    sources = {name: file_digest(here.parent / f"{name}.py") for name in
               ("mixture_imitation", "full_authority_imitation", "full_authority_train_v1", "roster_baseline")}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "seedBands": seed_bands(cfg),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s5_declaration.md"),
        "implementationDigest": file_digest(here), "sourceDigests": sources, "pinnedE3Digests": pinned,
        "teacherRunManifest": dr.verify_sealed(TRAINING / "runs" / TEACHER_RUN),
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, paired, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    condition = json.loads((root / f"condition-{CONDITION}" / "condition-report.json").read_text(encoding="utf-8"))
    critic = {seed: {key: report[key] for key in ("predictiveR2", "timeOnlyR2", "untrainedPredictiveR2", "gatePassed")}
              for seed, report in condition["criticPrecondition"].items()}
    report = {"format": "snowgym.m8-s5-roster-imitation-report.v0", "pairedEval": paired,
        "readings": readings(paired, cfg), "criticWarmStart": critic, "simulatorDecisions": decisions,
        "withinBudgetBound": decisions <= budget_bound(cfg)["total"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s5-manifest.v0")
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
            raise ValueError("M8-S5 budget exceeded")

    with SnowGymBatchClient() as client:
        mi.require_capabilities(client)
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            mi.train_condition(client, cfg, CONDITION, root, account)
        paired = {str(seed): measure_seed(client, cfg, root, seed, account) for seed in cfg["optimizerSeeds"]}
    return aggregate(root, cfg, paired, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
