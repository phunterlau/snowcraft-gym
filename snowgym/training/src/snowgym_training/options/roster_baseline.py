"""M8-S4: 3v3 Engage teacher achievability and random-init floors (`reviews/m8_s4_declaration.md`).

Trains and selects nothing. Every condition runs the same world-paired seeds against native
Red (random, scripted-easy, scripted-normal). No existing module is edited: collection reuses
`full_authority_train_v1.run_block` and reads each world's final state from the wrapper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .mixture_imitation import EVAL_ARMS
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

ROSTER = 3
ARMS = ("random", "easy", "normal")
MODES = ("deterministic", "stochastic")


def configuration():
    return {**v1.configuration(), "format": "snowgym.m8-s4-roster-baseline-config.v0",
        "roster": ROSTER, "arms": ARMS, "modes": MODES, "evaluationSeedBase": 2100000,
        "teacherWorlds": 400, "floorWorlds": 100, "initSeeds": (98101, 98102, 98103),
        "blockWorlds": 50, "budgetCap": 700000, "bootstrapSeed": 960004, "bootstrapSamples": 10000,
        "teacherSuccessGate": 0.90, "floorSuccessCeiling": 0.10,
        "assistType": "teacher is the native plan controller; floors are unassisted random init",
        "assistVersion": "snowgym.m8-s4.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    teacher = cfg["teacherWorlds"] * len(cfg["arms"]) * horizon
    floors = len(cfg["initSeeds"]) * len(cfg["modes"]) * cfg["floorWorlds"] * len(cfg["arms"]) * horizon
    return {"teacher": teacher, "floors": floors, "total": teacher + floors}


def world_seeds(cfg, count):
    return list(range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + count))


def arm_scenario(arm):
    return {**EVAL_ARMS[arm], "blueUnits": ROSTER, "redUnits": ROSTER}


# -- Metrics (declaration section 1) --------------------------------------------------


def units_lost_fraction(assigned, alive):
    if assigned <= 0 or not 0 <= alive <= assigned:
        raise ValueError("alive units must lie in 0..assigned")
    return (assigned - alive) / assigned


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("wilson interval needs a positive total")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bootstrap_mean(values, *, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows, cfg):
    total = len(rows)
    if total == 0:
        raise ValueError("no episodes to summarize")
    success = sum(bool(r["success"]) for r in rows)
    wipe = sum(r["blueAliveCount"] == 0 for r in rows)
    timeout = sum(bool(r["timedOut"]) for r in rows)
    lost = [r["unitsLostFraction"] for r in rows]
    submitted = sum(r["totalActions"] for r in rows)
    return {"episodes": total, "success": success / total, "successInterval": wilson(success, total),
        "teamWipe": wipe / total, "teamWipeInterval": wilson(wipe, total),
        "timeout": timeout / total, "meanUnitsLostFraction": float(np.mean(lost)),
        "meanUnitsLostInterval": bootstrap_mean(lost, samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]),
        "meanDecisions": float(np.mean([r["finalDecision"] for r in rows])),
        "rejectedActionRate": (sum(r["rejectedActions"] for r in rows) / submitted) if submitted else None}


def evaluate_gates(cells, cfg):
    teacher = {arm: cells[f"teacher/{arm}"]["success"] for arm in cfg["arms"]}
    floors = {f"{seed}/{arm}": cells[f"floor-{seed}-deterministic/{arm}"]["success"]
              for seed in cfg["initSeeds"] for arm in ("easy", "normal")}
    return {"teacherAchievable": all(v >= cfg["teacherSuccessGate"] for v in teacher.values()),
        "teacherSuccess": teacher,
        "floorsShowNoFreeSignal": all(v <= cfg["floorSuccessCeiling"] for v in floors.values()),
        "floorScriptedDeterministicSuccess": floors}


# -- Collection -------------------------------------------------------------------------


def collect_cell(client, seeds, cfg, arm, *, model, mode, account):
    """One (condition, arm) cell. `model=None` runs the native teacher. Rows keep run_block's
    episode summary plus the option's end state (assigned and alive blue units)."""
    rows = []
    for start in range(0, len(seeds), cfg["blockWorlds"]):
        block = seeds[start:start + cfg["blockWorlds"]]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"], scripted=model is None)

        def choose(wrap, active, observation, raws):
            with torch.no_grad():
                action, *_ = model.act(observation, deterministic=(mode == "deterministic"))
            return numpy_actions(action)

        with scenario_override(arm_scenario(arm)):
            episodes, _, used = v1.run_block(wrapper, block, cfg, choose=None if model is None else choose,
                                             source=f"m8-s4-{mode}-{arm}")
        account(used)
        for index, episode in enumerate(episodes):
            assigned = wrapper.trackers[index].assigned_ids
            final = wrapper.environment.raw_observations[index]["allies"]
            alive = sum(1 for u in final if u["alive"] and u["id"] in assigned)
            rows.append(v1.episode_row(episode) | {"assignedUnits": len(assigned), "blueAliveCount": alive,
                "unitsLostFraction": units_lost_fraction(len(assigned), alive)})
    return rows


def write_rows(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episodes.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def floor_model(init_seed):
    torch.manual_seed(init_seed)
    return FullAuthorityPolicyV1(destination="global")


def run_cells(root, cfg, client, account):
    cells = {}
    plan = [("teacher", None, "deterministic", cfg["teacherWorlds"])]
    plan += [(f"floor-{seed}-{mode}", seed, mode, cfg["floorWorlds"])
             for seed in cfg["initSeeds"] for mode in cfg["modes"]]
    for name, init_seed, mode, worlds in plan:
        for arm_index, arm in enumerate(cfg["arms"]):
            model = None if init_seed is None else floor_model(init_seed)
            if init_seed is not None:
                torch.manual_seed(init_seed * 10 + arm_index)  # sampling stream, fixed per cell
            rows = collect_cell(client, world_seeds(cfg, worlds), cfg, arm, model=model, mode=mode, account=account)
            write_rows(root / name / arm, rows)
            cells[f"{name}/{arm}"] = summarize(rows, cfg)
    return cells


# -- Declaration, run, archive ---------------------------------------------------------


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
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s4_declaration.md"),
        "implementationDigest": file_digest(here),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, cells, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    report = {"format": "snowgym.m8-s4-roster-baseline-report.v0", "cells": cells,
        "gates": evaluate_gates(cells, cfg), "simulatorDecisions": decisions,
        "withinBudgetBound": decisions <= bound["total"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s4-manifest.v0")
    return report


def run(output):
    root = Path(output)
    cfg = configuration()
    declare(root, cfg)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("M8-S4 budget exceeded")

    with SnowGymBatchClient() as client:
        cells = run_cells(root, cfg, client, account)
    return aggregate(root, cfg, cells, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
