"""M8-S8: per-channel teacher substitution on the S5 3v3 learners (`reviews/m8_s8_declaration.md`).

Diagnostic assistance only: every arm here uses the teacher's action for one channel, so no result is
an autonomous one. Reuses `roster_trace.collect_traced` with a substituting chooser; no existing module
is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..ppo import living_unit_mask
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from . import roster_imitation as ri
from . import roster_trace as rt
from .full_authority_diagnostics import TRAINING
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

S4_RUN, S5_RUN, S7_RUN = rt.S4_RUN, rt.S5_RUN, "m8_s7_trace_diagnosis_v0"
RULES = ("none", "aim", "power", "aim+power", "timing", "throw-all", "move")
ALL_ARM = "all"
ARM = "normal"


def configuration():
    return {**rb.configuration(), "format": "snowgym.m8-s8-channel-intervention-config.v0", "arm": ARM,
        "rules": RULES, "learnerSeeds": rt.SEEDS, "interventionWorlds": 100, "blockWorlds": 50,
        "allControlSeed": 97101, "budgetCap": 600_000, "bootstrapSamples": 10_000, "bootstrapSeed": 973001,
        "engageRange": 9.0,
        "assistType": "per-channel teacher substitution (diagnostic assistance, never autonomous)",
        "assistVersion": "snowgym.m8-s8.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    worlds = cfg["interventionWorlds"]
    arms = len(cfg["learnerSeeds"]) * len(cfg["rules"]) * worlds * horizon
    controls = 2 * worlds * horizon
    return {"arms": arms, "allControls": controls, "total": arms + controls}


def world_seeds(cfg):
    return list(range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + cfg["interventionWorlds"]))


# -- Substitution rules (declaration section 1) --------------------------------------------


def substitute(rule, learner, teacher, throw_target, throw_power, living):
    """Apply one rule to the learner's action arrays (`action_type` [n, u], `target` [n, u, 2],
    `power` [n, u]) using the teacher's. Returns the new action dict and per-call counts. Only living
    units are ever changed."""
    if rule not in RULES and rule != ALL_ARM:
        raise ValueError(f"unknown rule {rule!r}")
    action = {key: value.copy() for key, value in learner.items()}
    throws_learner = living & (learner["action_type"] == ACTION_THROW)
    throws_teacher = living & (teacher["action_type"] == ACTION_THROW)
    moves_learner = living & (learner["action_type"] == ACTION_MOVE)
    moves_teacher = living & (teacher["action_type"] == ACTION_MOVE)
    counts = {"learnerThrows": int(throws_learner.sum()), "comparableThrows": int((throws_learner & throws_teacher).sum()),
              "learnerMoves": int(moves_learner.sum()), "comparableMoves": int((moves_learner & moves_teacher).sum()),
              "teacherThrows": int(throws_teacher.sum()), "unitDecisions": int(living.sum()), "changed": 0}

    def take(mask, keys):
        for key in keys:
            action[key][mask] = teacher[key][mask]

    if rule in ("aim", "aim+power"):
        take(throws_learner & throws_teacher, ("target",))
    if rule in ("power", "aim+power"):
        take(throws_learner & throws_teacher, ("power",))
    if rule in ("timing", "throw-all"):
        add = throws_teacher & ~throws_learner
        action["action_type"][add] = ACTION_THROW
        action["target"][add] = throw_target[add]
        action["power"][add] = throw_power[add]
        take(throws_learner & ~throws_teacher, ("action_type", "target", "power"))
        if rule == "throw-all":
            now = living & (action["action_type"] == ACTION_THROW)
            take(now & throws_teacher, ("target", "power"))
    if rule == "move":
        take(moves_learner & moves_teacher, ("target",))
    if rule == ALL_ARM:
        take(living, ("action_type", "target", "power"))
    differs = ((action["action_type"] != learner["action_type"]) | np.any(action["target"] != learner["target"], axis=-1)
               | (action["power"] != learner["power"]))
    counts["changed"] = int((differs & living).sum())
    return action, counts


class Substituter:
    """`run_block`/`collect_traced` chooser: the learner acts, the teacher is queried in the same state,
    and one rule mixes them. The teacher is queried for every rule, including `none`."""

    def __init__(self, model, rule):
        self.model, self.rule = model, rule
        self.totals = {}

    def __call__(self, wrapper, active, rows, _raws):
        with torch.no_grad():
            prediction = self.model(rows, with_value=False)
            learner, _, _, _ = self.model.act(rows, deterministic=True)
        learner = numpy_actions(learner)
        teacher = wrapper.environment.plan_teacher_tensor_actions_indices(active)
        living = living_unit_mask(rows).cpu().numpy().astype(bool)
        throw_target = torch.tanh(prediction["throw_raw"]).cpu().numpy().astype(np.float32)
        throw_power = torch.sigmoid(prediction["power_raw"]).cpu().numpy().astype(np.float32)
        action, counts = substitute(self.rule, learner, {k: np.asarray(v) for k, v in teacher.items()},
                                    throw_target, throw_power, living)
        for key, value in counts.items():
            self.totals[key] = self.totals.get(key, 0) + value
        return {key: np.asarray(value).astype(np.int64 if key == "action_type" else np.float32)
                for key, value in action.items()}


# -- Collection -----------------------------------------------------------------------------


def collect_cell(client, cfg, model, rule, *, account, arm=None):
    substituter = Substituter(model, rule)
    arm = arm or cfg["arm"]
    rows, traces = [], []
    seeds = world_seeds(cfg)
    for start in range(0, len(seeds), cfg["blockWorlds"]):
        block = seeds[start:start + cfg["blockWorlds"]]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
        with scenario_override(rb.arm_scenario(arm)):
            found, found_traces, used = rt.collect_traced(wrapper, block, cfg, choose=substituter,
                                                          source=f"s8-{rule.replace('+', '-and-')}")
        account(used)
        rows.extend(found)
        traces.extend(found_traces)
    for row in rows:
        row["assignedUnits"] = rb.ROSTER
        row["unitsLostFraction"] = rb.units_lost_fraction(rb.ROSTER, row["blueAliveCount"])
    return rows, traces, substituter.totals


def coverage(totals):
    def share(numerator, denominator):
        return totals.get(numerator, 0) / totals[denominator] if totals.get(denominator) else None
    return {"throwCoverage": share("comparableThrows", "learnerThrows"),
            "moveCoverage": share("comparableMoves", "learnerMoves"),
            "changedShareOfUnitDecisions": share("changed", "unitDecisions"), **totals}


def recovery(arm_yield, none_yield, teacher_yield):
    if None in (arm_yield, none_yield, teacher_yield) or teacher_yield <= none_yield:
        return None
    return (arm_yield - none_yield) / (teacher_yield - none_yield)


def cell_report(rows, traces, totals, none_rows, cfg):
    summary = rb.summarize(rows, cfg)
    metrics = rt.cell_summary([rt.episode_metrics(t, cfg) for t in traces], cfg)
    return {"summary": summary, "traceMetrics": metrics, "coverage": coverage(totals),
            "successMinusNone": ri.paired_difference(rows, none_rows, lambda r: r["success"], cfg),
            "unitsLostMinusNone": ri.paired_difference(rows, none_rows, lambda r: r["unitsLostFraction"], cfg)}


# -- Predictions, scored mechanically (declaration section 3) --------------------------------------


def score_predictions(report):
    seeds = list(report["learners"])

    def check(rule, fn, test):
        if any(rule not in report["learners"][s] for s in seeds):
            return {"holds": None, "reason": "arm not collected"}
        found = {s: fn(report["learners"][s][rule]) for s in seeds}
        if any(v is None for v in found.values()):
            return {"holds": None, "reason": "undefined", "values": found}
        return {"holds": bool(all(test(v) for v in found.values())), "values": found}

    recover = lambda cell: cell["yieldRecovery"]
    success = lambda cell: cell["summary"]["success"]
    return {"P1": check("aim", recover, lambda v: v >= 0.5), "P2": check("power", recover, lambda v: v < 0.25),
            "P3": check("timing", recover, lambda v: v < 0.25), "P4": check("throw-all", recover, lambda v: v >= 0.75),
            "P5": check("throw-all", success, lambda v: v >= 0.25), "P6": check("move", success, lambda v: v < 0.10)}


# -- Declaration, run, archive ----------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "s4Manifest": dr.verify_sealed(TRAINING / "runs" / S4_RUN),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / S5_RUN),
        "s7Manifest": dr.verify_sealed(TRAINING / "runs" / S7_RUN),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s8_declaration.md"),
        "implementationDigest": file_digest(here),
        "sourceDigests": {name: file_digest(here.parent / f"{name}.py")
                          for name in ("full_authority_train_v1", "roster_baseline", "roster_imitation", "roster_trace")},
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def native_teacher_yield(cfg):
    report = json.loads((TRAINING / "runs" / S7_RUN / "report.json").read_text(encoding="utf-8"))
    return report["teacher"][cfg["arm"]]["blueYield"]["mean"]


def run_all_controls(root, cfg, client, account):
    """The tensor-path teacher, once per Red arm. The `normal` yield is the recovery reference."""
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / S5_RUN, cfg["allControlSeed"])
    controls = {}
    for arm in ("normal", "random"):
        rows, traces, totals = collect_cell(client, cfg, model, ALL_ARM, account=account, arm=arm)
        rt.write_traces(Path(root) / "all-control" / arm, traces)
        archived = rt.archived_rows("teacher", arm, None, cfg["interventionWorlds"])
        controls[arm] = {"reproducesNativeS4Teacher": rt.reproduction_gate(rows, archived),
            "summary": rb.summarize(rows, cfg),
            "traceMetrics": rt.cell_summary([rt.episode_metrics(t, cfg) for t in traces], cfg)}
    return controls


def run_cells(root, cfg, client, account):
    s5_cfg = ri.configuration()
    controls = run_all_controls(root, cfg, client, account)
    reference = controls[cfg["arm"]]["traceMetrics"]["blueYield"]["mean"]
    report = {"learners": {}, "reproduction": {}, "allControls": controls, "tensorPathTeacherYield": reference,
              "nativeTeacherYield": native_teacher_yield(cfg)}
    for seed in cfg["learnerSeeds"]:
        model = ri.load_final(s5_cfg, TRAINING / "runs" / S5_RUN, seed)
        collected = {}
        for rule in cfg["rules"]:
            collected[rule] = collect_cell(client, cfg, model, rule, account=account)
            rt.write_traces(Path(root) / f"learner-{seed}" / rule, collected[rule][1])
        none_rows = collected["none"][0]
        report["reproduction"][f"{seed}/none"] = rt.reproduction_gate(
            none_rows, rt.archived_rows("learner", cfg["arm"], seed, cfg["interventionWorlds"]))
        cells, none_yield = {}, None
        for rule in cfg["rules"]:
            rows, traces, totals = collected[rule]
            cells[rule] = cell_report(rows, traces, totals, none_rows, cfg)
            if rule == "none":
                none_yield = cells[rule]["traceMetrics"]["blueYield"]["mean"]
        for cell in cells.values():
            cell["yieldRecovery"] = recovery(cell["traceMetrics"]["blueYield"]["mean"], none_yield, reference)
        report["learners"][str(seed)] = cells
        del collected
    return report


def aggregate(root, cfg, report, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    final = {"format": "snowgym.m8-s8-channel-intervention-report.v0", **report,
             "noneReproducesS5": all(g["passed"] for g in report["reproduction"].values()),
             "predictions": score_predictions(report), "simulatorDecisions": decisions,
             "withinBudgetBound": decisions <= bound["total"], "assistType": cfg["assistType"],
             "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", final)
    dr.seal(root, "snowgym.m8-s8-manifest.v0")
    return final


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("M8-S8 budget exceeded")

    with SnowGymBatchClient() as client:
        report = run_cells(root, cfg, client, account)
    return aggregate(root, cfg, report, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
