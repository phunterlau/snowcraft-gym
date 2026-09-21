"""M8-S9: splitting the throw target into which enemy vs. angular offset (`reviews/m8_s9_declaration.md`).

Teacher-assisted diagnosis only. The simulator's throw uses only the direction from the thrower to the target
point (speed and arc come from power), so the split is defined on headings. Builds on `roster_intervention`
and `roster_trace`; no existing module is edited."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_THROW
from ..ppo import living_unit_mask
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from . import roster_imitation as ri
from . import roster_intervention as rv
from . import roster_trace as rt
from .full_authority_diagnostics import TRAINING
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

S4_RUN, S5_RUN, S7_RUN, S8_RUN = rt.S4_RUN, rt.S5_RUN, rv.S7_RUN, "m8_s8_channel_intervention_v0"
RULES = ("none", "aim", "aim-heading", "enemy", "offset")
MIN_RAY = 0.5


def configuration():
    return {**rv.configuration(), "format": "snowgym.m8-s9-target-split-config.v0", "rules": RULES,
        "budgetCap": 420_000,
        "assistType": "throw-heading substitution by chosen enemy / angular offset (diagnostic assistance, never autonomous)",
        "assistVersion": "snowgym.m8-s9.v0"}


def budget_bound(cfg):
    return {"total": len(cfg["learnerSeeds"]) * len(cfg["rules"]) * cfg["interventionWorlds"] * cfg["optionHorizon"]}


# -- Geometry (declaration section 1) -----------------------------------------------------------


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def bearing(origin, point):
    return math.atan2(point[1] - origin[1], point[0] - origin[0])


def decompose(theta, thrower, enemies):
    """(chosen enemy index, signed offset, all bearings) for a heading `theta`; the chosen enemy is the one
    whose bearing is angularly nearest."""
    bearings = [bearing(thrower, e) for e in enemies]
    offsets = [wrap(theta - b) for b in bearings]
    k = int(np.argmin([abs(o) for o in offsets]))
    return k, offsets[k], bearings


def place_on_ray(thrower, theta, distance, half):
    """A point on the ray from `thrower` at heading `theta`, at most `distance` away and inside the arena.
    Returns (point, ok); `ok` is False when the ray leaves the arena in under MIN_RAY."""
    dx, dy = math.cos(theta), math.sin(theta)
    limit = math.inf
    for delta, position, extent in ((dx, thrower[0], half[0]), (dy, thrower[1], half[1])):
        if delta > 1e-12:
            limit = min(limit, (extent - position) / delta)
        elif delta < -1e-12:
            limit = min(limit, (-extent - position) / delta)
    radius = min(distance, limit)
    if radius < MIN_RAY:
        return None, False
    return (thrower[0] + radius * dx, thrower[1] + radius * dy), True


def new_heading(rule, theta_l, theta_t, thrower, enemies):
    """The heading a rule assigns to one comparable throw, plus the decomposition used to sample statistics."""
    k_l, delta_l, bearings = decompose(theta_l, thrower, enemies)
    k_t, delta_t, _ = decompose(theta_t, thrower, enemies)
    if rule == "aim-heading":
        theta = theta_t
    elif rule == "enemy":
        theta = wrap(bearings[k_t] + delta_l)
    elif rule == "offset":
        theta = wrap(bearings[k_l] + delta_t)
    else:
        raise ValueError(f"{rule!r} does not assign a heading")
    return theta, {"sameEnemy": k_l == k_t, "learnerOffset": abs(delta_l), "teacherOffset": abs(delta_t),
                   "headingGap": abs(wrap(theta_l - theta_t))}


def split_actions(rule, learner, teacher, thrower_xy, enemies_xy, half, living):
    """Apply one rule to arrays `action_type` [n, u], `target` [n, u, 2] (arena-normalized), `power` [n, u].
    `thrower_xy` [n, u, 2] and `half` [n, 2] are world units; `enemies_xy` is a list (per world) of living
    enemy positions. Returns (action, counts, samples)."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    action = {key: value.copy() for key, value in learner.items()}
    n, u = learner["action_type"].shape
    counts = {"learnerThrows": 0, "comparableThrows": 0, "teacherThrows": 0, "learnerMoves": 0,
              "comparableMoves": 0, "unitDecisions": int(living.sum()), "fallbacks": 0}
    samples = []
    for i in range(n):
        for j in range(u):
            if not living[i, j]:
                continue
            l_throw = learner["action_type"][i, j] == ACTION_THROW
            t_throw = teacher["action_type"][i, j] == ACTION_THROW
            counts["learnerThrows"] += int(l_throw)
            counts["teacherThrows"] += int(t_throw)
            if not (l_throw and t_throw):
                continue
            counts["comparableThrows"] += 1
            enemies = enemies_xy[i]
            if not enemies:
                continue
            origin = (float(thrower_xy[i, j, 0]), float(thrower_xy[i, j, 1]))
            h = (float(half[i, 0]), float(half[i, 1]))
            l_point = (learner["target"][i, j, 0] * h[0], learner["target"][i, j, 1] * h[1])
            t_point = (teacher["target"][i, j, 0] * h[0], teacher["target"][i, j, 1] * h[1])
            if math.hypot(l_point[0] - origin[0], l_point[1] - origin[1]) < 1e-6 or \
                    math.hypot(t_point[0] - origin[0], t_point[1] - origin[1]) < 1e-6:
                continue
            theta_l, theta_t = bearing(origin, l_point), bearing(origin, t_point)
            _, stats = new_heading("aim-heading", theta_l, theta_t, origin, enemies)
            samples.append(stats)
            if rule == "aim":
                action["target"][i, j] = teacher["target"][i, j]
            elif rule != "none":
                theta, _ = new_heading(rule, theta_l, theta_t, origin, enemies)
                distance = math.hypot(t_point[0] - origin[0], t_point[1] - origin[1])
                point, ok = place_on_ray(origin, theta, distance, h)
                if not ok:
                    counts["fallbacks"] += 1
                    continue
                action["target"][i, j] = (point[0] / h[0], point[1] / h[1])
    differs = ((action["action_type"] != learner["action_type"]) | np.any(action["target"] != learner["target"], axis=-1)
               | (action["power"] != learner["power"]))
    counts["changed"] = int((differs & living).sum())
    return action, counts, samples


def summarize_samples(samples):
    if not samples:
        return {"comparableThrows": 0}
    q = lambda key: {str(p): float(np.percentile([math.degrees(s[key]) for s in samples], p)) for p in (25, 50, 75)}
    return {"comparableThrows": len(samples), "sameEnemyShare": float(np.mean([s["sameEnemy"] for s in samples])),
            "learnerOffsetDegrees": q("learnerOffset"), "teacherOffsetDegrees": q("teacherOffset"),
            "headingGapDegrees": q("headingGap")}


class SplitSubstituter:
    """`collect_traced` chooser: the learner acts, the teacher is queried in the same state, one rule mixes them.
    The teacher is queried for every rule, including `none`."""

    def __init__(self, model, rule):
        self.model, self.rule = model, rule
        self.totals, self.samples = {}, []

    def __call__(self, wrapper, active, rows, raws):
        with torch.no_grad():
            action, _, _, _ = self.model.act(rows, deterministic=True)
        learner = numpy_actions(action)
        teacher = {k: np.asarray(v) for k, v in wrapper.environment.plan_teacher_tensor_actions_indices(active).items()}
        living = living_unit_mask(rows).cpu().numpy().astype(bool)
        n, capacity = learner["action_type"].shape
        thrower = np.zeros((n, capacity, 2), dtype=np.float64)
        for i, raw in enumerate(raws):
            for j, unit in enumerate(raw["allies"][:capacity]):
                thrower[i, j] = (unit["x"], unit["y"])
        enemies = [[(u["x"], u["y"]) for u in raw["enemies"] if u["alive"]] for raw in raws]
        half = np.array([[raw["arena"]["width"] / 2, raw["arena"]["height"] / 2] for raw in raws])
        new, counts, samples = split_actions(self.rule, learner, teacher, thrower, enemies, half, living)
        for key, value in counts.items():
            self.totals[key] = self.totals.get(key, 0) + value
        self.samples.extend(samples)
        return {key: np.asarray(value).astype(np.int64 if key == "action_type" else np.float32)
                for key, value in new.items()}


# -- Collection -----------------------------------------------------------------------------


def safe_source(rule):
    return "s9-" + re.sub(r"[^A-Za-z0-9-]", "-", rule)


def collect_cell(client, cfg, model, rule, *, account):
    chooser = SplitSubstituter(model, rule)
    rows, traces = [], []
    seeds = rv.world_seeds(cfg)
    for start in range(0, len(seeds), cfg["blockWorlds"]):
        block = seeds[start:start + cfg["blockWorlds"]]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
        with scenario_override(rb.arm_scenario(cfg["arm"])):
            found, found_traces, used = rt.collect_traced(wrapper, block, cfg, choose=chooser, source=safe_source(rule))
        account(used)
        rows.extend(found)
        traces.extend(found_traces)
    for row in rows:
        row["assignedUnits"] = rb.ROSTER
        row["unitsLostFraction"] = rb.units_lost_fraction(rb.ROSTER, row["blueAliveCount"])
    return rows, traces, chooser.totals, chooser.samples


# -- Predictions, scored mechanically (declaration section 3) ----------------------------------------


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
    same = lambda cell: cell["targetSplit"].get("sameEnemyShare")
    gap = lambda cell: (None if "learnerOffsetDegrees" not in cell["targetSplit"] else
                        cell["targetSplit"]["learnerOffsetDegrees"]["50"] - cell["targetSplit"]["teacherOffsetDegrees"]["50"])
    return {"P1": check("enemy", recover, lambda v: v >= 0.5), "P2": check("offset", recover, lambda v: v < 0.5),
            "P3": check("none", same, lambda v: v < 0.5), "P4": check("none", gap, lambda v: v > 0)}


# -- Declaration, run, archive ----------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "s4Manifest": dr.verify_sealed(TRAINING / "runs" / S4_RUN),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / S5_RUN), "s7Manifest": dr.verify_sealed(TRAINING / "runs" / S7_RUN),
        "s8Manifest": dr.verify_sealed(TRAINING / "runs" / S8_RUN),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s9_declaration.md"),
        "implementationDigest": file_digest(here),
        "sourceDigests": {name: file_digest(here.parent / f"{name}.py") for name in
                          ("full_authority_train_v1", "roster_baseline", "roster_imitation", "roster_trace",
                           "roster_intervention")},
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def s8_report():
    return json.loads((TRAINING / "runs" / S8_RUN / "report.json").read_text(encoding="utf-8"))


def traces_match(new, archived):
    """Episode-by-episode equality of two trace lists (the archive's rounding already applied to both)."""
    if len(new) != len(archived):
        return {"passed": False, "reason": f"episode count {len(new)} != {len(archived)}"}
    for a, b in zip(new, archived):
        if a != b:
            return {"passed": False, "firstMismatchSeed": a["seed"] if a["seed"] == b["seed"] else None}
    return {"passed": True, "episodes": len(new)}


def aim_matches_s8(root, seed, count=None):
    new = rt.read_traces(Path(root) / f"learner-{seed}" / "aim" / "traces.jsonl.gz")
    archived = rt.read_traces(TRAINING / "runs" / S8_RUN / f"learner-{seed}" / "aim" / "traces.jsonl.gz")
    return traces_match(new if count is None else new[:count], archived if count is None else archived[:count])


def run_cells(root, cfg, client, account):
    s5_cfg = ri.configuration()
    reference = s8_report()["tensorPathTeacherYield"]
    report = {"learners": {}, "reproduction": {}, "tensorPathTeacherYield": reference}
    for seed in cfg["learnerSeeds"]:
        model = ri.load_final(s5_cfg, TRAINING / "runs" / S5_RUN, seed)
        collected = {}
        for rule in cfg["rules"]:
            collected[rule] = collect_cell(client, cfg, model, rule, account=account)
            rt.write_traces(Path(root) / f"learner-{seed}" / rule, collected[rule][1])
        none_rows = collected["none"][0]
        report["reproduction"][f"{seed}/none-vs-S5"] = rt.reproduction_gate(
            none_rows, rt.archived_rows("learner", cfg["arm"], seed, cfg["interventionWorlds"]))
        cells, none_yield = {}, None
        for rule in cfg["rules"]:
            rows, traces, totals, samples = collected[rule]
            cells[rule] = rv.cell_report(rows, traces, {**totals, "changed": totals.get("changed", 0)}, none_rows, cfg)
            cells[rule]["targetSplit"] = summarize_samples(samples)
            cells[rule]["fallbacks"] = totals.get("fallbacks", 0)
            if rule == "none":
                none_yield = cells[rule]["traceMetrics"]["blueYield"]["mean"]
        for cell in cells.values():
            cell["yieldRecovery"] = rv.recovery(cell["traceMetrics"]["blueYield"]["mean"], none_yield, reference)
        report["reproduction"][f"{seed}/aim-vs-S8"] = aim_matches_s8(root, seed)
        report["learners"][str(seed)] = cells
        del collected
    return report


def aggregate(root, cfg, report, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    final = {"format": "snowgym.m8-s9-target-split-report.v0", **report,
             "requiredGatesPassed": all(g["passed"] for g in report["reproduction"].values()),
             "predictions": score_predictions(report), "simulatorDecisions": decisions,
             "withinBudgetBound": decisions <= bound["total"], "assistType": cfg["assistType"],
             "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", final)
    dr.seal(root, "snowgym.m8-s9-manifest.v0")
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
            raise ValueError("M8-S9 budget exceeded")

    with SnowGymBatchClient() as client:
        report = run_cells(root, cfg, client, account)
    return aggregate(root, cfg, report, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
