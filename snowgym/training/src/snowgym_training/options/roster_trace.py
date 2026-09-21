"""M8-S7: trace-level diagnosis of the post-contact fight at 3v3 (`reviews/m8_s7_declaration.md`).

Observation only. The loop mirrors `full_authority_train_v1.run_block` (reimplemented because
`opponent_transfer.collect_logged` is 1v1-specific) and additionally records the raw simulator
state of every unit and projectile before each decision. Nothing is trained or intervened on."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from . import roster_imitation as ri
from .full_authority_diagnostics import TRAINING
from .opponent_transfer import scenario_override
from .plans import teacher_option_plan
from .reservoir import file_digest
from .supervised_probe import write_json

S4_RUN, S5_RUN = "m8_s4_roster_baseline_v0", "m8_s5_roster_imitation_v0"
ARMS = ("normal", "random")
SEEDS = (97101, 97102, 97103)
ROW_KEYS = ("success", "blueAliveCount", "finalDecision", "finalTargetDamage", "firstHitDecision")
FULL_HEALTH = 100.0


def configuration():
    return {**rb.configuration(), "format": "snowgym.m8-s7-trace-diagnosis-config.v0", "arms": ARMS,
        "learnerSeeds": SEEDS, "traceWorlds": 100, "blockWorlds": 50, "engageRange": 9.0,
        "budgetCap": 220_000, "bootstrapSamples": 10_000, "bootstrapSeed": 973001,
        "assistType": "none; observation of frozen checkpoints and the native teacher",
        "assistVersion": "snowgym.m8-s7.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    learners = len(cfg["learnerSeeds"]) * len(cfg["arms"]) * cfg["traceWorlds"] * horizon
    teacher = len(cfg["arms"]) * cfg["traceWorlds"] * horizon
    return {"learners": learners, "teacher": teacher, "total": learners + teacher}


def trace_seeds(cfg):
    return list(range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + cfg["traceWorlds"]))


# -- Collection ------------------------------------------------------------------------


def unit_state(unit):
    return [int(unit["alive"]), max(0.0, float(unit["health"])), unit["x"], unit["y"], unit["vx"], unit["vy"],
            unit["stunRemaining"], unit["throwPhaseRemaining"], unit["throwCooldown"]]


def snapshot(raw):
    return {"blue": [unit_state(u) for u in raw["allies"]], "red": [unit_state(u) for u in raw["enemies"]],
            "proj": [[p["id"], p["ownerId"], p["team"], p["x"], p["y"], p["vx"], p["vy"]] for p in raw["projectiles"]]}


def collect_traced(wrapper, seeds, cfg, *, choose, source):
    """Returns (episode rows, traces, decisions). Each trace holds `states` (one per decision
    plus the final state) and `acts` (per decision, blue (unitId, type, accepted) triples)."""
    count = len(seeds)
    if wrapper.batch_size != count:
        raise ValueError("block seeds must match the wrapper batch size")
    plan, _ = teacher_option_plan("engage")
    spec = v1.engage_spec(cfg)
    observation, _ = wrapper.reset(list(seeds), [v1.scenario()] * count, [f"m8-s7-{source}-{s}" for s in seeds],
                                   [plan] * count, [spec] * count)
    current = v1.tensor_dict(observation)
    episodes = [{"seed": int(seed), "source": source, "rewards": [], "remainingFraction": [], "distances": [],
                 "targetDamage": [], "rejectedActions": 0, "totalActions": 0} for seed in seeds]
    traces = [{"seed": int(seed), "states": [], "acts": []} for seed in seeds]
    decisions = 0
    while True:
        active = [i for i in range(count) if not wrapper.trackers[i].finished]
        if not active:
            break
        rows = {key: value[active] for key, value in current.items()}
        raws = [wrapper.environment.raw_observations[i] for i in active]
        distances = [v1.target_distance(raw, wrapper.trackers[i]) for raw, i in zip(raws, active)]
        for i, raw in zip(active, raws):
            if len(raw["allies"]) != rb.ROSTER or len(raw["enemies"]) != rb.ROSTER:
                raise RuntimeError("roster trace expects 3v3")
            traces[i]["states"].append(snapshot(raw))
        if choose is None:
            raw_next, rewards, _, _, infos = wrapper.step_scripted_indices(active)
        else:
            raw_next, rewards, _, _, infos = wrapper.step_indices(active, choose(wrapper, active, rows, raws))
        decisions += len(active)
        following = v1.tensor_dict(raw_next)
        for key in current:
            current[key][active] = following[key]
        for row, index in enumerate(active):
            episode, option = episodes[index], infos[row]["option"]
            episode["remainingFraction"].append(float(rows["option_state"][row, 0]))
            episode["rewards"].append(float(rewards[row]))
            episode["distances"].append(distances[row])
            episode["targetDamage"].append(float(option["metrics"]["targetDamage"]))
            results = infos[row].get("actionResults", [])
            episode["rejectedActions"] += sum(r.get("accepted") is False for r in results)
            episode["totalActions"] += len(results)
            traces[index]["acts"].append([[r["action"]["unitId"], r["action"]["type"], bool(r["accepted"])]
                                          for r in results])
            if wrapper.trackers[index].finished:
                final = wrapper.environment.raw_observations[index]
                traces[index]["states"].append(snapshot(final))
                episode.update(success=bool(option["success"]), failed=bool(option["failed"]),
                    timedOut=bool(option["timedOut"]), finalDecision=int(option["decision"]),
                    blueAliveAtEnd=any(u["alive"] and u["id"] in wrapper.trackers[index].assigned_ids
                                       for u in final["allies"]))
                episode["blueAliveCount"] = sum(1 for u in final["allies"]
                                                if u["alive"] and u["id"] in wrapper.trackers[index].assigned_ids)
    rows_out = [v1.episode_row(e) | {"blueAliveCount": e["blueAliveCount"]} for e in episodes]
    return rows_out, traces, decisions


def collect_cell(client, seeds, cfg, arm, *, model, account):
    rows, traces = [], []
    for start in range(0, len(seeds), cfg["blockWorlds"]):
        block = seeds[start:start + cfg["blockWorlds"]]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"], scripted=model is None)

        def choose(wrap, active, observation, raws):
            with torch.no_grad():
                action, *_ = model.act(observation, deterministic=True)
            return numpy_actions(action)

        with scenario_override(rb.arm_scenario(arm)):
            found, found_traces, used = collect_traced(wrapper, block, cfg, choose=None if model is None else choose,
                                                       source=f"{'teacher' if model is None else 'learner'}-{arm}")
        account(used)
        rows.extend(found)
        traces.extend(found_traces)
    return rows, traces


# -- Measures (declaration section 2) ------------------------------------------------------


def _dist(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def _alive(unit):
    return unit[0] == 1


def red_total(state):
    return sum(u[1] for u in state["red"])


def first_hit_decision(states):
    for t in range(len(states) - 1):
        if red_total(states[t + 1]) < red_total(states[t]) - 1e-9:
            return t + 1
    return None


def _mean(values):
    return float(np.mean(values)) if values else None


def episode_metrics(trace, cfg):
    states = trace["states"]
    horizon = len(states) - 1
    hit = first_hit_decision(states)
    metrics = {"firstHitDecision": hit}
    first_death = wipe = None
    for t in range(1, len(states)):
        alive_before = sum(_alive(u) for u in states[t - 1]["blue"])
        alive_now = sum(_alive(u) for u in states[t]["blue"])
        if first_death is None and alive_now < alive_before:
            first_death = t
        if wipe is None and alive_now == 0:
            wipe = t
    metrics["firstDeathAfterHit"] = None if hit is None or first_death is None else first_death - hit
    metrics["wipeAfterHit"] = None if hit is None or wipe is None else wipe - hit
    final = states[-1]
    damage = [max(0.0, FULL_HEALTH - u[1]) for u in final["red"]]
    total_damage = sum(damage)
    dead = [not _alive(u) for u in final["red"]]
    metrics["redKills"] = sum(dead)
    metrics["wastedDamageShare"] = (sum(d for d, k in zip(damage, dead) if not k) / total_damage
                                    if total_damage > 0 else None)
    blue_lost = sum(max(0.0, FULL_HEALTH - u[1]) for u in final["blue"])
    first_seen = {}
    for k, state in enumerate(states):
        for pid, owner, team, *_ in state["proj"]:
            first_seen.setdefault(pid, (k, team))
    blue_shots = [k for k, team in first_seen.values() if team == "blue"]
    red_shots = [k for k, team in first_seen.values() if team == "red"]
    metrics["blueYield"] = total_damage / len(blue_shots) if blue_shots else None
    metrics["redYield"] = blue_lost / len(red_shots) if red_shots else None
    if hit is None:
        return metrics | {k: None for k in ("spacingAtFirstHit", "nearestRedDistance", "blueShotsPerUnitDecision",
            "redShotsPerUnitDecision", "displacementUnderThreat", "displacementNoThreat", "incapacitatedShare",
            "throwActionShare", "moveActionShare")}
    window = range(hit - 1, horizon)
    living_blue = living_red = 0
    incapacitated = 0
    nearest, under, calm = [], [], []
    for t in window:
        state, nxt = states[t], states[t + 1]
        reds = [u for u in state["red"] if _alive(u)]
        threats = [p for p in state["proj"] if p[2] == "red"]
        living_red += len(reds)
        for j, unit in enumerate(state["blue"]):
            if not _alive(unit):
                continue
            living_blue += 1
            incapacitated += int(unit[6] > 0 or unit[7] > 0)
            if reds:
                nearest.append(min(_dist(unit, r) for r in reds))
            after = nxt["blue"][j]
            if _alive(after):
                step = _dist(unit, after)
                near = any(math.hypot(p[3] - unit[2], p[4] - unit[3]) <= cfg["engageRange"] for p in threats)
                (under if near else calm).append(step)
    pre = states[hit - 1]["blue"]
    living = [u for u in pre if _alive(u)]
    pairs = [_dist(a, b) for i, a in enumerate(living) for b in living[i + 1:]]
    metrics["spacingAtFirstHit"] = _mean(pairs)
    metrics["nearestRedDistance"] = _mean(nearest)
    metrics["blueShotsPerUnitDecision"] = (sum(1 for k in blue_shots if k >= hit) / living_blue) if living_blue else None
    metrics["redShotsPerUnitDecision"] = (sum(1 for k in red_shots if k >= hit) / living_red) if living_red else None
    metrics["displacementUnderThreat"] = _mean(under)
    metrics["displacementNoThreat"] = _mean(calm)
    metrics["incapacitatedShare"] = incapacitated / living_blue if living_blue else None
    types = [kind for t in range(hit - 1, horizon) for _uid, kind, _ok in trace["acts"][t]]
    metrics["throwActionShare"] = types.count("throw") / len(types) if types else None
    metrics["moveActionShare"] = types.count("move") / len(types) if types else None
    return metrics


METRIC_KEYS = ("firstHitDecision", "firstDeathAfterHit", "wipeAfterHit", "redKills", "wastedDamageShare",
               "blueYield", "redYield", "spacingAtFirstHit", "nearestRedDistance", "blueShotsPerUnitDecision",
               "redShotsPerUnitDecision", "displacementUnderThreat", "displacementNoThreat", "incapacitatedShare",
               "throwActionShare", "moveActionShare")


def cell_summary(per_episode, cfg):
    out = {"episodes": len(per_episode)}
    for key in METRIC_KEYS:
        values = [m[key] for m in per_episode if m[key] is not None]
        out[key] = {"defined": len(values), "mean": float(np.mean(values)) if values else None,
                    "interval": rb.bootstrap_mean(values, samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
                    if values else None}
    return out


def reproduction_gate(rows, archived):
    by_seed = {r["seed"]: r for r in archived}
    mismatches = [{"seed": r["seed"], "key": key, "traced": r[key], "archived": by_seed[r["seed"]][key]}
                  for r in rows for key in ROW_KEYS if r[key] != by_seed[r["seed"]][key]]
    return {"worlds": len(rows), "mismatches": mismatches, "passed": not mismatches}


# -- Predictions, scored mechanically (declaration section 3) -----------------------------------


def _value(cell, key):
    return cell[key]["mean"]


def score_predictions(report):
    teacher, learners = report["teacher"], report["learners"]

    def check(arm, key, rule):
        if arm not in teacher or any(arm not in cells for cells in learners.values()):
            return {"holds": None, "reason": "arm not collected"}
        t = _value(teacher[arm], key)
        values = {seed: _value(cells[arm], key) for seed, cells in learners.items()}
        if t is None or any(v is None for v in values.values()):
            return {"holds": None, "reason": "undefined", "teacher": t, "learners": values}
        if rule != "waste" and t == 0:
            return {"holds": None, "reason": "teacher value is zero", "teacher": t, "learners": values}
        holds = {
            "waste": all(v >= 0.5 and v >= t + 0.30 for v in values.values()),
            "le80": all(v <= 0.8 * t for v in values.values()),
            "le75": all(v <= 0.75 * t for v in values.values()),
            "ge80": all(v >= 0.8 * t for v in values.values()),
            "ge125": all(v >= 1.25 * t for v in values.values()),
            "band": all(0.75 * t <= v <= 1.25 * t for v in values.values()),
        }[rule]
        return {"holds": bool(holds), "teacher": t, "learners": values}

    return {
        "P1": check("normal", "wastedDamageShare", "waste"),
        "P2": check("normal", "spacingAtFirstHit", "le80"),
        "P3": check("normal", "blueShotsPerUnitDecision", "le75"),
        "P4": check("normal", "displacementUnderThreat", "ge80"),
        "P5": check("normal", "redYield", "ge125"),
        "P6": check("normal", "incapacitatedShare", "ge125"),
        "P7": check("random", "nearestRedDistance", "le80"),
        "P8": check("random", "redShotsPerUnitDecision", "band"),
    }


# -- Declaration, run, archive --------------------------------------------------------------


def write_traces(directory, traces):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "traces.jsonl.gz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")

    def rounded(value):
        if isinstance(value, float):
            return round(value, 2)
        if isinstance(value, list):
            return [rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value

    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        for trace in traces:
            handle.write((json.dumps(rounded(trace), separators=(",", ":"), allow_nan=False) + "\n").encode())


def read_traces(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "s4Manifest": dr.verify_sealed(TRAINING / "runs" / S4_RUN),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / S5_RUN),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s7_declaration.md"),
        "implementationDigest": file_digest(here),
        "sourceDigests": {name: file_digest(here.parent / f"{name}.py")
                          for name in ("full_authority_train_v1", "roster_baseline", "roster_imitation")},
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def archived_rows(kind, arm, seed, worlds):
    if kind == "teacher":
        path = TRAINING / "runs" / S4_RUN / "teacher" / arm / "episodes.jsonl"
    else:
        path = TRAINING / "runs" / S5_RUN / "paired-eval" / f"seed-{seed}" / arm / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:worlds]


def run_cells(root, cfg, client, account):
    seeds = trace_seeds(cfg)
    s5_cfg = ri.configuration()
    report = {"teacher": {}, "learners": {}, "reproduction": {}}
    for arm in cfg["arms"]:
        rows, traces = collect_cell(client, seeds, cfg, arm, model=None, account=account)
        write_traces(root / "teacher" / arm, traces)
        report["reproduction"][f"teacher/{arm}"] = reproduction_gate(rows, archived_rows("teacher", arm, None, len(seeds)))
        report["teacher"][arm] = cell_summary([episode_metrics(t, cfg) for t in traces], cfg)
    for seed in cfg["learnerSeeds"]:
        model = ri.load_final(s5_cfg, TRAINING / "runs" / S5_RUN, seed)
        for arm in cfg["arms"]:
            rows, traces = collect_cell(client, seeds, cfg, arm, model=model, account=account)
            write_traces(root / f"learner-{seed}" / arm, traces)
            report["reproduction"][f"{seed}/{arm}"] = reproduction_gate(rows, archived_rows("learner", arm, seed, len(seeds)))
            report["learners"].setdefault(str(seed), {})[arm] = cell_summary(
                [episode_metrics(t, cfg) for t in traces], cfg)
    return report


def aggregate(root, cfg, report, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    final = {"format": "snowgym.m8-s7-trace-diagnosis-report.v0", **report,
             "allReproductionGatesPassed": all(g["passed"] for g in report["reproduction"].values()),
             "predictions": score_predictions(report), "simulatorDecisions": decisions,
             "withinBudgetBound": decisions <= bound["total"],
             "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", final)
    dr.seal(root, "snowgym.m8-s7-manifest.v0")
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
            raise ValueError("M8-S7 budget exceeded")

    with SnowGymBatchClient() as client:
        report = run_cells(root, cfg, client, account)
    return aggregate(root, cfg, report, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
