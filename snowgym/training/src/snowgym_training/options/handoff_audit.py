"""R1m-S7: inspect exact archived actions around the teacher/source handoff."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import duration_probe as d
from .control_channels import recommend_movement, validate_movement_agreement
from .throw_channels import recommend_shots, validate_teacher_agreement
from .opportunity_audit import plain, write_jsonl
from .supervised_probe import write_json

b = d.b
ARCHIVE = b.TRAINING / "runs/m7b_engage_r1m_s6_v0"
ARMS = ("keep", "squad-30")
WINDOWS = {"correction": (0, 30), "lateCorrection": (20, 30),
           "immediateTail": (30, 40), "firstTail": (30, 60), "allTail": (30, None)}


def inputs():
    _, metadata, selection, s3, _ = d.inputs()
    manifest = b.verified_manifest(ARCHIVE)
    cfg = json.loads((ARCHIVE/"declaration.json").read_text())
    if cfg["implementationDigest"] != b.file_digest(Path(d.__file__)) or cfg["source"] != metadata:
        raise ValueError("S6 implementation or checkpoint drift")
    if json.loads((ARCHIVE/"selection.json").read_text()) != selection:
        raise ValueError("S6 selection drift")
    return selection, s3, manifest


def load_row(seed, arm):
    if arm not in ARMS:
        raise ValueError("unknown inspection arm")
    with gzip.open(ARCHIVE/f"branch-{seed}-{arm}.jsonl.gz", "rt") as stream:
        return json.loads(stream.readline())


def enemy_snapshot(wrapper):
    raw = wrapper.environment.raw_observations[0]
    tracker = wrapper.trackers[0]
    return {"objectiveHealth": tracker.target_health_fraction(raw),
        "remainingBudget": tracker.spec.horizon-tracker.decision,
        "activatedTargetIds": list(tracker.activated_target_ids),
        "activatedHealth": tracker.activated_target_health,
        "enemies": copy.deepcopy(raw["enemies"]), "allies": copy.deepcopy(raw["allies"])}


def unit_metrics(raw, actions, observation, movement, shots):
    enemies = {e["id"]: e for e in raw["enemies"] if e["alive"]}
    scale = np.asarray([raw["arena"]["width"]/2, raw["arena"]["height"]/2])
    rows = []
    for i, unit in enumerate(raw["allies"]):
        if not unit["alive"]:
            continue
        enemy = enemies.get(int(shots["enemyIds"][0, i]))
        if enemy is None:
            raise ValueError("living fighter has no inspectable enemy before terminal")
        delta = np.asarray([enemy["x"]-unit["x"], enemy["y"]-unit["y"]])
        distance = float(np.linalg.norm(delta)); axis = delta/max(distance, 1e-9)
        kind = int(actions["action_type"][0, i])
        point = actions["target"][0, i]*scale
        if kind == 1:
            point = np.clip(point, -scale+.5, scale-.5)
        mask = observation["unit_action_mask"][0, i]
        rows.append({"unit": copy.deepcopy(unit), "type": kind, "mask": plain(mask),
            "target": point.tolist(), "enemyId": enemy["id"], "enemyHealth": enemy["health"],
            "distance": distance, "inRange": distance <= 9, "throwReady": bool(mask[2]),
            "moveAvailable": bool(mask[1]), "recommendationAvailable": bool(movement["valid"][0, i]),
            "shotAvailable": bool(shots["valid"][0, i]), "threat": bool(movement["threat"][0, i]),
            "outwardMove": bool(kind == 1 and np.dot(point-np.asarray([unit["x"], unit["y"]]), axis) < -1e-6),
            "rangeOpening": bool(np.dot(np.asarray([enemy["vx"]-unit["vx"], enemy["vy"]-unit["vy"]]), axis) > 1e-6)})
    return rows


def window_metrics(rows, start, end):
    if start < 0 or end is not None and end <= start:
        raise ValueError("invalid inspection window")
    selected = rows[start:end]
    units = [u for row in selected for u in row["units"]]
    def fraction(numerator, denominator):
        return numerator/denominator if denominator else None
    throws = [u for u in units if u["type"] == 2]
    ready = [u for u in units if u["throwReady"] and u["inRange"]]
    outside_moves = [u for u in units if u["type"] == 1 and not u["inRange"]]
    outside = [u for u in units if not u["inRange"]]
    gain = sum(r["progressGain"] for r in selected)
    return {"decisions": len(selected), "complete": bool(selected) and (end is None or len(rows) >= end),
        "unitOpportunities": len(units), "throwOrders": len(throws), "readyInRange": len(ready),
        "readyInRangeNotThrowing": sum(u["type"] != 2 for u in ready),
        "outsideMoves": len(outside_moves), "outwardOutsideMoves": sum(u["outwardMove"] for u in outside_moves),
        "progressGain": gain, "progressPerDecision": fraction(gain, len(selected)),
        "damageDealt": sum(r["damageDealt"] for r in selected),
        "damageReceived": sum(r["damageReceived"] for r in selected),
        "rangeOccupancy": fraction(sum(u["inRange"] for u in units), len(units)),
        "meanRangeError": fraction(sum(abs(u["distance"]-6.5) for u in units), len(units)),
        "throwOrderFraction": fraction(len(throws), len(units)),
        "throwsOutsideFraction": fraction(sum(not u["inRange"] for u in throws), len(throws)),
        "readyInRangeNotThrowingFraction": fraction(sum(u["type"] != 2 for u in ready), len(ready)),
        "outwardOutsideMoveFraction": fraction(sum(u["outwardMove"] for u in outside_moves), len(outside_moves)),
        "outsideRangeOpeningFraction": fraction(sum(u["rangeOpening"] for u in outside), len(outside))}


def inspect(wrapper, frame, pick, row):
    if row["seed"] != pick["seed"] or row["arm"] not in ARMS:
        raise ValueError("inspection row identity mismatch")
    if not row["trace"] or len(row["stateHashes"]) != len(row["trace"])+1:
        raise ValueError("invalid archived trace length")
    if json_digest([e["action"] for e in row["trace"]]) != row["actionsDigest"]:
        raise ValueError("archived action digest mismatch")
    obs = b.post_hit.restore(wrapper, frame["seed"], frame["trigger"]["prefix"], pick["identity"])
    if wrapper.environment.state_hashes[0] != row["stateHashes"][0]:
        raise ValueError("initial hash mismatch")
    snapshots = {"start": enemy_snapshot(wrapper), "handoff": None}
    raw = wrapper.environment.raw_observations[0]
    initial_health = (b.team_health(raw["allies"]), b.team_health(raw["enemies"]))
    records = []
    for offset, saved in enumerate(row["trace"]):
        if saved["offset"] != offset:
            raise ValueError("noncontiguous archived offsets")
        actions = {k: np.asarray(v, dtype=np.int64 if k == "action_type" else np.float32) for k, v in saved["action"].items()}
        raw = wrapper.environment.raw_observations[0]
        identity = b.post_hit.identity(wrapper, obs)
        movement = recommend_movement(raw, actions["action_type"].shape[1])
        shots = recommend_shots(raw, actions["action_type"].shape[1])
        teacher = wrapper.environment.plan_teacher_tensor_actions()
        validate_movement_agreement(teacher, movement); validate_teacher_agreement(teacher, shots)
        if b.post_hit.identity(wrapper, obs) != identity:
            raise ValueError("inspection labeling mutated state")
        units = unit_metrics(raw, actions, obs, movement, shots)
        previous = wrapper.trackers[0].target_health_fraction(raw)
        health = (b.team_health(raw["allies"]), b.team_health(raw["enemies"]))
        obs, reward, terminated, truncated, infos = wrapper.step(actions)
        info = infos[0]; done = bool(terminated[0] or truncated[0])
        if (wrapper.environment.state_hashes[0] != saved["stateHash"] or saved["stateHash"] != row["stateHashes"][offset+1]
            or float(reward[0]) != saved["reward"] or plain(info["option"]) != saved["option"]
            or plain(info["actionResults"]) != saved["actionResults"] or done != (offset == len(row["trace"])-1)):
            raise ValueError("archived transition mismatch")
        raw = wrapper.environment.raw_observations[0]
        current = wrapper.trackers[0].target_health_fraction(raw)
        records.append({"offset": offset, "units": units, "stateHash": saved["stateHash"],
            "preObjectiveHealth": previous, "postObjectiveHealth": current, "progressGain": previous-current,
            "damageDealt": health[1]-b.team_health(raw["enemies"]), "damageReceived": health[0]-b.team_health(raw["allies"])})
        if offset == 29:
            snapshots["handoff"] = enemy_snapshot(wrapper)
    if b.post_hit.outcome(wrapper, info, initial_health, len(records)) != row["final"]:
        raise ValueError("archived final outcome mismatch")
    snapshots["final"] = enemy_snapshot(wrapper)
    gains = [r["offset"] for r in records if r["progressGain"] > 1e-9]
    return {"seed": row["seed"], "arm": row["arm"], **d.LABEL, "final": row["final"],
        "snapshots": snapshots, "records": records,
        "windows": {k: window_metrics(records, *bounds) for k, bounds in WINDOWS.items()},
        "lastGainOffset": gains[-1] if gains else None,
        "finalProgressFreeDecisions": len(records)-gains[-1]-1 if gains else len(records),
        "remainingHealthAboveSuccess": max(0., snapshots["final"]["objectiveHealth"]-.2),
        "simulatorDecisions": len(frame["trigger"]["prefix"])+len(records)}


def statistic(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(991001)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"count": len(values), "mean": float(values.mean()), "ci95": np.quantile(means, [.025,.975]).tolist()}


METRICS = ("progressGain", "progressPerDecision", "damageDealt", "damageReceived", "rangeOccupancy", "meanRangeError",
           "throwOrderFraction", "throwsOutsideFraction", "readyInRangeNotThrowingFraction",
           "outwardOutsideMoveFraction", "outsideRangeOpeningFraction")


def summarize(groups):
    complete = [g for g in groups if all(g[a]["windows"][w]["complete"] for a in ARMS for w in ("lateCorrection", "immediateTail"))]
    result = {"seeds": len(groups), "completePairedSeeds": [g["keep"]["seed"] for g in complete], "arms": {}, "temporalDifference": {}}
    result["completePairedWindows"] = {a: {w: {k: statistic([g[a]["windows"][w][k] for g in complete])
        for k in METRICS} for w in ("lateCorrection", "immediateTail")} for a in ARMS}
    for arm in ARMS:
        result["arms"][arm] = {w: {"trajectories": sum(g[arm]["windows"][w]["decisions"] > 0 for g in groups),
            "complete": sum(g[arm]["windows"][w]["complete"] for g in groups),
            "metrics": {k: statistic([g[arm]["windows"][w][k] for g in groups if g[arm]["windows"][w]["decisions"] > 0]) for k in METRICS}}
            for w in WINDOWS}
        result["temporalDifference"][arm] = {k: statistic([g[arm]["windows"]["immediateTail"][k]-g[arm]["windows"]["lateCorrection"][k]
            for g in complete if g[arm]["windows"]["immediateTail"][k] is not None and g[arm]["windows"]["lateCorrection"][k] is not None]) for k in METRICS}
    result["pairedTemporalContrast"] = {}
    for k in METRICS:
        values = []
        for g in complete:
            if all(g[a]["windows"][w][k] is not None for a in ARMS for w in ("lateCorrection", "immediateTail")):
                values.append((g["squad-30"]["windows"]["immediateTail"][k]-g["squad-30"]["windows"]["lateCorrection"][k])-
                              (g["keep"]["windows"]["immediateTail"][k]-g["keep"]["windows"]["lateCorrection"][k]))
        result["pairedTemporalContrast"][k] = statistic(values)
    result["strata"] = {}
    predicates = {"success": lambda r:r["final"]["success"], "failure":lambda r:not r["final"]["success"],
        "budget<=60": lambda r:r["snapshots"]["start"]["remainingBudget"]<=60,
        "budget61-100": lambda r:60<r["snapshots"]["start"]["remainingBudget"]<=100,
        "budget>100": lambda r:r["snapshots"]["start"]["remainingBudget"]>100}
    for name, predicate in predicates.items():
        rows = [g["squad-30"] for g in groups if predicate(g["squad-30"])]
        result["strata"][name] = {"count": len(rows), "successes": sum(r["final"]["success"] for r in rows),
            "progress": statistic([r["final"]["progress"] for r in rows]),
            "remainingBudgetAtStart": statistic([r["snapshots"]["start"]["remainingBudget"] for r in rows]),
            "finalProgressFreeDecisions": statistic([r["finalProgressFreeDecisions"] for r in rows]),
            "tail": {k: statistic([r["windows"]["allTail"][k] for r in rows if r["windows"]["allTail"]["decisions"]]) for k in METRICS}}
    return result


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    selection, s3, manifest = inputs()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True)
    write_json(root/"declaration.json", {"format": "snowgym.handoff-audit-config.v0", **d.LABEL,
        "sourceManifest": b.file_digest(ARCHIVE/"manifest.json"), "implementationDigest": b.file_digest(Path(__file__)),
        "declarationDigest": b.file_digest(b.TRAINING/"reviews/m7b_r1m_s7_declaration.md"),
        "gitCommit": resolve_git_commit(), "windows": WINDOWS, "simulatorBudget": 9600})
    frames = {f["seed"]:f for f in selection["frames"]}; groups=[]; steps=0
    with SnowGymBatchClient() as client:
        b.require_capabilities(client); wrapper=b.make_wrapper(client, 1, s3["config"]["gamma"])
        for pick in selection["selected"]:
            group={}
            for arm in ARMS:
                row = inspect(wrapper, frames[pick["seed"]], pick, load_row(pick["seed"], arm))
                steps += row["simulatorDecisions"]
                if steps > 9600:
                    raise ValueError("inspection budget exceeded")
                write_jsonl(root/f"inspection-{pick['seed']}-{arm}.jsonl.gz", [row]); group[arm]=row
            groups.append(group)
            print(json.dumps({"seed":pick["seed"],"inspectedPairs":len(groups),"simulatorDecisions":steps}),flush=True)
    if inputs()[-1] != manifest:
        raise ValueError("inspection changed source archive")
    report={"format":"snowgym.handoff-audit-report.v0",**d.LABEL,**summarize(groups),"simulatorDecisions":steps}
    write_json(root/"report.json",report)
    final={"format":"snowgym.handoff-audit-manifest.v0",**d.LABEL,
        "artifacts":{str(p.relative_to(root)):b.file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    final["manifestDigest"]=json_digest(final);write_json(root/"manifest.json",final);b.verified_manifest(root)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
