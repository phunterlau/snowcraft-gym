"""R1m-S6: frozen teacher movement duration-by-scope factorial."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import boundary_probe as b
from .control_channels import recommend_movement, validate_movement_agreement
from .opportunity_audit import plain, write_jsonl
from .supervised_probe import write_json

ARCHIVE = b.TRAINING / "runs/m7b_engage_r1m_s5_v0"
ARMS = ("keep", "single-1", "squad-1", "single-30", "squad-30")
LABEL = {"assistType": "corrected-shots-with-teacher-duration-scope-probe",
         "assistVersion": "snowgym.duration-scope.v0", "autonomousQualificationEligible": False}


def inputs():
    source, metadata, frames, s3, _ = b.frozen_source()
    manifest = b.verified_manifest(ARCHIVE)
    s5 = json.loads((ARCHIVE/"declaration.json").read_text())
    if s5["implementationDigest"] != b.file_digest(Path(b.__file__)) or s5["source"] != metadata:
        raise ValueError("S5 implementation or checkpoint differs")
    selection = json.loads((ARCHIVE/"selection.json").read_text())
    if len(selection["selected"]) != 24 or len({p["seed"] for p in selection["selected"]}) != 24:
        raise ValueError("S5 selection count mismatch")
    original = {f["seed"]: f for f in frames}
    b.recovery_train.validate_frames(selection["frames"], training=True)
    if len(selection["frames"]) != 24 or any(f != original.get(f["seed"]) for f in selection["frames"]):
        raise ValueError("S5 frames differ from S3")
    for pick in selection["selected"]:
        frame = original.get(pick["seed"])
        if frame is None or pick["frameDigest"] != frame["frameDigest"] or pick["identity"] != frame["trigger"]["identity"]:
            raise ValueError("S5 selected identity mismatch")
    return source, metadata, selection, s3, manifest


def active(arm, offset):
    if arm not in ARMS or type(offset) is not int or offset < 0:
        raise ValueError("invalid duration arm/offset")
    return arm != "keep" and offset < int(arm.split("-")[1])


def intervene(first, raw, recommendation, arm, offset, unit_id):
    enabled = active(arm, offset)
    result = {k: v.copy() for k, v in first.items()}
    overridden, changed = [], []
    moves = []
    for i, unit in enumerate(raw["allies"]):
        if not unit["alive"] or first["action_type"][0, i] != 1:
            continue
        moves.append(unit["id"])
        if not enabled or (arm.startswith("single") and unit["id"] != unit_id):
            continue
        if not recommendation["valid"][0, i]:
            raise ValueError("selected correction lacks a recommendation")
        overridden.append(unit["id"])
        if not np.array_equal(result["target"][0, i], recommendation["target"][0, i]):
            changed.append(unit["id"])
        result["target"][0, i] = recommendation["target"][0, i]
    return result, {"windowActive": enabled, "livingMoveIds": moves,
                    "overriddenIds": overridden, "changedIds": changed}


def archived_row(seed, arm):
    name = "keep" if arm == "keep" else "teacher"
    with gzip.open(ARCHIVE/f"branch-{seed}-{name}.jsonl.gz", "rt") as stream:
        return json.loads(stream.readline())


def branch(source, wrapper, frame, pick, arm, gamma):
    active(arm, 0)
    obs = b.post_hit.restore(wrapper, frame["seed"], frame["trigger"]["prefix"], pick["identity"])
    raw = wrapper.environment.raw_observations[0]
    if raw["allies"][pick["index"]]["id"] != pick["unitId"]:
        raise ValueError("selected fighter identity mismatch")
    initial_health = (b.team_health(raw["allies"]), b.team_health(raw["enemies"]))
    trace, hashes, actions = [], [wrapper.environment.state_hashes[0]], []
    components = dict.fromkeys(("mission", "combat", "shaping", "canonical", "executor"), 0.)
    local = None
    for offset in range(200-frame["trigger"]["decision"]):
        raw = wrapper.environment.raw_observations[0]
        identity = b.post_hit.identity(wrapper, obs)
        first = b.action(source, obs, wrapper)
        if offset == 0 and plain(first) != pick["firstAction"]:
            raise ValueError("initial source action changed")
        rec = recommend_movement(raw, first["action_type"].shape[1])
        validate_movement_agreement(wrapper.environment.plan_teacher_tensor_actions(), rec)
        if b.post_hit.identity(wrapper, obs) != identity:
            raise ValueError("labeling changed state")
        executed, exposure = intervene(first, raw, rec, arm, offset, pick["unitId"])
        distances = rec["distance"][rec["valid"]]
        exposure.update({"rangeCount": len(distances), "inRange": int((distances <= 9).sum()),
            "rangeErrorSum": float(np.abs(distances-6.5).sum()),
            "recommendationIds": [u["id"] for i, u in enumerate(raw["allies"]) if rec["valid"][0, i]],
            "throwReadyIds": [u["id"] for i, u in enumerate(raw["allies"]) if rec["ready"][0, i]]})
        obs, reward, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]
        hashes.append(wrapper.environment.state_hashes[0]); actions.append(plain(executed))
        for key in components:
            components[key] += gamma**offset*info["option"]["rewards"][key]
        trace.append({"offset": offset, "action": plain(executed), "stateHash": hashes[-1],
            "reward": float(reward[0]), "option": plain(info["option"]), "exposure": exposure,
            "actionResults": plain(info["actionResults"])})
        done = bool(terminated[0] or truncated[0])
        if offset == 29 or done and local is None:
            local = b.post_hit.outcome(wrapper, info, initial_health, offset+1)
        if done:
            break
    if not done:
        raise ValueError("continuation exceeded original option budget")
    final = b.post_hit.outcome(wrapper, info, initial_health, len(trace))
    digest = json_digest(actions)
    if arm in ("keep", "single-1"):
        expected = archived_row(frame["seed"], arm)
        if hashes != expected["stateHashes"] or digest != expected["actionsDigest"] or final != expected["final"]:
            raise ValueError("S5 reference mismatch")
    if not np.isclose(components["executor"], components["mission"]+.1*components["combat"]+components["shaping"], atol=1e-6):
        raise ValueError("reward accounting mismatch")
    exposure = {"windowDecisions": sum(t["exposure"]["windowActive"] for t in trace),
        "overriddenMoves": sum(len(t["exposure"]["overriddenIds"]) for t in trace),
        "changedTargets": sum(len(t["exposure"]["changedIds"]) for t in trace),
        "windowLivingMoves": sum(len(t["exposure"]["livingMoveIds"]) for t in trace if t["exposure"]["windowActive"]),
        "uniqueOverriddenIds": sorted({i for t in trace for i in t["exposure"]["overriddenIds"]})}
    count = sum(t["exposure"]["rangeCount"] for t in trace)
    exposure.update({"rangeCount": count, "rangeOccupancy": sum(t["exposure"]["inRange"] for t in trace)/max(count, 1),
        "meanRangeError": sum(t["exposure"]["rangeErrorSum"] for t in trace)/max(count, 1)})
    return {"seed": frame["seed"], "arm": arm, "unitId": pick["unitId"], **LABEL,
        "local": local, "final": final, "discounted": components, "exposure": exposure,
        "earlyTerminationBefore30": len(trace) < 30, "trace": trace, "stateHashes": hashes, "actionsDigest": digest,
        "simulatorDecisions": len(frame["trigger"]["prefix"])+len(trace),
        "rejectedActions": sum(a.get("accepted") is False for t in trace for a in t["actionResults"]),
        "totalActions": sum(len(t["actionResults"]) for t in trace)}


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(990001)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"count": len(values), "mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def summarize(groups):
    keys = ("success", "return", "progress", "damageDealt", "damageReceived", "livingFraction")
    def metric(row, key):
        return row["discounted"]["executor"] if key == "return" else float(row["final"][key])
    arms = {}
    for arm in ARMS:
        rows = [g[arm] for g in groups]
        arms[arm] = {"successes": sum(r["final"]["success"] for r in rows),
            "versusKeep": {k: interval([metric(g[arm], k)-metric(g["keep"], k) for g in groups]) for k in keys},
            "discounted": {k: interval([r["discounted"][k] for r in rows]) for k in rows[0]["discounted"]},
            "local": {k: interval([float(r["local"][k]) for r in rows]) for k in rows[0]["local"]},
            "earlyTerminationBefore30": sum(r["earlyTerminationBefore30"] for r in rows),
            "exposure": {k: interval([r["exposure"][k] for r in rows]) for k in
                ("windowDecisions", "overriddenMoves", "changedTargets", "windowLivingMoves", "rangeOccupancy", "meanRangeError")}}
    contrasts = {"durationSingle": {"single-30": 1, "single-1": -1},
        "durationSquad": {"squad-30": 1, "squad-1": -1},
        "scopeOne": {"squad-1": 1, "single-1": -1},
        "scopeThirty": {"squad-30": 1, "single-30": -1},
        "interaction": {"squad-30": 1, "squad-1": -1, "single-30": -1, "single-1": 1}}
    effects = {name: {k: interval([sum(weight*metric(g[arm], k) for arm, weight in weights.items()) for g in groups])
                     for k in keys} for name, weights in contrasts.items()}
    primary = arms["squad-30"]["versusKeep"]
    rejects = sum(r["rejectedActions"] for g in groups for r in g.values())
    total = sum(r["totalActions"] for g in groups for r in g.values())
    criteria = {"successGain": primary["success"]["mean"] >= .1-1e-12,
        "positiveSuccessInterval": primary["success"]["ci95"][0] > 0,
        "positiveReturnInterval": primary["return"]["ci95"][0] > 0,
        "rejectedActions": rejects/max(total, 1) < .001}
    return {"seeds": len(groups), "arms": arms, "contrasts": effects,
        "primaryTransferGate": {"passed": all(criteria.values()), "criteria": criteria},
        "rejectedActions": rejects, "totalActions": total}


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    source, metadata, selected, s3, manifest = inputs()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = semantic_state_digest(source.state_dict()); gamma = s3["config"]["gamma"]
    root.mkdir(parents=True)
    write_json(root/"declaration.json", {"format": "snowgym.duration-scope-config.v0", **LABEL,
        "source": metadata, "sourceManifest": b.file_digest(ARCHIVE/"manifest.json"),
        "sourceFiles": s3["sourceFiles"], "implementationDigest": b.file_digest(Path(__file__)),
        "declarationDigest": b.file_digest(b.TRAINING/"reviews/m7b_r1m_s6_declaration.md"),
        "gitCommit": resolve_git_commit(), "arms": list(ARMS), "gamma": gamma,
        "simulatorBudget": 48000, "repeats": 2, "bootstrapSeed": 990001, "bootstrapSamples": 10000})
    write_json(root/"selection.json", selected)
    frames = {f["seed"]: f for f in selected["frames"]}
    groups, repeats, steps = [], [], 0
    with SnowGymBatchClient() as client:
        b.require_capabilities(client); wrapper = b.make_wrapper(client, 1, gamma)
        for pick in selected["selected"]:
            group = {}
            for arm in ARMS:
                first = branch(source, wrapper, frames[pick["seed"]], pick, arm, gamma)
                second = branch(source, wrapper, frames[pick["seed"]], pick, arm, gamma)
                if json_digest(first) != json_digest(second):
                    raise ValueError("duration probe duplicate mismatch")
                steps += first["simulatorDecisions"]+second["simulatorDecisions"]
                if steps > 48000:
                    raise ValueError("duration probe budget exceeded")
                repeats.append({"seed": pick["seed"], "arm": arm, "first": json_digest(first), "second": json_digest(second)})
                write_jsonl(root/f"branch-{pick['seed']}-{arm}.jsonl.gz", [first]); group[arm] = first
            groups.append(group)
            print(json.dumps({"seed": pick["seed"], "completedSnapshots": len(groups), "simulatorDecisions": steps}), flush=True)
    if semantic_state_digest(source.state_dict()) != before or inputs()[-1] != manifest:
        raise ValueError("source/archive changed during experiment")
    report = {"format": "snowgym.duration-scope-report.v0", **LABEL, **summarize(groups),
              "simulatorDecisions": steps, "duplicatePairs": len(repeats)}
    write_json(root/"report.json", report); write_json(root/"duplicates.json", repeats)
    inventory = {"format": "snowgym.duration-scope-manifest.v0", **LABEL,
        "artifacts": {str(p.relative_to(root)): b.file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    inventory["manifestDigest"] = json_digest(inventory)
    write_json(root/"manifest.json", inventory); b.verified_manifest(root)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
