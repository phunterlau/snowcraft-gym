"""R1m-S5: frozen one-fighter destination-to-motion boundary experiment."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import post_hit, recovery_train
from .control_channels import recommend_movement, validate_movement_agreement
from .identity import checkpoint_model
from .interventions import require_capabilities, team_health
from .movement_collect import corrected_shots
from .movement_train import REFERENCE, TRAINING, make_wrapper
from .opportunity_audit import plain, write_jsonl
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json

ARCHIVE = TRAINING / "runs/m7b_engage_r1m_s3_v0"
ARMS = ("keep", "radial+1", "radial-1", "lateral+1", "lateral-1",
        "radial+5", "radial-5", "lateral+5", "lateral-5", "teacher")
LABEL = {"assistType": "corrected-shots-with-one-decision-movement-probe",
         "assistVersion": "snowgym.boundary-probe.v0", "autonomousQualificationEligible": False}


def verified_manifest(root):
    root = Path(root)
    value = audit_artifact_manifest(root, "manifest.json")
    if value.get("manifestDigest") != json_digest({k: v for k, v in value.items() if k != "manifestDigest"}):
        raise ValueError("manifest self digest mismatch")
    inventory = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root/"manifest.json"}
    if inventory != set(value["artifacts"]):
        raise ValueError("manifest inventory mismatch")
    return value


def frozen_source():
    manifest = verified_manifest(ARCHIVE)
    declaration = json.loads((ARCHIVE/"declaration.json").read_text())
    repository = TRAINING.parents[1]
    if any(file_digest(repository/p) != d for p, d in declaration["sourceFiles"].items()):
        raise ValueError("archived S3 source changed")
    dataset = json.loads((ARCHIVE/"snapshots.json").read_text())
    if dataset["digest"] != json_digest(dataset["datasets"]):
        raise ValueError("snapshot inventory digest mismatch")
    frames = dataset["datasets"]["training"]
    recovery_train.validate_frames(frames, training=True)
    if len(frames) != 57:
        raise ValueError("training snapshot count mismatch")
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata != declaration["source"]:
        raise ValueError("source checkpoint mismatch")
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    source.eval().requires_grad_(False)
    return source, metadata, frames, declaration, manifest


def action(source, obs, wrapper):
    with torch.no_grad():
        result = source.act(tensor_dict(obs), deterministic=True)[0]
    return corrected_shots(numpy_actions(result), wrapper.environment.raw_observations)


def desired_velocity(position, target):
    delta = np.asarray(target, dtype=float)-np.asarray(position, dtype=float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("invalid geometry")
    length = np.linalg.norm(delta)
    return delta * (6 / max(2.1, length))


def arm_geometry(position, target, teacher, scale, arm):
    if arm not in ARMS:
        raise ValueError("unknown boundary arm")
    position, target, teacher, scale = map(lambda x: np.asarray(x, dtype=float), (position, target, teacher, scale))
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (position, target, teacher, scale)) or (scale <= .5).any():
        raise ValueError("invalid boundary geometry")
    base = np.clip(target, -scale+.5, scale-.5)
    ray = base-position
    distance = np.linalg.norm(ray)
    if distance <= 1e-6:
        raise ValueError("degenerate movement ray")
    radial = ray/distance
    lateral = np.asarray([-radial[1], radial[0]])
    requested = target.copy() if arm == "keep" else teacher.copy() if arm == "teacher" else (
        base + (radial if arm.startswith("radial") else lateral)*int(arm[-2:]))
    # Model the float32 tensor serialization before the authoritative engine clamp.
    serialized = (requested/scale).astype(np.float32)
    serialized = np.clip(serialized, -1, 1)
    effective = np.clip(serialized*scale, -scale+.5, scale-.5)
    shift = float(np.linalg.norm(effective-base))
    candidate_ray = effective-position
    same_ray = np.dot(candidate_ray, radial) > 2.1 and abs(np.dot(candidate_ray, lateral)) < 1e-5
    clipped = bool(np.linalg.norm(requested-effective) > 1e-5)
    return {"requested": requested.tolist(), "serialized": serialized.tolist(), "effective": effective.tolist(),
        "baselineEffective": base.tolist(), "effectiveShift": shift, "clipped": clipped,
        "radialUnclippedFar": bool(arm.startswith("radial") and distance > 2.1 and same_ray and not clipped),
        "desiredVelocityDelta": float(np.linalg.norm(desired_velocity(position, effective)-desired_velocity(position, base)))}


def substitute(first, index, geometry, arm):
    if arm not in ARMS or first["action_type"][0, index] != 1:
        raise ValueError("substitution requires selected MOVE")
    result = {k: v.copy() for k, v in first.items()}
    if arm != "keep":
        result["target"][0, index] = geometry["serialized"]
    return result


def choose(raw, first, movement):
    scale = np.asarray([raw["arena"]["width"]/2, raw["arena"]["height"]/2])
    eligible = []
    for i, unit in enumerate(raw["allies"]):
        if unit["alive"] and first["action_type"][0, i] == 1 and movement["valid"][0, i]:
            target = np.clip(first["target"][0, i]*scale, -scale+.5, scale-.5)
            if np.linalg.norm(target-np.asarray([unit["x"], unit["y"]])) > 1e-6:
                eligible.append((unit["id"], i))
    return min(eligible)[1] if eligible else None


def selection(source, wrapper, frames):
    selected, exclusions, steps = [], [], 0
    for frame in sorted(frames, key=lambda f: f["seed"]):
        obs = post_hit.restore(wrapper, frame["seed"], frame["trigger"]["prefix"], frame["trigger"]["identity"])
        steps += len(frame["trigger"]["prefix"])
        raw = wrapper.environment.raw_observations[0]
        before = post_hit.identity(wrapper, obs)
        first = action(source, obs, wrapper)
        movement = recommend_movement(raw, first["action_type"].shape[1])
        validate_movement_agreement(wrapper.environment.plan_teacher_tensor_actions(), movement)
        if post_hit.identity(wrapper, obs) != before:
            raise ValueError("labeling changed branch state")
        index = choose(raw, first, movement)
        if index is None:
            exclusions.append({"seed": frame["seed"], "reason": "no-living-MOVE-valid-nondegenerate-ray"})
            continue
        selected.append({"seed": frame["seed"], "frameDigest": frame["frameDigest"], "index": index,
            "unitId": raw["allies"][index]["id"], "firstAction": plain(first),
            "teacherTarget": movement["target"][0, index].tolist(), "rawUnit": copy.deepcopy(raw["allies"][index]),
            "throwReady": bool(movement["ready"][0, index]), "recommendationAvailable": True,
            "actionMask": plain(obs["unit_action_mask"]), "identity": before})
        if len(selected) == 24:
            break
    if len(selected) != 24:
        raise ValueError("fewer than 24 eligible snapshots")
    return selected, exclusions, steps


def physical(raw, unit_id):
    unit = next(u for u in raw["allies"] if u["id"] == unit_id)
    ranges = [np.hypot(e["x"]-unit["x"], e["y"]-unit["y"]) for e in raw["enemies"] if e["alive"]]
    return {"position": [unit["x"], unit["y"]], "velocity": [unit["vx"], unit["vy"]],
            "health": unit["health"], "alive": unit["alive"], "state": unit["state"],
            "moveTarget": plain(unit.get("moveTarget")), "range": float(min(ranges)) if ranges and unit["alive"] else None}


def branch(source, wrapper, frame, selected, arm, gamma):
    obs = post_hit.restore(wrapper, frame["seed"], frame["trigger"]["prefix"], selected["identity"])
    raw = wrapper.environment.raw_observations[0]
    index, unit_id = selected["index"], selected["unitId"]
    if raw["allies"][index]["id"] != unit_id:
        raise ValueError("selected unit identity changed")
    first = action(source, obs, wrapper)
    if plain(first) != selected["firstAction"]:
        raise ValueError("frozen first action changed")
    scale = np.asarray([raw["arena"]["width"]/2, raw["arena"]["height"]/2])
    unit = raw["allies"][index]
    geo = arm_geometry([unit["x"], unit["y"]], first["target"][0, index]*scale,
                       np.asarray(selected["teacherTarget"])*scale, scale, arm)
    start_health = (team_health(raw["allies"]), team_health(raw["enemies"]))
    initial = physical(raw, unit_id)
    trace, hashes, actions = [], [wrapper.environment.state_hashes[0]], []
    components = dict.fromkeys(("mission", "combat", "shaping", "canonical", "executor"), 0.)
    checkpoints = {str(k): None for k in (1, 5, 30)}
    for offset in range(200-frame["trigger"]["decision"]):
        executed = substitute(first, index, geo, arm) if offset == 0 else action(source, obs, wrapper)
        obs, reward, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]
        raw = wrapper.environment.raw_observations[0]
        state = physical(raw, unit_id)
        hashes.append(wrapper.environment.state_hashes[0]); actions.append(plain(executed))
        for key in components:
            components[key] += gamma**offset*info["option"]["rewards"][key]
        trace.append({"offset": offset, "action": plain(executed), "stateHash": hashes[-1],
            "physical": state, "reward": float(reward[0]), "option": plain(info["option"]),
            "actionResults": plain(info["actionResults"])})
        if str(offset+1) in checkpoints:
            checkpoints[str(offset+1)] = {"physical": state, **post_hit.outcome(wrapper, info, start_health, offset+1)}
        if terminated[0] or truncated[0]:
            break
    if not (terminated[0] or truncated[0]):
        raise ValueError("continuation did not terminate within original budget")
    if arm == "keep" and (hashes != frame["suffixHashes"] or json_digest(actions) != frame["suffixActionsDigest"]):
        raise ValueError("baseline suffix differs from archive")
    if not np.isclose(components["executor"], components["mission"]+.1*components["combat"]+components["shaping"], atol=1e-6):
        raise ValueError("reward components disagree")
    return {"seed": frame["seed"], "arm": arm, "unitId": unit_id, **LABEL,
        "geometry": geo, "initial": initial, "checkpoints": checkpoints, "discounted": components,
        "final": post_hit.outcome(wrapper, info, start_health, len(trace)), "trace": trace,
        "stateHashes": hashes, "actionsDigest": json_digest(actions),
        "rejectedActions": sum(a.get("accepted") is False for e in trace for a in e["actionResults"]),
        "totalActions": sum(len(e["actionResults"]) for e in trace),
        "simulatorDecisions": len(frame["trigger"]["prefix"])+len(trace)}


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(980001)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"count": len(values), "mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def motion(row, control, time):
    current, baseline = row["checkpoints"][str(time)], control["checkpoints"][str(time)]
    if current is None or baseline is None:
        return None
    p, b = current["physical"], baseline["physical"]
    return {"positionDelta": float(np.linalg.norm(np.asarray(p["position"])-b["position"])),
        "velocityDelta": float(np.linalg.norm(np.asarray(p["velocity"])-b["velocity"])),
        "rangeErrorDelta": (abs(p["range"]-6.5)-abs(b["range"]-6.5)
                            if p["range"] is not None and b["range"] is not None else None),
        "bothAlive": bool(p["alive"] and b["alive"])}


def summarize(groups):
    result = {"seeds": len(groups), "arms": {}, "scales": {}}
    for arm in ARMS:
        rows = [g[arm] for g in groups]; controls = [g["keep"] for g in groups]
        item = {"returnDelta": interval([r["discounted"]["executor"]-b["discounted"]["executor"] for r, b in zip(rows, controls)]),
            "finalDelta": {key: interval([float(r["final"][key])-float(b["final"][key]) for r, b in zip(rows, controls)])
                for key in ("success", "progress", "damageDealt", "damageReceived")},
            "effectiveTargetShift": interval([r["geometry"]["effectiveShift"] for r in rows]),
            "analyticVelocityDelta": interval([r["geometry"]["desiredVelocityDelta"] for r in rows]),
            "clipped": sum(r["geometry"]["clipped"] for r in rows), "motion": {}}
        for time in (1, 5, 30):
            paired = [motion(r, b, time) for r, b in zip(rows, controls)]
            valid = [p for p in paired if p is not None]
            item["motion"][str(time)] = {"censoredPairs": len(rows)-len(valid),
                "bothAlivePairs": sum(p["bothAlive"] for p in valid),
                "rangeErrorDelta": interval([p["rangeErrorDelta"] for p in valid if p["rangeErrorDelta"] is not None]),
                **{key: interval([p[key] for p in valid]) for key in ("positionDelta", "velocityDelta")}}
        first = [motion(r, b, 1) for r, b in zip(rows, controls)]
        item["nearZeroPositionFraction"] = float(np.mean([p["positionDelta"] <= 1e-6 for p in first]))
        item["positionGain"] = interval([p["positionDelta"]/r["geometry"]["effectiveShift"] for p, r in zip(first, rows) if r["geometry"]["effectiveShift"] > 1e-6])
        subset = [p["positionDelta"] for p, r in zip(first, rows) if r["geometry"]["radialUnclippedFar"]]
        item["unclippedFarRadial"] = {"positionDelta": interval(subset), "nearZeroCount": sum(x <= 1e-6 for x in subset)}
        result["arms"][arm] = item
    for scale in (1, 5):
        arms = [a for a in ARMS if a.endswith(str(scale))]
        physical_means, returns, any_change = [], [], []
        for group in groups:
            physical_means.append(float(np.mean([motion(group[a], group["keep"], 1)["positionDelta"] for a in arms])))
            differences = [group[a]["discounted"]["executor"]-group["keep"]["discounted"]["executor"] for a in arms]
            returns.append(float(np.mean(differences))); any_change.append(any(abs(d) > 1e-6 for d in differences))
        result["scales"][str(scale)] = {"positionDelta": interval(physical_means), "signedReturnDelta": interval(returns),
            "anyReturnChangeFraction": float(np.mean(any_change))}
    result["scalePositionDifference"] = interval([np.mean([motion(g[a], g["keep"], 1)["positionDelta"] for a in ARMS if a.endswith("5")])-
        np.mean([motion(g[a], g["keep"], 1)["positionDelta"] for a in ARMS if a.endswith("1")]) for g in groups])
    return result


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    source, metadata, frames, archived, manifest = frozen_source()
    before = semantic_state_digest(source.state_dict())
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    gamma = archived["config"]["gamma"]
    root.mkdir(parents=True)
    write_json(root/"declaration.json", {"format": "snowgym.boundary-config.v0", **LABEL,
        "source": metadata, "sourceManifest": file_digest(ARCHIVE/"manifest.json"),
        "sourceFiles": archived["sourceFiles"], "gitCommit": resolve_git_commit(),
        "implementationDigest": file_digest(Path(__file__)),
        "declarationDigest": file_digest(TRAINING/"reviews/m7b_r1m_s5_declaration.md"),
        "arms": list(ARMS), "gamma": gamma, "repeats": 2, "sampleCount": 24, "simulatorBudget": 107400})
    groups, duplicates = [], []
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        wrapper = make_wrapper(client, 1, gamma)
        selected, exclusions, steps = selection(source, wrapper, frames)
        write_json(root/"selection.json", {"selected": selected, "exclusions": exclusions,
            "selectionDecisions": steps, "frames": [f for f in frames if f["seed"] in {s["seed"] for s in selected}]})
        by_seed = {f["seed"]: f for f in frames}
        for pick in selected:
            group = {}
            for arm in ARMS:
                first = branch(source, wrapper, by_seed[pick["seed"]], pick, arm, gamma)
                second = branch(source, wrapper, by_seed[pick["seed"]], pick, arm, gamma)
                if json_digest(first) != json_digest(second):
                    raise ValueError("duplicate trajectory mismatch")
                steps += first["simulatorDecisions"]+second["simulatorDecisions"]
                if steps > 107400:
                    raise ValueError("simulator budget exceeded")
                duplicates.append({"seed": pick["seed"], "arm": arm, "first": json_digest(first), "second": json_digest(second)})
                write_jsonl(root/f"branch-{pick['seed']}-{arm}.jsonl.gz", [first])
                group[arm] = first
            groups.append(group)
            print(json.dumps({"seed": pick["seed"], "completedSnapshots": len(groups), "simulatorDecisions": steps}), flush=True)
    if semantic_state_digest(source.state_dict()) != before or frozen_source()[-1] != manifest:
        raise ValueError("frozen source or archive changed")
    report = {"format": "snowgym.boundary-report.v0", **LABEL, **summarize(groups),
        "simulatorDecisions": steps, "duplicatePairs": len(duplicates),
        "rejectedActions": sum(r["rejectedActions"] for g in groups for r in g.values()),
        "totalActions": sum(r["totalActions"] for g in groups for r in g.values())}
    write_json(root/"duplicates.json", duplicates); write_json(root/"report.json", report)
    final = {"format": "snowgym.boundary-manifest.v0", **LABEL,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    final["manifestDigest"] = json_digest(final)
    write_json(root/"manifest.json", final); verified_manifest(root)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
