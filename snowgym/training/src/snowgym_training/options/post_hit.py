"""R1m-S2 frozen post-hit choice/movement continuation experiment."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import tensor_dict
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from .control_channels import compose_action, recommend_movement, validate_movement_agreement
from .throw_channels import recommend_shots, validate_teacher_agreement
from .identity import checkpoint_model
from .interventions import require_capabilities, team_health
from .movement_train import REFERENCE, TRAINING, make_wrapper, load_config
from .movement_collect import world_identities
from .plans import teacher_option_plan, teacher_option_scenario
from .opportunity_audit import plain, write_jsonl
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json

ARMS = ("keep", "choice-30", "move-30", "both-30", "move-rest")
LABEL = {"assistType": "corrected-shots-with-post-hit-channel-intervention",
         "assistVersion": "snowgym.post-hit.v0", "autonomousQualificationEligible": False}


def config():
    return {"format": "snowgym.post-hit-config.v0", "checkpointDigest": load_config()["checkpointDigest"],
        "seeds": [200000, 200039], "arms": list(ARMS), "window": 30, "horizon": 200,
        "gamma": .9976921765, "bootstrapSeed": 960001, "bootstrapSamples": 10000,
        "rangeReference": 6.5, "rangeThreshold": 9., "repeats": 2, **LABEL}


def active_channel(arm, offset):
    if arm not in ARMS or offset < 0:
        raise ValueError("invalid arm or offset")
    if arm != "move-rest" and offset >= 30:
        return "shot-only"
    return {"keep": "shot-only", "choice-30": "teacher-choice", "move-30": "teacher-move",
            "both-30": "teacher-choice-move", "move-rest": "teacher-move"}[arm]


def reset(wrapper, seed):
    plan, spec = teacher_option_plan("engage")
    return wrapper.reset([seed], [teacher_option_scenario("engage")],
                         [f"movement-{seed}"], [plan], [spec])[0]


def identity(wrapper, observation):
    return {**world_identities(wrapper)[0], "observation": json_digest(plain(observation))}


def restore(wrapper, seed, prefix, expected):
    observation = reset(wrapper, seed)
    for action in prefix:
        arrays = {k: np.asarray(v, dtype=np.int64 if k == "action_type" else np.float32)
                  for k, v in action.items()}
        observation, _, terminated, truncated, _ = wrapper.step(arrays)
        if terminated[0] or truncated[0]:
            raise ValueError("branch prefix reaches terminal option")
    if identity(wrapper, observation) != expected:
        raise ValueError("physical/plan/option/observation identity mismatch")
    return observation


def action_and_metrics(model, wrapper, observation, channel):
    raw = wrapper.environment.raw_observations[0]
    capacity = wrapper.environment.max_team_units
    movement, shot = recommend_movement(raw, capacity), recommend_shots(raw, capacity)
    teacher = wrapper.environment.plan_teacher_tensor_actions()
    validate_movement_agreement(teacher, movement)
    validate_teacher_agreement(teacher, shot)
    with torch.no_grad():
        prediction = model(tensor_dict(observation))
        action = compose_action(channel, prediction, teacher, shot, movement)
    live = np.zeros_like(action["action_type"], dtype=bool)
    live[0, :len(raw["allies"])] = [u["alive"] for u in raw["allies"]]
    valid = live & movement["valid"]
    throws = live & (action["action_type"] == 2)
    learner = prediction["action_logits"].argmax(-1).numpy()
    cross = np.zeros((4, 4), dtype=int)
    np.add.at(cross, (learner[live], teacher["action_type"][live]), 1)
    return action, {"rangeErrorSum": float(np.abs(movement["distance"][valid]-6.5).sum()),
        "rangeCount": int(valid.sum()), "inRange": int((valid & (movement["distance"] <= 9)).sum()),
        "readyInRange": int((valid & movement["ready"] & (movement["distance"] <= 9)).sum()),
        "readyInRangeNotThrowing": int((valid & movement["ready"] & (movement["distance"] <= 9) & ~throws).sum()),
        "throws": int(throws.sum()), "throwsNotReady": int((throws & ~movement["ready"]).sum()),
        "throwsBeyondThreshold": int((throws & (movement["distance"] > 9)).sum()),
        "choiceChanges": int((live & (action["action_type"] != learner)).sum()),
        "choiceCrossTab": cross.tolist(),
        "opportunities": [{"unitId": u["id"], "learnerType": int(learner[0, i]),
            "teacherType": int(teacher["action_type"][0, i]), "executedType": int(action["action_type"][0, i]),
            "ready": bool(movement["ready"][0, i]), "distance": float(movement["distance"][0, i]),
            "threat": bool(movement["threat"][0, i]), "moveRecommendationAvailable": bool(movement["valid"][0, i]),
            "shotRecommendationAvailable": bool(shot["valid"][0, i]),
            "moveRecommendation": movement["target"][0, i].tolist(),
            "shotRecommendation": shot["target"][0, i].tolist(), "shotPower": float(shot["power"][0, i]),
            "previousMoveTarget": plain(u.get("moveTarget"))}
            for i, u in enumerate(raw["allies"]) if u["alive"]]}


def outcome(wrapper, info, start_health, elapsed):
    raw = wrapper.environment.raw_observations[0]
    tracker = wrapper.trackers[0]
    blue, red = team_health(raw["allies"]), team_health(raw["enemies"])
    return {"success": info["option"]["success"], "progress": info["option"]["progress"],
        "physicalWin": not any(u["alive"] for u in raw["enemies"]), "exposureDecisions": elapsed,
        "damageDealt": start_health[1]-red, "damageReceived": start_health[0]-blue,
        "livingFraction": float(tracker.option_state(raw)[2]),
        "remainingBudget": tracker.spec.horizon-tracker.decision}


def continuation(model, wrapper, observation, seed, arm="keep", *, baseline=False):
    raw = wrapper.environment.raw_observations[0]
    initial_health = (team_health(raw["allies"]), team_health(raw["enemies"]))
    initial_target = wrapper.trackers[0].target_health_fraction(raw)
    start_decision = wrapper.trackers[0].decision
    hashes, actions, trace = [wrapper.environment.state_hashes[0]], [], []
    trigger = local = None
    for offset in range(200-start_decision):
        action, metrics = action_and_metrics(model, wrapper, observation, active_channel(arm, offset))
        actions.append(plain(action))
        observation, _, terminated, truncated, infos = wrapper.step(action)
        info = infos[0]
        done = bool(terminated[0] or truncated[0])
        hashes.append(wrapper.environment.state_hashes[0])
        trace.append({"offset": offset, "action": plain(action), "stateHash": hashes[-1],
            "metrics": metrics, "option": plain(info["option"]), "actionResults": plain(info["actionResults"])})
        if baseline and trigger is None and wrapper.trackers[0].target_health_fraction(
                wrapper.environment.raw_observations[0]) < initial_target:
            trigger = {"decision": offset+1, "terminal": done, "prefix": copy.deepcopy(actions),
                       "identity": identity(wrapper, observation)}
        if offset == 29 or done and local is None:
            local = outcome(wrapper, info, initial_health, offset+1)
        if done:
            break
    final = outcome(wrapper, info, initial_health, len(actions))
    final.update({"seed": seed, "completionDecisions": len(actions) if final["success"] else None,
                  "completionCensored": not final["success"]})
    totals = {k: sum(r["metrics"][k] for r in trace) for k in trace[0]["metrics"]
              if k not in ("choiceCrossTab", "opportunities")}
    totals["choiceCrossTab"] = np.sum([r["metrics"]["choiceCrossTab"] for r in trace], axis=0).tolist()
    totals["meanRangeError"] = totals["rangeErrorSum"]/max(1, totals["rangeCount"])
    totals["rangeOccupancy"] = totals["inRange"]/max(1, totals["rangeCount"])
    return {"seed": seed, "arm": arm, **LABEL, "startDecision": start_decision, "trigger": trigger,
        "stateHashes": hashes, "actionsDigest": json_digest(actions), "trace": trace,
        "local": local, "final": final, "diagnostics": totals,
        "rejectedActions": sum(a.get("accepted") is False for r in trace for a in r["actionResults"]),
        "totalActions": sum(len(r["actionResults"]) for r in trace)}


def paired_interval(values):
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(960001)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def report_effects(branches):
    effects = {}
    for arm in ARMS[1:]:
        effects[arm] = {}
        for phase in ("local", "final"):
            effects[arm][phase] = {key: paired_interval([float(b[arm][phase][key])-float(b["keep"][phase][key])
                for b in branches]) for key in ("success", "progress", "damageDealt", "damageReceived", "livingFraction")}
    interaction = {phase: {key: paired_interval([float(b["both-30"][phase][key])-float(b["choice-30"][phase][key])
        -float(b["move-30"][phase][key])+float(b["keep"][phase][key]) for b in branches])
        for key in ("success", "progress")} for phase in ("local", "final")}
    supported = {arm: bool((v := effects[arm]["final"]["success"]) is not None
        and v["mean"] >= .1-1e-12 and v["ci95"][0] > 0) for arm in ARMS[1:]}
    return {"effects": effects, "interaction": interaction, "supportedForFurtherStudy": supported}


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    archive = TRAINING / "runs/m7b_engage_r1m_movement_v0"
    archive_manifest = audit_artifact_manifest(archive, "manifest.json")
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata["checkpointDigest"] != config()["checkpointDigest"]:
        raise ValueError("source mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    frozen = semantic_state_digest(model.state_dict())
    repository = TRAINING.parents[1]
    paths = [p for base in (repository / "snowgym", repository / "src") for p in base.rglob("*")
             if p.suffix in (".ts", ".py") and not any(x in p.parts for x in (".venv", "runs", "node_modules", "__pycache__"))]
    sources = {str(p.relative_to(repository)): file_digest(p) for p in sorted(paths)}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"configuration": config(), "sourceFiles": sources,
        "gitCommit": resolve_git_commit(), "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1m_s2_declaration.md"),
        "archiveManifestDigest": file_digest(archive / "manifest.json"), "sourceCheckpoint": metadata["checkpointDigest"]})
    baselines, branches, coverage = [], [], []
    archived = json.loads((archive / "assisted-initialization.json").read_text())["historical"]
    if [r["seed"] for r in archived] != list(range(200000, 200040)):
        raise ValueError("archived seed allocation mismatch")
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        wrapper = make_wrapper(client, 1, config()["gamma"])
        # All baselines must reproduce before any intervention.
        for expected in archived:
            seed = expected["seed"]
            baseline = continuation(model, wrapper, reset(wrapper, seed), seed, baseline=True)
            if any(baseline[k] != expected[k] for k in ("stateHashes", "actionsDigest")):
                raise ValueError("archived baseline trajectory mismatch")
            if any(baseline["final"][k] != expected[k] for k in ("success", "progress")):
                raise ValueError("archived baseline scoring mismatch")
            baselines.append(baseline)
        for baseline in baselines:
            seed, trigger = baseline["seed"], baseline["trigger"]
            write_jsonl(root / f"baseline-{seed}.jsonl.gz", [baseline])
            eligible = trigger is not None and not trigger["terminal"]
            coverage.append({"seed": seed, "eligible": eligible,
                             "firstHit": None if trigger is None else trigger["decision"]})
            if not eligible:
                continue
            pair = {}
            for arm in ARMS:
                repeats = []
                for _ in range(2):
                    observation = restore(wrapper, seed, trigger["prefix"], trigger["identity"])
                    repeats.append(continuation(model, wrapper, observation, seed, arm))
                if repeats[0] != repeats[1]:
                    raise ValueError("duplicate continuation mismatch")
                if arm == "keep" and (repeats[0]["stateHashes"] != baseline["stateHashes"][trigger["decision"]:]
                    or [r["action"] for r in repeats[0]["trace"]] != [r["action"] for r in baseline["trace"]][trigger["decision"]:]):
                    raise ValueError("keep continuation differs from baseline")
                pair[arm] = repeats[0]
                write_jsonl(root / f"branch-{seed}-{arm}.jsonl.gz", [{**repeats[0], "exactDuplicate": True,
                    "duplicateDigest": json_digest(repeats[1]), "startIdentity": trigger["identity"]}])
            branches.append(pair)
            print(json.dumps({"seed": seed, "firstHit": trigger["decision"],
                              "success": {a: pair[a]["final"]["success"] for a in ARMS}}), flush=True)
    if semantic_state_digest(model.state_dict()) != frozen or any(file_digest(repository/p) != d for p, d in sources.items()):
        raise ValueError("source changed during experiment")
    if audit_artifact_manifest(archive, "manifest.json") != archive_manifest:
        raise ValueError("archive changed")
    report = {"format": "snowgym.post-hit-report.v0", **LABEL, "coverage": coverage,
        "baselineReproduction": True, "eligibleEpisodes": len(branches), "duplicateBranches": len(branches)*len(ARMS),
        "summaries": [{a: {k: b[a][k] for k in ("seed", "local", "final", "diagnostics", "totalActions", "rejectedActions")}
                       for a in ARMS} for b in branches], **report_effects(branches)}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.post-hit-manifest.v0", **LABEL,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({"eligible": result["eligibleEpisodes"], "supported": result["supportedForFurtherStudy"]}))
