"""Frozen R1k learner-action-conditioned audit; never promotes fitted weights."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..checkpoint import semantic_state_digest
from ..ppo_collect import numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .control_channels import recommend_movement, validate_movement_agreement
from .environment import FixedPlanOptionBatchEnv
from .geometry_probe import load_probe
from .interventions import require_capabilities, require_raw, team_health
from .opportunity_metrics import cross_tabs, describe, hard_fit, physical_errors, select_opportunities
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol
from .reservoir import file_digest, load_teacher_bc_reservoir
from .supervised_probe import write_json
from .throw_channels import recommend_shots, validate_teacher_agreement

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1k_opportunities_v0.json"
REFERENCE_DIGEST = "sha256:774102a64d32997cb346ff374bf6e219dba6a3028af9aa32f9b9961a438b9203"
RESERVOIR_DIGEST = "sha256:a1410c32a718c53664b91878852a2203247454ae0bf2dcb4caeb904b0ac334a6"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    expected = {
        "format": "snowgym.opportunity-audit-config.v0", "checkpointDigest": REFERENCE_DIGEST,
        "reservoirDigest": RESERVOIR_DIGEST, "simulationVersion": "snowgym.sim.v2",
        "stateHashVersion": "snowgym.state.v2", "trainingSeeds": [100000, 100039],
        "channels": ["move", "aim", "power"], "opportunitiesPerChannel": 64,
        "perEpisodeCap": 4, "branchDecisions": 30, "hardFitSteps": 200,
        "hardFitLearningRate": .001, "hardFitTrainingSeeds": [100000, 100031],
        "hardFitValidationSeeds": [100032, 100039], "bootstrapSamples": 10000,
        "bootstrapSeed": 780001, "freshHoldoutsCollected": False, "qualificationEligible": False,
    }
    if config != expected:
        raise ValueError("configuration differs from frozen R1k protocol")
    return config


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_jsonl(path, rows):
    # Deterministic gzip header: neither wall time nor temporary filename enters it.
    with Path(path).open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(plain(row), sort_keys=True, allow_nan=False)+"\n").encode())


def reset_world(client, seed, kind):
    plan, spec = teacher_option_plan("engage")
    base = SnowGymBatchEnv(1, client=client, observation_version=3)
    wrapped = FixedPlanOptionBatchEnv(base, gamma=.9976921765)
    plan_id = f"engage-r1-teacher-{seed}" if kind == "teacher" else f"r1k-learner-{seed}"
    observation, _ = wrapped.reset([seed], [teacher_option_scenario("engage")], [plan_id], [plan], [spec])
    return base, wrapped, observation


def identity(base, wrapped):
    body = base.plan_observations()[1][0]
    tracker = plain(vars(wrapped.trackers[0]))
    return {"physical": base.state_hashes[0], "plan": json_digest(plain(body)),
            "tracker": json_digest(tracker)}


def label_state(model, base, wrapped, observation, *, seed, decision, state_index, kind):
    before = identity(base, wrapped)
    raw = copy.deepcopy(require_raw(base))
    teacher = base.plan_teacher_tensor_actions()
    movement = recommend_movement(raw, base.max_team_units)
    shots = recommend_shots(raw, base.max_team_units)
    validate_movement_agreement(teacher, movement)
    validate_teacher_agreement(teacher, shots)
    with torch.no_grad():
        prediction = model(tensor_dict(observation))
        action, _, _ = model.act(tensor_dict(observation), deterministic=True)
    learner = numpy_actions(action)
    targets = prediction["target_by_action"][0].cpu().numpy()
    power = prediction["power"][0].cpu().numpy()
    available_move = movement["valid"] & observation["unit_action_mask"][..., 1].astype(bool)
    available_shot = shots["valid"] & observation["unit_action_mask"][..., 2].astype(bool)
    labels = {"move_target": movement["target"], "shot_target": shots["target"], "power": shots["power"],
              "move_mask": available_move, "shot_mask": available_shot}
    state = {"stateIndex": state_index, "kind": kind, "seed": seed, "decision": decision,
             "identity": before, "observation": copy.deepcopy(observation), "raw": raw,
             "plan": plain(base.plan_observations()[1][0]), "optionState": plain(vars(wrapped.trackers[0])),
             "teacher": teacher, "learner": learner, "labels": labels,
             "allConditionalTargets": targets, "allConditionalRawTargets": prediction["target_raw_by_action"][0].detach().numpy(),
             "powerPrediction": power, "recommendationProvenance": "R1h-movement/R1g-shot-open-main-engage"}
    rows = []
    for slot, unit in enumerate(raw["allies"]):
        if not unit["alive"]:
            continue
        distances = sorted(math.hypot(u["x"]-unit["x"], u["y"]-unit["y"])
                           for u in raw["enemies"] if u["alive"])
        teacher_type, learner_type = int(teacher["action_type"][0, slot]), int(learner["action_type"][0, slot])
        rows.append({"opportunityId": f"{seed}:{decision}:{unit['id']}", "stateIndex": state_index,
            "seed": seed, "decision": decision, "unitId": unit["id"], "slot": slot,
            "stateHash": before["physical"], "teacherType": teacher_type, "learnerType": learner_type,
            "legal": observation["unit_action_mask"][0, slot].astype(bool).tolist(),
            "ready": bool(movement["ready"][0, slot]), "moveAvailable": bool(movement["valid"][0, slot]),
            "shotAvailable": bool(shots["valid"][0, slot]), "selectedEnemyId": int(shots["enemyIds"][0, slot]),
            "targetDistanceMargin": distances[1]-distances[0] if len(distances) > 1 else None,
            "range": distances[0] if distances else None, "threat": bool(movement["threat"][0, slot]),
            "oldMoveMask": teacher_type == 1, "oldShotMask": teacher_type == 2,
            **physical_errors(observation["allies"][0, slot, 2:4], targets[slot, 1], targets[slot, 2], power[slot],
                              movement["target"][0, slot], shots["target"][0, slot], shots["power"][0, slot])})
    if identity(base, wrapped) != before:
        raise RuntimeError("labeling mutated physical/plan/option state")
    return state, rows


def collect(model, client, config, *, kind, reservoir=None):
    states, rows, episodes = [], [], []
    offset = 0
    for seed in range(config["trainingSeeds"][0], config["trainingSeeds"][1]+1):
        base, wrapped, observation = reset_world(client, seed, kind)
        initial_raw = require_raw(base)
        blue_initial, red_initial = team_health(initial_raw["allies"]), team_health(initial_raw["enemies"])
        first_hit, first_contact, actions, episode_rows = None, None, [], []
        damage, received, ready, legal_throws, in_range, live_opportunities = 0., 0., 0, 0, 0, 0
        rejections, action_count = 0, 0
        for decision in range(200):
            state, opportunities = label_state(model, base, wrapped, observation, seed=seed,
                decision=decision, state_index=len(states), kind=kind)
            if reservoir is not None:
                if offset >= reservoir.size:
                    raise ValueError("teacher reconstruction exceeds archived reservoir")
                for name, expected in reservoir.observations.items():
                    if not np.array_equal(observation[name][0], expected[offset].numpy()):
                        raise ValueError(f"teacher reservoir observation mismatch at {offset}: {name}")
                for name, expected in reservoir.actions.items():
                    if not np.array_equal(state["teacher"][name][0], expected[offset].numpy()):
                        raise ValueError(f"teacher reservoir action mismatch at {offset}: {name}")
                offset += 1
            for row in opportunities:
                if row["range"] is not None and row["range"] <= 9 and first_contact is None:
                    first_contact = decision
                row["phase"] = ("recovery" if row["threat"] or state["raw"]["allies"][row["slot"]]["state"] in {"stunned", "recovering"}
                                else "sustained" if first_hit is not None else "contact" if first_contact is not None else "approach")
            episode_rows.extend(opportunities)
            states.append(state)
            rows.extend(opportunities)
            selected = state["teacher"] if kind == "teacher" else state["learner"]
            actions.append(plain(selected))
            old_blue, old_red = team_health(state["raw"]["allies"]), team_health(state["raw"]["enemies"])
            if kind == "teacher":
                observation, _, terminated, truncated, infos = wrapped.step(teacher=True)
            else:
                observation, _, terminated, truncated, infos = wrapped.step(selected)
            raw = require_raw(base)
            dealt = max(0., old_red-team_health(raw["enemies"]))
            damage += dealt
            received += max(0., old_blue-team_health(raw["allies"]))
            if dealt > 0 and first_hit is None:
                first_hit = decision+1
            ready += sum(r["ready"] for r in opportunities)
            legal_throws += sum(selected["action_type"][0, r["slot"]] == 2 and r["legal"][2] for r in opportunities)
            in_range += sum(r["range"] is not None and 5 <= r["range"] <= 9 for r in opportunities)
            live_opportunities += len(opportunities)
            results = infos[0].get("actionResults", [])
            rejections += sum(r.get("accepted") is False for r in results)
            action_count += len(results)
            if bool(terminated[0] or truncated[0]):
                break
        option = infos[0]["option"]
        if kind == "teacher" and not option["success"]:
            raise ValueError("teacher reconstruction failed its archived successful episode")
        episodes.append({"seed": seed, "kind": kind, "actions": actions, "decisions": len(actions),
            "success": option["success"], "progress": option["progress"], "firstContactDecision": first_contact,
            "firstHitDecision": first_hit, "completionAfterFirstHit": len(actions)-first_hit if first_hit is not None and option["success"] else None,
            "completionCensored": not option["success"], "damageDealt": damage, "damageReceived": received,
            "damagePerReadyOpportunity": damage/ready if ready else None, "readyOpportunities": ready,
            "legalThrows": int(legal_throws), "damagePerLegalThrow": damage/legal_throws if legal_throws else None,
            "usefulRangeOccupancy": in_range/max(live_opportunities, 1),
            "blueHealthFraction": team_health(raw["allies"])/blue_initial,
            "redHealthFraction": team_health(raw["enemies"])/red_initial,
            "rejectedActions": rejections, "totalActions": action_count,
            "tailErrors": {k: describe([r[k] for r in episode_rows if r[k] is not None])
                           for k in ("moveErrorWorld", "angleDegrees", "powerError")}})
        print(json.dumps({"phase": f"collect-{kind}", "seed": seed, "states": len(states),
                          "success": option["success"]}), flush=True)
    if reservoir is not None and offset != reservoir.size:
        raise ValueError("teacher reconstruction does not cover entire reservoir")
    return states, rows, episodes


def substitute(state, row, channel):
    action = {k: np.array(v, copy=True) for k, v in state["learner"].items()}
    slot = row["slot"]
    expected_type = 1 if channel == "move" else 2
    if channel not in {"move", "aim", "power"} or action["action_type"][0, slot] != expected_type:
        raise ValueError("substitution does not match the executed conditional head")
    if not row["legal"][expected_type] or not row["moveAvailable" if channel == "move" else "shotAvailable"]:
        raise ValueError("substitution requires legal action and available recommendation")
    if channel in {"move", "aim"}:
        action["target"][0, slot] = state["labels"]["move_target" if channel == "move" else "shot_target"][0, slot]
    else:
        action["power"][0, slot] = state["labels"]["power"][0, slot]
    return action


def branch(model, client, state, row, episode, channel, *, replace, horizon):
    base, wrapped, observation = reset_world(client, state["seed"], "learner")
    for action in episode["actions"][:state["decision"]]:
        observation, _, terminated, truncated, _ = wrapped.step({k: np.asarray(v) for k, v in action.items()})
        if bool(terminated[0] or truncated[0]):
            raise ValueError("prefix terminated before branch point")
    if identity(base, wrapped) != state["identity"]:
        raise RuntimeError("prefix replay failed physical/plan/tracker identity")
    raw = require_raw(base)
    blue, red = team_health(raw["allies"]), team_health(raw["enemies"])
    hashes, action_digests, range_errors = [], [], []
    for step in range(horizon):
        if step == 0:
            action = substitute(state, row, channel) if replace else {k: np.array(v, copy=True) for k, v in state["learner"].items()}
        else:
            with torch.no_grad():
                prediction, _, _ = model.act(tensor_dict(observation), deterministic=True)
            action = numpy_actions(prediction)
        action_digests.append(json_digest(plain(action)))
        observation, _, terminated, truncated, infos = wrapped.step(action)
        hashes.append(base.state_hashes[0])
        raw = require_raw(base)
        unit = next((u for u in raw["allies"] if u["id"] == row["unitId"] and u["alive"]), None)
        enemies = [u for u in raw["enemies"] if u["alive"]]
        if unit is not None and enemies:
            nearest = min(math.hypot(e["x"]-unit["x"], e["y"]-unit["y"]) for e in enemies)
            range_errors.append(abs(nearest-6.5))
        if bool(terminated[0] or truncated[0]):
            break
    return {"startIdentity": state["identity"], "stateHashes": hashes, "actionDigests": action_digests,
            "decisions": len(hashes), "damageDealt": red-team_health(raw["enemies"]),
            "damageReceived": blue-team_health(raw["allies"]),
            "rangeError": float(np.mean(range_errors)) if range_errors else None,
            "progress": infos[0]["option"]["progress"], "success": infos[0]["option"]["success"]}


def branch_summary(records, config):
    summary = {}
    for channel in config["channels"]:
        rows = [r for r in records if r["channel"] == channel]
        grouped = {}
        for row in rows:
            base, corrected = row["base"], row["corrected"]
            metrics = {"damageDealt": corrected["damageDealt"]-base["damageDealt"],
                       "damageReceived": corrected["damageReceived"]-base["damageReceived"],
                       "progress": corrected["progress"]-base["progress"],
                       "rangeErrorImprovement": None if base["rangeError"] is None or corrected["rangeError"] is None else base["rangeError"]-corrected["rangeError"]}
            metrics["netDamage"] = metrics["damageDealt"]-metrics["damageReceived"]
            grouped.setdefault(row["seed"], []).append(metrics)
        estimates = {}
        for metric in ("damageDealt", "damageReceived", "netDamage", "progress", "rangeErrorImprovement"):
            values = np.asarray([np.mean([r[metric] for r in group if r[metric] is not None])
                                 for group in grouped.values() if any(r[metric] is not None for r in group)])
            if not len(values):
                estimates[metric] = None
                continue
            rng = np.random.default_rng(config["bootstrapSeed"])
            bootstrap = values[rng.integers(len(values), size=(config["bootstrapSamples"], len(values)))].mean(1)
            estimates[metric] = {"mean": float(values.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                 "episodes": len(values)}
        net, distance = estimates["netDamage"], estimates["rangeErrorImprovement"]
        useful = bool(net and (net["ci95"][0] > 0 or (channel == "move" and distance
                      and distance["ci95"][0] > 0 and net["mean"] >= 0)))
        summary[channel] = {"opportunities": len(rows), "episodes": len(grouped), "effects": estimates,
                            "useful": useful, "bootstrapUnit": "episode means; selected hard opportunities"}
    return summary


def run_audit(*, checkpoint, reservoir_path, output, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite opportunity audit {destination}")
    model, metadata = load_probe(checkpoint)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    if metadata["checkpointDigest"] != config["checkpointDigest"] or metadata["relative"] or reservoir.metadata["digest"] != config["reservoirDigest"]:
        raise ValueError("R1k checkpoint/reservoir identity mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_digest = semantic_state_digest(model.state_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        with SnowGymBatchClient() as client:
            capabilities = require_capabilities(client)
            if any(capabilities[k] != config[k] for k in ("simulationVersion", "stateHashVersion")):
                raise ValueError("R1k simulator version mismatch")
            teacher_states, teacher_rows, teacher_episodes = collect(model, client, config, kind="teacher", reservoir=reservoir)
            write_jsonl(root / "teacher-states.jsonl.gz", teacher_states)
            write_jsonl(root / "teacher-opportunities.jsonl.gz", teacher_rows)
            write_jsonl(root / "teacher-episodes.jsonl.gz", teacher_episodes)
            del teacher_states
            states, rows, episodes = collect(model, client, config, kind="learner")
            write_jsonl(root / "learner-states.jsonl.gz", states)
            write_jsonl(root / "learner-opportunities.jsonl.gz", rows)
            write_jsonl(root / "learner-episodes.jsonl.gz", episodes)
            fitting = hard_fit(model, states, rows, steps=config["hardFitSteps"], learning_rate=config["hardFitLearningRate"])
            write_json(root / "hard-fit.json", fitting)
            print(json.dumps({"phase": "hard-fit", "passed": fitting["passed"], "reduction": fitting.get("reduction")}), flush=True)
            indexed_episodes = {e["seed"]: e for e in episodes}
            branches = []
            for channel in config["channels"]:
                selected = select_opportunities(rows, channel, limit=config["opportunitiesPerChannel"], per_episode=config["perEpisodeCap"])
                for index, row in enumerate(selected):
                    state, episode = states[row["stateIndex"]], indexed_episodes[row["seed"]]
                    result = {"channel": channel, "seed": row["seed"], "opportunityId": row["opportunityId"],
                              "oldMaskExcluded": row["teacherType"] != row["learnerType"]}
                    for replace in (False, True):
                        result["corrected" if replace else "base"] = branch(model, client, state, row, episode, channel,
                            replace=replace, horizon=config["branchDecisions"])
                    branches.append(result)
                    if (index+1) % 8 == 0:
                        print(json.dumps({"phase": "branches", "channel": channel, "completed": index+1}), flush=True)
            write_jsonl(root / "branches.jsonl.gz", branches)
        summary = branch_summary(branches, config)
        unchanged = semantic_state_digest(model.state_dict()) == source_digest
        if not unchanged:
            raise RuntimeError("audit mutated frozen reference")
        gate = fitting["passed"] and all(summary[k]["useful"] for k in ("move", "aim"))
        report = {"format": "snowgym.opportunity-audit-report.v0", "r1lAllowed": gate,
            "qualificationEligible": False, "productionUpdates": 0, "sourceUnchanged": unchanged,
            "teacherReservoirExact": True, "teacherStates": reservoir.size, "learnerStates": len(states),
            "learnerOpportunities": len(rows), "learnerSuccesses": sum(e["success"] for e in episodes),
            "teacherCrossTabs": cross_tabs(teacher_rows), "learnerCrossTabs": cross_tabs(rows),
            "branches": summary, "hardFitPassed": fitting["passed"],
            "decision": "predeclare R1l factorial" if gate else "stop corrective-data branch; review failed audit gates",
            "freshHoldoutsUsed": False,
            "limitations": ["Recommendations are scenario-specific and not asserted optimal.",
                "Hard-state episode bootstrap is conditional on selected opportunities, not population coverage proof.",
                "Same reset seed does not imply identical opponent actions after a substitution.",
                "Eight audit-validation episodes were held out of this fit, not necessarily source training.",
                "Damage per legal throw is not projectile hit probability."]}
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        manifest = {"format": "snowgym.opportunity-audit-run.v0", "gitCommit": resolve_git_commit(),
            "config": config, "configDigest": json_digest(config), "checkpoint": metadata,
            "reservoir": reservoir.metadata, "capabilities": capabilities,
            "optionProtocolDigest": json_digest(load_option_protocol()),
            "sourceFiles": {str(p.relative_to(repository)): file_digest(p) for p in sorted(module_root.rglob("*.py"))},
            "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
        manifest["manifestDigest"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        root.replace(destination)
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_audit(checkpoint=args.checkpoint, reservoir_path=args.reservoir, output=args.output, config_path=args.config)
    print(json.dumps({k: result[k] for k in ("r1lAllowed", "decision", "learnerSuccesses")}), flush=True)


if __name__ == "__main__":
    main()
