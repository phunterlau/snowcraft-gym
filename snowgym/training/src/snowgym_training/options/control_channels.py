"""Frozen R1h conditional action-choice/movement experiment; no policy fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..checkpoint import semantic_state_digest
from ..ppo import HybridActorCritic, conditioned_target_mean
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .environment import FixedPlanOptionBatchEnv
from .identity import checkpoint_model
from .interventions import contact_distance, require_capabilities, require_raw, team_health
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol
from .reservoir import file_digest
from .supervised_probe import write_json
from .throw_channels import recommend_shots, validate_teacher_agreement

ARMS = ("shot-only", "teacher-move", "teacher-choice", "teacher-choice-move", "teacher")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1h_control_channels_v0.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    expected = {
        "format": "snowgym.engage-control-channels-config.v0",
        "checkpointDigest": "sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697",
        "seeds": [200000, 200039], "arms": list(ARMS),
        "simulationVersion": "snowgym.sim.v2", "stateHashVersion": "snowgym.state.v2",
        "bootstrapSeed": 750001, "bootstrapSamples": 10000,
    }
    if value != expected:
        raise ValueError("control-channel configuration differs from frozen R1h design")
    return value


def recommend_movement(raw: dict[str, Any], capacity: int) -> dict[str, np.ndarray]:
    """Conditional destination for frozen single-main Engage/direct/balanced/normal.

    Mirrors ReactiveUnitPolicy and TacticalFrame for this open 5v5 scenario only.
    Does not decide MOVE versus HOLD/THROW/NOOP or issue simulator commands.
    All initial enemies are the assigned cluster; all living allies are main.
    """
    target = np.zeros((1, capacity, 2), dtype=np.float32)
    valid = np.zeros((1, capacity), dtype=bool)
    threat = np.zeros((1, capacity), dtype=bool)
    ready = np.zeros((1, capacity), dtype=bool)
    distance = np.full((1, capacity), np.inf)
    allies = [u for u in raw["allies"] if u["alive"]]
    enemies = [u for u in raw["enemies"] if u["alive"]]
    if len(raw["allies"]) > capacity:
        raise ValueError("movement recommendation capacity is too small")
    if not allies or not enemies:
        return {"target": target, "valid": valid, "threat": threat, "ready": ready, "distance": distance}
    own = np.asarray([sum(u[k] for u in allies) / len(allies) for k in ("x", "y")])
    enemy_center = np.asarray([sum(u[k] for u in enemies) / len(enemies) for k in ("x", "y")])
    axis = enemy_center - own
    length = np.hypot(*axis)
    axis = axis / length if length > 1e-9 else np.asarray([1.0, 0.0])
    left = np.asarray([-axis[1], axis[0]])
    members = sorted(u["id"] for u in allies)
    scale = np.asarray([raw["arena"]["width"] / 2, raw["arena"]["height"] / 2])
    for i, unit in enumerate(raw["allies"]):
        if not unit["alive"]:
            continue
        pos = np.asarray([unit["x"], unit["y"]])
        enemy = min(enemies, key=lambda u: ((u["x"] - unit["x"]) ** 2 + (u["y"] - unit["y"]) ** 2, u["id"]))
        point = np.asarray([enemy["x"], enemy["y"]])
        delta = point - pos
        distance[0, i] = np.hypot(*delta)
        ready[0, i] = unit["throwCooldown"] <= 0 and unit["state"] in {"idle", "moving", "preparingThrow"}
        destination = point - delta / max(distance[0, i], 1e-9) * 6.5
        destination += left * (members.index(unit["id"]) - (len(members) - 1) / 2) * 1.2
        destination += (own - pos) * 0.12
        incoming = None
        nearest_squared = 3.5 ** 2
        for projectile in raw["projectiles"]:
            if projectile["team"] == unit["team"]:
                continue
            dx, dy = unit["x"] - projectile["x"], unit["y"] - projectile["y"]
            squared = dx * dx + dy * dy
            if projectile["vx"] * dx + projectile["vy"] * dy > 0 and squared <= nearest_squared:
                incoming, nearest_squared = projectile, squared
        threat[0, i] = incoming is not None
        if incoming is not None and unit["state"] in {"idle", "moving", "recovering"}:
            speed = np.hypot(incoming["vx"], incoming["vy"])
            if speed > 1e-9:
                side = 1 if unit["id"] % 2 == 0 else -1
                destination = pos + np.asarray([-incoming["vy"], incoming["vx"]]) / speed * 2.4 * side
            else:
                destination = pos
        target[0, i] = np.clip(destination, -scale + 0.5, scale - 0.5) / scale
        valid[0, i] = True
    return {"target": target, "valid": valid, "threat": threat, "ready": ready, "distance": distance}


def validate_movement_agreement(teacher: dict[str, np.ndarray], movement: dict[str, np.ndarray]) -> int:
    selected = teacher["action_type"] == 1
    if bool((selected & ~movement["valid"]).any()) or not np.allclose(
        teacher["target"][selected], movement["target"][selected], rtol=0, atol=2e-6
    ):
        raise ValueError("movement oracle disagrees with production teacher")
    return int(selected.sum())


def compose_action(arm: str, prediction: dict[str, torch.Tensor], teacher: dict[str, np.ndarray],
                   shot: dict[str, np.ndarray], movement: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if arm not in ARMS:
        raise ValueError("unknown control-channel arm")
    if arm == "teacher":
        return {k: v.copy() for k, v in teacher.items()}
    action_type = (torch.as_tensor(teacher["action_type"], device=prediction["action_logits"].device)
                   if arm in {"teacher-choice", "teacher-choice-move"}
                   else prediction["action_logits"].argmax(-1))
    # Select the conditional head AFTER replacing the action type.
    result = numpy_actions({"action_type": action_type,
                            "target": torch.tanh(conditioned_target_mean(prediction, action_type)),
                            "power": torch.sigmoid(prediction["power_raw"])})
    throws = result["action_type"] == 2
    if bool((throws & ~shot["valid"]).any()):
        raise ValueError("executed throw lacks a living recommended target")
    result["target"][throws], result["power"][throws] = shot["target"][throws], shot["power"][throws]
    if arm in {"teacher-move", "teacher-choice-move"}:
        moves = result["action_type"] == 1
        if bool((moves & ~movement["valid"]).any()):
            raise ValueError("executed move lacks a conditional destination")
        result["target"][moves] = movement["target"][moves]
    return result


def evaluate_episode(model: HybridActorCritic, *, seed: int, arm: str, client: SnowGymBatchClient) -> dict[str, Any]:
    plan, spec = teacher_option_plan("engage")
    base = SnowGymBatchEnv(1, client=client, observation_version=3)
    wrapper = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
    observation, _ = wrapper.reset([seed], [teacher_option_scenario("engage")],
                                  [f"control-channels-{arm}-{seed}"], [plan], [spec])
    raw = require_raw(base)
    assigned = tuple(int(u["id"]) for u in raw["allies"])
    initial_health = team_health(raw["enemies"])
    hashes, actions = [base.state_hashes[0]], []
    confusion = np.zeros((4, 4), dtype=np.int64)  # living units: learner row, teacher column
    counts = {key: 0 for key in ("teacherMoveAgreements", "teacherThrowAgreements", "moveReplacements",
        "choiceChanges", "executedThrows", "executedMoves", "throwsNotReady", "throwsOutOfRange",
        "throwsWithThreat", "throwsWhileTeacherDoesNotThrow", "rejectedActions", "totalActions")}
    shot_distances: list[float] = []
    first_contact = first_hit = final = None
    for decision in range(spec.horizon):
        raw = require_raw(base)
        shot = recommend_shots(raw, base.max_team_units)
        movement = recommend_movement(raw, base.max_team_units)
        teacher = base.plan_teacher_tensor_actions()
        counts["teacherThrowAgreements"] += validate_teacher_agreement(teacher, shot)
        counts["teacherMoveAgreements"] += validate_movement_agreement(teacher, movement)
        with torch.no_grad():
            prediction = model(tensor_dict(observation))
            executed = compose_action(arm, prediction, teacher, shot, movement)
        learner_type = prediction["action_logits"].argmax(-1).cpu().numpy()
        living = np.zeros_like(learner_type, dtype=bool)
        living[0, :len(raw["allies"])] = [u["alive"] for u in raw["allies"]]
        np.add.at(confusion, (learner_type[living], teacher["action_type"][living]), 1)
        moves, throws = executed["action_type"] == 1, executed["action_type"] == 2
        counts["executedMoves"] += int(moves.sum())
        counts["executedThrows"] += int(throws.sum())
        counts["choiceChanges"] += int(((executed["action_type"] != learner_type) & living).sum())
        counts["moveReplacements"] += int(moves.sum()) if arm in {"teacher-move", "teacher-choice-move"} else 0
        counts["throwsNotReady"] += int((throws & ~movement["ready"]).sum())
        counts["throwsOutOfRange"] += int((throws & (movement["distance"] > 9)).sum())
        counts["throwsWithThreat"] += int((throws & movement["threat"]).sum())
        counts["throwsWhileTeacherDoesNotThrow"] += int((throws & (teacher["action_type"] != 2)).sum())
        shot_distances.extend(movement["distance"][throws].tolist())
        actions.append({k: v.tolist() for k, v in executed.items()})
        observation, _, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]
        hashes.append(base.state_hashes[0])
        results = info.get("actionResults", [])
        counts["rejectedActions"] += sum(r.get("accepted") is False for r in results)
        counts["totalActions"] += len(results)
        raw = require_raw(base)
        if first_contact is None and contact_distance(raw, assigned) <= 9:
            first_contact = decision + 1
        if first_hit is None and team_health(raw["enemies"]) < initial_health:
            first_hit = decision + 1
        if terminated[0] or truncated[0]:
            final = info["option"]
            break
    if final is None:
        raise RuntimeError("control intervention did not reach option boundary")
    return {"seed": seed, "arm": arm, "success": final["success"], "progress": final["progress"],
            "decisions": final["decision"], "firstContactDecision": first_contact, "firstHitDecision": first_hit,
            **counts, "choiceConfusion": confusion.tolist(), "shotDistanceSum": sum(shot_distances),
            "shotDistanceMax": max(shot_distances, default=None), "stateHashes": hashes,
            "actionsDigest": json_digest(actions)}


def summarize(records: dict[str, list[dict[str, Any]]], *, samples: int = 10000, bootstrap_seed: int = 750001) -> dict[str, Any]:
    if set(records) != set(ARMS):
        raise ValueError("control matrix requires every frozen arm")
    seeds = [r["seed"] for r in records["shot-only"]]
    if not seeds or len(seeds) != len(set(seeds)) or any([r["seed"] for r in rows] != seeds for rows in records.values()):
        raise ValueError("control-channel seeds must be unique and paired")
    summaries = {}
    for arm, rows in records.items():
        throws = sum(r["executedThrows"] for r in rows)
        summaries[arm] = {
            "episodes": len(rows), "successRate": float(np.mean([r["success"] for r in rows])),
            "hitRate": float(np.mean([r["firstHitDecision"] is not None for r in rows])),
            "contactRate": float(np.mean([r["firstContactDecision"] is not None for r in rows])),
            "meanProgress": float(np.mean([r["progress"] for r in rows])),
            "meanShotDistance": sum(r["shotDistanceSum"] for r in rows) / throws if throws else None,
            "choiceConfusion": np.sum([r["choiceConfusion"] for r in rows], axis=0).tolist(),
            **{k: sum(r[k] for r in rows) for k in rows[0] if k in {
                "teacherMoveAgreements", "teacherThrowAgreements", "moveReplacements", "choiceChanges",
                "executedThrows", "executedMoves", "throwsNotReady", "throwsOutOfRange", "throwsWithThreat",
                "throwsWhileTeacherDoesNotThrow", "rejectedActions", "totalActions"}},
        }
    indices = np.random.default_rng(bootstrap_seed).integers(0, len(seeds), (samples, len(seeds)))
    contrasts = {
        "movementWithLearnerChoice": {"teacher-move": 1, "shot-only": -1},
        "choiceWithLearnerMovement": {"teacher-choice": 1, "shot-only": -1},
        "movementWithTeacherChoice": {"teacher-choice-move": 1, "teacher-choice": -1},
        "choiceWithTeacherMovement": {"teacher-choice-move": 1, "teacher-move": -1},
        "interaction": {"teacher-choice-move": 1, "teacher-choice": -1, "teacher-move": -1, "shot-only": 1},
    }
    comparisons = {}
    for name, weights in contrasts.items():
        comparisons[name] = {}
        for metric in ("success", "progress"):
            delta = sum(weight * np.asarray([float(r[metric]) for r in records[arm]]) for arm, weight in weights.items())
            comparisons[name][metric] = {"mean": float(delta.mean()), "bootstrap95": np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    return {"arms": summaries, "pairedContrasts": comparisons}


def run_matrix(checkpoint: str | Path, *, output: str | Path, config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite control intervention {destination}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    protocol = load_option_protocol()
    if (metadata["checkpointDigest"] != config["checkpointDigest"] or metadata["curriculumDigest"] != json_digest(protocol)
        or config["seeds"] != [protocol["seeds"]["development"][0], protocol["seeds"]["development"][0] + 39]):
        raise ValueError("control-channel checkpoint/protocol mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    before = semantic_state_digest(model.state_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    seeds = list(range(config["seeds"][0], config["seeds"][1] + 1))
    try:
        records = {}
        with SnowGymBatchClient() as client:
            capabilities = require_capabilities(client)
            if any(capabilities[k] != config[k] for k in ("simulationVersion", "stateHashVersion")):
                raise ValueError("control-channel simulator provenance mismatch")
            for arm in ARMS:
                records[arm] = [evaluate_episode(model, seed=seed, arm=arm, client=client) for seed in seeds]
                write_json(root / f"{arm}.json", records[arm])
                print(json.dumps({"arm": arm, "successes": sum(r["success"] for r in records[arm])}), flush=True)
        if semantic_state_digest(model.state_dict()) != before:
            raise RuntimeError("no-training experiment changed model state")
        # Full restoration must reproduce the teacher's transitions exactly.
        if any(a["stateHashes"] != b["stateHashes"] for a, b in zip(records["teacher-choice-move"], records["teacher"], strict=True)):
            raise RuntimeError("combined intervention fails full-teacher trajectory parity")
        report = {"format": "snowgym.engage-control-channels-report.v0", "trainingUpdates": 0,
                  "modelUnchanged": True, "qualificationEligible": False, "teacherTrajectoryParity": True,
                  "summary": summarize(records, samples=config["bootstrapSamples"], bootstrap_seed=config["bootstrapSeed"])}
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        sources = {str(p.relative_to(repository)): file_digest(p) for p in sorted(module_root.rglob("*.py"))}
        for relative in ("snowgym/orchestration/execution/ReactiveUnitPolicy.ts", "snowgym/orchestration/grounding/TacticalFrame.ts"):
            sources[relative] = file_digest(repository / relative)
        manifest = {"format": "snowgym.engage-control-channels-run.v0", "gitCommit": resolve_git_commit(),
                    "config": config, "configDigest": json_digest(config), "checkpoint": metadata,
                    "protocolDigest": json_digest(protocol), "seeds": seeds, "seedPartition": "development",
                    "capabilities": capabilities, "sourceFiles": sources,
                    "artifacts": {p.name: file_digest(p) for p in sorted(root.iterdir())}}
        manifest["manifestDigest"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        root.replace(destination)
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(run_matrix(args.checkpoint, output=args.output, config_path=args.config), sort_keys=True))


if __name__ == "__main__":
    main()
