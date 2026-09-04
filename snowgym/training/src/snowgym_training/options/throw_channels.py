"""Frozen, no-training R1g throw recommendation interventions on the R1f actor."""

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
from ..ppo import HybridActorCritic
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

ARMS = ("learner", "direction", "power", "direction-power", "teacher")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1g_throw_channels_v0.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "format": "snowgym.engage-throw-channels-config.v0",
        "checkpointDigest": "sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697",
        "seeds": [200000, 200039], "arms": list(ARMS),
        "simulationVersion": "snowgym.sim.v2", "stateHashVersion": "snowgym.state.v2",
        "teacherLeadSeconds": 0.18, "teacherMaximumRange": 9.0,
        "bootstrapSeed": 740001, "bootstrapSamples": 10000,
    }
    if value != expected:
        raise ValueError("throw-channel configuration differs from frozen R1g design")
    return value


def recommend_shots(raw: dict[str, Any], capacity: int) -> dict[str, np.ndarray]:
    """Production Engage/opportunistic/medium shot heuristic, independent of firing choice.

    Only valid for the frozen open 5v5, single-main-group protocol in this module.
    All five initial enemies form its selected cluster; casualties never add targets.
    This is an intervention oracle, not a learned policy or a general plan teacher.
    """
    target = np.zeros((1, capacity, 2), dtype=np.float32)
    power = np.zeros((1, capacity), dtype=np.float32)
    valid = np.zeros((1, capacity), dtype=np.bool_)
    enemy_ids = np.full((1, capacity), -1, dtype=np.int64)
    enemies = [unit for unit in raw["enemies"] if unit["alive"]]
    if len(raw["allies"]) > capacity:
        raise ValueError("shot recommendation capacity is too small")
    scale = np.asarray([raw["arena"]["width"] / 2, raw["arena"]["height"] / 2])
    for index, ally in enumerate(raw["allies"]):
        if not ally["alive"] or not enemies:
            continue
        enemy = min(enemies, key=lambda unit: (
            (unit["x"] - ally["x"]) ** 2 + (unit["y"] - ally["y"]) ** 2, unit["id"]))
        distance = np.hypot(enemy["x"] - ally["x"], enemy["y"] - ally["y"])
        target[0, index] = np.clip(np.asarray([
            enemy["x"] + enemy["vx"] * 0.18, enemy["y"] + enemy["vy"] * 0.18,
        ]) / scale, -1, 1)
        power[0, index] = np.clip(((distance - 1.5) / (9.0 - 1.5)) * 0.9 + 0.1, 0.18, 1)
        valid[0, index] = True
        enemy_ids[0, index] = enemy["id"]
    return {"target": target, "power": power, "valid": valid, "enemyIds": enemy_ids}


def validate_teacher_agreement(teacher: dict[str, np.ndarray], recommendation: dict[str, np.ndarray]) -> int:
    throws = teacher["action_type"] == 2
    if bool((throws & ~recommendation["valid"]).any()):
        raise ValueError("production teacher throw lacks an oracle target")
    if (not np.allclose(teacher["target"][throws], recommendation["target"][throws], rtol=0, atol=2e-6)
        or not np.allclose(teacher["power"][throws], recommendation["power"][throws], rtol=0, atol=2e-6)):
        raise ValueError("shot oracle disagrees with production teacher")
    return int(throws.sum())


def compose_action(arm: str, learner: dict[str, np.ndarray], teacher: dict[str, np.ndarray],
                   recommendation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if arm not in ARMS:
        raise ValueError("unknown throw-channel arm")
    if arm == "teacher":
        return {name: value.copy() for name, value in teacher.items()}
    result = {name: value.copy() for name, value in learner.items()}
    selected = learner["action_type"] == 2
    if bool((selected & ~recommendation["valid"]).any()):
        raise ValueError("learner-selected throw has no living oracle target")
    if arm in {"direction", "direction-power"}:
        result["target"][selected] = recommendation["target"][selected]
    if arm in {"power", "direction-power"}:
        result["power"][selected] = recommendation["power"][selected]
    return result


def evaluate_episode(model: HybridActorCritic, *, seed: int, arm: str, client: SnowGymBatchClient) -> dict[str, Any]:
    plan, spec = teacher_option_plan("engage")
    base = SnowGymBatchEnv(1, client=client, observation_version=3)
    wrapper = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
    observation, _ = wrapper.reset([seed], [teacher_option_scenario("engage")],
                                  [f"throw-channels-{arm}-{seed}"], [plan], [spec])
    raw = require_raw(base)
    assigned = tuple(int(unit["id"]) for unit in raw["allies"])
    initial_health = team_health(raw["enemies"])
    hashes = [base.state_hashes[0]]
    action_rows = []
    counts = {"learnerThrows": 0, "teacherThrowAgreements": 0, "learnerThrowsWhileTeacherDoesNotThrow": 0,
              "directionReplacements": 0, "powerReplacements": 0, "rejectedActions": 0, "totalActions": 0}
    first_contact = first_hit = None
    final = None
    for decision in range(spec.horizon):
        raw = require_raw(base)
        recommendation = recommend_shots(raw, base.max_team_units)
        teacher = base.plan_teacher_tensor_actions()
        counts["teacherThrowAgreements"] += validate_teacher_agreement(teacher, recommendation)
        with torch.no_grad():
            action, _, _ = model.act(tensor_dict(observation), deterministic=True)
        learner = numpy_actions(action)
        throws = learner["action_type"] == 2
        counts["learnerThrows"] += int(throws.sum())
        counts["learnerThrowsWhileTeacherDoesNotThrow"] += int((throws & (teacher["action_type"] != 2)).sum())
        executed = compose_action(arm, learner, teacher, recommendation)
        for channel, arms in (("direction", {"direction", "direction-power"}), ("power", {"power", "direction-power"})):
            if arm in arms:
                counts[f"{channel}Replacements"] += int(throws.sum())
        action_rows.append({name: value.tolist() for name, value in executed.items()})
        observation, _, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]
        hashes.append(base.state_hashes[0])
        results = info.get("actionResults", [])
        counts["rejectedActions"] += sum(result.get("accepted") is False for result in results)
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
        raise RuntimeError("throw intervention did not reach option boundary")
    return {"seed": seed, "arm": arm, "success": final["success"], "progress": final["progress"],
            "decisions": final["decision"], "firstContactDecision": first_contact, "firstHitDecision": first_hit,
            **counts, "stateHashes": hashes, "actionsDigest": json_digest(action_rows)}


def summarize(records: dict[str, list[dict[str, Any]]], *, bootstrap_seed: int = 740001, samples: int = 10000) -> dict[str, Any]:
    seeds = [row["seed"] for row in records["learner"]]
    if len(seeds) != len(set(seeds)) or any([row["seed"] for row in rows] != seeds for rows in records.values()):
        raise ValueError("throw-channel seeds must be unique and paired")
    result = {}
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(0, len(seeds), size=(samples, len(seeds)))
    for arm, rows in records.items():
        count = len(rows)
        total = sum(row["totalActions"] for row in rows)
        item = {"episodes": count, "successRate": float(np.mean([row["success"] for row in rows])),
                "hitRate": float(np.mean([row["firstHitDecision"] is not None for row in rows])),
                "contactRate": float(np.mean([row["firstContactDecision"] is not None for row in rows])),
                "meanProgress": float(np.mean([row["progress"] for row in rows])),
                "rejectedActionRate": sum(row["rejectedActions"] for row in rows) / total if total else 1.0}
        for key in ("learnerThrows", "teacherThrowAgreements", "learnerThrowsWhileTeacherDoesNotThrow", "directionReplacements", "powerReplacements"):
            item[key] = sum(row[key] for row in rows)
        item["pairedDifferences"] = {}
        for name, field in (("success", "success"), ("hit", "firstHitDecision"), ("progress", "progress")):
            def value(row):
                return float(row[field] is not None) if name == "hit" else float(row[field])
            delta = np.asarray([value(row) - value(base) for row, base in zip(rows, records["learner"], strict=True)])
            bounds = np.quantile(delta[indices].mean(1), [.025, .975]).tolist()
            item["pairedDifferences"][name] = {"mean": float(delta.mean()), "bootstrap95": bounds}
        result[arm] = item
    return result


def run_matrix(checkpoint: str | Path, *, output: str | Path, config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite throw intervention {destination}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    if metadata["checkpointDigest"] != config["checkpointDigest"]:
        raise ValueError("throw-channel checkpoint digest mismatch")
    protocol = load_option_protocol()
    if metadata["curriculumDigest"] != json_digest(protocol) or config["seeds"] != [protocol["seeds"]["development"][0], protocol["seeds"]["development"][0] + 39]:
        raise ValueError("throw-channel protocol or seed partition mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    before = semantic_state_digest(model.state_dict())
    seeds = list(range(config["seeds"][0], config["seeds"][1] + 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        records = {}
        with SnowGymBatchClient() as client:
            capabilities = require_capabilities(client)
            if any(capabilities[key] != config[key] for key in ("simulationVersion", "stateHashVersion")):
                raise ValueError("throw-channel simulator provenance mismatch")
            for arm in ARMS:
                records[arm] = []
                for seed in seeds:
                    records[arm].append(evaluate_episode(model, seed=seed, arm=arm, client=client))
                write_json(root / f"{arm}.json", records[arm])
                print(json.dumps({"arm": arm, "successes": sum(row["success"] for row in records[arm]),
                                  "hits": sum(row["firstHitDecision"] is not None for row in records[arm])}), flush=True)
        if semantic_state_digest(model.state_dict()) != before:
            raise RuntimeError("no-training intervention changed model state")
        report = {"format": "snowgym.engage-throw-channels-report.v0", "trainingUpdates": 0,
                  "modelUnchanged": True, "qualificationEligible": False,
                  "summary": summarize(records, bootstrap_seed=config["bootstrapSeed"], samples=config["bootstrapSamples"]),
                  "interpretation": "Direction includes teacher-style enemy selection; intervals are paired development diagnostics, not qualification or multiplicity-adjusted inference."}
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        source_files = {str(path.relative_to(repository)): file_digest(path) for path in sorted(module_root.rglob("*.py"))}
        teacher_source = repository / "snowgym/orchestration/execution/ReactiveUnitPolicy.ts"
        source_files[str(teacher_source.relative_to(repository))] = file_digest(teacher_source)
        manifest = {"format": "snowgym.engage-throw-channels-run.v0", "gitCommit": resolve_git_commit(),
                    "config": config, "configDigest": json_digest(config), "checkpoint": metadata,
                    "protocolDigest": json_digest(protocol), "seeds": seeds, "seedPartition": "development",
                    "capabilities": capabilities, "sourceFiles": source_files,
                    "artifacts": {path.name: file_digest(path) for path in sorted(root.iterdir())}}
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
