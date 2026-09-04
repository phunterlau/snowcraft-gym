"""Export immutable Engage teacher, stochastic, and deterministic state datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from ..executor import model_config
from ..ppo import HybridActorCritic
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import numpy_actions, tensor_dict
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from .environment import FixedPlanOptionBatchEnv
from .interventions import (
    assigned_units,
    centroid,
    contact_distance,
    grounded_goal_target,
    objective_distance,
    require_raw,
    require_capabilities,
)
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol

FORMAT = "snowgym.engage.diagnostics.v0"
KINDS = ("teacher", "stochastic-learner", "deterministic-learner")
FILENAMES = {
    "teacher": "teacher_states.npz",
    "stochastic-learner": "stochastic_learner_states.npz",
    "deterministic-learner": "deterministic_learner_states.npz",
}


def export_engage_diagnostics(
    checkpoint: str | Path,
    *,
    output: str | Path,
    seed_count: int = 40,
) -> dict[str, Any]:
    if not 1 <= seed_count <= 40:
        raise ValueError("Engage diagnostic seed_count must be in [1,40]")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic directory {destination}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    config = model_config(metadata["architecture"])
    ppo_config = metadata["ppoConfig"]
    model = HybridActorCritic(
        config,
        initial_target_log_std=float(ppo_config["initial_target_log_std"]),
        initial_power_log_std=float(ppo_config["initial_power_log_std"]),
    ).eval()
    model.load_state_dict(state["model"])
    protocol = load_option_protocol()
    first_seed = int(protocol["seeds"]["development"][0])
    seeds = list(range(first_seed, first_seed + seed_count))
    destination.mkdir(parents=True)
    datasets: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, Any] = {}
    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        for kind in KINDS:
            if kind == "stochastic-learner":
                torch.set_rng_state(state["torchRngState"].detach().clone())
            dataset = collect_dataset(model, kind=kind, seeds=seeds, client=client)
            if kind == "teacher" and not bool(dataset["episode_success"].all()):
                raise RuntimeError("D1 requires successful production-teacher episodes")
            datasets[kind] = dataset
            summaries[kind] = dataset_metrics(dataset)
            np.savez_compressed(destination / FILENAMES[kind], **dataset)
    coverage = coverage_report(
        datasets["stochastic-learner"]["diagnostic_vector"],
        datasets["deterministic-learner"]["diagnostic_vector"],
        datasets["teacher"]["diagnostic_vector"],
    )
    report = {
        "format": "snowgym.engage.distribution-report.v0",
        "checkpointDigest": metadata["checkpointDigest"],
        "summaries": summaries,
        "coverage": coverage,
    }
    report["reportDigest"] = json_digest(report)
    report_path = destination / "distribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_hashes = {
        name: file_digest(destination / name)
        for name in [*FILENAMES.values(), "distribution_report.json"]
    }
    manifest = {
        "format": FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "checkpointStateDigest": metadata["stateDigest"],
        "checkpointGitCommit": metadata["gitCommit"],
        "implementationGitCommit": resolve_git_commit(),
        "simulationVersion": capabilities["simulationVersion"],
        "stateHashVersion": capabilities["stateHashVersion"],
        "upstreamBaseCommit": capabilities["upstreamBaseCommit"],
        "protocolDigest": json_digest(protocol),
        "seedPartition": "development",
        "seeds": seeds,
        "stochasticRngSource": "checkpoint torchRngState",
        "artifacts": artifact_hashes,
    }
    manifest["manifestDigest"] = json_digest(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest, "report": report}


def collect_dataset(
    model: HybridActorCritic,
    *,
    kind: str,
    seeds: list[int],
    client: SnowGymBatchClient,
) -> dict[str, np.ndarray]:
    if kind not in KINDS:
        raise ValueError(f"unknown Engage diagnostic kind {kind!r}")
    rows: list[dict[str, Any]] = []
    episode_success: list[bool] = []
    for episode_index, seed in enumerate(seeds):
        plan, spec = teacher_option_plan("engage")
        scenario = teacher_option_scenario("engage")
        base = SnowGymBatchEnv(1, client=client, observation_version=3)
        wrapped = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
        observation, _ = wrapped.reset(
            [seed], [scenario], [f"engage-diagnostic-{kind}-{seed}"], [plan], [spec]
        )
        episode_rows: list[dict[str, Any]] = []
        success = False
        contacted = False
        for decision in range(spec.horizon):
            tensors = tensor_dict(observation)
            teacher = base.plan_teacher_tensor_actions()
            with torch.no_grad():
                prediction = model(tensors)
                learner, _, _ = model.act(
                    tensors, deterministic=kind != "stochastic-learner"
                )
            learner_action = numpy_actions(learner)
            action = teacher if kind == "teacher" else learner_action
            raw = require_raw(base)
            plan_body = base.plan_observations()[1][0]
            goal, anchor, assigned = grounded_goal_target(
                raw, plan_body, capacity=base.max_team_units
            )
            nearest = contact_distance(raw, assigned)
            contacted = contacted or nearest <= 9.0
            row: dict[str, Any] = {
                "seed": seed,
                "episode": episode_index,
                "decision": decision,
                "teacher_action_type": teacher["action_type"][0].copy(),
                "teacher_target": teacher["target"][0].copy(),
                "teacher_power": teacher["power"][0].copy(),
                "learner_action_type": learner_action["action_type"][0].copy(),
                "learner_target": learner_action["target"][0].copy(),
                "learner_power": learner_action["power"][0].copy(),
                "action_probabilities": torch.softmax(
                    prediction["action_logits"], dim=-1
                )[0].cpu().numpy(),
                "move_target": torch.tanh(
                    prediction["target_raw_by_action"][0, :, ACTION_MOVE]
                ).cpu().numpy(),
                "move_target_raw": prediction[
                    "target_raw_by_action"
                ][0, :, ACTION_MOVE].cpu().numpy(),
                "throw_target": torch.tanh(
                    prediction["target_raw_by_action"][0, :, ACTION_THROW]
                ).cpu().numpy(),
                "throw_target_raw": prediction[
                    "target_raw_by_action"
                ][0, :, ACTION_THROW].cpu().numpy(),
                "base_move_target": torch.tanh(
                    prediction["base_target_raw_by_action"][0, :, ACTION_MOVE]
                ).cpu().numpy(),
                "base_move_target_raw": prediction[
                    "base_target_raw_by_action"
                ][0, :, ACTION_MOVE].cpu().numpy(),
                "base_throw_target": torch.tanh(
                    prediction["base_target_raw_by_action"][0, :, ACTION_THROW]
                ).cpu().numpy(),
                "base_throw_target_raw": prediction[
                    "base_target_raw_by_action"
                ][0, :, ACTION_THROW].cpu().numpy(),
                "plan_residual": prediction.get(
                    "plan_ppo_residual",
                    torch.zeros((1, base.max_team_units, 9)),
                )[0].cpu().numpy(),
                "power_prediction": prediction["power"][0].cpu().numpy(),
                "value_prediction": float(prediction["value"][0]),
                "goal_target": goal,
                "objective_distance": objective_distance(raw, assigned, anchor),
                "signed_x": centroid(assigned_units(raw, assigned))[0],
                "inside_preferred_range": nearest <= 9.0,
                "post_contact": contacted,
                "diagnostic_vector": diagnostic_vector(
                    raw, assigned, anchor, prediction, plan_body
                ),
            }
            for name, value in observation.items():
                row[f"observation__{name}"] = value[0].copy()
            observation, reward, terminated, truncated, infos = wrapped.step(action)
            row["reward"] = float(reward[0])
            episode_rows.append(row)
            if bool(terminated[0] or truncated[0]):
                success = bool(infos[0]["option"]["success"])
                break
        running = 0.0
        for row in reversed(episode_rows):
            running = float(row["reward"]) + 0.9976921765 * running
            row["return_to_go"] = running
            row["episode_success"] = success
        rows.extend(episode_rows)
        episode_success.append(success)
    keys = sorted(rows[0])
    return {
        **{key: np.asarray([row[key] for row in rows]) for key in keys},
        "episode_success": np.asarray(episode_success, dtype=np.bool_),
    }


def diagnostic_vector(
    raw: dict[str, Any],
    assigned: tuple[int, ...],
    anchor: tuple[float, float],
    prediction: dict[str, torch.Tensor],
    plan_body: dict[str, Any],
) -> np.ndarray:
    own = assigned_units(raw, assigned)
    own_center = centroid(own)
    width = float(raw["arena"]["width"])
    height = float(raw["arena"]["height"])
    spread = 0.0 if not own else float(np.mean([
        math.dist((float(unit["x"]), float(unit["y"])), own_center) for unit in own
    ])) / math.hypot(width, height)
    alive_fraction = len(own) / max(len(assigned), 1)
    move = torch.tanh(prediction["target_raw_by_action"][0, :, ACTION_MOVE]).cpu().numpy()
    role_state = plan_body["planRoleState"]
    return np.asarray(
        [
            own_center[0] / (width / 2),
            own_center[1] / (height / 2),
            anchor[0] / (width / 2),
            anchor[1] / (height / 2),
            (anchor[0] - own_center[0]) / width,
            (anchor[1] - own_center[1]) / height,
            spread,
            alive_fraction,
            float(move[:, 0].mean()),
            float(move[:, 1].mean()),
            float(1 - role_state[11]),
        ],
        dtype=np.float32,
    )


def dataset_metrics(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    alive = (
        dataset["observation__ally_mask"].astype(bool)
        & (dataset["observation__allies"][..., 1] > 0.5)
    )
    teacher_type = dataset["teacher_action_type"]
    agreement = masked_mean(
        (dataset["learner_action_type"] == teacher_type).astype(np.float64), alive
    )
    move = alive & (teacher_type == ACTION_MOVE)
    throw = alive & (teacher_type == ACTION_THROW)
    move_error = (dataset["move_target"] - dataset["teacher_target"]) ** 2
    throw_error = (dataset["throw_target"] - dataset["teacher_target"]) ** 2
    power_error = (dataset["power_prediction"] - dataset["teacher_power"]) ** 2
    value_error = (dataset["value_prediction"] - dataset["return_to_go"]) ** 2
    positions = dataset["observation__allies"][..., 2:4]
    predicted_direction = dataset["move_target"] - positions
    teacher_direction = dataset["teacher_target"] - positions
    direction_cosine = vector_cosine(predicted_direction, teacher_direction)
    return {
        "states": int(dataset["decision"].shape[0]),
        "episodes": int(dataset["episode_success"].shape[0]),
        "successRate": float(dataset["episode_success"].mean()),
        "actionAgreement": agreement,
        "moveTargetMseNormalized": masked_vector_mean(move_error, move),
        "moveTargetRmseWorldUnits": world_rmse(move_error, move),
        "moveTargetDirectionCosine": masked_mean(direction_cosine, move),
        "throwTargetMseNormalized": masked_vector_mean(throw_error, throw),
        "powerMse": masked_mean(power_error, throw),
        "objectiveRelativeMoveMse": masked_vector_mean(
            (dataset["move_target"] - dataset["goal_target"]) ** 2, alive
        ),
        "baseMoveOutputMean": float(dataset["base_move_target"].mean()),
        "planResidualMeanAbsolute": float(np.abs(dataset["plan_residual"]).mean()),
        "criticValueRmse": float(np.sqrt(value_error.mean())),
        "bins": binned_metrics(dataset, alive, move),
    }


def binned_metrics(
    dataset: dict[str, np.ndarray], alive: np.ndarray, move: np.ndarray
) -> dict[str, Any]:
    distance = dataset["objective_distance"]
    definitions = {
        "time:early": dataset["decision"] < 67,
        "time:middle": (dataset["decision"] >= 67) & (dataset["decision"] < 134),
        "time:late": dataset["decision"] >= 134,
        "distance:0-9": distance <= 9,
        "distance:9-20": (distance > 9) & (distance <= 20),
        "distance:20-40": (distance > 20) & (distance <= 40),
        "distance:40+": distance > 40,
        "x:negative": dataset["signed_x"] < 0,
        "x:nonnegative": dataset["signed_x"] >= 0,
        "range:inside": dataset["inside_preferred_range"].astype(bool),
        "range:outside": ~dataset["inside_preferred_range"].astype(bool),
        "contact:pre": ~dataset["post_contact"].astype(bool),
        "contact:post": dataset["post_contact"].astype(bool),
    }
    result: dict[str, Any] = {}
    for name, selected in definitions.items():
        if not bool(selected.any()):
            continue
        selected_alive = alive & selected[:, None]
        selected_move = move & selected[:, None]
        result[name] = {
            "states": int(selected.sum()),
            "actionAgreement": masked_mean(
                (dataset["learner_action_type"] == dataset["teacher_action_type"]).astype(float),
                selected_alive,
            ),
            "moveTargetMse": masked_vector_mean(
                (dataset["move_target"] - dataset["teacher_target"]) ** 2,
                selected_move,
            ),
        }
    return result


def coverage_report(
    stochastic: np.ndarray, deterministic: np.ndarray, teacher: np.ndarray
) -> dict[str, Any]:
    mean = stochastic.mean(axis=0)
    scale = stochastic.std(axis=0)
    scale[scale < 1e-8] = 1.0
    d2 = (stochastic - mean) / scale
    d3 = (deterministic - mean) / scale
    d1 = (teacher - mean) / scale
    d2_nearest, _ = nearest_distances(d2, d2, exclude_self=True, k=1)
    d3_nearest, d3_five = nearest_distances(d3, d2, exclude_self=False, k=5)
    d1_nearest, d1_five = nearest_distances(d1, d2, exclude_self=False, k=5)
    threshold = float(np.quantile(d2_nearest, 0.95))
    return {
        "features": [
            "ownCentroidX", "ownCentroidY", "objectiveX", "objectiveY",
            "objectiveDx", "objectiveDy", "spread", "livingFraction",
            "meanMoveTargetX", "meanMoveTargetY", "roleProgress",
        ],
        "d2LeaveOneOutNearestP95": threshold,
        "d3NearestMean": float(d3_nearest.mean()),
        "d3NearestP95": float(np.quantile(d3_nearest, 0.95)),
        "d3FiveNeighborMeanDistance": float(d3_five.mean()),
        "d3OutsideD2P95Fraction": float((d3_nearest > threshold).mean()),
        "d1NearestMean": float(d1_nearest.mean()),
        "d1NearestP95": float(np.quantile(d1_nearest, 0.95)),
        "d1FiveNeighborMeanDistance": float(d1_five.mean()),
        "d1OutsideD2P95Fraction": float((d1_nearest > threshold).mean()),
    }


def nearest_distances(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    exclude_self: bool,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.empty(len(queries), dtype=np.float64)
    kth_mean = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), 128):
        stop = min(start + 128, len(queries))
        squared = ((queries[start:stop, None, :] - references[None, :, :]) ** 2).sum(axis=-1)
        if exclude_self:
            indices = np.arange(start, stop)
            squared[np.arange(stop - start), indices] = np.inf
        count = min(k, squared.shape[1] - int(exclude_self))
        values = np.partition(squared, count - 1, axis=1)[:, :count]
        distances = np.sqrt(values)
        nearest[start:stop] = distances.min(axis=1)
        kth_mean[start:stop] = distances.mean(axis=1)
    return nearest, kth_mean


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return None if selected.size == 0 else float(selected.mean())


def masked_vector_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return None if selected.size == 0 else float(selected.mean())


def world_rmse(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    if selected.size == 0:
        return None
    scale = np.asarray([50.0, 40.0])
    return float(np.sqrt((selected * scale**2).mean()))


def vector_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-9,
    )


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    result = export_engage_diagnostics(
        args.checkpoint, output=args.output, seed_count=args.seed_count
    )
    print(json.dumps(result["report"], sort_keys=True))


if __name__ == "__main__":
    main()
