"""Export exact scripted-blue transitions from the guarded SnowGym server."""

from __future__ import annotations

import argparse
import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.client import SnowGymClient, SnowGymHttpClient
from snowgym_client.encoding import GymAction, GymObservation, decode_action, encode_action
from snowgym_client.env import SnowGymEnv

from .trajectory import TrajectoryWriter, audit_dataset, json_digest, load_export_spec

TEACHER_REASON_CODES = {
    None: 0,
    "duplicate_unit": 1,
    "invalid_value": 2,
    "missing_unit": 3,
    "wrong_team": 4,
    "unavailable": 5,
}


def default_spec_path() -> Path:
    return Path(
        str(
            files("snowgym_training").joinpath(
                "configs/teacher_1v1_v0.json"
            )
        )
    )


def export_scripted_dataset(
    *,
    output: str | Path,
    split: str,
    spec_path: str | Path | None = None,
    server_url: str = "http://127.0.0.1:8787",
    limit_episodes: int | None = None,
    max_decisions: int | None = None,
    client: SnowGymClient | None = None,
) -> dict[str, Any]:
    spec_source = Path(spec_path) if spec_path is not None else default_spec_path()
    spec = load_export_spec(spec_source)
    if split not in spec["splits"]:
        raise ValueError(f"unknown split {split!r}")
    episodes = list(spec["splits"][split])
    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError("limit_episodes must be positive")
        episodes = episodes[:limit_episodes]
    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    if not episodes:
        raise ValueError(f"split {split} selected no episodes")

    capacity = int(spec["maxTeamUnits"])
    writer = TrajectoryWriter(output, shard_size=int(spec["shardSize"]))
    environment = SnowGymEnv(
        client=client or SnowGymHttpClient(server_url),
        max_team_units=capacity,
        configurable=True,
    )
    episode_records: list[dict[str, Any]] = []
    versions: dict[str, Any] | None = None
    try:
        for episode_index, episode in enumerate(episodes):
            observation, status = environment.reset(
                seed=int(episode["seed"]),
                options={"scenario": episode["scenario"]},
            )
            if versions is None:
                versions = {
                    key: status.get(key)
                    for key in (
                        "apiVersion",
                        "simulationVersion",
                        "stateHashVersion",
                        "upstreamBaseCommit",
                    )
                }
            decisions = 0
            start_transition = writer.transition_count
            terminated = truncated = False
            current_info = status
            while not (terminated or truncated):
                if max_decisions is not None and decisions >= max_decisions:
                    break
                if environment.raw_observation is None:
                    raise ValueError("SnowGym environment lost its raw observation")
                raw_before = copy.deepcopy(environment.raw_observation)
                observation_before = clone_tensors(observation)
                pre_hash = required_string(current_info, "stateHash")
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    next_info,
                ) = environment.step_scripted()
                semantic_action = next_info.get("action")
                if not isinstance(semantic_action, dict):
                    raise ValueError("scripted step did not return info.action")
                tensor_action = decode_action(semantic_action, raw_before, capacity)
                assert_action_round_trip(
                    semantic_action,
                    encode_action(tensor_action, raw_before, capacity),
                )
                accepted, reasons = action_results(
                    next_info.get("actionResults"), raw_before, capacity
                )
                if not bool(np.all(accepted)):
                    raise ValueError("scripted teacher produced a rejected action")
                writer.add(
                    transition_tensors(
                        observation_before,
                        tensor_action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        seed=int(episode["seed"]),
                        episode_index=episode_index,
                        pre_hash=pre_hash,
                        post_hash=required_string(next_info, "stateHash"),
                        next_tick=int(next_observation["tick"][0]),
                        accepted=accepted,
                        reasons=reasons,
                    )
                )
                observation = next_observation
                current_info = next_info
                decisions += 1
            episode_records.append(
                {
                    "index": episode_index,
                    "seed": episode["seed"],
                    "scenario": episode["scenario"],
                    "startTransition": start_transition,
                    "transitions": decisions,
                    "terminated": terminated,
                    "truncated": truncated,
                    "decisionLimited": not (terminated or truncated),
                    "winner": current_info.get("winner"),
                    "finalStateHash": current_info.get("stateHash"),
                }
            )
    finally:
        environment.close()

    split_seeds = {
        name: [int(episode["seed"]) for episode in split_episodes]
        for name, split_episodes in spec["splits"].items()
    }
    manifest = writer.finish(
        {
            "name": spec["name"],
            "teacher": spec["teacher"],
            "split": split,
            "splitSeeds": split_seeds,
            "sourceSpecDigest": json_digest(spec),
            "maxTeamUnits": capacity,
            "versions": versions,
            "episodes": episode_records,
        }
    )
    audit_dataset(output)
    return manifest


def transition_tensors(
    observation: GymObservation,
    action: GymAction,
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    seed: int,
    episode_index: int,
    pre_hash: str,
    post_hash: str,
    next_tick: int,
    accepted: np.ndarray,
    reasons: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        f"observation__{name}": np.asarray(value)
        for name, value in observation.items()
    }
    result.update({f"action__{name}": np.asarray(value) for name, value in action.items()})
    result.update(
        {
            "reward": np.asarray(reward, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "truncated": np.asarray(truncated, dtype=np.bool_),
            "seed": np.asarray(seed, dtype=np.int64),
            "episode_index": np.asarray(episode_index, dtype=np.int32),
            "tick": np.asarray(int(observation["tick"][0]), dtype=np.int64),
            "next_tick": np.asarray(next_tick, dtype=np.int64),
            "pre_state_hash": np.asarray(pre_hash),
            "post_state_hash": np.asarray(post_hash),
            "teacher_accepted": accepted.astype(np.bool_),
            "teacher_reason": reasons.astype(np.int8),
        }
    )
    return result


def action_results(
    value: Any, raw_observation: dict[str, Any], capacity: int
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, list):
        raise ValueError("scripted step did not return actionResults")
    allies = raw_observation.get("allies")
    if not isinstance(allies, list):
        raise ValueError("raw observation is missing allies")
    indices = {unit["id"]: index for index, unit in enumerate(allies)}
    expected = {unit["id"] for unit in allies if unit.get("alive") is True}
    accepted = np.ones(capacity, dtype=np.bool_)
    reasons = np.zeros(capacity, dtype=np.int8)
    seen: set[int] = set()
    for raw_result in value:
        if not isinstance(raw_result, dict) or not isinstance(raw_result.get("action"), dict):
            raise ValueError("invalid scripted action result")
        unit_id = raw_result["action"].get("unitId")
        if unit_id not in indices or unit_id in seen:
            raise ValueError("action results do not match teacher units")
        seen.add(unit_id)
        index = indices[unit_id]
        is_accepted = raw_result.get("accepted")
        reason = raw_result.get("reason")
        if not isinstance(is_accepted, bool) or reason not in TEACHER_REASON_CODES:
            raise ValueError("invalid scripted action result fields")
        accepted[index] = is_accepted
        reasons[index] = TEACHER_REASON_CODES[reason]
    if seen != expected:
        raise ValueError("action results are missing teacher units")
    return accepted, reasons


def assert_action_round_trip(original: dict[str, Any], encoded: dict[str, Any]) -> None:
    original_actions = original.get("actions")
    encoded_actions = encoded.get("actions")
    if not isinstance(original_actions, list) or not isinstance(encoded_actions, list):
        raise ValueError("team action round trip is missing actions")
    if len(original_actions) != len(encoded_actions):
        raise ValueError("team action round trip changed action count")
    encoded_by_unit = {action.get("unitId"): action for action in encoded_actions}
    for action in original_actions:
        if not isinstance(action, dict):
            raise ValueError("teacher action must be an object")
        replayed = encoded_by_unit.get(action.get("unitId"))
        if replayed is None or replayed.get("type") != action.get("type"):
            raise ValueError("team action round trip changed unit/type")
        for key in ("x", "y", "power"):
            if key in action and not np.isclose(
                float(action[key]), float(replayed.get(key, np.nan)), atol=1e-5, rtol=0
            ):
                raise ValueError(f"team action round trip changed {key}")


def clone_tensors(value: GymObservation) -> GymObservation:
    return {name: np.array(tensor, copy=True) for name, tensor in value.items()}


def required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"SnowGym info is missing {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scripted SnowGym trajectories")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        manifest = export_scripted_dataset(
            output=args.output,
            split=args.split,
            spec_path=args.spec,
            server_url=args.server,
            limit_episodes=args.limit_episodes,
            max_decisions=args.max_decisions,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": manifest["format"],
        "split": manifest["split"],
        "transitions": manifest["transitions"],
        "shards": len(manifest["shards"]),
        "datasetDigest": manifest["datasetDigest"],
        "output": str(args.output.resolve()),
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"SnowGym scripted dataset: {summary['transitions']} transitions, "
            f"{summary['shards']} shards -> {summary['output']}"
        )


if __name__ == "__main__":
    main()
