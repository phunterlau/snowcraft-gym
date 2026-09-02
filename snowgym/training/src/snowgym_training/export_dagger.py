"""Export teacher labels on states visited by a learned blue policy."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.client import SnowGymClient, SnowGymHttpClient
from snowgym_client.encoding import decode_action, encode_action
from snowgym_client.env import SnowGymEnv

from .checkpoint import load_checkpoint
from .export_scripted import (
    assert_action_round_trip,
    clone_tensors,
    required_string,
    transition_tensors,
)
from .policy import TorchPolicy
from .trajectory import TrajectoryWriter, audit_dataset, json_digest, load_export_spec


def export_dagger_dataset(
    *,
    output: str | Path,
    checkpoint: str | Path,
    split: str,
    spec_path: str | Path,
    server_url: str = "http://127.0.0.1:8787",
    max_decisions: int | None = None,
    client: SnowGymClient | None = None,
) -> dict[str, Any]:
    spec_source = Path(spec_path)
    spec = load_export_spec(spec_source)
    if split not in spec["splits"]:
        raise ValueError(f"unknown split {split!r}")
    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    metadata, _ = load_checkpoint(checkpoint)
    policy = TorchPolicy(checkpoint)
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
        for episode_index, episode in enumerate(spec["splits"][split]):
            observation, info = environment.reset(
                seed=int(episode["seed"]), options={"scenario": episode["scenario"]}
            )
            if versions is None:
                versions = {
                    key: info.get(key)
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
            while not (terminated or truncated):
                if max_decisions is not None and decisions >= max_decisions:
                    break
                raw = environment.raw_observation
                if raw is None:
                    raise ValueError("SnowGym environment lost its raw observation")
                raw_before = copy.deepcopy(raw)
                observation_before = clone_tensors(observation)
                pre_hash = required_string(info, "stateHash")
                semantic_label = environment.teacher_action()
                teacher_action = decode_action(semantic_label, raw_before, capacity)
                assert_action_round_trip(
                    semantic_label, encode_action(teacher_action, raw_before, capacity)
                )
                learner_action = policy.act(observation)
                observation, reward, terminated, truncated, info = environment.step(
                    learner_action
                )
                rejected = sum(
                    isinstance(item, dict) and item.get("accepted") is False
                    for item in info.get("actionResults", [])
                )
                if rejected:
                    raise ValueError("learned rollout produced a rejected action")
                writer.add(
                    transition_tensors(
                        observation_before,
                        teacher_action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        seed=int(episode["seed"]),
                        episode_index=episode_index,
                        pre_hash=pre_hash,
                        post_hash=required_string(info, "stateHash"),
                        next_tick=int(observation["tick"][0]),
                        accepted=np.ones(capacity, dtype=np.bool_),
                        reasons=np.zeros(capacity, dtype=np.int8),
                    )
                )
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
                    "rolloutWinner": info.get("winner"),
                    "finalStateHash": info.get("stateHash"),
                }
            )
    finally:
        environment.close()

    manifest = writer.finish(
        {
            "name": f"{spec['name']}-dagger",
            "teacher": "teacher-action.v0",
            "rolloutPolicy": "learned-checkpoint",
            "rolloutCheckpointDigest": metadata["checkpointDigest"],
            "rolloutCheckpointStateDigest": metadata["stateDigest"],
            "split": split,
            "splitSeeds": {
                name: [int(episode["seed"]) for episode in episodes]
                for name, episodes in spec["splits"].items()
            },
            "sourceSpecDigest": json_digest(spec),
            "maxTeamUnits": capacity,
            "versions": versions,
            "episodes": episode_records,
        }
    )
    audit_dataset(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export teacher labels on learned-policy SnowGym states"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = export_dagger_dataset(
            output=args.output,
            checkpoint=args.checkpoint,
            split=args.split,
            spec_path=args.spec,
            server_url=args.server,
            max_decisions=args.max_decisions,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "split": result["split"],
        "transitions": result["transitions"],
        "datasetDigest": result["datasetDigest"],
        "rolloutCheckpointDigest": result["rolloutCheckpointDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
