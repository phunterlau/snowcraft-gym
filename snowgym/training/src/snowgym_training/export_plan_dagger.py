"""Export plan-aware teacher labels on states visited by a learned executor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_client.encoding import decode_action, encode_action

from .checkpoint import load_checkpoint
from .export_scripted import (
    assert_action_round_trip,
    clone_tensors,
    required_string,
    transition_tensors,
)
from .policy import TorchPolicy
from .trajectory import TrajectoryWriter, audit_dataset, json_digest

PLAN_DAGGER_SPEC_FORMAT = "snowgym.plan-dagger-export.v0"


def load_plan_dagger_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan DAgger spec {source}: {error}") from error
    required = {"format", "name", "teacher", "maxTeamUnits", "shardSize", "plans", "splits"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("plan DAgger spec fields are invalid")
    if value["format"] != PLAN_DAGGER_SPEC_FORMAT:
        raise ValueError(f"plan DAgger spec format must be {PLAN_DAGGER_SPEC_FORMAT}")
    if value["teacher"] != "plan-teacher-action.v0":
        raise ValueError("plan DAgger teacher must be plan-teacher-action.v0")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("plan DAgger spec name is invalid")
    for field, maximum in (("maxTeamUnits", 10), ("shardSize", None)):
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"plan DAgger {field} must be positive")
        if maximum is not None and item > maximum:
            raise ValueError(f"plan DAgger {field} exceeds {maximum}")
    plans = value["plans"]
    if not isinstance(plans, dict) or not plans or any(
        not isinstance(name, str) or not name or not isinstance(plan, dict)
        for name, plan in plans.items()
    ):
        raise ValueError("plan DAgger plan catalog is invalid")
    splits = value["splits"]
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "evaluation"}:
        raise ValueError("plan DAgger splits are invalid")
    seen: set[int] = set()
    for split, episodes in splits.items():
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"plan DAgger split {split} is empty")
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict) or set(episode) != {"seed", "scenario", "plan"}:
                raise ValueError(f"plan DAgger {split}[{index}] fields are invalid")
            seed = episode["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed in seen:
                raise ValueError(f"plan DAgger {split}[{index}] seed is invalid or duplicate")
            seen.add(seed)
            if not isinstance(episode["scenario"], dict) or episode["plan"] not in plans:
                raise ValueError(f"plan DAgger {split}[{index}] scenario/plan is invalid")
    return value


def export_plan_dagger_dataset(
    *,
    output: str | Path,
    checkpoint: str | Path,
    split: str,
    spec_path: str | Path,
    max_decisions: int | None = None,
    client: SnowGymBatchClient | None = None,
) -> dict[str, Any]:
    spec = load_plan_dagger_spec(spec_path)
    if split not in spec["splits"]:
        raise ValueError(f"unknown split {split!r}")
    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    metadata, _ = load_checkpoint(checkpoint)
    policy = TorchPolicy(checkpoint)
    if not policy.model.plan_conditioned:
        raise ValueError("plan DAgger rollout requires a plan-conditioned checkpoint")
    capacity = int(spec["maxTeamUnits"])
    writer = TrajectoryWriter(output, shard_size=int(spec["shardSize"]))
    batch_client = client or SnowGymBatchClient()
    owns_client = client is None
    environment = SnowGymBatchEnv(1, max_team_units=capacity, client=batch_client)
    episode_records: list[dict[str, Any]] = []
    versions: dict[str, Any] | None = None
    try:
        for episode_index, episode in enumerate(spec["splits"][split]):
            plan_name = episode["plan"]
            plan_id = f"{plan_name}-{episode['seed']}"
            plan = spec["plans"][plan_name]
            batch_observation, infos = environment.reset(
                [int(episode["seed"])], [episode["scenario"]]
            )
            environment.activate_plans([plan_id], [plan])
            info = infos[0]
            if versions is None:
                versions = {
                    key: info.get(key)
                    for key in (
                        "apiVersion", "simulationVersion", "stateHashVersion", "upstreamBaseCommit"
                    )
                }
            decisions = 0
            start_transition = writer.transition_count
            terminated = truncated = False
            while not (terminated or truncated):
                if max_decisions is not None and decisions >= max_decisions:
                    break
                raw = environment.raw_observations[0]
                if raw is None:
                    raise ValueError("SnowGym batch environment lost its raw observation")
                raw_before = copy.deepcopy(raw)
                pre_hash = environment.state_hashes[0]
                if pre_hash is None:
                    raise ValueError("SnowGym batch environment lost its state hash")
                plan_tensors, plan_metadata = environment.plan_observations()
                if required_string(plan_metadata[0], "stateHash") != pre_hash:
                    raise ValueError("plan tensor does not match learner state")
                semantic_label = environment.plan_teacher_actions()[0]
                teacher_action = decode_action(semantic_label, raw_before, capacity)
                assert_action_round_trip(
                    semantic_label, encode_action(teacher_action, raw_before, capacity)
                )
                observation_before = {
                    **clone_tensors({name: values[0] for name, values in batch_observation.items()}),
                    **clone_tensors({name: values[0] for name, values in plan_tensors.items()}),
                }
                learner_action = policy.act(observation_before)
                batch_observation, rewards, terms, truncs, infos = environment.step(
                    {name: value[None, ...] for name, value in learner_action.items()}
                )
                info = infos[0]
                rejected = sum(
                    item.get("accepted") is False for item in info.get("actionResults", [])
                    if isinstance(item, dict)
                )
                if rejected:
                    raise ValueError("learned plan rollout produced a rejected action")
                writer.add(
                    transition_tensors(
                        observation_before,
                        teacher_action,
                        reward=float(rewards[0]),
                        terminated=bool(terms[0]),
                        truncated=bool(truncs[0]),
                        seed=int(episode["seed"]),
                        episode_index=episode_index,
                        pre_hash=pre_hash,
                        post_hash=required_string(info, "stateHash"),
                        next_tick=int(batch_observation["tick"][0, 0]),
                        accepted=np.ones(capacity, dtype=np.bool_),
                        reasons=np.zeros(capacity, dtype=np.int8),
                    )
                )
                terminated, truncated = bool(terms[0]), bool(truncs[0])
                decisions += 1
            episode_records.append(
                {
                    "index": episode_index,
                    "seed": episode["seed"],
                    "scenario": episode["scenario"],
                    "planName": plan_name,
                    "planId": plan_id,
                    "plan": plan,
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
        if owns_client:
            batch_client.close()

    manifest = writer.finish(
        {
            "name": f"{spec['name']}-plan-dagger",
            "teacher": "plan-teacher-action.v0",
            "rolloutPolicy": "plan-conditioned-checkpoint",
            "rolloutCheckpointDigest": metadata["checkpointDigest"],
            "rolloutCheckpointStateDigest": metadata["stateDigest"],
            "planConditioned": True,
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
    parser = argparse.ArgumentParser(description="Export plan-aware DAgger trajectories")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = export_plan_dagger_dataset(
            output=args.output,
            checkpoint=args.checkpoint,
            split=args.split,
            spec_path=args.spec,
            max_decisions=args.max_decisions,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "transitions": result["transitions"],
        "datasetDigest": result["datasetDigest"],
        "rolloutCheckpointDigest": result["rolloutCheckpointDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
