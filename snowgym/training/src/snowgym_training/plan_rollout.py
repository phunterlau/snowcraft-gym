"""Audit and convert TypeScript plan-caused rollouts into training shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.encoding import decode_action, encode_observation
from snowgym_client.state_hash import hash_observation

from .export_scripted import action_results, transition_tensors
from .plan_data import (
    PLAN_FEATURE_VECTOR_SIZE,
    PLAN_GROUP_SLOTS,
    javascript_json_digest,
)
from .trajectory import TrajectoryWriter, audit_dataset

PLAN_ROLLOUT_DATASET_FORMAT = "snowgym.plan-rollout-dataset.v0"
SYNTHETIC_PLAN_CURRICULUM_FORMAT = "snowgym.synthetic-plan-curriculum.v0"


def load_plan_rollouts(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load plan rollout dataset {source}: {error}") from error
    return audit_plan_rollouts(value)


def audit_plan_rollouts(value: Any) -> dict[str, Any]:
    required = {
        "format", "apiVersion", "simulationVersion", "stateHashVersion",
        "upstreamBaseCommit", "scenario", "environmentSeed", "decisionHz",
        "configuration", "sourceStateHash", "maxDecisions", "curriculum",
        "episodes", "datasetDigest",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("plan rollout dataset fields are invalid")
    if value["format"] != PLAN_ROLLOUT_DATASET_FORMAT:
        raise ValueError(f"plan rollout format must be {PLAN_ROLLOUT_DATASET_FORMAT}")
    curriculum = value["curriculum"]
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("format") != SYNTHETIC_PLAN_CURRICULUM_FORMAT
        or not isinstance(curriculum.get("samples"), list)
        or curriculum.get("sampleCount") != len(curriculum["samples"])
    ):
        raise ValueError("plan rollout curriculum is invalid")
    episodes = value["episodes"]
    if not isinstance(episodes, list) or len(episodes) != len(curriculum["samples"]):
        raise ValueError("plan rollout sample counts do not match")
    if not _positive_integer(value["maxDecisions"]):
        raise ValueError("plan rollout maxDecisions must be a positive integer")
    for index, (sample, episode) in enumerate(zip(curriculum["samples"], episodes, strict=True)):
        _audit_episode(value, sample, episode, index)
    body = {name: item for name, item in value.items() if name != "datasetDigest"}
    if value["datasetDigest"] != javascript_json_digest(body):
        raise ValueError("plan rollout dataset digest mismatch")
    return value


def convert_plan_rollouts(
    *, source: str | Path, output: str | Path, max_team_units: int, shard_size: int = 1024
) -> dict[str, Any]:
    value = load_plan_rollouts(source)
    configuration = value["configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("plan rollout configuration is invalid")
    roster = max(configuration.get("blueUnits", 0), configuration.get("redUnits", 0))
    if not _positive_integer(max_team_units) or max_team_units > 10 or max_team_units < roster:
        raise ValueError("max_team_units must cover the rollout roster and be in [1, 10]")
    writer = TrajectoryWriter(output, shard_size=shard_size)
    episode_records: list[dict[str, Any]] = []
    environment_seed = int(value["environmentSeed"])
    for episode_index, episode in enumerate(value["episodes"]):
        start = writer.transition_count
        for transition in episode["transitions"]:
            raw = transition["observation"]
            observation = encode_observation(
                raw, max_team_units, include_unit_masks=True
            )
            observation["plan_groups"] = np.asarray(
                transition["planGroups"], dtype=np.float32
            ).reshape(PLAN_GROUP_SLOTS, PLAN_FEATURE_VECTOR_SIZE)
            observation["plan_group_mask"] = np.asarray(
                transition["planGroupMask"], dtype=np.int8
            )
            action = decode_action(transition["action"], raw, max_team_units)
            accepted, reasons = action_results(
                transition["actionResults"], raw, max_team_units
            )
            tensors = transition_tensors(
                observation,
                action,
                reward=float(transition["reward"]),
                terminated=bool(transition["terminated"]),
                truncated=bool(transition["truncated"]),
                seed=environment_seed,
                episode_index=episode_index,
                pre_hash=transition["preStateHash"],
                post_hash=transition["postStateHash"],
                next_tick=int(transition["nextTick"]),
                accepted=accepted,
                reasons=reasons,
            )
            tensors["plan_source_seed"] = np.asarray(
                episode["sourceSeed"], dtype=np.int64
            )
            writer.add(tensors)
        outcome = episode["outcome"]
        episode_records.append(
            {
                "index": episode_index,
                "seed": environment_seed,
                "planSourceSeed": episode["sourceSeed"],
                "planId": episode["planId"],
                "startTransition": start,
                "transitions": len(episode["transitions"]),
                **outcome,
            }
        )
    versions = {
        name: value[name]
        for name in ("apiVersion", "simulationVersion", "stateHashVersion", "upstreamBaseCommit")
    }
    manifest = writer.finish(
        {
            "name": f"plan-rollouts-{value['scenario']}",
            "teacher": "production-plan-aware-controller.v0",
            "split": "train",
            "splitSeeds": {"train": [environment_seed], "validation": [], "evaluation": []},
            "sourcePlanRolloutDigest": value["datasetDigest"],
            "maxTeamUnits": max_team_units,
            "planConditioned": True,
            "versions": versions,
            "episodes": episode_records,
        }
    )
    audit_dataset(output)
    return manifest


def _audit_episode(dataset: dict[str, Any], sample: Any, episode: Any, index: int) -> None:
    if not isinstance(sample, dict) or not isinstance(episode, dict):
        raise ValueError(f"plan rollout episode {index} is invalid")
    for name in ("sourceSeed", "planId", "plan", "assignments"):
        if episode.get(name) != sample.get(name):
            raise ValueError(f"plan rollout episode {index} is misaligned")
    transitions = episode.get("transitions")
    outcome = episode.get("outcome")
    if not isinstance(transitions, list) or not isinstance(outcome, dict):
        raise ValueError(f"plan rollout episode {index} transitions are invalid")
    if episode.get("initialStateHash") != dataset["sourceStateHash"]:
        raise ValueError(f"plan rollout episode {index} initial state hash differs")
    if outcome.get("decisions") != len(transitions) or len(transitions) > dataset["maxDecisions"]:
        raise ValueError(f"plan rollout episode {index} outcome is inconsistent")
    terminal = bool(outcome.get("terminated")) or bool(outcome.get("truncated"))
    if bool(outcome.get("decisionLimited")) == terminal:
        raise ValueError(f"plan rollout episode {index} ending is inconsistent")
    expected_hash = episode["initialStateHash"]
    previous_tick: int | None = None
    for decision, transition in enumerate(transitions):
        if not isinstance(transition, dict) or transition.get("decision") != decision:
            raise ValueError(f"plan rollout episode {index} decision indices are invalid")
        observation = transition.get("observation")
        if not isinstance(observation, dict):
            raise ValueError(f"plan rollout episode {index} observation is invalid")
        if (
            transition.get("preStateHash") != expected_hash
            or hash_observation(observation) != expected_hash
        ):
            raise ValueError(f"plan rollout episode {index} state hash is invalid")
        groups = np.asarray(transition.get("planGroups"))
        mask = np.asarray(transition.get("planGroupMask"))
        if (
            groups.shape != (PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE,)
            or groups.dtype.kind not in "iuf"
            or not np.isfinite(groups).all()
            or np.any((groups < -1) | (groups > 1))
            or mask.shape != (PLAN_GROUP_SLOTS,)
            or mask.dtype.kind not in "iu"
            or not np.isin(mask, (0, 1)).all()
        ):
            raise ValueError(f"plan rollout episode {index} plan tensor is invalid")
        results = transition.get("actionResults")
        if (
            not isinstance(results, list)
            or not results
            or any(not isinstance(item, dict) or item.get("accepted") is not True for item in results)
        ):
            raise ValueError(f"plan rollout episode {index} action result is invalid")
        tick = observation.get("tick")
        next_tick = transition.get("nextTick")
        if not isinstance(tick, int) or not isinstance(next_tick, int) or next_tick <= tick:
            raise ValueError(f"plan rollout episode {index} tick is invalid")
        if previous_tick is not None and tick != previous_tick:
            raise ValueError(f"plan rollout episode {index} tick continuity is invalid")
        previous_tick = next_tick
        expected_hash = transition.get("postStateHash")
    if expected_hash != outcome.get("finalStateHash"):
        raise ValueError(f"plan rollout episode {index} final state hash differs")


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audited SnowGym plan rollouts")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-team-units", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = convert_plan_rollouts(
            source=args.source,
            output=args.output,
            max_team_units=args.max_team_units,
            shard_size=args.shard_size,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "transitions": result["transitions"],
        "datasetDigest": result["datasetDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
