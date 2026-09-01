"""Versioned SnowGym evaluation suites and reproducible benchmark runner."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import ACTION_NOOP, GymAction, GymObservation
from .parallel_env import SnowGymParallelEnv
from .research_env import SnowGymResearchParallelEnv

EVALUATION_SUITE_FORMAT = "snowgym.evaluation-suite.v0"
BENCHMARK_RESULT_FORMAT = "snowgym.benchmark-result.v0"
SUPPORTED_POLICIES = frozenset({"masked_random", "noop"})
AGENTS = ("blue", "red")


def default_suite_path() -> Path:
    return Path(
        str(files("snowgym_client").joinpath("evaluation_suites/baseline_v0.json"))
    )


def load_evaluation_suite(path: str | Path | None = None) -> dict[str, Any]:
    suite_path = Path(path) if path is not None else default_suite_path()
    try:
        value = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation suite {suite_path}: {error}") from error
    validate_evaluation_suite(value)
    return value


def validate_evaluation_suite(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("evaluation suite must be an object")
    if value.get("format") != EVALUATION_SUITE_FORMAT:
        raise ValueError(f"evaluation suite format must be {EVALUATION_SUITE_FORMAT}")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError("evaluation suite name must be a non-empty string")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("evaluation suite episodes must be a non-empty array")
    seen_ids: set[str] = set()
    for index, episode in enumerate(episodes):
        context = f"episodes[{index}]"
        if not isinstance(episode, dict):
            raise ValueError(f"{context} must be an object")
        episode_id = episode.get("id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{context}.id must be a non-empty string")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate evaluation episode id: {episode_id}")
        seen_ids.add(episode_id)
        seed = episode.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"{context}.seed must be a non-negative integer")
        if not isinstance(episode.get("scenario"), dict):
            raise ValueError(f"{context}.scenario must be an object")
        policies = episode.get("policies")
        if not isinstance(policies, dict) or set(policies) != set(AGENTS):
            raise ValueError(f"{context}.policies must contain exactly blue and red")
        for agent in AGENTS:
            if policies[agent] not in SUPPORTED_POLICIES:
                raise ValueError(
                    f"{context}.policies.{agent} must be one of "
                    f"{', '.join(sorted(SUPPORTED_POLICIES))}"
                )
        validate_profile(episode.get("profile", {}), context)


def validate_profile(value: Any, context: str = "profile") -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context}.profile must be an object")
    allowed = {
        "visibilityRadius",
        "actionDelaySteps",
        "observationDelaySteps",
        "semanticRasterSize",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"{context}.profile has unknown fields: {', '.join(sorted(unknown))}"
        )
    radius = value.get("visibilityRadius")
    if radius is not None and (
        not isinstance(radius, int | float) or isinstance(radius, bool) or radius <= 0
    ):
        raise ValueError(f"{context}.profile.visibilityRadius must be positive")
    for key in ("actionDelaySteps", "observationDelaySteps"):
        delay = value.get(key, 0)
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError(f"{context}.profile.{key} must be a non-negative integer")
    raster_size = value.get("semanticRasterSize")
    if raster_size is not None and (
        not isinstance(raster_size, int)
        or isinstance(raster_size, bool)
        or not 8 <= raster_size <= 128
    ):
        raise ValueError(f"{context}.profile.semanticRasterSize must be in [8, 128]")


def run_evaluation_suite(
    suite: dict[str, Any],
    *,
    server_url: str = "http://127.0.0.1:8787",
    repeat: int = 1,
    max_decisions: int = 1000,
    environment_factory: Callable[[], SnowGymParallelEnv] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    validate_evaluation_suite(suite)
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    results: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    benchmark_started = clock()
    for repeat_index in range(repeat):
        for episode_index, episode in enumerate(suite["episodes"]):
            base_environment = (
                environment_factory()
                if environment_factory is not None
                else SnowGymParallelEnv(server_url=server_url)
            )
            environment = apply_research_profile(base_environment, episode["profile"])
            started = clock()
            try:
                result = run_evaluation_episode(
                    environment,
                    episode,
                    episode_index=episode_index,
                    repeat_index=repeat_index,
                    max_decisions=max_decisions,
                )
            finally:
                environment.close()
            elapsed = max(0.0, clock() - started)
            results.append(result)
            timings.append(
                {
                    "episodeId": episode["id"],
                    "repeatIndex": repeat_index,
                    "elapsedSeconds": elapsed,
                    "decisionsPerSecond": (
                        result["decisions"] / elapsed if elapsed else None
                    ),
                }
            )
    total_elapsed = max(0.0, clock() - benchmark_started)
    total_decisions = sum(result["decisions"] for result in results)
    winners = {winner: 0 for winner in ("blue", "red", "draw", "none")}
    for result in results:
        winner = (
            result["winner"]
            if result["winner"] in ("blue", "red", "draw")
            else "none"
        )
        winners[winner] += 1
    return {
        "format": BENCHMARK_RESULT_FORMAT,
        "suite": {"format": suite["format"], "name": suite["name"]},
        "repeat": repeat,
        "maxDecisions": max_decisions,
        "results": results,
        "summary": {
            "episodes": len(results),
            "decisions": total_decisions,
            "winners": winners,
            "terminated": sum(bool(result["terminated"]) for result in results),
            "truncated": sum(bool(result["truncated"]) for result in results),
            "decisionLimited": sum(
                bool(result["decisionLimited"]) for result in results
            ),
        },
        "performance": {
            "transport": "http" if environment_factory is None else "injected",
            "elapsedSeconds": total_elapsed,
            "decisionsPerSecond": total_decisions / total_elapsed if total_elapsed else None,
            "episodes": timings,
        },
    }


def apply_research_profile(
    base_environment: SnowGymParallelEnv, profile: dict[str, Any]
) -> SnowGymParallelEnv | SnowGymResearchParallelEnv:
    if not profile:
        return base_environment
    return SnowGymResearchParallelEnv(
        base_environment,
        visibility_radius=profile.get("visibilityRadius"),
        action_delay_steps=profile.get("actionDelaySteps", 0),
        observation_delay_steps=profile.get("observationDelaySteps", 0),
        semantic_raster_size=profile.get("semanticRasterSize"),
    )


def run_evaluation_episode(
    environment: SnowGymParallelEnv | SnowGymResearchParallelEnv,
    episode: dict[str, Any],
    *,
    episode_index: int,
    repeat_index: int,
    max_decisions: int,
) -> dict[str, Any]:
    observations, infos = environment.reset(
        seed=episode["seed"], options={"scenario": episode["scenario"]}
    )
    generators = {
        agent: np.random.default_rng(
            np.random.SeedSequence([episode["seed"], episode_index, agent_index])
        )
        for agent_index, agent in enumerate(AGENTS)
    }
    decisions = 0
    rejected_actions = {agent: 0 for agent in AGENTS}
    rewards = {agent: 0.0 for agent in AGENTS}
    terminations = {agent: False for agent in AGENTS}
    truncations = {agent: False for agent in AGENTS}
    while environment.agents and decisions < max_decisions:
        actions = {
            agent: policy_action(
                episode["policies"][agent], observations[agent], generators[agent]
            )
            for agent in environment.agents
        }
        observations, step_rewards, terminations, truncations, infos = environment.step(
            actions
        )
        decisions += 1
        for agent in AGENTS:
            rewards[agent] += float(step_rewards[agent])
        count_rejected_actions(infos, rejected_actions)

    info = infos[AGENTS[0]]
    return {
        "episodeId": episode["id"],
        "repeatIndex": repeat_index,
        "seed": episode["seed"],
        "scenario": episode["scenario"],
        "policies": episode["policies"],
        "profile": episode["profile"],
        "decisions": decisions,
        "tick": int(info["tick"]),
        "stateHash": info["stateHash"],
        "winner": info.get("winner"),
        "terminated": any(terminations.values()),
        "truncated": any(truncations.values()),
        "decisionLimited": bool(environment.agents),
        "rewards": rewards,
        "rejectedActions": rejected_actions,
    }


def policy_action(
    policy: str, observation: GymObservation, generator: np.random.Generator
) -> GymAction:
    capacity = int(observation["ally_mask"].shape[0])
    action_types = np.full(capacity, ACTION_NOOP, dtype=np.int64)
    targets = np.zeros((capacity, 2), dtype=np.float32)
    powers = np.zeros(capacity, dtype=np.float32)
    if policy == "noop":
        return {"action_type": action_types, "target": targets, "power": powers}
    if policy != "masked_random":
        raise ValueError(f"unsupported evaluation policy: {policy}")
    for index in range(capacity):
        if not observation["ally_mask"][index]:
            continue
        valid = np.flatnonzero(observation["unit_action_mask"][index])
        if valid.size:
            action_types[index] = int(generator.choice(valid))
        targets[index] = generator.uniform(-1.0, 1.0, size=2).astype(np.float32)
        powers[index] = np.float32(generator.random())
    return {"action_type": action_types, "target": targets, "power": powers}


def count_rejected_actions(
    infos: dict[str, dict[str, Any]], totals: dict[str, int]
) -> None:
    action_results = infos[AGENTS[0]].get("actionResults")
    if not isinstance(action_results, dict):
        return
    for agent in AGENTS:
        results = action_results.get(agent, [])
        if isinstance(results, list):
            totals[agent] += sum(
                isinstance(result, dict) and result.get("accepted") is False
                for result in results
            )


def write_benchmark(path: str | Path, result: dict[str, Any], *, force: bool) -> None:
    output_path = Path(path)
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a versioned deterministic SnowGym evaluation suite"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the full JSON result")
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.max_decisions <= 0:
        parser.error("--max-decisions must be positive")

    try:
        result = run_evaluation_suite(
            load_evaluation_suite(args.suite),
            server_url=args.server,
            repeat=args.repeat,
            max_decisions=args.max_decisions,
        )
        if args.output is not None:
            write_benchmark(args.output, result, force=args.force)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        summary = result["summary"]
        performance = result["performance"]
        print(f"SnowGym benchmark: {result['suite']['name']}")
        print(f"  episodes:  {summary['episodes']}")
        print(f"  decisions: {summary['decisions']}")
        print(f"  winners:   {summary['winners']}")
        print(f"  decisions/s: {performance['decisionsPerSecond']:.1f}")


if __name__ == "__main__":
    main()
