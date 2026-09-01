"""Held-out scripted-teacher and masked-random blue baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.client import SnowGymClient, SnowGymHttpClient
from snowgym_client.env import SnowGymEnv
from snowgym_client.opponents import masked_random_action

from .export_scripted import default_spec_path
from .trajectory import json_digest, load_export_spec

BASELINE_FORMAT = "snowgym.teacher-baseline.v0"


def run_teacher_baseline(
    *,
    spec_path: str | Path | None = None,
    split: str = "evaluation",
    server_url: str = "http://127.0.0.1:8787",
    max_decisions: int = 1000,
    client: SnowGymClient | None = None,
) -> dict[str, Any]:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    spec_source = Path(spec_path) if spec_path is not None else default_spec_path()
    spec = load_export_spec(spec_source)
    if split not in spec["splits"]:
        raise ValueError(f"unknown split {split!r}")
    environment = SnowGymEnv(
        client=client or SnowGymHttpClient(server_url),
        max_team_units=int(spec["maxTeamUnits"]),
        configurable=True,
    )
    results: list[dict[str, Any]] = []
    versions: dict[str, Any] | None = None
    try:
        for policy in ("scripted_teacher", "masked_random"):
            for episode in spec["splits"][split]:
                observation, info = environment.reset(
                    seed=int(episode["seed"]),
                    options={"scenario": episode["scenario"]},
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
                generator = np.random.default_rng(
                    np.random.SeedSequence([int(episode["seed"]), 0x424C5545])
                )
                decisions = 0
                terminated = truncated = False
                rejected = 0
                reward = 0.0
                while not (terminated or truncated) and decisions < max_decisions:
                    if policy == "scripted_teacher":
                        observation, reward, terminated, truncated, info = (
                            environment.step_scripted()
                        )
                    else:
                        action = masked_random_action(observation, generator)
                        observation, reward, terminated, truncated, info = environment.step(
                            action
                        )
                    action_results = info.get("actionResults", [])
                    if isinstance(action_results, list):
                        rejected += sum(
                            isinstance(result, dict)
                            and result.get("accepted") is False
                            for result in action_results
                        )
                    decisions += 1
                results.append(
                    {
                        "policy": policy,
                        "seed": episode["seed"],
                        "scenario": episode["scenario"],
                        "decisions": decisions,
                        "tick": info["tick"],
                        "stateHash": info["stateHash"],
                        "winner": info.get("winner"),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "decisionLimited": not (terminated or truncated),
                        "blueAlive": info["blueAlive"],
                        "redAlive": info["redAlive"],
                        "rejectedActions": rejected,
                    }
                )
    finally:
        environment.close()
    summary = {
        policy: summarize([result for result in results if result["policy"] == policy])
        for policy in ("scripted_teacher", "masked_random")
    }
    value = {
        "format": BASELINE_FORMAT,
        "sourceSpecDigest": json_digest(spec),
        "versions": versions,
        "split": split,
        "maxDecisions": max_decisions,
        "results": results,
        "summary": summary,
    }
    value["resultDigest"] = json_digest(value)
    return value


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(results)
    return {
        "episodes": episodes,
        "blueWins": sum(result["winner"] == "blue" for result in results),
        "redWins": sum(result["winner"] == "red" for result in results),
        "draws": sum(result["winner"] is None for result in results),
        "meanDecisions": (
            sum(int(result["decisions"]) for result in results) / episodes
            if episodes
            else 0.0
        ),
        "rejectedActions": sum(int(result["rejectedActions"]) for result in results),
    }


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scripted-teacher and masked-random blue policies"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_teacher_baseline(
            spec_path=args.spec,
            split=args.split,
            server_url=args.server,
            max_decisions=args.max_decisions,
        )
        if args.output is not None:
            write_result(args.output, result)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"SnowGym teacher baseline: {result['split']}")
        for policy, summary in result["summary"].items():
            print(
                f"  {policy}: wins={summary['blueWins']}/{summary['episodes']} "
                f"rejections={summary['rejectedActions']}"
            )


if __name__ == "__main__":
    main()
