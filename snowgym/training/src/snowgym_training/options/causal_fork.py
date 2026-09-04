"""Same-state HOLD/WITHDRAW/ADVANCE production-teacher causal fork."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..trajectory import json_digest
from .plans import teacher_option_plan, teacher_option_scenario

FORK_NAMES = ("hold", "withdraw", "advance")


def run_causal_fork(*, seed: int = 42_001, decisions: int = 30) -> dict[str, Any]:
    if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions <= 0:
        raise ValueError("causal fork decisions must be positive")
    scenario = teacher_option_scenario()
    plans = [teacher_option_plan(name)[0] for name in FORK_NAMES]
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(3, client=client, observation_version=3)
        environment.reset([seed] * 3, [dict(scenario) for _ in FORK_NAMES])
        initial_hashes = list(environment.state_hashes)
        if len(set(initial_hashes)) != 1:
            raise RuntimeError("causal fork did not begin from an equal physical state")
        environment.activate_plans(
            [f"causal-{name}-{seed}" for name in FORK_NAMES], plans
        )
        traces = {
            name: {
                "stateHashes": [initial_hashes[index]],
                "actions": [],
                "blueCentroids": [],
            }
            for index, name in enumerate(FORK_NAMES)
        }
        rejected = 0
        for _ in range(decisions):
            actions = environment.plan_teacher_actions()
            _, _, terminated, truncated, infos = environment.step_team_actions(actions)
            for index, name in enumerate(FORK_NAMES):
                raw = environment.raw_observations[index]
                if raw is None:
                    raise RuntimeError("causal fork lost its raw observation")
                living = [unit for unit in raw["allies"] if unit["alive"]]
                traces[name]["actions"].append(actions[index])
                traces[name]["stateHashes"].append(environment.state_hashes[index])
                traces[name]["blueCentroids"].append(
                    [
                        sum(unit["x"] for unit in living) / len(living),
                        sum(unit["y"] for unit in living) / len(living),
                    ]
                )
                rejected += sum(
                    result.get("accepted") is False
                    for result in infos[index].get("actionResults", [])
                )
            if bool(terminated.any() or truncated.any()):
                raise RuntimeError("causal fork episode ended before its fixed horizon")
    result = {
        "format": "snowgym.option-causal-fork.v0",
        "seed": seed,
        "decisions": decisions,
        "scenario": scenario,
        "initialStateHash": initial_hashes[0],
        "rejectedActions": rejected,
        "forks": traces,
    }
    result["artifactDigest"] = json_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42_001)
    parser.add_argument("--decisions", type=int, default=30)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite causal fork {destination}")
    result = run_causal_fork(seed=args.seed, decisions=args.decisions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
