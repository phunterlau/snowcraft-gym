"""Run PettingZoo's official Parallel API checker against a live server."""

from __future__ import annotations

import argparse
import json

from pettingzoo.test import parallel_api_test

from .parallel_env import SnowGymParallelEnv
from .research_env import SnowGymResearchParallelEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SnowGym PettingZoo contract")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--visibility-radius", type=float)
    parser.add_argument("--action-delay-steps", type=int, default=0)
    parser.add_argument("--observation-delay-steps", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    if args.visibility_radius is not None and args.visibility_radius <= 0:
        parser.error("--visibility-radius must be positive")
    if args.action_delay_steps < 0 or args.observation_delay_steps < 0:
        parser.error("latency steps must be non-negative")

    base_environment = SnowGymParallelEnv(server_url=args.server)
    profile_enabled = (
        args.visibility_radius is not None
        or args.action_delay_steps > 0
        or args.observation_delay_steps > 0
    )
    environment = (
        SnowGymResearchParallelEnv(
            base_environment,
            visibility_radius=args.visibility_radius,
            action_delay_steps=args.action_delay_steps,
            observation_delay_steps=args.observation_delay_steps,
        )
        if profile_enabled
        else base_environment
    )
    try:
        parallel_api_test(environment, num_cycles=args.cycles)
    finally:
        environment.close()
    result = {
        "ok": True,
        "environment": environment.metadata["name"],
        "agents": environment.possible_agents,
        "cycles": args.cycles,
        "profile": {
            "visibilityRadius": args.visibility_radius,
            "actionDelaySteps": args.action_delay_steps,
            "observationDelaySteps": args.observation_delay_steps,
        },
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"PettingZoo checker passed: {result['environment']}")


if __name__ == "__main__":
    main()
