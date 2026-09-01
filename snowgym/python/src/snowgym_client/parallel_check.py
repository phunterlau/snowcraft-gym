"""Run PettingZoo's official Parallel API checker against a live server."""

from __future__ import annotations

import argparse
import json

from pettingzoo.test import parallel_api_test

from .parallel_env import SnowGymParallelEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SnowGym PettingZoo contract")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    environment = SnowGymParallelEnv(server_url=args.server)
    try:
        parallel_api_test(environment, num_cycles=args.cycles)
    finally:
        environment.close()
    result = {
        "ok": True,
        "environment": environment.metadata["name"],
        "agents": environment.possible_agents,
        "cycles": args.cycles,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"PettingZoo checker passed: {result['environment']}")


if __name__ == "__main__":
    main()
