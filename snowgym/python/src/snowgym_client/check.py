"""Run Gymnasium's official environment checker against a live server."""

from __future__ import annotations

import argparse
import json
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from . import CONFIGURABLE_ENV_ID, ENV_ID, TEN_UNIT_ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SnowGym Gymnasium contract")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    args = parser.parse_args()

    checked: list[str] = []
    environments = (
        (ENV_ID, {}),
        (CONFIGURABLE_ENV_ID, {}),
        (TEN_UNIT_ENV_ID, {"map": "arena6.json"}),
    )
    for environment_id, kwargs in environments:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*SnowGym/Squad-v[01] is out of date.*",
                category=DeprecationWarning,
            )
            environment = gym.make(
                environment_id, server_url=args.server, **kwargs
            ).unwrapped
        try:
            check_env(environment, skip_render_check=True)
        finally:
            environment.close()
        checked.append(environment_id)
        if not args.json:
            print(f"Gymnasium checker passed: {environment_id}")
    if args.json:
        print(json.dumps({"ok": True, "environments": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
