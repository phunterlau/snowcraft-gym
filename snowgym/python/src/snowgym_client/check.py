"""Run Gymnasium's official environment checker against a live server."""

from __future__ import annotations

import argparse
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from . import CONFIGURABLE_ENV_ID, ENV_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SnowGym Gymnasium contract")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    args = parser.parse_args()

    for environment_id in (ENV_ID, CONFIGURABLE_ENV_ID):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*SnowGym/Squad-v0 is out of date.*",
                category=DeprecationWarning,
            )
            environment = gym.make(environment_id, server_url=args.server).unwrapped
        try:
            check_env(environment, skip_render_check=True)
        finally:
            environment.close()
        print(f"Gymnasium checker passed: {environment_id}")


if __name__ == "__main__":
    main()
