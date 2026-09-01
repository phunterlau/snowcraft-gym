"""Gymnasium registration and public API for SnowGym."""

from gymnasium.envs.registration import register, registry

from .client import SnowGymHttpClient, SnowGymProtocolError
from .env import SnowGymEnv
from .parallel_env import SnowGymParallelEnv

ENV_ID = "SnowGym/Squad-v0"
CONFIGURABLE_ENV_ID = "SnowGym/Squad-v1"
TEN_UNIT_ENV_ID = "SnowGym/Squad-v2"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="snowgym_client.env:SnowGymEnv",
        nondeterministic=False,
    )

if CONFIGURABLE_ENV_ID not in registry:
    register(
        id=CONFIGURABLE_ENV_ID,
        entry_point="snowgym_client.env:SnowGymEnv",
        kwargs={"max_team_units": 8, "configurable": True},
        nondeterministic=False,
    )

if TEN_UNIT_ENV_ID not in registry:
    register(
        id=TEN_UNIT_ENV_ID,
        entry_point="snowgym_client.env:SnowGymEnv",
        kwargs={"max_team_units": 10, "configurable": True},
        nondeterministic=False,
    )

__all__ = [
    "CONFIGURABLE_ENV_ID",
    "ENV_ID",
    "TEN_UNIT_ENV_ID",
    "SnowGymEnv",
    "SnowGymHttpClient",
    "SnowGymParallelEnv",
    "SnowGymProtocolError",
]
