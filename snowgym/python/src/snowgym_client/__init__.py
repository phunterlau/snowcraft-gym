"""Gymnasium registration and public API for SnowGym."""

from gymnasium.envs.registration import register, registry

from .client import SnowGymHttpClient, SnowGymProtocolError
from .env import SnowGymEnv

ENV_ID = "SnowGym/Squad-v0"
CONFIGURABLE_ENV_ID = "SnowGym/Squad-v1"

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

__all__ = [
    "CONFIGURABLE_ENV_ID",
    "ENV_ID",
    "SnowGymEnv",
    "SnowGymHttpClient",
    "SnowGymProtocolError",
]
