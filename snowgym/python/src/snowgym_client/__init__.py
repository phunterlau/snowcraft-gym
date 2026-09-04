"""Gymnasium registration and public API for SnowGym."""

from gymnasium.envs.registration import register, registry

from .batch import BatchOperationError, SnowGymBatchClient, SnowGymBatchEnv
from .client import SnowGymHttpClient, SnowGymProtocolError
from .env import SnowGymEnv
from .opponents import (
    LearnedOpponent,
    MaskedRandomOpponent,
    NoopOpponent,
    RemoteOpponent,
    SnowGymSingleTeamEnv,
)
from .parallel_env import SnowGymParallelEnv
from .research_env import SnowGymResearchParallelEnv

ENV_ID = "SnowGym/Squad-v0"
CONFIGURABLE_ENV_ID = "SnowGym/Squad-v1"
TEN_UNIT_ENV_ID = "SnowGym/Squad-v2"
FULL_STATE_ENV_ID = "SnowGym/Squad-v3"

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

if FULL_STATE_ENV_ID not in registry:
    register(
        id=FULL_STATE_ENV_ID,
        entry_point="snowgym_client.env:SnowGymEnv",
        kwargs={
            "max_team_units": 10,
            "configurable": True,
            "observation_version": 3,
        },
        nondeterministic=False,
    )

__all__ = [
    "CONFIGURABLE_ENV_ID",
    "ENV_ID",
    "FULL_STATE_ENV_ID",
    "TEN_UNIT_ENV_ID",
    "SnowGymEnv",
    "SnowGymBatchClient",
    "SnowGymBatchEnv",
    "BatchOperationError",
    "SnowGymHttpClient",
    "SnowGymSingleTeamEnv",
    "SnowGymParallelEnv",
    "SnowGymResearchParallelEnv",
    "SnowGymProtocolError",
    "LearnedOpponent",
    "MaskedRandomOpponent",
    "NoopOpponent",
    "RemoteOpponent",
]
