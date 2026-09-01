from __future__ import annotations

import numpy as np

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv


def scenario(max_ticks: int = 120) -> dict[str, object]:
    return {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": max_ticks,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }


def noop_actions(batch_size: int, capacity: int = 10) -> dict[str, np.ndarray]:
    return {
        "action_type": np.zeros((batch_size, capacity), dtype=np.int64),
        "target": np.zeros((batch_size, capacity, 2), dtype=np.float32),
        "power": np.zeros((batch_size, capacity), dtype=np.float32),
    }


def test_batch_subprocess_handshake_and_independent_worlds() -> None:
    with SnowGymBatchClient() as client:
        assert client.capabilities["protocolVersion"] == "snowgym.batch.v0"
        environment = SnowGymBatchEnv(2, client=client)
        observation, infos = environment.reset([11, 12], [scenario(), scenario()])
        assert observation["allies"].shape == (2, 10, 10)
        assert [info["tick"] for info in infos] == [0, 0]
        assert infos[0]["stateHash"] == infos[1]["stateHash"]

        observation, rewards, terminated, truncated, infos = environment.step(
            noop_actions(2)
        )
        assert observation["tick"].tolist() == [[6], [6]]
        assert rewards.tolist() == [0.0, 0.0]
        assert not terminated.any()
        assert not truncated.any()
        assert [info["tick"] for info in infos] == [6, 6]

        changed, reset_infos = environment.reset_indices([1], [99], [scenario()])
        assert changed["tick"].tolist() == [[0]]
        assert reset_infos[0]["seed"] == 99
        assert environment._observations[0]["tick"].tolist() == [6]
