from __future__ import annotations

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.curriculum import load_curriculum


def test_every_frozen_curriculum_gate_resets_in_authoritative_batch_server() -> None:
    curriculum = load_curriculum()
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(len(curriculum["gates"]), client=client)
        observation, infos = environment.reset(
            [gate["evaluationSeeds"][0] for gate in curriculum["gates"]],
            [gate["scenario"] for gate in curriculum["gates"]],
        )
    assert observation["allies"].shape == (7, 10, 10)
    assert [info["blueAlive"] for info in infos] == [1, 1, 3, 3, 3, 5, 10]
    assert [info["redAlive"] for info in infos] == [1, 1, 3, 3, 3, 5, 10]
    assert [info["configuration"]["map"] for info in infos] == [
        None,
        None,
        None,
        None,
        "arena4.json",
        "arena6.json",
        "arena6.json",
    ]
