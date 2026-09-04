from __future__ import annotations

import json
from pathlib import Path

import pytest

from snowgym_training.ppo_demo import run_ppo_demo


@pytest.mark.parametrize(
    ("checkpoint", "gate_id", "seed", "winner", "decisions"),
    [
        (
            "runs/ppo_1v1_bc_v0/checkpoints/update-000025/checkpoint",
            "1v1-random",
            3101,
            "red",
            64,
        ),
        (
            "runs/ppo_1v1_easy_bc_v0/checkpoints/update-000010/checkpoint",
            "1v1-easy-scripted",
            4103,
            "blue",
            43,
        ),
        (
            "runs/ppo_3v3_random_bc_v0/checkpoints/update-000010/checkpoint",
            "3v3-random",
            5101,
            "red",
            86,
        ),
        (
            "runs/ppo_3v3_scripted_bc_v0/checkpoints/update-000010/checkpoint",
            "3v3-scripted",
            6108,
            "blue",
            87,
        ),
    ],
)
def test_historical_ppo_checkpoint_has_deterministic_v2_replay(
    tmp_path,
    checkpoint: str,
    gate_id: str,
    seed: int,
    winner: str,
    decisions: int,
) -> None:
    training_root = Path(__file__).resolve().parents[1]
    result = run_ppo_demo(
        checkpoint=training_root / checkpoint,
        output=tmp_path / "replay.json",
        gate_id=gate_id,
        seed=seed,
    )
    replay = json.loads((tmp_path / "replay.json").read_text())
    assert result["winner"] == winner
    assert result["rejectedActions"] == 0
    assert result["decisions"] == decisions
    assert replay["outcome"]["winner"] == winner
    assert replay["simulationVersion"] == "snowgym.sim.v2"
    assert len(replay["frames"]) == result["decisions"] + 1
    assert len(replay["stateHashes"]) == len(replay["frames"])
