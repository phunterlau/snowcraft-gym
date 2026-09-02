from __future__ import annotations

import json
from pathlib import Path

from snowgym_training.ppo_demo import run_ppo_demo


def test_qualified_ppo_checkpoint_records_blue_win(tmp_path) -> None:
    training_root = Path(__file__).resolve().parents[1]
    result = run_ppo_demo(
        checkpoint=training_root / "runs/ppo_1v1_bc_v0/checkpoints/update-000025/checkpoint",
        output=tmp_path / "replay.json",
        seed=3101,
    )
    replay = json.loads((tmp_path / "replay.json").read_text())
    assert result["winner"] == "blue"
    assert result["rejectedActions"] == 0
    assert result["decisions"] == 60
    assert replay["outcome"]["winner"] == "blue"
    assert len(replay["frames"]) == result["decisions"] + 1
    assert len(replay["stateHashes"]) == len(replay["frames"])
