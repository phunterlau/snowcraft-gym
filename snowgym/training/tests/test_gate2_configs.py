from __future__ import annotations

from pathlib import Path

from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import load_export_spec


def test_gate2_teacher_and_bc_configs_are_valid_and_disjoint() -> None:
    config_root = Path(__file__).resolve().parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_1v1_easy_v0.json")
    learner = load_training_config(config_root / "bc_1v1_easy_v0.json")
    assert teacher["name"] == "scripted-blue-1v1-easy-v0"
    assert learner["evaluationSuite"] == "teacher_1v1_easy_v0/evaluation"
    assert {
        episode["scenario"]["redController"]
        for episodes in teacher["splits"].values()
        for episode in episodes
    } == {"scripted"}
    split_seeds = [
        {episode["seed"] for episode in teacher["splits"][split]}
        for split in ("train", "validation", "evaluation")
    ]
    assert not split_seeds[0] & split_seeds[1]
    assert not split_seeds[0] & split_seeds[2]
    assert not split_seeds[1] & split_seeds[2]
