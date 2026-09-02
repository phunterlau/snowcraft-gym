from __future__ import annotations

from pathlib import Path

from snowgym_training.curriculum import load_curriculum
from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import load_export_spec


def test_gate3_teacher_and_bc_configs_are_valid_and_disjoint() -> None:
    config_root = Path(__file__).resolve().parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_3v3_random_v0.json")
    learner = load_training_config(config_root / "bc_3v3_random_v0.json")
    curriculum = load_curriculum()
    gate = next(item for item in curriculum["gates"] if item["id"] == "3v3-random")

    assert teacher["name"] == "scripted-blue-3v3-random-v0"
    assert learner["evaluationSuite"] == "teacher_3v3_random_v0/evaluation"
    scenarios = [
        episode["scenario"]
        for episodes in teacher["splits"].values()
        for episode in episodes
    ]
    assert {scenario["blueUnits"] for scenario in scenarios} == {3}
    assert {scenario["redUnits"] for scenario in scenarios} == {3}
    assert {scenario["redController"] for scenario in scenarios} == {"random"}

    split_seeds = [
        {episode["seed"] for episode in teacher["splits"][split]}
        for split in ("train", "validation", "evaluation")
    ]
    assert all(
        not split_seeds[left] & split_seeds[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    assert not set().union(*split_seeds) & set(gate["evaluationSeeds"])
