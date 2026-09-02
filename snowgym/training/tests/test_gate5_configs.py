from pathlib import Path

from snowgym_training.curriculum import load_curriculum
from snowgym_training.trajectory import load_export_spec


def test_gate5_teacher_config_is_valid_terrain_and_seed_disjoint() -> None:
    config_root = Path(__file__).parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_3v3_terrain_v0.json")
    curriculum = load_curriculum()
    gate = next(item for item in curriculum["gates"] if item["id"] == "3v3-terrain")

    assert teacher["name"] == "scripted-blue-3v3-terrain-v0"
    scenarios = [
        episode["scenario"]
        for episodes in teacher["splits"].values()
        for episode in episodes
    ]
    assert {scenario["map"] for scenario in scenarios} == {"arena4.json"}
    assert {scenario["redController"] for scenario in scenarios} == {"random"}
    assert all(
        {key: scenario[key] for key in gate["scenario"]} == gate["scenario"]
        for scenario in scenarios
    )

    split_seeds = [
        {episode["seed"] for episode in teacher["splits"][split]}
        for split in ("train", "validation", "evaluation")
    ]
    assert all(left.isdisjoint(right) for index, left in enumerate(split_seeds) for right in split_seeds[index + 1 :])
    assert set(gate["evaluationSeeds"]).isdisjoint(set().union(*split_seeds))
