from pathlib import Path

from snowgym_training.curriculum import load_curriculum
from snowgym_training.trajectory import load_export_spec


def test_gate7_teacher_config_is_valid_terrain_and_seed_disjoint() -> None:
    config_root = Path(__file__).parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_10v10_terrain_v0.json")
    curriculum = load_curriculum()
    gate = next(
        item for item in curriculum["gates"] if item["id"] == "10v10-random-terrain"
    )

    assert teacher["name"] == "scripted-blue-10v10-terrain-v0"
    scenarios = [
        episode["scenario"]
        for episodes in teacher["splits"].values()
        for episode in episodes
    ]
    assert {scenario["map"] for scenario in scenarios} == {"arena6.json"}
    assert {scenario["blueUnits"] for scenario in scenarios} == {10}
    assert {scenario["redUnits"] for scenario in scenarios} == {10}
    assert all(
        {key: scenario[key] for key in gate["scenario"]} == gate["scenario"]
        for scenario in scenarios
    )

    split_seeds = [
        {episode["seed"] for episode in teacher["splits"][split]}
        for split in ("train", "validation", "evaluation")
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(split_seeds)
        for right in split_seeds[index + 1 :]
    )
    assert set(gate["evaluationSeeds"]).isdisjoint(set().union(*split_seeds))
