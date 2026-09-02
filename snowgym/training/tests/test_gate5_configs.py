import json
from pathlib import Path

from snowgym_training.curriculum import load_curriculum
from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import json_digest, load_export_spec


def test_gate5_teacher_config_is_valid_terrain_and_seed_disjoint() -> None:
    config_root = Path(__file__).parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_3v3_terrain_v0.json")
    learner = load_training_config(config_root / "bc_3v3_terrain_v0.json")
    curriculum = load_curriculum()
    gate = next(item for item in curriculum["gates"] if item["id"] == "3v3-terrain")

    assert teacher["name"] == "scripted-blue-3v3-terrain-v0"
    assert learner["evaluationSuite"] == "teacher_3v3_terrain_v0/evaluation"
    assert learner["steps"] == 10_000
    assert learner["loss"] == {
        "action_weight": 1.0,
        "target_weight": 10.0,
        "power_weight": 1.0,
    }
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


def test_committed_gate5_teacher_baseline_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    spec = load_export_spec(
        training_root / "src/snowgym_training/configs/teacher_3v3_terrain_v0.json"
    )
    baseline = json.loads(
        (training_root / "baselines/teacher_3v3_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = baseline.pop("resultDigest")
    assert claimed == json_digest(baseline)
    assert baseline["sourceSpecDigest"] == json_digest(spec)
    assert baseline["summary"]["scripted_teacher"]["blueWins"] == 2
    assert baseline["summary"]["scripted_teacher"]["rejectedActions"] == 0
    assert baseline["summary"]["masked_random"]["blueWins"] == 0
