import json
from pathlib import Path

from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.curriculum import load_curriculum
from snowgym_training.ppo_series_config import load_series_config
from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import json_digest, load_export_spec


def test_gate6_teacher_config_is_valid_terrain_and_seed_disjoint() -> None:
    config_root = Path(__file__).parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_5v5_terrain_v0.json")
    learner = load_training_config(config_root / "bc_5v5_terrain_v0.json")
    curriculum = load_curriculum()
    gate = next(
        item for item in curriculum["gates"] if item["id"] == "5v5-random-terrain"
    )

    assert teacher["name"] == "scripted-blue-5v5-terrain-v0"
    assert learner["steps"] == 20_000
    assert learner["architecture"]["action_conditioned_targets"] is True
    assert learner["evaluationSuite"] == "teacher_5v5_terrain_v0/evaluation"
    scenarios = [
        episode["scenario"]
        for episodes in teacher["splits"].values()
        for episode in episodes
    ]
    assert {scenario["map"] for scenario in scenarios} == {"arena6.json"}
    assert {scenario["blueUnits"] for scenario in scenarios} == {5}
    assert {scenario["redUnits"] for scenario in scenarios} == {5}
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


def test_committed_gate6_teacher_baseline_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    spec = load_export_spec(
        training_root / "src/snowgym_training/configs/teacher_5v5_terrain_v0.json"
    )
    baseline = json.loads(
        (training_root / "baselines/teacher_5v5_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = baseline.pop("resultDigest")
    assert claimed == json_digest(baseline)
    assert baseline["sourceSpecDigest"] == json_digest(spec)
    assert baseline["summary"]["scripted_teacher"]["blueWins"] == 2
    assert baseline["summary"]["scripted_teacher"]["rejectedActions"] == 0
    assert baseline["summary"]["masked_random"]["blueWins"] == 0


def test_committed_gate6_initializer_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    metadata, _ = load_checkpoint(training_root / "checkpoints/bc_5v5_terrain_v0")
    evaluation = json.loads(
        (training_root / "evaluations/bc_5v5_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = evaluation.pop("resultDigest")
    assert claimed == json_digest(evaluation)
    assert evaluation["checkpointDigest"] == metadata["checkpointDigest"]
    assert metadata["gitCommit"].startswith("f7bb43a")
    assert metadata["datasetManifestHash"] == (
        "sha256:4e1a0352ed5d6e21c09e4599b27c783abe14009258214f8e87dc5bdd2f134ac4"
    )
    assert metadata["architecture"]["action_conditioned_targets"] is True
    assert evaluation["summary"]["learned"]["blueWins"] == 2
    assert evaluation["summary"]["learned"]["rejectedActions"] == 0


def test_gate6_ppo_config_is_frozen_to_accepted_initializer() -> None:
    training_root = Path(__file__).parents[1]
    config = load_series_config(
        training_root / "src/snowgym_training/configs/ppo_5v5_terrain_bc_v0.json"
    )
    assert config["gateId"] == "5v5-random-terrain"
    assert config["checkpointUpdates"] == [1, 5, 10]
    assert config["trainingSeed"] == 78
    assert config["architecture"]["action_conditioned_targets"] is True
    assert config["warmStart"] == {
        "path": "checkpoints/bc_5v5_terrain_v0",
        "checkpointDigest": "sha256:8fec9129582ad3709733158d919a544a833caab24d1fa0e744a4b4d5eaa01bcd",
    }
