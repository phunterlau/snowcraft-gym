import json
from pathlib import Path

from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.curriculum import load_curriculum
from snowgym_training.ppo_series_config import load_series_config
from snowgym_training.trainer import load_training_config
from snowgym_training.trajectory import json_digest, load_export_spec


def test_gate7_teacher_config_is_valid_terrain_and_seed_disjoint() -> None:
    config_root = Path(__file__).parents[1] / "src/snowgym_training/configs"
    teacher = load_export_spec(config_root / "teacher_10v10_terrain_v0.json")
    learner = load_training_config(config_root / "bc_10v10_terrain_v0.json")
    relational = load_training_config(
        config_root / "bc_10v10_terrain_relational_v0.json"
    )
    ppo = load_series_config(
        config_root / "ppo_10v10_terrain_relational_bc_v0.json"
    )
    curriculum = load_curriculum()
    gate = next(
        item for item in curriculum["gates"] if item["id"] == "10v10-random-terrain"
    )

    assert teacher["name"] == "scripted-blue-10v10-terrain-v0"
    assert learner["steps"] == 30_000
    assert learner["architecture"]["action_conditioned_targets"] is True
    assert learner["architecture"]["nearest_enemy_throw_target"] is True
    assert learner["loss"]["throw_action_weight"] == 10.0
    assert learner["evaluationSuite"] == "teacher_10v10_terrain_v0/evaluation"
    assert relational["steps"] == 20_000
    assert relational["architecture"]["last_enemy_move_target"] is True
    assert relational["architecture"]["nearest_enemy_throw_target"] is True
    assert relational["architecture"]["nearest_enemy_features"] is True
    assert relational["evaluationSuite"] == "teacher_10v10_terrain_v0/evaluation"
    assert ppo["gateId"] == "10v10-random-terrain"
    assert ppo["checkpointUpdates"] == [1, 5, 10]
    assert ppo["rolloutSteps"] == 600
    assert ppo["architecture"] == relational["architecture"]
    assert ppo["warmStart"]["checkpointDigest"] == (
        "sha256:a7f1362cf163fbf23ebc3c8290bb0f772e57fb3a763b6dcf97b21135aa47bc08"
    )
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


def test_committed_gate7_teacher_baseline_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    spec = load_export_spec(
        training_root / "src/snowgym_training/configs/teacher_10v10_terrain_v0.json"
    )
    baseline = json.loads(
        (training_root / "baselines/teacher_10v10_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = baseline.pop("resultDigest")
    assert claimed == json_digest(baseline)
    assert baseline["sourceSpecDigest"] == json_digest(spec)
    assert baseline["summary"]["scripted_teacher"]["blueWins"] == 2
    assert baseline["summary"]["scripted_teacher"]["rejectedActions"] == 0
    assert baseline["summary"]["masked_random"]["blueWins"] == 0


def test_committed_gate7_initializer_is_digest_bound_and_near_terminal() -> None:
    training_root = Path(__file__).parents[1]
    metadata, _ = load_checkpoint(training_root / "checkpoints/bc_10v10_terrain_v0")
    evaluation = json.loads(
        (training_root / "evaluations/bc_10v10_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = evaluation.pop("resultDigest")
    assert claimed == json_digest(evaluation)
    assert evaluation["checkpointDigest"] == metadata["checkpointDigest"]
    assert metadata["gitCommit"].startswith("c514757")
    assert metadata["datasetManifestHash"] == (
        "sha256:6f2e7a7ed096d64e980343818d446e721c2bd70468fe25644b47052c0b89df78"
    )
    assert evaluation["summary"]["learned"]["blueWins"] == 0
    assert evaluation["summary"]["learned"]["meanRedHealthDealt"] > 9.5
    assert evaluation["summary"]["learned"]["rejectedActions"] == 0


def test_committed_gate7_relational_initializer_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    metadata, _ = load_checkpoint(
        training_root / "checkpoints/bc_10v10_terrain_relational_v0"
    )
    evaluation = json.loads(
        (
            training_root / "evaluations/bc_10v10_terrain_relational_v0.json"
        ).read_text(encoding="utf-8")
    )
    claimed = evaluation.pop("resultDigest")
    assert claimed == json_digest(evaluation)
    assert evaluation["checkpointDigest"] == metadata["checkpointDigest"]
    assert metadata["gitCommit"].startswith("9f28de6")
    assert metadata["datasetManifestHash"] == (
        "sha256:6f2e7a7ed096d64e980343818d446e721c2bd70468fe25644b47052c0b89df78"
    )
    assert metadata["architecture"]["last_enemy_move_target"] is True
    assert evaluation["summary"]["learned"]["blueWins"] == 2
    assert evaluation["summary"]["learned"]["meanDecisions"] == 145.0
    assert evaluation["summary"]["learned"]["meanRedHealthDealt"] == 10.0
    assert evaluation["summary"]["learned"]["rejectedActions"] == 0
