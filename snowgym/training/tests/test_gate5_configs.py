import json
from pathlib import Path

from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.curriculum import load_curriculum
from snowgym_training.ppo_series import audit_ppo_series
from snowgym_training.ppo_series_config import load_series_config
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


def test_committed_gate5_initializer_is_digest_bound_and_wins() -> None:
    training_root = Path(__file__).parents[1]
    metadata, _ = load_checkpoint(training_root / "checkpoints/bc_3v3_terrain_v0")
    evaluation = json.loads(
        (training_root / "evaluations/bc_3v3_terrain_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = evaluation.pop("resultDigest")
    assert claimed == json_digest(evaluation)
    assert evaluation["checkpointDigest"] == metadata["checkpointDigest"]
    assert metadata["gitCommit"].startswith("b60112f")
    assert metadata["datasetManifestHash"] == (
        "sha256:cccab3a390331277484e251867b42e2f017b14ed0b21836454c2632d20626957"
    )
    assert evaluation["summary"]["learned"]["blueWins"] == 2
    assert evaluation["summary"]["learned"]["rejectedActions"] == 0


def test_gate5_ppo_config_is_frozen_to_accepted_initializer() -> None:
    training_root = Path(__file__).parents[1]
    config = load_series_config(
        training_root / "src/snowgym_training/configs/ppo_3v3_terrain_bc_v0.json"
    )
    assert config["gateId"] == "3v3-terrain"
    assert config["checkpointUpdates"] == [1, 5, 10]
    assert config["trainingSeed"] == 77
    assert config["warmStart"] == {
        "path": "checkpoints/bc_3v3_terrain_v0",
        "checkpointDigest": "sha256:757f46ca1d2c361814c15f11327ffcc88bf73b4218e633bbc8d16b2424cd9cbd",
    }


def test_committed_gate5_ppo_series_is_auditable_and_qualifies() -> None:
    training_root = Path(__file__).parents[1]
    audit = audit_ppo_series(training_root / "runs/ppo_3v3_terrain_bc_v0")
    assert audit["mode"] == "qualifying"
    assert audit["gate"] == "3v3-terrain"
    assert audit["updates"] == [1, 5, 10]
    assert audit["finalThresholdPassed"] is True

    for update in (1, 5, 10):
        evaluation = json.loads(
            (
                training_root
                / f"runs/ppo_3v3_terrain_bc_v0/evaluations/update-{update:06d}.json"
            ).read_text(encoding="utf-8")
        )
        assert evaluation["summary"]["ppo"]["blueWins"] == 8
        assert evaluation["summary"]["masked_random"]["blueWins"] == 0
        assert evaluation["summary"]["ppo"]["rejectedActions"] == 0
