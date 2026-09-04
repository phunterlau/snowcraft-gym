from __future__ import annotations

import json
import math

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.executor import model_config
from snowgym_training.options.evaluate import evaluate_option_episode
from snowgym_training.options.train import OPTION_PPO_RUN_FORMAT, train_option_ppo
from snowgym_training.options.train import DEFAULT_INITIALIZER
from snowgym_training.plan_ppo import freeze_initializer, initialize_plan_ppo_policy
from snowgym_training.ppo import HybridActorCritic, PPOConfig
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


def test_option_ppo_smoke_and_exact_resume_match_uninterrupted(tmp_path) -> None:
    config = PPOConfig(update_epochs=1, minibatch_size=2)
    first = train_option_ppo(
        output=tmp_path / "first",
        option="engage",
        worlds=1,
        rollout_steps=2,
        target_updates=1,
        anchor_total_updates=4,
        ppo_config=config,
        training_seed=93_001,
        git_commit="test",
        infrastructure_smoke=True,
    )
    resumed = train_option_ppo(
        output=tmp_path / "resumed",
        option="engage",
        worlds=1,
        rollout_steps=2,
        target_updates=2,
        anchor_total_updates=4,
        ppo_config=config,
        training_seed=93_001,
        resume=tmp_path / "first" / "checkpoint",
        git_commit="test",
        infrastructure_smoke=True,
    )
    uninterrupted = train_option_ppo(
        output=tmp_path / "uninterrupted",
        option="engage",
        worlds=1,
        rollout_steps=2,
        target_updates=2,
        anchor_total_updates=4,
        ppo_config=config,
        training_seed=93_001,
        git_commit="test",
        infrastructure_smoke=True,
    )
    assert first["format"] == OPTION_PPO_RUN_FORMAT
    assert first["mode"] == "infrastructure-smoke"
    assert sum(first["updates"][0]["actionCounts"].values()) == 10
    assert sum(first["updates"][0]["teacherActionCounts"].values()) == 10
    assert set(first["updates"][0]["metrics"]) >= {
        "bcAction", "bcTarget", "bcPower", "targetStd", "powerStd"
    }
    assert first["initialization"]["checkpointDigest"] == (
        "sha256:4119b9a3f8c0d3df69b704a8f3f813a1b38ab11ca56faffb4c4cebc2cb235133"
    )
    assert resumed["startUpdate"] == 1
    assert resumed["updates"] == uninterrupted["updates"][1:]
    assert resumed["seedSchedule"] == uninterrupted["seedSchedule"]
    assert resumed["optionSchedule"] == uninterrupted["optionSchedule"]
    assert resumed["checkpoint"]["stateDigest"] == uninterrupted["checkpoint"]["stateDigest"]
    assert json.loads((tmp_path / "resumed" / "manifest.json").read_text()) == resumed

    metadata, state = load_ppo_checkpoint(tmp_path / "first" / "checkpoint")
    architecture = model_config(metadata["architecture"])
    model = HybridActorCritic(architecture)
    model.load_state_dict(state["model"])
    _, source_state = load_checkpoint(DEFAULT_INITIALIZER)
    initializer = HybridActorCritic(architecture)
    initialize_plan_ppo_policy(initializer, source_state["model"])
    initializer = freeze_initializer(initializer)
    with SnowGymBatchClient() as client:
        episode = evaluate_option_episode(
            model,
            initializer,
            option="engage",
            seed=200_000,
            condition="correct",
            client=client,
        )
    assert set(episode) == {
        "seed", "success", "progress", "physicalWin", "rejectedActions", "totalActions"
    }
    assert episode["rejectedActions"] == 0
    assert episode["totalActions"] > 0


def test_option_ppo_stage_three_requires_recorded_gates(tmp_path) -> None:
    try:
        train_option_ppo(
            output=tmp_path / "blocked",
            option="hold",
            worlds=1,
            rollout_steps=1,
            target_updates=1,
            stage=3,
            ppo_config=PPOConfig(update_epochs=1, minibatch_size=1),
            git_commit="test",
            infrastructure_smoke=True,
        )
    except ValueError as error:
        assert "both physical and plan gates" in str(error)
    else:
        raise AssertionError("stage 3 trained without physical and plan gates")


def test_option_ppo_research_run_cannot_reset_before_option_horizon(tmp_path) -> None:
    try:
        train_option_ppo(
            output=tmp_path / "partial",
            option="support",
            worlds=1,
            rollout_steps=32,
            target_updates=1,
            ppo_config=PPOConfig(update_epochs=1, minibatch_size=1),
            git_commit="test",
        )
    except ValueError as error:
        assert "full option horizon" in str(error)
    else:
        raise AssertionError("research run accepted a partial option horizon")


def test_option_ppo_stage_two_transfers_state_and_opens_inherited_heads(tmp_path) -> None:
    config = PPOConfig(update_epochs=1, minibatch_size=1)
    first = train_option_ppo(
        output=tmp_path / "stage1",
        option="advance",
        worlds=1,
        rollout_steps=1,
        target_updates=1,
        anchor_total_updates=4,
        stage=1,
        ppo_config=config,
        git_commit="test",
        infrastructure_smoke=True,
    )
    second = train_option_ppo(
        output=tmp_path / "stage2",
        option="advance",
        worlds=1,
        rollout_steps=1,
        target_updates=2,
        anchor_total_updates=4,
        stage=2,
        ppo_config=config,
        ppo_warm_start=tmp_path / "stage1" / "checkpoint",
        git_commit="test",
        infrastructure_smoke=True,
    )
    assert second["startUpdate"] == 1
    assert second["initialization"] == {
        "type": "ppo-transfer",
        "checkpointDigest": first["checkpoint"]["checkpointDigest"],
        "stateDigest": first["checkpoint"]["stateDigest"],
        "curriculumDigest": first["checkpoint"]["curriculumDigest"],
        "sourceGate": "m7b-advance-stage1",
        "updateIndex": 1,
    }
    assert second["learningRates"]["new"] == 0.0003
    assert math.isclose(second["learningRates"]["heads"], 0.00003)
    assert second["seedSchedule"]["nextSeed"] == first["seedSchedule"]["nextSeed"] + 1
    third = train_option_ppo(
        output=tmp_path / "hold-stage2",
        option="hold",
        worlds=1,
        rollout_steps=1,
        target_updates=3,
        anchor_total_updates=4,
        stage=2,
        ppo_config=config,
        ppo_warm_start=tmp_path / "stage2" / "checkpoint",
        git_commit="test",
        infrastructure_smoke=True,
    )
    assert third["startUpdate"] == 2
    assert third["seedSchedule"] == {
        "minimum": 120_000,
        "maximum": 129_999,
        "nextSeed": 120_001,
    }
    assert third["optionSchedule"]["prefix"] == "m7b-hold"
