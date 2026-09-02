from __future__ import annotations

import json

from snowgym_training.model import ModelConfig
from snowgym_training.ppo import PPOConfig
from snowgym_training.ppo_train import PPO_RUN_FORMAT, train_ppo


def test_ppo_smoke_run_and_resume_match_uninterrupted_training(tmp_path) -> None:
    architecture = ModelConfig(16, 12, 24)
    config = PPOConfig(update_epochs=1, minibatch_size=4)
    first = train_ppo(
        output=tmp_path / "first",
        worlds=2,
        rollout_steps=2,
        target_updates=1,
        training_seed=19,
        model_config=architecture,
        ppo_config=config,
        git_commit="test",
    )
    resumed = train_ppo(
        output=tmp_path / "resumed",
        worlds=2,
        rollout_steps=2,
        target_updates=2,
        training_seed=19,
        resume=tmp_path / "first" / "checkpoint",
        model_config=architecture,
        ppo_config=config,
        git_commit="test",
    )
    uninterrupted = train_ppo(
        output=tmp_path / "uninterrupted",
        worlds=2,
        rollout_steps=2,
        target_updates=2,
        training_seed=19,
        model_config=architecture,
        ppo_config=config,
        git_commit="test",
    )

    assert first["format"] == PPO_RUN_FORMAT
    assert first["mode"] == "infrastructure-smoke"
    assert first["environmentSteps"] == 4
    assert first["seedSchedule"]["nextSeed"] == 10_002
    assert resumed["startUpdate"] == 1
    assert resumed["environmentSteps"] == 8
    assert resumed["seedSchedule"]["nextSeed"] == 10_004
    assert resumed["updates"] == uninterrupted["updates"][1:]
    assert resumed["checkpoint"]["stateDigest"] == uninterrupted["checkpoint"]["stateDigest"]
    stored = json.loads((tmp_path / "resumed" / "manifest.json").read_text())
    assert stored == resumed


def test_ppo_smoke_run_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    try:
        train_ppo(output=output)
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("PPO trainer overwrote an existing run")
