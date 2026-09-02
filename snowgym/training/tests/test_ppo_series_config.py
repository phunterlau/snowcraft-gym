from __future__ import annotations

from snowgym_training.ppo_series_config import load_series_config, validate_series_config


def test_frozen_ppo_series_config_binds_conservative_bc_run() -> None:
    config = load_series_config()
    assert config["gateId"] == "1v1-random"
    assert config["checkpointUpdates"] == [1, 5, 10, 25]
    assert config["rolloutSteps"] == 200
    assert config["ppoConfig"]["learning_rate"] == 0.00003
    assert config["ppoConfig"]["update_epochs"] == 1
    assert config["warmStart"]["checkpointDigest"].startswith("sha256:")


def test_ppo_series_config_rejects_unsorted_checkpoints() -> None:
    config = load_series_config()
    config["checkpointUpdates"] = [5, 1]
    try:
        validate_series_config(config)
    except ValueError as error:
        assert "checkpointUpdates" in str(error)
    else:
        raise AssertionError("series config accepted unsorted checkpoints")
