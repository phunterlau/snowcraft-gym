from __future__ import annotations

import json
from pathlib import Path

import snowgym_training.ppo_series_config as series_config_module
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


def test_gate2_series_config_stops_before_observed_regression() -> None:
    config_path = Path(__file__).resolve().parents[1] / (
        "src/snowgym_training/configs/ppo_1v1_easy_bc_v0.json"
    )
    config = load_series_config(config_path)
    assert config["gateId"] == "1v1-easy-scripted"
    assert config["checkpointUpdates"] == [1, 5, 10]
    assert config["rolloutSteps"] == 300
    assert config["ppoConfig"]["minibatch_size"] == 2400


def test_ppo_series_config_accepts_exactly_one_ppo_transfer_initializer() -> None:
    config = load_series_config()
    config["ppoWarmStart"] = config.pop("warmStart")
    validate_series_config(config)

    config["warmStart"] = dict(config["ppoWarmStart"])
    try:
        validate_series_config(config)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("series config accepted two initializers")


def test_configured_series_dispatches_digest_checked_ppo_transfer(tmp_path, monkeypatch) -> None:
    config = load_series_config()
    config["ppoWarmStart"] = {
        "path": "runs/source/checkpoint",
        "checkpointDigest": "sha256:source",
    }
    config.pop("warmStart")
    config_path = tmp_path / "ppo-transfer.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        series_config_module,
        "load_ppo_checkpoint",
        lambda path: ({"checkpointDigest": "sha256:source"}, {}),
    )

    def fake_run_ppo_series(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(series_config_module, "run_ppo_series", fake_run_ppo_series)
    assert series_config_module.run_configured_series(
        output=tmp_path / "output", config_path=config_path
    ) == {"ok": True}
    assert captured["warm_start"] is None
    assert captured["ppo_warm_start"].as_posix().endswith("runs/source/checkpoint")
