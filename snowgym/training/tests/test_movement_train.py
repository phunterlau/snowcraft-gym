import copy
import json
from pathlib import Path

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.options.movement_checkpoint import load_movement
from snowgym_training.options.movement_train import (DEFAULT_CONFIG, REFERENCE, load_config,
    train_run, result_gate, run_experiment)
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


def test_frozen_movement_configuration_and_existing_output(tmp_path):
    config = load_config()
    assert config["updates"] == 100 and config["batchSize"] == 8
    assert config["bcCoefficient"] == config["entropyCoefficient"] == 0
    for key, value in (("latentStd", .03), ("updates", 101), ("trainingEpisodeSeeds", [100000, 108000])):
        changed = {**config, key: value}
        path = tmp_path / "changed.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="frozen"):
            load_config(path)
    with pytest.raises(FileExistsError):
        run_experiment(output=tmp_path)


def test_reward_only_updates_resume_identically_and_preserve_source(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata, state = load_ppo_checkpoint(REFERENCE)
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    original = semantic_state_digest(source.state_dict())
    # Explicit small contract test, not a run eligible for experiment reporting.
    config = {**load_config(), "updates": 2, "batchSize": 2, "rolloutDecisions": 4,
              "minibatchSize": 4, "epochs": 2}
    with SnowGymBatchClient() as client:
        for name in ("full", "partial", "resumed"):
            (tmp_path / name).mkdir()
        full, report = train_run(source, metadata, client, config, tmp_path / "full", 94001)
        assert report["actorParameterL2Change"] > 0 and report["sourceUnchanged"]
        assert all(row["optimizerSteps"] > 0 for row in report["trace"])
        _, partial = train_run(source, metadata, client, config, tmp_path / "partial", 94001, pause_after=2)
        assert partial["paused"]
        resumed, resumed_report = train_run(source, metadata, client, config, tmp_path / "resumed", 94001,
                                           resume=tmp_path / "partial/partial")
        assert report["trace"] == resumed_report["trace"]
        assert semantic_state_digest(full.state_dict()) == semantic_state_digest(resumed.state_dict())
        left = load_movement(tmp_path / "full/final")[2]
        right = load_movement(tmp_path / "resumed/final")[2]
        assert left["stateDigest"] == right["stateDigest"]
        assert semantic_state_digest(source.state_dict()) == original
        assert not left["autonomousQualificationEligible"]
        with pytest.raises(ValueError, match="identity mismatch"):
            train_run(source, metadata, client, {**config, "latentStd": .03}, tmp_path / "bad", 94001,
                      resume=tmp_path / "partial/partial")


def test_assisted_gates_are_paired_and_never_autonomous():
    baseline = [{"seed": i, "success": False, "progress": 0., "physicalWin": False,
                 "rejectedActions": 0, "totalActions": 100} for i in range(40)]
    good = [{**r, "success": i < 20, "progress": .8} for i, r in enumerate(baseline)]
    config = load_config()
    assert result_gate(good, baseline, 1., config)["passed"]
    assert not result_gate(good, baseline, 1., config)["autonomousQualificationEligible"]
    assert not result_gate(good, baseline, 0., config)["passed"]
    assert not result_gate(baseline, baseline, 1., config)["passed"]
    bad = [{**r, "rejectedActions": 1} for r in good]
    assert not result_gate(bad, baseline, 1., config)["passed"]
    with pytest.raises(ValueError, match="unpaired"):
        result_gate(good[::-1], baseline, 1., config)
