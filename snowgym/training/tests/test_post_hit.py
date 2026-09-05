import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options import post_hit as audit
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint


def test_windows_configuration_empty_coverage_and_refusal(tmp_path):
    assert audit.active_channel("both-30", 29) == "teacher-choice-move"
    assert audit.active_channel("both-30", 30) == "shot-only"
    assert audit.active_channel("choice-30", 0) == "teacher-choice"
    assert audit.active_channel("move-rest", 199) == "teacher-move"
    with pytest.raises(ValueError):
        audit.active_channel("unknown", 0)
    with pytest.raises(ValueError):
        audit.active_channel("keep", -1)
    with pytest.raises(FileExistsError):
        audit.run(tmp_path)
    assert not any(audit.report_effects([])["supportedForFurtherStudy"].values())
    assert audit.config()["autonomousQualificationEligible"] is False
    assert audit.config()["seeds"] == [200000, 200039]


def test_paired_interaction_and_support_are_episode_level():
    episodes = []
    for i in range(40):
        pair = {}
        for arm in audit.ARMS:
            metric = {"success": arm != "keep" and i < 20, "progress": float(arm != "keep"),
                      "damageDealt": 1, "damageReceived": 0, "livingFraction": 1}
            pair[arm] = {"local": metric, "final": metric}
        episodes.append(pair)
    result = audit.report_effects(episodes)
    assert all(result["supportedForFurtherStudy"].values())
    assert result["interaction"]["final"]["success"]["mean"] == -.5
    assert result == audit.report_effects(episodes)


def test_live_baseline_prefix_readonly_channels_and_exact_branches():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata, state = load_ppo_checkpoint(audit.REFERENCE)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    digest = semantic_state_digest(model.state_dict())
    seed = 200000
    with SnowGymBatchClient() as client:
        wrapper = audit.make_wrapper(client, 1, audit.config()["gamma"])
        baseline = audit.continuation(model, wrapper, audit.reset(wrapper, seed), seed, baseline=True)
        archived = json.loads((audit.TRAINING / "runs/m7b_engage_r1m_movement_v0/assisted-initialization.json").read_text())["historical"][0]
        assert baseline["stateHashes"] == archived["stateHashes"]
        assert baseline["actionsDigest"] == archived["actionsDigest"]
        trigger = baseline["trigger"]
        assert trigger and not trigger["terminal"]
        for key in ("physical", "plan", "option", "observation"):
            changed = {**trigger["identity"], key: "tampered"}
            with pytest.raises(ValueError, match="identity mismatch"):
                audit.restore(wrapper, seed, trigger["prefix"], changed)
        observation = audit.restore(wrapper, seed, trigger["prefix"], trigger["identity"])
        before = audit.identity(wrapper, observation)
        base, _ = audit.action_and_metrics(model, wrapper, observation, "shot-only")
        move, _ = audit.action_and_metrics(model, wrapper, observation, "teacher-move")
        assert audit.identity(wrapper, observation) == before
        np.testing.assert_array_equal(base["action_type"], move["action_type"])
        np.testing.assert_array_equal(base["power"], move["power"])
        not_move = base["action_type"] != 1
        np.testing.assert_array_equal(base["target"][not_move], move["target"][not_move])
        for arm in ("keep", "both-30", "move-rest"):
            runs = []
            for _ in range(2):
                obs = audit.restore(wrapper, seed, trigger["prefix"], trigger["identity"])
                runs.append(audit.continuation(model, wrapper, obs, seed, arm))
            assert runs[0] == runs[1]
            row = runs[0]
            assert row["final"]["exposureDecisions"] <= 200-trigger["decision"]
            assert row["local"]["exposureDecisions"] <= 30
            assert row["final"]["completionCensored"] == (row["final"]["completionDecisions"] is None)
            if arm == "keep":
                assert row["stateHashes"] == baseline["stateHashes"][trigger["decision"]:]
    assert semantic_state_digest(model.state_dict()) == digest
