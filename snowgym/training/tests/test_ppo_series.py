from __future__ import annotations

import json

from snowgym_training.model import ModelConfig
from snowgym_training.ppo import PPOConfig
from snowgym_training.ppo_series import PPO_SERIES_FORMAT, run_ppo_series


def test_ppo_series_keeps_and_evaluates_every_checkpoint(tmp_path) -> None:
    curriculum = {
        "format": "snowgym.ppo-curriculum.v0",
        "name": "series-test",
        "trainingSeedRanges": {"1v1-random": [100, 199]},
        "gates": [{
            "id": "1v1-random",
            "evaluationSeeds": [7],
            "scenario": {"blueUnits": 1, "redUnits": 1, "arenaWidth": 40, "arenaHeight": 30, "maxTicks": 120, "decisionHz": 10, "redDifficulty": "normal", "redController": "random"},
            "minimumWinRate": 0.75,
            "minimumImprovementOverMaskedRandom": 0.5,
        }],
    }
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum))
    result = run_ppo_series(
        output=tmp_path / "series",
        checkpoints=[1, 2],
        curriculum_path=curriculum_path,
        worlds=2,
        rollout_steps=2,
        max_decisions=2,
        model_config=ModelConfig(16, 12, 24),
        ppo_config=PPOConfig(update_epochs=1, minibatch_size=4),
        git_commit="test",
    )
    assert result["format"] == PPO_SERIES_FORMAT
    assert result["mode"] == "development"
    assert len(result["learningCurve"]) == 2
    assert [item["update"] for item in result["checkpoints"]] == [1, 2]
    for item in result["checkpoints"]:
        assert (tmp_path / "series" / item["runPath"] / "checkpoint" / "state.pt").is_file()
        assert (tmp_path / "series" / item["evaluationPath"]).is_file()
    assert json.loads((tmp_path / "series" / "manifest.json").read_text()) == result


def test_ppo_series_rejects_ambiguous_checkpoint_order(tmp_path) -> None:
    try:
        run_ppo_series(output=tmp_path / "bad", checkpoints=[2, 1])
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("PPO series accepted ambiguous checkpoint order")
