from __future__ import annotations

import json

from snowgym_training.model import ModelConfig
from snowgym_training.ppo import PPOConfig
from snowgym_training.ppo_evaluate import PPO_EVALUATION_FORMAT, evaluate_ppo_checkpoint
from snowgym_training.ppo_train import train_ppo


def test_ppo_evaluation_uses_frozen_seeds_and_canonical_returns(tmp_path) -> None:
    curriculum = {
        "format": "snowgym.ppo-curriculum.v0",
        "name": "evaluation-test",
        "trainingSeedRanges": {"1v1-random": [100, 199]},
        "gates": [
            {
                "id": "1v1-random",
                "evaluationSeeds": [7, 8],
                "scenario": {
                    "blueUnits": 1,
                    "redUnits": 1,
                    "arenaWidth": 40,
                    "arenaHeight": 30,
                    "maxTicks": 120,
                    "decisionHz": 10,
                    "redDifficulty": "normal",
                    "redController": "random",
                },
                "minimumWinRate": 0.75,
                "minimumImprovementOverMaskedRandom": 0.5,
            }
        ],
    }
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum))
    train_ppo(
        output=tmp_path / "run",
        curriculum_path=curriculum_path,
        worlds=2,
        rollout_steps=2,
        target_updates=1,
        model_config=ModelConfig(16, 12, 24),
        ppo_config=PPOConfig(update_epochs=1, minibatch_size=4),
        git_commit="test",
    )
    result = evaluate_ppo_checkpoint(
        checkpoint=tmp_path / "run" / "checkpoint",
        curriculum_path=curriculum_path,
        max_decisions=2,
    )

    assert result["format"] == PPO_EVALUATION_FORMAT
    assert len(result["results"]) == 6
    assert {item["seed"] for item in result["results"]} == {7, 8}
    assert {item["policy"] for item in result["results"]} == {
        "ppo",
        "masked_random",
        "scripted_teacher",
    }
    assert {item["canonicalReturn"] for item in result["results"]} <= {-1.0, 0.0, 1.0}
    assert all(summary["episodes"] == 2 for summary in result["summary"].values())
    assert isinstance(result["threshold"]["passed"], bool)
