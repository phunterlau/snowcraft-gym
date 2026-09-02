from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from snowgym_training.checkpoint import load_checkpoint
from snowgym_training.data import TrajectoryDataset
from snowgym_training.demo import run_learned_demo
from snowgym_training.evaluate import EVALUATION_FORMAT, run_checkpoint_evaluation
from snowgym_training.loss import LossConfig, behavior_clone_loss
from snowgym_training.model import (
    EntityPolicy,
    ModelConfig,
    living_enemy_mask,
    model_config,
    nearest_enemy_target,
)
from snowgym_training.policy import TorchPolicy
from snowgym_training.trainer import TRAINING_CONFIG_FORMAT, train_behavior_clone
from snowgym_training.trajectory import (
    EXPORT_SPEC_FORMAT,
    TrajectoryWriter,
    json_digest,
)


def test_model_masks_outputs_and_has_finite_gradients(tmp_path: Path) -> None:
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, action = dataset.batch(np.asarray([0, 1]))
    model = EntityPolicy(ModelConfig(16, 12, 24))
    prediction = model(observation)

    assert prediction["action_logits"].shape == (2, 2, 4)
    assert torch.all(prediction["target"].abs() <= 1)
    assert torch.all((prediction["power"] >= 0) & (prediction["power"] <= 1))
    assert prediction["action_logits"][0, 0, 2] == torch.finfo(torch.float32).min
    assert prediction["action_logits"][0, 1].argmax() == 0
    loss = behavior_clone_loss(prediction, action, observation, LossConfig())
    loss["total"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pairwise_enemy_attention_is_masked_and_permutation_invariant(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, _ = dataset.batch(np.asarray([0, 1]))
    model = EntityPolicy(ModelConfig(16, 12, 24, pairwise_enemy_attention=True))
    original = model(observation)

    permuted = {name: value.clone() for name, value in observation.items()}
    order = torch.tensor([1, 0])
    permuted["enemies"] = permuted["enemies"][:, order]
    permuted["enemy_mask"] = permuted["enemy_mask"][:, order]
    reordered = model(permuted)

    assert torch.isfinite(original["hidden"]).all()
    assert torch.allclose(original["action_logits"], reordered["action_logits"])
    assert torch.allclose(original["target"], reordered["target"])
    assert torch.allclose(original["power"], reordered["power"])


def test_model_config_preserves_legacy_checkpoint_shape() -> None:
    legacy = {"entity_hidden": 16, "entity_embedding": 12, "actor_hidden": 24}
    assert model_config(legacy).as_dict() == legacy
    assert model_config({**legacy, "pairwise_enemy_attention": True}).as_dict() == {
        **legacy,
        "pairwise_enemy_attention": True,
    }
    assert model_config({**legacy, "action_conditioned_targets": True}).as_dict() == {
        **legacy,
        "action_conditioned_targets": True,
    }
    assert model_config({**legacy, "nearest_enemy_features": True}).as_dict() == {
        **legacy,
        "nearest_enemy_features": True,
    }
    assert model_config(
        {**legacy, "action_conditioned_targets": True, "last_enemy_move_target": True}
    ).as_dict() == {
        **legacy,
        "action_conditioned_targets": True,
        "last_enemy_move_target": True,
    }


def test_action_conditioned_target_heads_select_distinct_means(tmp_path: Path) -> None:
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, action = dataset.batch(np.asarray([0, 1]))
    model = EntityPolicy(
        ModelConfig(16, 12, 24, action_conditioned_targets=True)
    )
    prediction = model(observation)

    assert prediction["target_by_action"].shape == (2, 2, 4, 2)
    assert not torch.equal(
        prediction["target_by_action"][..., 1, :],
        prediction["target_by_action"][..., 2, :],
    )
    loss = behavior_clone_loss(prediction, action, observation, LossConfig())
    loss["total"].backward()
    assert model.move_target_head.weight.grad is not None
    assert model.throw_target_head.weight.grad is not None


def test_nearest_enemy_throw_prior_uses_live_relative_geometry() -> None:
    allies = torch.tensor([[[0.0, 0.0], [0.8, 0.8]]])
    enemies = torch.tensor([[[0.2, 0.1], [0.9, 0.7], [-0.9, -0.9]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.int8)
    target = nearest_enemy_target(allies, enemies, mask.bool())
    torch.testing.assert_close(target, torch.tensor([[[0.2, 0.1], [0.9, 0.7]]]))

    no_enemies = nearest_enemy_target(allies, enemies, torch.zeros_like(mask).bool())
    torch.testing.assert_close(no_enemies, torch.zeros_like(allies))


def test_relational_targets_ignore_defeated_roster_slots(tmp_path: Path) -> None:
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, _ = dataset.batch(np.asarray([0]))
    observation["enemy_mask"][0] = torch.tensor([1, 1])
    observation["enemies"][0, 0, 1] = 0.0
    observation["enemies"][0, 0, 2:4] = torch.tensor([0.1, 0.1])
    observation["enemies"][0, 1, 1] = 1.0
    observation["enemies"][0, 1, 2:4] = torch.tensor([0.8, -0.6])
    observation["team_alive"][0, 1] = 1
    model = EntityPolicy(
        ModelConfig(
            16,
            12,
            24,
            action_conditioned_targets=True,
            last_enemy_move_target=True,
            nearest_enemy_throw_target=True,
        )
    )

    assert living_enemy_mask(observation).tolist() == [[False, True]]
    prediction = model(observation)
    expected = torch.tensor([0.8, -0.6]).expand(2, -1)
    torch.testing.assert_close(prediction["target_by_action"][0, :, 1], expected)
    torch.testing.assert_close(prediction["target_by_action"][0, :, 2], expected)


def test_nearest_enemy_throw_prior_requires_conditioned_targets() -> None:
    legacy = {"entity_hidden": 16, "entity_embedding": 12, "actor_hidden": 24}
    with pytest.raises(ValueError, match="requires action_conditioned_targets"):
        model_config({**legacy, "nearest_enemy_throw_target": True})
    with pytest.raises(ValueError, match="requires action_conditioned_targets"):
        model_config({**legacy, "last_enemy_move_target": True})


def test_one_batch_overfit_reduces_hybrid_loss(tmp_path: Path) -> None:
    torch.manual_seed(9)
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, action = dataset.batch(np.arange(len(dataset)))
    model = EntityPolicy(ModelConfig(16, 12, 24))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = float(
        behavior_clone_loss(model(observation), action, observation, LossConfig())[
            "total"
        ].detach()
    )
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = behavior_clone_loss(model(observation), action, observation, LossConfig())[
            "total"
        ]
        loss.backward()
        optimizer.step()
    final = float(
        behavior_clone_loss(model(observation), action, observation, LossConfig())[
            "total"
        ].detach()
    )
    assert final < initial * 0.1


def test_throw_class_weight_increases_rare_throw_penalty(tmp_path: Path) -> None:
    dataset = TrajectoryDataset(make_dataset(tmp_path / "dataset"))
    observation, action = dataset.batch(np.asarray([0, 1]))
    model = EntityPolicy(ModelConfig(16, 12, 24))
    prediction = model(observation)
    action["action_type"][0, 0] = 1
    action["action_type"][1, 0] = 2
    with torch.no_grad():
        prediction["action_logits"][:, 0] = torch.tensor(
            [[0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]
        )
    ordinary = behavior_clone_loss(
        prediction, action, observation, LossConfig()
    )["action"]
    weighted = behavior_clone_loss(
        prediction,
        action,
        observation,
        LossConfig(throw_action_weight=10.0),
    )["action"]
    assert weighted > ordinary


def test_checkpoint_resume_is_exact_and_policy_respects_masks(tmp_path: Path) -> None:
    dataset_path = make_dataset(tmp_path / "dataset")
    config = training_config(steps=8)
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    resumed = tmp_path / "resumed"
    train_behavior_clone(
        dataset_path=dataset_path,
        output=full,
        config=config,
        git_commit="test-commit",
    )
    train_behavior_clone(
        dataset_path=dataset_path,
        output=partial,
        config=config,
        target_step=3,
        git_commit="test-commit",
    )
    train_behavior_clone(
        dataset_path=dataset_path,
        output=resumed,
        config=config,
        resume=partial,
        git_commit="test-commit",
    )
    full_metadata, full_state = load_checkpoint(full)
    resumed_metadata, resumed_state = load_checkpoint(resumed)
    assert full_metadata["stateDigest"] == resumed_metadata["stateDigest"]
    for name, tensor in full_state["model"].items():
        assert torch.equal(tensor, resumed_state["model"][name])

    observation, _ = TrajectoryDataset(dataset_path).batch(np.asarray([0]))
    numpy_observation = {name: tensor[0].numpy() for name, tensor in observation.items()}
    action = TorchPolicy(full).act(numpy_observation)
    selected = numpy_observation["unit_action_mask"][
        np.arange(2), action["action_type"]
    ]
    assert selected[numpy_observation["ally_mask"].astype(bool)].tolist() == [1]
    assert action["action_type"][1] == 0


def test_checkpoint_detects_state_corruption(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    train_behavior_clone(
        dataset_path=make_dataset(tmp_path / "dataset"),
        output=checkpoint,
        config=training_config(steps=2),
        git_commit="test-commit",
    )
    state = torch.load(checkpoint / "state.pt", weights_only=True)
    first = next(iter(state["model"]))
    state["model"][first] = state["model"][first] + 1
    torch.save(state, checkpoint / "state.pt")
    with pytest.raises(ValueError, match="state digest mismatch"):
        load_checkpoint(checkpoint)


def test_committed_checkpoint_evaluation_and_replays_are_consistent() -> None:
    training_root = Path(__file__).parents[1]
    metadata, _ = load_checkpoint(training_root / "checkpoints" / "bc_1v1_v0")
    evaluation = json.loads(
        (training_root / "evaluations" / "bc_1v1_v0.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = evaluation.pop("resultDigest")
    assert claimed == json_digest(evaluation)
    assert evaluation["checkpointDigest"] == metadata["checkpointDigest"]
    assert evaluation["summary"]["learned"]["blueWins"] == 2
    assert evaluation["summary"]["masked_random"]["blueWins"] == 0
    assert evaluation["summary"]["learned"]["rejectedActions"] == 0
    for replay_name in evaluation["replays"]:
        replay_path = (
            training_root.parents[1]
            / "public"
            / "replays"
            / "bc_1v1_v0"
            / replay_name
        )
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        assert replay["format"] == "snowgym.replay.v0"
        assert replay["outcome"]["winner"] == "blue"
        assert len(replay["stateHashes"]) == len(replay["frames"])


def test_learned_evaluation_records_normal_replay(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    train_behavior_clone(
        dataset_path=make_dataset(tmp_path / "dataset"),
        output=checkpoint,
        config=training_config(steps=4),
        git_commit="test-commit",
    )
    spec = evaluation_spec()
    spec_path = tmp_path / "teacher_1v1_v0.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    baseline = {
        "format": "snowgym.teacher-baseline.v0",
        "sourceSpecDigest": json_digest(spec),
        "split": "evaluation",
        "maxDecisions": 4,
        "summary": {
            "scripted_teacher": baseline_summary(1),
            "masked_random": baseline_summary(0),
        },
    }
    baseline["resultDigest"] = json_digest(baseline)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    result = run_checkpoint_evaluation(
        checkpoint=checkpoint,
        spec_path=spec_path,
        client=FakeLearningClient(),
        baseline_path=baseline_path,
        replay_directory=tmp_path / "replays",
        max_decisions=4,
    )
    assert result["format"] == EVALUATION_FORMAT
    assert result["summary"]["learned"]["blueWins"] == 1
    assert result["summary"]["learned"]["rejectedActions"] == 0
    replay = json.loads(
        (tmp_path / "replays" / result["replays"][0]).read_text(encoding="utf-8")
    )
    assert replay["format"] == "snowgym.replay.v0"
    assert len(replay["frames"]) == 3


def test_learned_demo_records_checkpoint_provenance(tmp_path: Path) -> None:
    result = run_learned_demo(
        output=tmp_path / "learned-demo.json",
        checkpoint=Path(__file__).parents[1] / "checkpoints" / "bc_1v1_v0",
        seed=42,
        client=FakeLearningClient(),
    )
    replay = json.loads((tmp_path / "learned-demo.json").read_text(encoding="utf-8"))
    assert result["winner"] == "blue"
    assert result["rejectedActions"] == 0
    assert result["checkpointDigest"].startswith("sha256:")
    assert replay["format"] == "snowgym.replay.v0"
    assert replay["outcome"]["winner"] == "blue"


def make_dataset(path: Path) -> Path:
    writer = TrajectoryWriter(path, shard_size=4)
    for index in range(8):
        observation = observation_tensors(index)
        writer.add(
            {
                **{f"observation__{name}": value for name, value in observation.items()},
                "action__action_type": np.asarray([1, 0], dtype=np.int64),
                "action__target": np.asarray([[0.4, -0.2], [0, 0]], dtype=np.float32),
                "action__power": np.zeros(2, dtype=np.float32),
                "reward": np.asarray(0, dtype=np.float32),
                "terminated": np.asarray(index == 7, dtype=np.bool_),
                "truncated": np.asarray(False, dtype=np.bool_),
                "seed": np.asarray(11, dtype=np.int64),
                "episode_index": np.asarray(0, dtype=np.int32),
                "tick": np.asarray(index, dtype=np.int64),
                "next_tick": np.asarray(index + 1, dtype=np.int64),
                "pre_state_hash": np.asarray(state_hash(index)),
                "post_state_hash": np.asarray(state_hash(index + 1)),
                "teacher_accepted": np.ones(2, dtype=np.bool_),
                "teacher_reason": np.zeros(2, dtype=np.int8),
            }
        )
    writer.finish(
        {
            "name": "learning-test-v0",
            "teacher": "test-teacher",
            "split": "train",
            "splitSeeds": {"train": [11], "validation": [101], "evaluation": [201]},
            "sourceSpecDigest": "sha256:test",
            "maxTeamUnits": 2,
            "versions": versions(),
            "episodes": [
                {
                    "index": 0,
                    "seed": 11,
                    "transitions": 8,
                    "finalStateHash": state_hash(8),
                }
            ],
        }
    )
    return path


def observation_tensors(tick: int) -> dict[str, np.ndarray]:
    allies = np.zeros((2, 10), dtype=np.float32)
    enemies = np.zeros((2, 10), dtype=np.float32)
    allies[0, [0, 1, 2, 6]] = [1, 1, -0.5 + tick * 0.01, 1]
    enemies[0, [0, 1, 2, 6]] = [1, 1, 0.5, 1]
    action_mask = np.zeros((2, 4), dtype=np.int8)
    action_mask[0] = [1, 1, 0, 1]
    return {
        "allies": allies,
        "ally_mask": np.asarray([1, 0], dtype=np.int8),
        "enemies": enemies,
        "enemy_mask": np.asarray([1, 0], dtype=np.int8),
        "projectiles": np.zeros((64, 8), dtype=np.float32),
        "projectile_mask": np.zeros(64, dtype=np.int8),
        "unit_action_mask": action_mask,
        "tick": np.asarray([tick], dtype=np.int64),
        "team_alive": np.asarray([1, 1], dtype=np.int32),
        "obstacles": np.zeros((64, 9), dtype=np.float32),
        "obstacle_mask": np.zeros(64, dtype=np.int8),
    }


def training_config(*, steps: int) -> dict[str, Any]:
    return {
        "format": TRAINING_CONFIG_FORMAT,
        "name": "test-bc-v0",
        "seed": 17,
        "steps": steps,
        "batchSize": 4,
        "learningRate": 0.003,
        "architecture": {
            "entity_hidden": 16,
            "entity_embedding": 12,
            "actor_hidden": 24,
        },
        "loss": {
            "action_weight": 1.0,
            "target_weight": 1.0,
            "power_weight": 0.25,
        },
        "evaluationSuite": "teacher_1v1_v0/evaluation",
    }


def versions() -> dict[str, str]:
    return {
        "apiVersion": "snowgym.v0",
        "simulationVersion": "snowgym.sim.v1",
        "stateHashVersion": "snowgym.state.v1",
        "upstreamBaseCommit": "test",
    }


def evaluation_spec() -> dict[str, Any]:
    scenario = scenario_config()
    return {
        "format": EXPORT_SPEC_FORMAT,
        "name": "test-evaluation-v0",
        "teacher": "test-teacher",
        "maxTeamUnits": 2,
        "shardSize": 4,
        "splits": {
            "train": [{"seed": 11, "scenario": scenario}],
            "validation": [{"seed": 101, "scenario": scenario}],
            "evaluation": [{"seed": 201, "scenario": scenario}],
        },
    }


def scenario_config() -> dict[str, Any]:
    return {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 12,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }


def baseline_summary(wins: int) -> dict[str, Any]:
    return {
        "episodes": 1,
        "blueWins": wins,
        "redWins": 0,
        "draws": 1 - wins,
        "meanDecisions": 2,
        "rejectedActions": 0,
    }


class FakeLearningClient:
    def __init__(self) -> None:
        self.tick = 0
        self.seed = 0
        self.scenario: dict[str, Any] = {}

    def reset(
        self,
        seed: int,
        scenario: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del idempotency_key
        self.tick = 0
        self.seed = seed
        self.scenario = dict(scenario or scenario_config())
        return fake_payload(self.seed, self.tick, self.scenario)

    def step(
        self,
        action: dict[str, Any],
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del idempotency_key
        assert expected_state_hash == state_hash(self.tick)
        self.tick += 6
        payload = fake_payload(self.seed, self.tick, self.scenario)
        terminated = self.tick >= 12
        info = payload["status"] | {
            "terminated": terminated,
            "winner": "blue" if terminated else None,
            "redAlive": 0 if terminated else 1,
            "action": action,
            "actionResults": [
                {"action": item, "accepted": True} for item in action["actions"]
            ],
        }
        return {
            "observation": payload["observation"],
            "reward": 1.0 if terminated else 0.0,
            "terminated": terminated,
            "truncated": False,
            "info": info,
        }


def fake_payload(seed: int, tick: int, scenario: dict[str, Any]) -> dict[str, Any]:
    red_alive = tick < 12
    raw = {
        "tick": tick,
        "selfTeam": "blue",
        "simulationHz": 60,
        "arena": {"width": 40, "height": 30},
        "allies": [raw_unit(1, "blue", -10, 1.0)],
        "enemies": [raw_unit(2, "red", 10, 1.0 if red_alive else 0.0, red_alive)],
        "projectiles": [],
        "obstacles": [],
        "match": {"blueAlive": 1, "redAlive": int(red_alive)},
    }
    status = {
        **versions(),
        "stateHash": state_hash(tick),
        "scenario": "test-1v1",
        "seed": seed,
        "tick": tick,
        "simulationHz": 60,
        "decisionHz": 10,
        "ticksPerDecision": 6,
        "configuration": scenario,
        "blueAlive": 1,
        "redAlive": int(red_alive),
        "terminated": tick >= 12,
        "truncated": False,
        "winner": "blue" if tick >= 12 else None,
    }
    return {"observation": raw, "status": status}


def raw_unit(
    unit_id: int, team: str, x: float, health: float, alive: bool = True
) -> dict[str, Any]:
    return {
        "id": unit_id,
        "team": team,
        "x": x,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "health": health * 100,
        "maxHealth": 100.0,
        "throwCooldown": 0.0,
        "charge": 0.0,
        "state": "idle" if alive else "defeated",
        "alive": alive,
    }


def state_hash(tick: int) -> str:
    return f"fnv1a64:{tick:016x}"
