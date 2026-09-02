from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from snowgym_training.baseline import BASELINE_FORMAT, run_teacher_baseline
from snowgym_training.export_dagger import export_dagger_dataset
from snowgym_training.merge_trajectory import merge_datasets
from snowgym_training.export_scripted import (
    action_results,
    assert_action_round_trip,
    export_scripted_dataset,
)
from snowgym_training.trajectory import (
    DATASET_FORMAT,
    EXPORT_SPEC_FORMAT,
    audit_dataset,
    json_digest,
    tensor_digest,
    validate_export_spec,
)


class FakeScriptedClient:
    def __init__(self, *, reject: bool = False) -> None:
        self.seed = 0
        self.tick = 0
        self.scenario: dict[str, Any] = {}
        self.reject = reject

    def teacher_action(self) -> dict[str, Any]:
        value = snapshot(self.seed, self.tick, self.scenario)
        return {
            "status": value["status"],
            "action": {
                "actions": [
                    {"type": "move", "unitId": index + 1, "x": 5.0, "y": -2.0}
                    for index in range(int(self.scenario["blueUnits"]))
                ]
            },
        }

    def reset(
        self,
        seed: int,
        scenario: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del idempotency_key
        self.seed = seed
        self.tick = 0
        self.scenario = dict(scenario or {})
        return snapshot(self.seed, self.tick, self.scenario)

    def step_scripted(
        self,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del idempotency_key
        assert expected_state_hash == state_hash(self.tick)
        actions = {
            "actions": [
                {"type": "move", "unitId": index + 1, "x": 5.0, "y": -2.0}
                for index in range(int(self.scenario["blueUnits"]))
            ]
        }
        return self._advance(actions)

    def step(
        self,
        action: dict[str, Any],
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del idempotency_key
        assert expected_state_hash == state_hash(self.tick)
        return self._advance(action)

    def _advance(self, action: dict[str, Any]) -> dict[str, Any]:
        results = [
            {
                "action": item,
                "accepted": not self.reject,
                **({"reason": "unavailable"} if self.reject else {}),
            }
            for item in action["actions"]
        ]
        self.tick += 6
        value = snapshot(self.seed, self.tick, self.scenario)
        truncated = self.tick >= 18
        info = value["status"] | {
            "truncated": truncated,
            "action": action,
            "actionResults": results,
        }
        return {
            "observation": value["observation"],
            "reward": 0.0,
            "terminated": False,
            "truncated": truncated,
            "info": info,
        }


def test_action_round_trip_allows_only_extra_explicit_noops() -> None:
    original = {"actions": [{"type": "move", "unitId": 1, "x": 2.0, "y": 3.0}]}
    encoded = {
        "actions": [
            {"type": "move", "unitId": 1, "x": 2.0, "y": 3.0},
            {"type": "noop", "unitId": 2},
        ]
    }
    assert_action_round_trip(original, encoded)

    encoded["actions"][1] = {"type": "hold", "unitId": 2}
    with pytest.raises(ValueError, match="state-changing"):
        assert_action_round_trip(original, encoded)


def test_export_spec_rejects_seed_overlap() -> None:
    spec = export_spec()
    spec["splits"]["evaluation"][0]["seed"] = 11

    with pytest.raises(ValueError, match="overlaps splits"):
        validate_export_spec(spec)


def test_scripted_export_is_exactly_reproducible_and_auditable(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")

    first = export_scripted_dataset(
        output=tmp_path / "first",
        split="train",
        spec_path=spec_path,
        client=FakeScriptedClient(),
    )
    second = export_scripted_dataset(
        output=tmp_path / "second",
        split="train",
        spec_path=spec_path,
        client=FakeScriptedClient(),
    )

    assert first == second
    assert first["format"] == DATASET_FORMAT
    assert first["transitions"] == 6
    assert len(first["shards"]) == 2
    assert audit_dataset(tmp_path / "first")["datasetDigest"] == first["datasetDigest"]
    with np.load(tmp_path / "first" / "shard-00000.npz", allow_pickle=False) as left:
        with np.load(tmp_path / "second" / "shard-00000.npz", allow_pickle=False) as right:
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])


def test_dagger_export_labels_states_visited_by_committed_policy(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")
    training_root = Path(__file__).resolve().parents[1]
    checkpoint = training_root / "checkpoints/bc_1v1_v0"

    result = export_dagger_dataset(
        output=tmp_path / "dagger",
        checkpoint=checkpoint,
        split="train",
        spec_path=spec_path,
        client=FakeScriptedClient(),
    )

    assert result["transitions"] == 6
    assert result["teacher"] == "teacher-action.v0"
    assert result["rolloutPolicy"] == "learned-checkpoint"
    assert result["rolloutCheckpointDigest"].startswith("sha256:")
    assert audit_dataset(tmp_path / "dagger")["datasetDigest"] == result["datasetDigest"]


def test_merge_datasets_preserves_ordered_source_provenance(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_scripted_dataset(
        output=first,
        split="train",
        spec_path=spec_path,
        client=FakeScriptedClient(),
    )
    export_scripted_dataset(
        output=second,
        split="train",
        spec_path=spec_path,
        client=FakeScriptedClient(),
    )

    result = merge_datasets(output=tmp_path / "merged", inputs=[first, second, first])

    assert result["transitions"] == 18
    assert [source["datasetDigest"] for source in result["sources"]] == [
        result["sources"][0]["datasetDigest"],
        result["sources"][1]["datasetDigest"],
        result["sources"][0]["datasetDigest"],
    ]
    assert all("path" not in source for source in result["sources"])
    assert [episode["index"] for episode in result["episodes"]] == list(range(6))
    assert audit_dataset(tmp_path / "merged")["datasetDigest"] == result["datasetDigest"]


def test_dataset_audit_detects_tensor_corruption(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")
    export_scripted_dataset(
        output=tmp_path / "dataset",
        split="train",
        spec_path=spec_path,
        limit_episodes=1,
        client=FakeScriptedClient(),
    )
    shard = tmp_path / "dataset" / "shard-00000.npz"
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["reward"] = arrays["reward"].copy()
    arrays["reward"][0] = 1.0
    np.savez_compressed(shard, **arrays)

    with pytest.raises(ValueError, match="tensor digest mismatch"):
        audit_dataset(tmp_path / "dataset")


def test_export_rejects_teacher_action_before_qualifying_dataset(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")

    with pytest.raises(ValueError, match="rejected action"):
        export_scripted_dataset(
            output=tmp_path / "rejected",
            split="train",
            spec_path=spec_path,
            limit_episodes=1,
            client=FakeScriptedClient(reject=True),
        )


def test_action_results_allow_omitted_dead_roster_slots() -> None:
    raw = snapshot(
        1,
        0,
        {"blueUnits": 2, "redUnits": 1, "arenaWidth": 40, "arenaHeight": 30},
    )["observation"]
    raw["allies"][1]["alive"] = False
    results = [
        {
            "action": {"type": "noop", "unitId": 1},
            "accepted": True,
        }
    ]

    accepted, reasons = action_results(results, raw, capacity=3)
    assert accepted.tolist() == [True, True, True]
    assert reasons.tolist() == [0, 0, 0]


def test_teacher_baseline_is_deterministic(tmp_path: Path) -> None:
    spec_path = tmp_path / "fixture-spec.json"
    spec_path.write_text(json.dumps(export_spec()), encoding="utf-8")
    first = run_teacher_baseline(
        spec_path=spec_path,
        split="evaluation",
        client=FakeScriptedClient(),
    )
    second = run_teacher_baseline(
        spec_path=spec_path,
        split="evaluation",
        client=FakeScriptedClient(),
    )

    assert first == second
    assert first["format"] == BASELINE_FORMAT
    assert first["summary"]["scripted_teacher"]["episodes"] == 1
    assert first["summary"]["masked_random"]["episodes"] == 1


def test_tensor_digest_includes_name_dtype_shape_and_bytes() -> None:
    array = np.asarray([[1, 2]], dtype=np.int32)

    assert tensor_digest({"a": array}) == tensor_digest({"a": array.copy()})
    assert tensor_digest({"a": array}) != tensor_digest({"b": array})
    assert tensor_digest({"a": array}) != tensor_digest({"a": array.astype(np.int64)})


def test_committed_teacher_baseline_has_valid_digest_and_zero_rejections() -> None:
    path = Path(__file__).parents[1] / "baselines" / "teacher_1v1_v0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("resultDigest")

    assert claimed == json_digest(value)
    assert value["summary"]["scripted_teacher"]["blueWins"] == 2
    assert value["summary"]["scripted_teacher"]["rejectedActions"] == 0
    assert value["summary"]["masked_random"]["blueWins"] == 0


def export_spec() -> dict[str, Any]:
    scenario = {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 18,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }
    return {
        "format": EXPORT_SPEC_FORMAT,
        "name": "test-v0",
        "teacher": "simple-blue.v0",
        "maxTeamUnits": 3,
        "shardSize": 4,
        "splits": {
            "train": [
                {"seed": 11, "scenario": scenario},
                {"seed": 12, "scenario": scenario},
            ],
            "validation": [{"seed": 101, "scenario": scenario}],
            "evaluation": [{"seed": 201, "scenario": scenario}],
        },
    }


def snapshot(seed: int, tick: int, scenario: dict[str, Any]) -> dict[str, Any]:
    blue_units = int(scenario.get("blueUnits", 1))
    red_units = int(scenario.get("redUnits", 1))
    width = float(scenario.get("arenaWidth", 40))
    height = float(scenario.get("arenaHeight", 30))
    allies = [unit(index + 1, "blue", -10.0, float(index)) for index in range(blue_units)]
    enemies = [
        unit(blue_units + index + 1, "red", 10.0, float(index))
        for index in range(red_units)
    ]
    observation = {
        "tick": tick,
        "selfTeam": "blue",
        "simulationHz": 60,
        "arena": {"width": width, "height": height},
        "allies": allies,
        "enemies": enemies,
        "projectiles": [],
        "obstacles": [],
        "match": {"blueAlive": blue_units, "redAlive": red_units},
    }
    status = {
        "apiVersion": "snowgym.v0",
        "simulationVersion": "snowgym.sim.v1",
        "stateHashVersion": "snowgym.state.v1",
        "upstreamBaseCommit": "test",
        "stateHash": state_hash(tick),
        "scenario": "test-1v1",
        "seed": seed,
        "tick": tick,
        "simulationHz": 60,
        "decisionHz": 10,
        "ticksPerDecision": 6,
        "configuration": scenario,
        "blueAlive": blue_units,
        "redAlive": red_units,
        "terminated": False,
        "truncated": tick >= 18,
        "winner": None,
    }
    return {"status": status, "observation": observation}


def unit(unit_id: int, team: str, x: float, y: float) -> dict[str, Any]:
    return {
        "id": unit_id,
        "team": team,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "health": 100.0,
        "maxHealth": 100.0,
        "throwCooldown": 0.0,
        "charge": 0.0,
        "state": "idle",
        "alive": True,
    }


def state_hash(tick: int) -> str:
    return f"fnv1a64:{tick:016x}"
