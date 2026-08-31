from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

import snowgym_client
from snowgym_client.encoding import ACTION_MOVE, ACTION_NOOP, ACTION_THROW
from snowgym_client.env import SnowGymEnv
from snowgym_client.recording import REPLAY_FORMAT, ReplayRecorder, write_replay
from snowgym_client.state_hash import hash_observation


class FakeClient:
    def __init__(self) -> None:
        self.seed = 0
        self.tick = 0
        self.last_action: dict[str, Any] | None = None
        self.scenario: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        return make_snapshot(self.seed, self.tick)

    def reset(
        self,
        seed: int,
        scenario: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.seed = seed
        self.tick = 0
        self.last_action = None
        self.scenario = scenario
        return make_snapshot(seed, 0, scenario)

    def step(
        self,
        action: dict[str, Any],
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        assert expected_state_hash == make_snapshot(
            self.seed, self.tick, self.scenario
        )["status"]["stateHash"]
        self.last_action = action
        self.tick += 6
        snapshot = make_snapshot(self.seed, self.tick, self.scenario)
        return {
            "observation": snapshot["observation"],
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "info": snapshot["status"] | {"actionResults": [], "action": action},
        }

    def step_scripted(
        self,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.step(
            {"actions": []},
            expected_state_hash=expected_state_hash,
            idempotency_key=idempotency_key,
        )

    def autoplay(
        self,
        max_decisions: int,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return make_snapshot(self.seed, min(max_decisions, 1) * 6)


def test_registered_environment_passes_gymnasium_checker() -> None:
    environment = gym.make(
        snowgym_client.CONFIGURABLE_ENV_ID, client=FakeClient()
    ).unwrapped
    check_env(environment, skip_render_check=True)


def test_configurable_environment_has_fixed_capacity_and_presence_masks() -> None:
    client = FakeClient()
    environment = gym.make(
        snowgym_client.CONFIGURABLE_ENV_ID,
        client=client,
        blue_units=5,
        red_units=2,
    ).unwrapped
    observation, info = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)
    assert environment.action_space["action_type"].shape == (8,)
    assert observation["allies"].shape == (8, 10)
    assert observation["ally_mask"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert observation["enemy_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert info["configuration"]["blueUnits"] == 5
    assert client.scenario is not None
    assert client.scenario["redUnits"] == 2


def test_ten_unit_environment_supports_a_10v10_roster() -> None:
    client = FakeClient()
    environment = gym.make(
        snowgym_client.TEN_UNIT_ENV_ID,
        client=client,
        blue_units=10,
        red_units=10,
        arena_width=60,
        arena_height=50,
    ).unwrapped
    observation, info = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)
    assert environment.action_space["action_type"].shape == (10,)
    assert observation["allies"].shape == (10, 10)
    assert observation["enemies"].shape == (10, 10)
    assert observation["ally_mask"].tolist() == [1] * 10
    assert observation["enemy_mask"].tolist() == [1] * 10
    assert info["configuration"]["blueUnits"] == 10
    assert info["configuration"]["redUnits"] == 10


def test_configurable_reset_can_change_rosters_without_changing_spaces() -> None:
    environment = gym.make(
        snowgym_client.CONFIGURABLE_ENV_ID,
        client=FakeClient(),
    ).unwrapped
    first, _ = environment.reset(seed=1, options={"scenario": {"blueUnits": 1, "redUnits": 3}})
    second, _ = environment.reset(seed=1, options={"scenario": {"blueUnits": 8, "redUnits": 1}})

    assert first["allies"].shape == second["allies"].shape == (8, 10)
    assert int(first["ally_mask"].sum()) == 1
    assert int(second["ally_mask"].sum()) == 8


def test_red_controller_is_configurable_and_validated() -> None:
    client = FakeClient()
    environment = gym.make(
        snowgym_client.CONFIGURABLE_ENV_ID,
        client=client,
        red_controller="random",
    ).unwrapped
    environment.reset(seed=42)

    assert client.scenario is not None
    assert client.scenario["redController"] == "random"

    with pytest.raises(ValueError, match="redController"):
        gym.make(
            snowgym_client.CONFIGURABLE_ENV_ID,
            client=FakeClient(),
            red_controller="skynet",
        )

    with pytest.raises(ValueError, match="redController"):
        environment.reset(seed=1, options={"scenario": {"redController": "skynet"}})


def test_map_reset_loads_terrain_with_obstacle_tensor() -> None:
    client = FakeClient()
    environment = gym.make(
        snowgym_client.CONFIGURABLE_ENV_ID,
        client=client,
    ).unwrapped
    observation, info = environment.reset(seed=42, options={"scenario": {"map": "arena1.json"}})

    assert client.scenario is not None
    assert client.scenario["map"] == "arena1.json"
    # Map fixes terrain/native spawns; configurable tuning still reaches the server.
    assert "blueUnits" not in client.scenario
    assert client.scenario["decisionHz"] == 10
    assert client.scenario["redDifficulty"] == "normal"
    assert client.scenario["redController"] == "scripted"
    assert client.scenario["maxTicks"] == 60 * 180
    assert environment.observation_space.contains(observation)
    assert observation["obstacles"].shape == (64, 9)
    assert int(observation["obstacle_mask"].sum()) == 2
    assert info["configuration"]["map"] == "arena1.json"


def test_map_reset_can_select_a_smaller_native_roster() -> None:
    client = FakeClient()
    environment = gym.make(
        snowgym_client.TEN_UNIT_ENV_ID,
        client=client,
        map="arena6.json",
        blue_units=5,
        red_units=2,
    ).unwrapped
    observation, info = environment.reset(seed=42)

    assert client.scenario is not None
    assert client.scenario["map"] == "arena6.json"
    assert client.scenario["blueUnits"] == 5
    assert client.scenario["redUnits"] == 2
    assert int(observation["ally_mask"].sum()) == 5
    assert int(observation["enemy_mask"].sum()) == 2
    assert info["configuration"]["map"] == "arena6.json"


def test_open_arena_has_empty_obstacle_tensor() -> None:
    environment = SnowGymEnv(client=FakeClient())
    observation, _ = environment.reset(seed=42)

    assert observation["obstacles"].shape == (64, 9)
    assert int(observation["obstacle_mask"].sum()) == 0


def test_spaces_contain_encoded_observations_and_translate_actions() -> None:
    client = FakeClient()
    environment = SnowGymEnv(client=client)
    observation, info = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)
    assert info["seed"] == 42
    assert observation["unit_action_mask"].tolist() == [[1, 1, 1]] * 3

    action = {
        "action_type": np.asarray([ACTION_MOVE, ACTION_THROW, ACTION_NOOP], dtype=np.int64),
        "target": np.asarray([[0.5, -0.5], [0.25, 0.5], [0.0, 0.0]], dtype=np.float32),
        "power": np.asarray([0.0, 0.75, 0.0], dtype=np.float32),
    }
    next_observation, reward, terminated, truncated, _ = environment.step(action)

    assert environment.observation_space.contains(next_observation)
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert client.last_action == {
        "actions": [
            {"type": "move", "unitId": 1, "x": 10.0, "y": -7.5},
            {"type": "throw", "unitId": 2, "x": 5.0, "y": 7.5, "power": 0.75},
            {"type": "noop", "unitId": 3},
        ]
    }


def test_explicit_seed_reproduces_initial_observation() -> None:
    environment = SnowGymEnv(client=FakeClient())
    first, _ = environment.reset(seed=123)
    replay, _ = environment.reset(seed=123)

    for key in first:
        np.testing.assert_array_equal(first[key], replay[key])


def test_scripted_step_uses_server_policy_path() -> None:
    client = FakeClient()
    environment = SnowGymEnv(client=client)
    environment.reset(seed=7)

    observation, reward, terminated, truncated, _ = environment.step_scripted()

    assert environment.observation_space.contains(observation)
    assert client.last_action == {"actions": []}
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_visual_recording_contains_frames_actions_and_outcome(tmp_path: Any) -> None:
    client = FakeClient()
    environment = SnowGymEnv(client=client)
    _, initial_info = environment.reset(seed=42)
    assert environment.raw_observation is not None
    recorder = ReplayRecorder(environment.raw_observation, initial_info)

    _, _, _, _, info = environment.step_scripted()
    assert environment.raw_observation is not None
    recorder.append(environment.raw_observation, info)
    replay = recorder.finish(decisions=1)
    destination = write_replay(tmp_path / "episode.json", replay)

    assert destination.exists()
    assert replay["format"] == REPLAY_FORMAT
    assert [frame["tick"] for frame in replay["frames"]] == [0, 6]
    assert replay["actions"] == [{"actions": []}]
    assert replay["outcome"]["finalTick"] == 6
    assert replay["configuration"]["blueUnits"] == 3
    assert replay["simulationVersion"] == "snowgym.sim.v1"
    assert replay["stateHashVersion"] == "snowgym.state.v1"
    assert replay["upstreamBaseCommit"] == "7d9fca5"
    assert replay["stateHashes"] == [
        hash_observation(replay["frames"][0]),
        hash_observation(replay["frames"][1]),
    ]


def make_snapshot(
    seed: int, tick: int, scenario: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = {
        "blueUnits": 3,
        "redUnits": 3,
        "arenaWidth": 40.0,
        "arenaHeight": 30.0,
        "maxTicks": 60 * 180,
        "decisionHz": 10,
        "redDifficulty": "normal",
    } | (scenario or {})
    blue_units = int(config["blueUnits"])
    red_units = int(config["redUnits"])
    width = float(config["arenaWidth"])
    height = float(config["arenaHeight"])
    allies = [make_unit(index + 1, "blue", -width * 0.3, spawn_y(index, blue_units)) for index in range(blue_units)]
    enemies = [
        make_unit(index + blue_units + 1, "red", width * 0.3, spawn_y(index, red_units))
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
        "obstacles": make_obstacles(config.get("map")),
        "match": {"blueAlive": blue_units, "redAlive": red_units},
    }
    status = {
        "apiVersion": "snowgym.v0",
        "simulationVersion": "snowgym.sim.v1",
        "stateHashVersion": "snowgym.state.v1",
        "upstreamBaseCommit": "7d9fca5",
        "stateHash": hash_observation(observation),
        "scenario": f"{blue_units}-vs-{red_units}-open",
        "seed": seed,
        "tick": tick,
        "simulationHz": 60,
        "decisionHz": int(config["decisionHz"]),
        "ticksPerDecision": 60 // int(config["decisionHz"]),
        "configuration": config,
        "blueAlive": blue_units,
        "redAlive": red_units,
        "terminated": False,
        "truncated": False,
        "winner": None,
    }
    return {
        "status": status,
        "observation": observation,
    }


def make_obstacles(map_id: Any) -> list[dict[str, Any]]:
    if not map_id:
        return []
    return [
        {
            "id": 1,
            "type": "tree",
            "x": -5.0,
            "y": 0.0,
            "halfWidth": 0.35,
            "halfHeight": 0.35,
            "blocksSight": True,
            "blocksProjectiles": True,
            "blocksMovement": True,
        },
        {
            "id": 2,
            "type": "fort",
            "x": 0.0,
            "y": 0.0,
            "halfWidth": 2.5,
            "halfHeight": 0.6,
            "blocksSight": True,
            "blocksProjectiles": True,
            "blocksMovement": True,
        },
    ]


def make_unit(unit_id: int, team: str, x: float, y: float) -> dict[str, Any]:
    return {
        "id": unit_id,
        "team": team,
        "x": x,
        "y": y,
        "vx": 0,
        "vy": 0,
        "health": 100,
        "maxHealth": 100,
        "alive": True,
        "state": "idle",
        "throwCooldown": 0,
        "charge": 0,
    }


def spawn_y(index: int, count: int) -> float:
    return 0.0 if count == 1 else -5.0 * (count - 1) / 2 + index * 5.0
