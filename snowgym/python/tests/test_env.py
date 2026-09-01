from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

import snowgym_client
from snowgym_client.encoding import (
    ACTION_HOLD,
    ACTION_MOVE,
    ACTION_NOOP,
    ACTION_THROW,
    encode_action,
)
from snowgym_client.evaluation import (
    BENCHMARK_RESULT_FORMAT,
    EVALUATION_SUITE_FORMAT,
    load_evaluation_suite,
    policy_action,
    run_evaluation_suite,
    validate_evaluation_suite,
    write_benchmark,
)
from snowgym_client.env import SnowGymEnv
from snowgym_client.opponents import (
    REMOTE_ACTION_FORMAT,
    REMOTE_OBSERVATION_FORMAT,
    LearnedOpponent,
    MaskedRandomOpponent,
    NoopOpponent,
    RemoteOpponent,
    SnowGymSingleTeamEnv,
    noop_action,
)
from snowgym_client.parallel_env import SnowGymParallelEnv
from snowgym_client.research_env import (
    SEMANTIC_RASTER_CHANNELS,
    SnowGymResearchParallelEnv,
)
from snowgym_client.recording import REPLAY_FORMAT, ReplayRecorder, write_replay
from snowgym_client.state_hash import hash_observation


class FakeClient:
    def __init__(self) -> None:
        self.seed = 0
        self.tick = 0
        self.last_action: dict[str, Any] | None = None
        self.scenario: dict[str, Any] | None = None
        self.joint_actions: list[dict[str, Any]] = []

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
        self.joint_actions = []
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

    def step_joint(
        self,
        actions: dict[str, Any],
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        assert expected_state_hash == make_snapshot(
            self.seed, self.tick, self.scenario
        )["status"]["stateHash"]
        self.last_action = actions
        self.joint_actions.append(actions)
        self.tick += 6
        snapshot = make_snapshot(self.seed, self.tick, self.scenario)
        return {
            "observations": snapshot["observations"],
            "rewards": {"blue": 0.0, "red": 0.0},
            "terminations": {"blue": False, "red": False},
            "truncations": {"blue": False, "red": False},
            "info": snapshot["status"] | {"actionResults": {"blue": [], "red": []}},
        }

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


def test_parallel_environment_passes_pettingzoo_checker() -> None:
    parallel_api_test(SnowGymParallelEnv(client=FakeClient()), num_cycles=25)


def test_single_team_environment_passes_gymnasium_checker() -> None:
    environment = SnowGymSingleTeamEnv(
        SnowGymParallelEnv(client=FakeClient()), opponent=NoopOpponent()
    )
    check_env(environment, skip_render_check=True)


def test_single_team_environment_can_control_red_perspective() -> None:
    client = FakeClient()
    environment = SnowGymSingleTeamEnv(
        SnowGymParallelEnv(client=client, max_team_units=3),
        controlled_agent="red",
        opponent=NoopOpponent(),
    )
    observation, _ = environment.reset(seed=42)
    environment.step(noop_action(3))

    assert observation["allies"][0, 2] > 0
    assert observation["enemies"][0, 2] < 0
    assert client.joint_actions[0].keys() == {"blue", "red"}
    assert {item["unitId"] for item in client.joint_actions[0]["red"]["actions"]} == {
        4,
        5,
        6,
    }


def test_masked_random_opponent_exactly_replays_joint_actions() -> None:
    def rollout() -> list[dict[str, Any]]:
        client = FakeClient()
        environment = SnowGymSingleTeamEnv(
            SnowGymParallelEnv(client=client, max_team_units=3),
            opponent=MaskedRandomOpponent(),
        )
        environment.reset(seed=91)
        for _ in range(5):
            environment.step(noop_action(3))
        return client.joint_actions

    assert rollout() == rollout()


def test_learned_opponent_receives_detached_observation_and_info() -> None:
    seen: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []

    def policy(
        observation: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        seen.append((observation, info))
        observation["tick"][0] = 999
        info["tick"] = 999
        return noop_action(3)

    environment = SnowGymSingleTeamEnv(
        SnowGymParallelEnv(client=FakeClient(), max_team_units=3),
        opponent=LearnedOpponent(policy),
    )
    environment.reset(seed=4)
    next_observation, _, _, _, info = environment.step(noop_action(3))

    assert len(seen) == 1
    assert int(next_observation["tick"][0]) == 6
    assert info["tick"] == 6


def test_remote_opponent_uses_versioned_id_free_tensor_contract() -> None:
    class FakeRemoteClient:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def act(self, request: dict[str, Any]) -> dict[str, Any]:
            self.requests.append(request)
            return {
                "format": REMOTE_ACTION_FORMAT,
                "action": {
                    "action_type": [0, 0, 0],
                    "target": [[0, 0], [0, 0], [0, 0]],
                    "power": [0, 0, 0],
                },
            }

    remote_client = FakeRemoteClient()
    environment = SnowGymSingleTeamEnv(
        SnowGymParallelEnv(client=FakeClient(), max_team_units=3),
        opponent=RemoteOpponent(remote_client),
    )
    environment.reset(seed=12)
    environment.step(noop_action(3))
    request = remote_client.requests[0]

    assert request["format"] == REMOTE_OBSERVATION_FORMAT
    assert request["agent"] == "red"
    assert request["seed"] == 12
    assert request["decision"] == 0
    assert request["tick"] == 0
    assert set(request["observation"]) == set(environment.observation_space.spaces)
    assert isinstance(request["observation"]["allies"], list)
    assert "unitId" not in str(request)


def test_remote_opponent_rejects_invalid_response_before_server_step() -> None:
    class InvalidRemoteClient:
        def __init__(self) -> None:
            self.decisions: list[int] = []

        def act(self, request: dict[str, Any]) -> dict[str, Any]:
            self.decisions.append(request["decision"])
            return {"format": "wrong", "action": {}}

    client = FakeClient()
    remote_client = InvalidRemoteClient()
    environment = SnowGymSingleTeamEnv(
        SnowGymParallelEnv(client=client, max_team_units=3),
        opponent=RemoteOpponent(remote_client),
    )
    environment.reset(seed=12)

    with pytest.raises(ValueError, match="response format"):
        environment.step(noop_action(3))
    with pytest.raises(ValueError, match="response format"):
        environment.step(noop_action(3))
    assert client.joint_actions == []
    assert remote_client.decisions == [0, 0]


def test_parallel_agents_have_independently_seedable_spaces() -> None:
    environment = SnowGymParallelEnv(client=FakeClient())

    assert environment.action_space("blue") is not environment.action_space("red")
    assert environment.observation_space("blue") is not environment.observation_space(
        "red"
    )


def test_bundled_evaluation_suite_is_versioned_and_valid() -> None:
    suite = load_evaluation_suite()

    assert suite["format"] == EVALUATION_SUITE_FORMAT
    assert suite["name"] == "baseline-v0"
    assert [episode["id"] for episode in suite["episodes"]] == [
        "open-balanced-3v3",
        "winter-front-10v10",
        "winter-front-partial-latency-5v5",
    ]


def test_evaluation_suite_rejects_duplicate_ids_and_unknown_profile_fields() -> None:
    episode = {
        "id": "duplicate",
        "seed": 1,
        "scenario": {},
        "policies": {"blue": "noop", "red": "noop"},
        "profile": {},
    }
    with pytest.raises(ValueError, match="duplicate evaluation episode id"):
        validate_evaluation_suite(
            {
                "format": EVALUATION_SUITE_FORMAT,
                "name": "invalid",
                "episodes": [episode, dict(episode)],
            }
        )
    with pytest.raises(ValueError, match="unknown fields"):
        validate_evaluation_suite(
            {
                "format": EVALUATION_SUITE_FORMAT,
                "name": "invalid",
                "episodes": [{**episode, "profile": {"renderedPixels": True}}],
            }
        )


def test_masked_random_policy_is_seeded_and_respects_action_mask() -> None:
    environment = SnowGymParallelEnv(client=FakeClient(), max_team_units=3)
    observations, _ = environment.reset(seed=42)
    observations["blue"]["unit_action_mask"][1] = 0

    first = policy_action(
        "masked_random", observations["blue"], np.random.default_rng(123)
    )
    second = policy_action(
        "masked_random", observations["blue"], np.random.default_rng(123)
    )

    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert first["action_type"][1] == ACTION_NOOP
    for index, action_type in enumerate(first["action_type"]):
        if observations["blue"]["unit_action_mask"][index].any():
            assert observations["blue"]["unit_action_mask"][index, action_type] == 1


def test_evaluation_runner_exactly_replays_deterministic_fields() -> None:
    suite = {
        "format": EVALUATION_SUITE_FORMAT,
        "name": "unit-test-v0",
        "episodes": [
            {
                "id": "latency-3v3",
                "seed": 17,
                "scenario": {
                    "blueUnits": 3,
                    "redUnits": 3,
                    "arenaWidth": 40,
                    "arenaHeight": 30,
                    "maxTicks": 1200,
                    "decisionHz": 10,
                    "redDifficulty": "normal",
                    "redController": "scripted",
                },
                "policies": {"blue": "masked_random", "red": "masked_random"},
                "profile": {
                    "visibilityRadius": 8,
                    "actionDelaySteps": 1,
                    "observationDelaySteps": 1,
                    "semanticRasterSize": 16,
                },
            }
        ],
    }

    def rollout() -> dict[str, Any]:
        return run_evaluation_suite(
            suite,
            repeat=2,
            max_decisions=5,
            environment_factory=lambda: SnowGymParallelEnv(
                client=FakeClient(), max_team_units=3
            ),
            clock=lambda: 1.0,
        )

    first = rollout()
    second = rollout()
    assert first == second
    assert first["format"] == BENCHMARK_RESULT_FORMAT
    assert first["summary"] == {
        "episodes": 2,
        "decisions": 10,
        "winners": {"blue": 0, "red": 0, "draw": 0, "none": 2},
        "terminated": 0,
        "truncated": 0,
        "decisionLimited": 2,
    }
    assert first["results"][0]["tick"] == 30
    assert first["results"][0]["rejectedActions"] == {"blue": 0, "red": 0}
    repeated = [
        {key: value for key, value in result.items() if key != "repeatIndex"}
        for result in first["results"]
    ]
    assert repeated[0] == repeated[1]


def test_benchmark_writer_refuses_implicit_overwrite(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    write_benchmark(output, {"format": BENCHMARK_RESULT_FORMAT}, force=False)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_benchmark(output, {"format": BENCHMARK_RESULT_FORMAT}, force=False)
    write_benchmark(output, {"replaced": True}, force=True)
    assert output.read_text(encoding="utf-8") == '{\n  "replaced": true\n}\n'


def test_parallel_environment_encodes_each_team_from_its_own_perspective() -> None:
    client = FakeClient()
    environment = SnowGymParallelEnv(client=client, max_team_units=3)
    observations, infos = environment.reset(seed=42)
    actions = {
        agent: environment.action_space(agent).sample()
        for agent in environment.agents
    }
    next_observations, rewards, terminations, truncations, next_infos = (
        environment.step(actions)
    )

    assert set(observations) == {"blue", "red"}
    assert observations["blue"]["allies"].shape == (3, 10)
    assert observations["red"]["allies"].shape == (3, 10)
    assert infos["blue"]["stateHash"] == infos["red"]["stateHash"]
    assert set(next_observations) == set(rewards) == {"blue", "red"}
    assert rewards["red"] == -rewards["blue"]
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert next_infos["blue"]["tick"] == 6
    assert client.last_action is not None
    assert {action["unitId"] for action in client.last_action["blue"]["actions"]} == {
        1,
        2,
        3,
    }
    assert {action["unitId"] for action in client.last_action["red"]["actions"]} == {
        4,
        5,
        6,
    }


def test_research_environment_passes_parallel_checker_with_delays_and_local_view() -> None:
    environment = SnowGymResearchParallelEnv(
        SnowGymParallelEnv(client=FakeClient()),
        visibility_radius=5,
        action_delay_steps=2,
        observation_delay_steps=2,
        semantic_raster_size=16,
    )
    parallel_api_test(environment, num_cycles=25)


def test_research_profile_masks_remote_enemies_and_reports_source_tick() -> None:
    environment = SnowGymResearchParallelEnv(
        SnowGymParallelEnv(client=FakeClient(), max_team_units=3),
        visibility_radius=5,
        observation_delay_steps=2,
    )
    observations, infos = environment.reset(seed=42)

    assert observations["blue"]["enemy_mask"].tolist() == [0, 0, 0]
    assert observations["red"]["enemy_mask"].tolist() == [0, 0, 0]
    assert infos["blue"]["research"] == {
        "visibilityRadius": 5.0,
        "actionDelaySteps": 0,
        "observationDelaySteps": 2,
        "observationSourceTick": 0,
        "appliedActionSourceTick": None,
        "semanticRasterSize": None,
    }


def test_research_profile_applies_and_observes_exact_decision_delays() -> None:
    client = FakeClient()
    environment = SnowGymResearchParallelEnv(
        SnowGymParallelEnv(client=client, max_team_units=3),
        action_delay_steps=2,
        observation_delay_steps=2,
    )
    observations, _ = environment.reset(seed=42)
    submitted = {
        agent: environment.action_space(agent).sample()
        for agent in environment.agents
    }
    expected_blue = encode_action(
        submitted["blue"],
        environment.environment.raw_observations["blue"],
        3,
    )
    returned_ticks = []
    for _ in range(3):
        observations, _, _, _, infos = environment.step(submitted)
        returned_ticks.append(int(observations["blue"]["tick"][0]))

    assert returned_ticks == [0, 0, 6]
    assert all(
        action["type"] == "noop"
        for action in client.joint_actions[0]["blue"]["actions"]
    )
    assert all(
        action["type"] == "noop"
        for action in client.joint_actions[1]["blue"]["actions"]
    )
    assert client.joint_actions[2]["blue"] == expected_blue
    assert infos["blue"]["tick"] == 18
    assert infos["blue"]["research"]["observationSourceTick"] == 6
    assert infos["blue"]["research"]["appliedActionSourceTick"] == 0


def test_semantic_raster_respects_local_visibility_and_fixed_space() -> None:
    environment = SnowGymResearchParallelEnv(
        SnowGymParallelEnv(
            client=FakeClient(),
            max_team_units=3,
            map="arena1.json",
        ),
        visibility_radius=5,
        semantic_raster_size=16,
    )
    observations, infos = environment.reset(seed=42)
    raster = observations["blue"]["semantic_raster"]
    channels = {name: index for index, name in enumerate(SEMANTIC_RASTER_CHANNELS)}

    assert environment.observation_space("blue").contains(observations["blue"])
    assert raster.shape == (5, 16, 16)
    assert raster[channels["allies"]].sum() == 3
    assert raster[channels["enemies"]].sum() == 0
    assert raster[channels["obstacles"]].sum() > 0
    assert infos["blue"]["research"]["semanticRasterSize"] == 16


def test_research_profile_exactly_replays_actions_and_delayed_observations() -> None:
    def rollout() -> tuple[
        list[dict[str, Any]], list[tuple[int, int]], list[bytes]
    ]:
        client = FakeClient()
        environment = SnowGymResearchParallelEnv(
            SnowGymParallelEnv(client=client, max_team_units=3),
            visibility_radius=5,
            action_delay_steps=2,
            observation_delay_steps=1,
            semantic_raster_size=16,
        )
        environment.reset(seed=17)
        action = {
            "action_type": np.asarray(
                [ACTION_MOVE, ACTION_HOLD, ACTION_NOOP], dtype=np.int64
            ),
            "target": np.asarray(
                [[0.25, -0.25], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32
            ),
            "power": np.zeros(3, dtype=np.float32),
        }
        ticks = []
        rasters = []
        for _ in range(5):
            observations, _, _, _, infos = environment.step(
                {"blue": action, "red": action}
            )
            ticks.append(
                (
                    int(infos["blue"]["tick"]),
                    int(observations["blue"]["tick"][0]),
                )
            )
            rasters.append(observations["blue"]["semantic_raster"].tobytes())
        return client.joint_actions, ticks, rasters

    assert rollout() == rollout()


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
    assert observation["unit_action_mask"].tolist() == [[1, 1, 1, 1]] * 3

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


def test_hold_action_cancels_movement_without_using_target_fields() -> None:
    raw = make_snapshot(0, 0)["observation"]
    action = {
        "action_type": np.asarray([ACTION_HOLD, ACTION_NOOP, ACTION_NOOP], dtype=np.int64),
        "target": np.full((3, 2), np.nan, dtype=np.float32),
        "power": np.full(3, np.nan, dtype=np.float32),
    }

    assert encode_action(action, raw) == {
        "actions": [
            {"type": "hold", "unitId": 1},
            {"type": "noop", "unitId": 2},
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
        "observations": {
            "blue": observation,
            "red": {
                **observation,
                "selfTeam": "red",
                "allies": enemies,
                "enemies": allies,
            },
        },
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
