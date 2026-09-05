from __future__ import annotations

import numpy as np
import pytest

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv


def scenario(max_ticks: int = 120) -> dict[str, object]:
    return {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": max_ticks,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }


def noop_actions(batch_size: int, capacity: int = 10) -> dict[str, np.ndarray]:
    return {
        "action_type": np.zeros((batch_size, capacity), dtype=np.int64),
        "target": np.zeros((batch_size, capacity, 2), dtype=np.float32),
        "power": np.zeros((batch_size, capacity), dtype=np.float32),
    }


def one_group_plan() -> dict[str, object]:
    return {
        "schemaVersion": "snowgym.command-plan.v0",
        "intentSummary": "Advance together and engage the nearest enemy.",
        "groups": [
            {
                "role": "main",
                "allocationWeight": 1,
                "selection": "balanced",
                "order": {
                    "mission": "engage",
                    "objective": {"kind": "enemy_cluster", "select": "nearest"},
                    "approach": "direct",
                    "engagement": {
                        "posture": "balanced",
                        "fire": "focus",
                        "preferredRange": "medium",
                        "cohesion": "normal",
                    },
                },
            }
        ],
    }


def test_batch_subprocess_handshake_and_independent_worlds() -> None:
    with SnowGymBatchClient() as client:
        assert client.capabilities["protocolVersion"] == "snowgym.batch.v0"
        environment = SnowGymBatchEnv(2, client=client)
        observation, infos = environment.reset([11, 12], [scenario(), scenario()])
        assert observation["allies"].shape == (2, 10, 10)
        assert [info["tick"] for info in infos] == [0, 0]
        assert infos[0]["stateHash"] == infos[1]["stateHash"]

        observation, rewards, terminated, truncated, infos = environment.step(
            noop_actions(2)
        )
        assert observation["tick"].tolist() == [[6], [6]]
        assert rewards.tolist() == [0.0, 0.0]
        assert not terminated.any()
        assert not truncated.any()
        assert [info["tick"] for info in infos] == [6, 6]

        changed, reset_infos = environment.reset_indices([1], [99], [scenario()])
        assert changed["tick"].tolist() == [[0]]
        assert reset_infos[0]["seed"] == 99
        assert environment._observations[0]["tick"].tolist() == [6]


def test_batch_scripted_step_uses_native_blue_policy() -> None:
    with SnowGymBatchClient() as client:
        assert "stepScripted" in client.capabilities["operations"]
        environment = SnowGymBatchEnv(1, client=client)
        environment.reset([13], [scenario()])
        observation, rewards, terminated, truncated, infos = environment.step_scripted()
        assert observation["tick"].tolist() == [[6]]
        assert rewards.tolist() == [0.0]
        assert not terminated.any()
        assert not truncated.any()
        assert infos[0]["tick"] == 6
        assert all(result["accepted"] for result in infos[0]["actionResults"])


def test_selected_step_preserves_other_worlds_and_row_order():
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(3, client=client, observation_version=3)
        environment.reset([21, 22, 23], [scenario()] * 3)
        before = list(environment.state_hashes)
        observations, _, _, _, _ = environment.step_indices([2, 0], noop_actions(2))
        assert observations["tick"].tolist() == [[6], [6]]
        assert environment.state_hashes[1] == before[1]
        assert environment._observations[1]["tick"].tolist() == [0]
        for indices in ([], [0, 0], [3], [-1], [True]):
            with pytest.raises(ValueError, match="indices"):
                environment.step_indices(indices, noop_actions(len(indices)))
        with pytest.raises(ValueError, match="actions"):
            environment.step_indices([1], noop_actions(2))


def test_batch_joint_step_controls_both_teams_symmetrically() -> None:
    with SnowGymBatchClient() as client:
        assert "stepJoint" in client.capabilities["operations"]
        environment = SnowGymBatchEnv(2, client=client, observation_version=3)
        observation, _ = environment.reset([19, 19], [scenario(), scenario()])
        assert observation["allies"].shape == (2, 10, 21)

        observation, rewards, terminated, truncated, infos = environment.step_joint(
            noop_actions(2), noop_actions(2)
        )

        assert observation["tick"].tolist() == [[6], [6]]
        assert rewards.tolist() == [0.0, 0.0]
        assert not terminated.any()
        assert not truncated.any()
        assert infos[0]["stateHash"] == infos[1]["stateHash"]
        assert all(
            result["accepted"]
            for info in infos
            for team_results in info["actionResults"].values()
            for result in team_results
        )


def test_batch_can_execute_plan_teacher_semantic_actions() -> None:
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(1, client=client, observation_version=3)
        environment.reset([23], [scenario()])
        environment.activate_plans(["teacher-plan"], [one_group_plan()])
        actions = environment.plan_teacher_actions()
        observation, _, terminated, truncated, infos = environment.step_team_actions(
            actions
        )
        assert observation["tick"].tolist() == [[6]]
        assert not terminated.any()
        assert not truncated.any()
        assert all(result["accepted"] for result in infos[0]["actionResults"])


def test_batch_plan_tensors_are_host_owned_and_follow_world_ticks() -> None:
    with SnowGymBatchClient() as client:
        assert "activatePlan" in client.capabilities["operations"]
        assert "planObservation" in client.capabilities["operations"]
        assert "planTeacherAction" in client.capabilities["operations"]
        assert "previewPlan" in client.capabilities["operations"]
        environment = SnowGymBatchEnv(2, client=client)
        environment.reset([31, 32], [scenario(), scenario()])

        activated = environment.activate_plans(
            ["plan-left", "plan-right"], [one_group_plan(), one_group_plan()]
        )
        assert [body["planId"] for body in activated] == ["plan-left", "plan-right"]
        assert [body["tick"] for body in activated] == [0, 0]
        tensors, metadata = environment.plan_observations()
        assert tensors["plan_groups"].shape == (2, 3, 38)
        assert tensors["plan_group_mask"].tolist() == [[1, 0, 0], [1, 0, 0]]
        assert tensors["plan_unit_roles"].shape == (2, 10, 3)
        assert tensors["plan_role_state"].shape == (2, 3, 20)
        assert tensors["mission_progress"].shape == (2, 3)
        assert tensors["plan_unit_roles"][:, 0].tolist() == [[1, 0, 0], [1, 0, 0]]
        assert [body["stateHash"] for body in metadata] == environment.state_hashes
        assert np.all(tensors["plan_groups"][:, 0, 37] == 0)
        teachers = environment.plan_teacher_actions()
        assert len(teachers) == 2
        assert all(len(action["actions"]) == 1 for action in teachers)
        assert environment.plan_observations()[1][0]["tick"] == 0
        preview_tensors, preview_actions, preview_metadata = environment.preview_plans(
            ["preview-left", "preview-right"], [one_group_plan(), one_group_plan()]
        )
        assert preview_tensors["plan_groups"].shape == (2, 3, 38)
        assert preview_tensors["plan_unit_roles"].shape == (2, 10, 3)
        assert preview_tensors["plan_role_state"].shape == (2, 3, 20)
        assert preview_tensors["mission_progress"].shape == (2, 3)
        assert [body["planId"] for body in preview_metadata] == [
            "preview-left", "preview-right"
        ]
        assert all(len(action["actions"]) == 1 for action in preview_actions)
        assert [body["planId"] for body in environment.plan_observations()[1]] == [
            "plan-left", "plan-right"
        ]

        environment.step(noop_actions(2))
        advanced, metadata = environment.plan_observations()
        assert [body["tick"] for body in metadata] == [6, 6]
        np.testing.assert_allclose(advanced["plan_groups"][:, 0, 37], 1 / 300)

        reset, _ = environment.reset_indices([1], [44], [scenario()])
        assert reset["tick"].tolist() == [[0]]
        reactivated = environment.activate_plan_indices(
            [1], ["plan-replacement"], [one_group_plan()]
        )
        assert [body["planId"] for body in reactivated] == ["plan-replacement"]
        selected, selected_metadata = environment.plan_observations([1])
        assert selected["plan_groups"].shape == (1, 3, 38)
        assert selected["plan_unit_roles"].shape == (1, 10, 3)
        assert [body["tick"] for body in selected_metadata] == [0]
        _, all_metadata = environment.plan_observations()
        assert [(body["planId"], body["tick"]) for body in all_metadata] == [
            ("plan-left", 6),
            ("plan-replacement", 0),
        ]
