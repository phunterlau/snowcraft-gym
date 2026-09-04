from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.options import (
    FROZEN_OPTION_SPECS,
    FixedOptionTracker,
    FixedPlanOptionBatchEnv,
    OptionSpec,
    evaluate_teacher_option,
    load_option_protocol,
    OptionEntry,
    OptionSchedule,
    collect_option_rollout,
    qualify_m7b,
)
from snowgym_training.executor import ModelConfig
from snowgym_training.plan_ppo import target_only_plan_ppo_config
from snowgym_training.ppo import HybridActorCritic, PPOConfig, generalized_advantage_estimate
from snowgym_training.ppo_collect import SeedSchedule
from snowgym_training.options.causal_fork import run_causal_fork
from snowgym_training.options.evaluate import resolve_evaluation_options
from snowgym_training.options.interventions import compose_intervention_action
from snowgym_training.options.gradient_diagnostics import write_csv
from snowgym_training.options.recovery_report import audit_artifact_manifest
from snowgym_training.trajectory import json_digest


def unit(identifier: int, x: float, y: float, *, alive: bool = True) -> dict[str, object]:
    return {
        "id": identifier,
        "x": x,
        "y": y,
        "vx": 0,
        "vy": 0,
        "health": 100 if alive else 0,
        "maxHealth": 100,
        "alive": alive,
    }


def observation() -> dict[str, object]:
    return {
        "arena": {"width": 100, "height": 80},
        "allies": [unit(1, 0, -1), unit(2, 0, 1), unit(3, 0, 15)],
        "enemies": [unit(10, 20, -2), unit(11, 20, 2)],
    }


def plan(name: str, *, role: str = "main") -> dict[str, object]:
    mission = name if name in {"engage", "advance", "hold", "withdraw", "support"} else "engage"
    objective: dict[str, object]
    if mission == "engage":
        objective = {"kind": "enemy_cluster", "select": "nearest"}
    elif mission == "support":
        objective = {"kind": "ally_group", "role": "main"}
    elif mission == "hold":
        objective = {"kind": "current_position"}
    else:
        objective = {"kind": "region", "region": "own_backfield" if mission == "withdraw" else "center_lane"}
    approach = "left_flank" if name == "flank" else "direct"
    fire = name if name in {"focus", "distributed"} else "opportunistic"
    groups = []
    if role != "main":
        groups.append(
            {
                "role": "main",
                "allocationWeight": 2,
                "selection": "balanced",
                "order": {
                    "mission": "hold",
                    "objective": {"kind": "current_position"},
                    "approach": "direct",
                    "engagement": {"posture": "balanced", "fire": "focus", "preferredRange": "medium", "cohesion": "normal"},
                },
            }
        )
    groups.append(
        {
            "role": role,
            "allocationWeight": 1,
            "selection": "balanced",
            "order": {
                "mission": mission,
                "objective": objective,
                "approach": approach,
                "engagement": {"posture": "balanced", "fire": fire, "preferredRange": "medium", "cohesion": "normal"},
            },
        }
    )
    return {"schemaVersion": "snowgym.command-plan.v0", "intentSummary": name, "groups": groups}


def plan_observation(*, objective_health: float = 1.0, support: bool = False) -> dict[str, object]:
    role_state = [0.0] * 60
    role_state[8] = 0.2
    role_state[11] = objective_health
    assignments = [{"role": "main", "unitIds": [1, 2]}]
    if support:
        assignments.append({"role": "reserve", "unitIds": [3]})
    return {"planRoleState": role_state, "assignments": assignments}


def update(
    tracker: FixedOptionTracker,
    raw: dict[str, object],
    body: dict[str, object],
):
    return tracker.update(raw, body, canonical_reward=0, gamma=0.99)


def test_every_frozen_option_has_an_achievable_deterministic_trace() -> None:
    initial = observation()

    engage_body = plan_observation(objective_health=1)
    engage = FixedOptionTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), initial, engage_body)
    assert update(engage, initial, plan_observation(objective_health=0.2)).success

    advance = FixedOptionTracker(FROZEN_OPTION_SPECS["advance"], plan("advance"), initial, plan_observation())
    arrived = deepcopy(initial)
    arrived["allies"][0]["x"] = 20
    arrived["allies"][1]["x"] = 20
    assert [update(advance, arrived, plan_observation()).success for _ in range(10)][-1]

    hold = FixedOptionTracker(FROZEN_OPTION_SPECS["hold"], plan("hold"), initial, plan_observation())
    assert [update(hold, initial, plan_observation()).success for _ in range(150)][-1]

    withdraw = FixedOptionTracker(FROZEN_OPTION_SPECS["withdraw"], plan("withdraw"), initial, plan_observation())
    withdrawn = deepcopy(initial)
    withdrawn["allies"][0]["x"] = 20
    withdrawn["allies"][1]["x"] = 20
    assert [update(withdraw, withdrawn, plan_observation()).success for _ in range(20)][-1]

    flank = FixedOptionTracker(FROZEN_OPTION_SPECS["flank"], plan("flank"), initial, plan_observation())
    flanked = deepcopy(initial)
    flanked["allies"][0]["y"] += 15
    flanked["allies"][1]["y"] += 15
    assert not update(flank, flanked, plan_observation()).success
    damaged = deepcopy(flanked)
    damaged["enemies"][0]["health"] = 80
    assert update(flank, damaged, plan_observation()).success

    focus = FixedOptionTracker(FROZEN_OPTION_SPECS["focus"], plan("focus"), initial, plan_observation())
    focused = deepcopy(initial)
    focused["enemies"][0]["health"] = 80
    assert update(focus, focused, plan_observation()).success

    distributed = FixedOptionTracker(FROZEN_OPTION_SPECS["distributed"], plan("distributed"), initial, plan_observation())
    spread = deepcopy(initial)
    spread["enemies"][0]["health"] = 90
    spread["enemies"][1]["health"] = 90
    result = update(distributed, spread, plan_observation())
    assert result.success
    assert math.isclose(result.metrics["targetDamageEntropy"], 1)

    support_spec = OptionSpec("support", 300, role="reserve")
    support = FixedOptionTracker(support_spec, plan("support", role="reserve"), initial, plan_observation(support=True))
    assert [update(support, initial, plan_observation(support=True)).success for _ in range(30)][-1]


def test_option_reward_keeps_components_separate() -> None:
    initial = observation()
    tracker = FixedOptionTracker(FROZEN_OPTION_SPECS["engage"], plan("engage"), initial, plan_observation())
    damaged = deepcopy(initial)
    damaged["enemies"][0]["health"] = 80
    result = tracker.update(
        damaged,
        plan_observation(objective_health=0.2),
        canonical_reward=1,
        gamma=0.99,
    )
    assert result.mission_reward == 1
    assert math.isclose(result.combat_reward, 0.1)
    assert result.canonical_reward == 1
    assert math.isclose(
        result.executor_reward,
        result.mission_reward + 0.1 * result.combat_reward + result.shaping_reward,
    )
    assert result.shaping_reward == 0


def test_temporal_option_counters_advance_once_per_decision() -> None:
    initial = observation()
    body = plan_observation(support=True)
    hold = FixedOptionTracker(FROZEN_OPTION_SPECS["hold"], plan("hold"), initial, body)
    assert update(hold, initial, body).progress == 1 / 150

    withdrawn = deepcopy(initial)
    withdrawn["allies"][0]["x"] = 20
    withdrawn["allies"][1]["x"] = 20
    withdraw = FixedOptionTracker(
        FROZEN_OPTION_SPECS["withdraw"], plan("withdraw"), initial, body
    )
    assert update(withdraw, withdrawn, body).progress == 1 / 20

    support = FixedOptionTracker(
        OptionSpec("support", 300, role="reserve"),
        plan("support", role="reserve"),
        initial,
        body,
    )
    assert update(support, initial, body).progress == 1 / 30


def test_option_timeout_is_terminal_and_does_not_bootstrap_reset_value() -> None:
    initial = observation()
    tracker = FixedOptionTracker(
        OptionSpec("engage", 1), plan("engage"), initial, plan_observation()
    )
    result = update(tracker, initial, plan_observation())
    assert result.done and result.failed and result.timed_out
    advantages, _ = generalized_advantage_estimate(
        torch.tensor([[-1.0]]),
        torch.tensor([[0.4]]),
        torch.tensor([[0.8]]),
        torch.tensor([[True]]),
        torch.tensor([[False]]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages, torch.tensor([[-1.4]]))


def test_engage_intervention_changes_only_the_selected_action_channel() -> None:
    learner = {
        "action_type": torch.tensor([[1, 2]]).numpy(),
        "target": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]).numpy(),
        "power": torch.tensor([[0.5, 0.6]]).numpy(),
    }
    teacher = {
        "action_type": torch.tensor([[2, 1]]).numpy(),
        "target": torch.tensor([[[0.7, 0.8], [0.9, 1.0]]]).numpy(),
        "power": torch.tensor([[0.2, 0.3]]).numpy(),
    }
    goal = torch.tensor([[[0.4, -0.4], [0.4, -0.4]]]).numpy()

    teacher_move = compose_intervention_action(
        "teacher-move", learner, teacher, goal
    )
    assert teacher_move["action_type"].tolist() == [[1, 2]]
    assert teacher_move["target"].tolist()[0][0] == pytest.approx([0.7, 0.8])
    assert teacher_move["target"].tolist()[0][1] == pytest.approx([0.3, 0.4])
    assert teacher_move["power"][0].tolist() == pytest.approx([0.5, 0.6])

    teacher_action = compose_intervention_action(
        "teacher-action", learner, teacher, goal
    )
    assert teacher_action["action_type"].tolist() == [[2, 1]]
    torch.testing.assert_close(
        torch.from_numpy(teacher_action["target"]),
        torch.from_numpy(learner["target"]),
    )

    anchored = compose_intervention_action("goal-anchor", learner, teacher, goal)
    assert anchored["target"].tolist()[0][0] == pytest.approx([0.4, -0.4])
    assert anchored["target"].tolist()[0][1] == pytest.approx([0.3, 0.4])


def test_gradient_diagnostic_csv_uses_repository_line_endings(tmp_path: Path) -> None:
    destination = tmp_path / "diagnostic.csv"
    write_csv(destination, [{"component": "actor", "norm": 1.0}])
    assert destination.read_bytes() == b"component,norm\nactor,1.0\n"


def test_archived_failed_engage_evidence_passes_digest_audit() -> None:
    archive = Path(__file__).resolve().parents[1] / "runs" / "m7b_engage_failed_v0"
    audit_artifact_manifest(archive, "archive-manifest.json")
    audit_artifact_manifest(archive / "diagnostics", "manifest.json")
    audit_artifact_manifest(archive / "gradient-diagnostics", "manifest.json")


def test_live_fixed_option_wrapper_executes_production_teacher() -> None:
    scenario = {
        "blueUnits": 3,
        "redUnits": 3,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 300,
        "decisionHz": 10,
        "redDifficulty": "easy",
        "redController": "scripted",
    }
    engage = plan("engage")
    with SnowGymBatchClient() as client:
        wrapped = FixedPlanOptionBatchEnv(
            SnowGymBatchEnv(1, client=client, observation_version=3), gamma=0.99
        )
        observation_tensor, _ = wrapped.reset(
            [71], [scenario], ["engage-live"], [engage], [FROZEN_OPTION_SPECS["engage"]]
        )
        assert observation_tensor["allies"].shape == (1, 10, 21)
        _, reward, terminated, truncated, infos = wrapped.step(teacher=True)
    assert reward.shape == (1,)
    assert not terminated[0]
    assert not truncated[0]
    assert set(infos[0]["option"]["rewards"]) == {
        "mission", "combat", "shaping", "canonical", "executor"
    }
    assert all(result["accepted"] for result in infos[0]["actionResults"])


def test_production_teacher_proves_every_frozen_option_is_achievable() -> None:
    names = (
        "engage", "advance", "hold", "withdraw", "flank", "focus", "distributed", "support"
    )
    seeds = (41_000, 41_001, 41_002, 41_003, 41_000, 41_005, 41_006, 41_007)
    with SnowGymBatchClient() as client:
        results = [
            evaluate_teacher_option(name, seed=seed, client=client)
            for name, seed in zip(names, seeds, strict=True)
        ]
    assert all(result["success"] for result in results), results
    assert all(result["rejectedActions"] == 0 for result in results)


def test_frozen_option_protocol_has_disjoint_seed_partitions() -> None:
    protocol = load_option_protocol()
    assert protocol["pairedSeedsPerMission"] == {
        "development": 40,
        "qualification": 100,
    }
    assert protocol["qualification"]["missionSuccessMinimum"] == 0.75


def test_option_schedule_and_collector_restore_after_selective_timeout() -> None:
    option_plan = plan("engage")
    entries = tuple(
        OptionEntry(option_plan, OptionSpec("engage", 1)) for _ in range(6)
    )
    schedule = OptionSchedule(entries, prefix="engage")
    restored = OptionSchedule.restore(entries, schedule.state())
    assert restored.state() == schedule.state()
    base = ModelConfig(
        16,
        12,
        24,
        action_conditioned_targets=True,
        plan_conditioned=True,
        plan_target_only=True,
        separate_target_actor=True,
    )
    torch.manual_seed(79)
    model = HybridActorCritic(target_only_plan_ppo_config(base))
    scenario = {
        "blueUnits": 3,
        "redUnits": 3,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 300,
        "decisionHz": 10,
        "redDifficulty": "easy",
        "redController": "random",
    }
    with SnowGymBatchClient() as client:
        wrapped = FixedPlanOptionBatchEnv(
            SnowGymBatchEnv(2, client=client, observation_version=3), gamma=0.99
        )
        collection = collect_option_rollout(
            wrapped,
            model,
            scenario=scenario,
            seed_schedule=SeedSchedule(60_000, 60_099),
            option_schedule=schedule,
            rollout_steps=3,
            config=PPOConfig(update_epochs=1, minibatch_size=4),
        )
    assert collection.completed_options == 6
    assert collection.successful_options == 0
    assert collection.episode_seeds == tuple(range(60_000, 60_006))
    assert collection.option_schedule["nextIndex"] == 6
    assert collection.teacher_actions["action_type"].shape == (3, 2, 10)
    assert collection.rollout.observations["plan_role_state"].shape == (3, 2, 3, 20)
    assert collection.reward_sums["mission"] == -6
    assert collection.reward_sums["executor"] == float(
        collection.rollout.rewards.sum()
    )


def qualification_input() -> dict[str, object]:
    def records(successes: int, progress: float, physical_wins: int):
        return [
            {
                "seed": 300_000 + index,
                "success": index < successes,
                "progress": progress,
                "physicalWin": index < physical_wins,
                "rejectedActions": 0,
                "totalActions": 100,
            }
            for index in range(100)
        ]

    value = {
        "format": "snowgym.m7b-evaluation.v0",
        "checkpointDigest": "sha256:" + "1" * 64,
        "sourceDigest": "sha256:" + "2" * 64,
        "protocolDigest": "sha256:" + "3" * 64,
        "inheritedHeadLearningRate": 1e-5,
        "newModuleLearningRate": 1e-4,
        "parameterL2Change": 1.0,
        "missions": {
            name: {
                "correct": records(80, 0.9, 70),
                "shuffled": records(40, 0.1, 70),
                "initializer": records(60, 0.5, 75),
            }
            for name in FROZEN_OPTION_SPECS
        },
    }
    value["evaluationDigest"] = json_digest(value)
    return value


def test_m7b_qualification_requires_every_mission_independently() -> None:
    passing = qualification_input()
    report = qualify_m7b(passing)
    assert report["passed"]
    assert all(mission["passed"] for mission in report["missions"].values())

    failing = qualification_input()
    failing["missions"]["support"]["correct"] = [
        {**row, "success": index < 70}
        for index, row in enumerate(failing["missions"]["support"]["correct"])
    ]
    failing["evaluationDigest"] = json_digest(
        {name: item for name, item in failing.items() if name != "evaluationDigest"}
    )
    failed = qualify_m7b(failing)
    assert not failed["passed"]
    assert not failed["missions"]["support"]["passed"]
    assert failed["missions"]["engage"]["passed"]


def test_m7b_development_may_select_missions_but_qualification_may_not() -> None:
    assert resolve_evaluation_options("development", ("hold", "engage")) == (
        "engage",
        "hold",
    )
    assert resolve_evaluation_options("qualification", None) == tuple(
        FROZEN_OPTION_SPECS
    )
    for selected in (("engage", "engage"), (), ("unknown",)):
        try:
            resolve_evaluation_options("development", selected)
        except ValueError:
            pass
        else:
            raise AssertionError("development accepted an invalid option selection")
    try:
        resolve_evaluation_options("qualification", ("engage",))
    except ValueError as error:
        assert "every frozen mission" in str(error)
    else:
        raise AssertionError("qualification accepted a mission subset")


def test_same_state_hold_withdraw_advance_fork_is_deterministic_and_diverges() -> None:
    first = run_causal_fork(seed=42_001, decisions=12)
    second = run_causal_fork(seed=42_001, decisions=12)
    assert first == second
    assert first["rejectedActions"] == 0
    traces = first["forks"]
    assert len({tuple(trace["stateHashes"]) for trace in traces.values()}) == 3
    assert all(trace["stateHashes"][0] == first["initialStateHash"] for trace in traces.values())
