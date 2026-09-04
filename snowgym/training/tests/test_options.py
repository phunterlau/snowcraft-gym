from __future__ import annotations

from copy import deepcopy
import math

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_training.options import (
    FROZEN_OPTION_SPECS,
    FixedOptionTracker,
    FixedPlanOptionBatchEnv,
    OptionSpec,
    evaluate_teacher_option,
    load_option_protocol,
)


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
    with SnowGymBatchClient() as client:
        results = [
            evaluate_teacher_option(name, seed=41_000 + index, client=client)
            for index, name in enumerate(names)
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
