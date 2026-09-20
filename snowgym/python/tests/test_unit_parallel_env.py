from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from snowgym_client.encoding import ACTION_MOVE, ACTION_NOOP
from snowgym_client.state_hash import hash_observation
from snowgym_client.unit_parallel_env import SnowGymUnitParallelEnv, split_unit_agent, unit_agents


def make_unit(unit_id: int, team: str, x: float, alive: bool = True) -> dict[str, Any]:
    return {
        "id": unit_id, "team": team, "x": x, "y": 0.0, "vx": 0.0, "vy": 0.0,
        "health": 100.0 if alive else 0.0, "maxHealth": 100.0, "alive": alive,
        "state": "idle" if alive else "defeated", "throwCooldown": 0, "charge": 0,
        "moveTarget": None, "steeringTarget": None, "aimDirection": {"x": 1, "y": 0},
        "stunRemaining": 0, "throwPhaseRemaining": 0, "immunityRemaining": 0, "speedRemaining": 0,
    }


def make_snapshot(seed: int, tick: int, *, blue_units: int, red_units: int,
                   dead_blue_slot: int | None = None) -> dict[str, Any]:
    width, height = 40.0, 30.0
    allies = [make_unit(i + 1, "blue", -10.0, alive=(dead_blue_slot != i)) for i in range(blue_units)]
    enemies = [make_unit(i + blue_units + 1, "red", 10.0) for i in range(red_units)]
    blue_alive = sum(1 for u in allies if u["alive"])
    observation = {
        "observationVersion": "snowgym.observation.v1", "tick": tick, "selfTeam": "blue",
        "simulationHz": 60, "arena": {"width": width, "height": height},
        "allies": allies, "enemies": enemies, "projectiles": [], "obstacles": [],
        "match": {"blueAlive": blue_alive, "redAlive": red_units},
        "decision": {"hz": 10, "dt": 0.1, "maxTicks": 60 * 180, "remainingFraction": max(0.0, 1 - tick / (60 * 180))},
    }
    status = {
        "apiVersion": "snowgym.v0", "simulationVersion": "snowgym.sim.v1", "stateHashVersion": "snowgym.state.v2",
        "upstreamBaseCommit": "test", "stateHash": hash_observation(observation),
        "scenario": f"{blue_units}-vs-{red_units}-open", "seed": seed, "tick": tick, "simulationHz": 60,
        "decisionHz": 10, "ticksPerDecision": 6,
        "configuration": {"blueUnits": blue_units, "redUnits": red_units, "arenaWidth": width, "arenaHeight": height,
                          "maxTicks": 60 * 180, "decisionHz": 10, "redDifficulty": "normal"},
        "blueAlive": blue_alive, "redAlive": red_units, "terminated": False, "truncated": False, "winner": None,
    }
    return {"status": status, "observations": {
        "blue": observation,
        "red": {**observation, "selfTeam": "red", "allies": enemies, "enemies": allies}}}


class FakeJointClient:
    """A minimal joint-step client, self-contained (not importing test_env.py's
    private helpers). `die_after_step`, if set, kills blue unit slot 0 on that
    step. `terminate_after_step`, if set, returns terminations={"blue": True,
    "red": False} on that step (an episode-ending event unrelated to any single
    unit's death — e.g. a mission-level condition)."""

    def __init__(self, blue_units: int = 3, red_units: int = 3, die_after_step: int | None = None,
                 terminate_after_step: int | None = None):
        self.blue_units, self.red_units = blue_units, red_units
        self.die_after_step = die_after_step
        self.terminate_after_step = terminate_after_step
        self.seed = 0
        self.tick = 0
        self.step_count = 0
        self.joint_actions: list[dict[str, Any]] = []

    def _snapshot(self) -> dict[str, Any]:
        dead = 0 if self.die_after_step is not None and self.step_count >= self.die_after_step else None
        return make_snapshot(self.seed, self.tick, blue_units=self.blue_units, red_units=self.red_units,
                              dead_blue_slot=dead)

    def reset(self, seed: int, scenario: dict[str, Any] | None = None, *, idempotency_key: str | None = None):
        self.seed, self.tick, self.step_count, self.joint_actions = seed, 0, 0, []
        if scenario:
            self.blue_units = int(scenario.get("blueUnits", self.blue_units))
            self.red_units = int(scenario.get("redUnits", self.red_units))
        return self._snapshot()

    def step_joint(self, actions: dict[str, Any], *, expected_state_hash: str | None = None,
                   idempotency_key: str | None = None):
        before = self._snapshot()
        assert expected_state_hash == before["status"]["stateHash"]
        self.joint_actions.append(actions)
        self.tick += 6
        self.step_count += 1
        snapshot = self._snapshot()
        blue_terminated = self.terminate_after_step is not None and self.step_count >= self.terminate_after_step
        return {"observations": snapshot["observations"], "rewards": {"blue": 1.0, "red": -1.0},
                "terminations": {"blue": blue_terminated, "red": False}, "truncations": {"blue": False, "red": False},
                "info": snapshot["status"] | {"actionResults": {"blue": [], "red": []}}}


# -- pure helpers -----------------------------------------------------------------------


def test_unit_agents_lists_blue_then_red_by_slot():
    assert unit_agents(3, 2) == ["blue-0", "blue-1", "blue-2", "red-0", "red-1"]


def test_split_unit_agent_round_trips():
    assert split_unit_agent("blue-2") == ("blue", 2)
    assert split_unit_agent("red-0") == ("red", 0)


# -- live: PettingZoo checker -------------------------------------------------------------


def test_unit_parallel_environment_passes_pettingzoo_checker():
    env = SnowGymUnitParallelEnv(client=FakeJointClient(blue_units=3, red_units=3))
    parallel_api_test(env, num_cycles=25)


# -- live: reset/step contract, determinism, death and removal ----------------------------


def make_env(*, blue_units: int = 3, red_units: int = 3, die_after_step: int | None = None,
             terminate_after_step: int | None = None) -> SnowGymUnitParallelEnv:
    client = FakeJointClient(blue_units=blue_units, red_units=red_units, die_after_step=die_after_step,
                             terminate_after_step=terminate_after_step)
    return SnowGymUnitParallelEnv(client=client, blue_units=blue_units, red_units=red_units)


def sample_action():
    return {"action_type": ACTION_NOOP, "target": np.zeros(2, dtype=np.float32), "power": np.float32(0.0)}


def test_reset_starts_with_every_configured_unit_agent():
    env = make_env(blue_units=2, red_units=1)
    observations, infos = env.reset(seed=1)
    assert env.agents == ["blue-0", "blue-1", "red-0"]
    assert set(observations) == set(env.agents)
    assert set(infos) == set(env.agents)


def test_reset_is_deterministic_for_the_same_seed():
    env = make_env(blue_units=2, red_units=2)
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env.reset(seed=7)
    for agent in env.possible_agents:
        assert np.array_equal(obs_a[agent]["allies"], obs_b[agent]["allies"])
        assert np.array_equal(obs_a[agent]["enemies"], obs_b[agent]["enemies"])


def test_step_requires_exactly_the_live_agents():
    env = make_env(blue_units=1, red_units=1)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="actions must contain exactly"):
        env.step({"blue-0": sample_action()})  # missing red-0


def test_step_broadcasts_team_reward_to_every_living_unit_on_that_team():
    env = make_env(blue_units=2, red_units=1)
    env.reset(seed=1)
    actions = {agent: sample_action() for agent in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    assert rewards["blue-0"] == rewards["blue-1"] == pytest.approx(1.0)
    assert rewards["red-0"] == pytest.approx(-1.0)


def test_a_dying_unit_leaves_agents_and_is_terminated_exactly_once():
    env = make_env(blue_units=2, red_units=1, die_after_step=1)
    env.reset(seed=1)
    actions = {agent: sample_action() for agent in env.agents}
    _, _, terminations, _, _ = env.step(actions)  # step_count becomes 1: blue-0 dies
    assert terminations == {"blue-0": True, "blue-1": False, "red-0": False}
    assert set(env.agents) == {"blue-1", "red-0"}

    # The next step must not require an action from the now-removed agent.
    actions = {agent: sample_action() for agent in env.agents}
    assert "blue-0" not in actions
    env.step(actions)  # must not raise


def test_a_team_level_termination_ends_every_remaining_agent_on_both_teams():
    env = make_env(blue_units=2, red_units=2, terminate_after_step=1)
    env.reset(seed=1)
    actions = {agent: sample_action() for agent in env.agents}
    _, _, terminations, truncations, _ = env.step(actions)
    assert terminations == {"blue-0": True, "blue-1": True, "red-0": True, "red-1": True}
    assert truncations == {"blue-0": False, "blue-1": False, "red-0": False, "red-1": False}
    assert env.agents == []


def test_merged_team_action_places_a_missing_agent_as_noop_without_crashing():
    env = make_env(blue_units=2, red_units=1, die_after_step=1)
    env.reset(seed=1)
    env.step({agent: sample_action() for agent in env.agents})  # blue-0 dies here
    # blue-0 has no agent to submit an action next step; the merge must default it
    # to ACTION_NOOP rather than erroring, since encode_action treats a dead
    # unit's action as a no-op regardless of its value.
    merged = env._merge_team_actions({agent: sample_action() for agent in env.agents})
    assert merged["blue"]["action_type"][0] == ACTION_NOOP


def test_action_type_and_target_are_placed_in_the_correct_slot():
    env = make_env(blue_units=2, red_units=1)
    env.reset(seed=1)
    actions = {
        "blue-0": {"action_type": ACTION_MOVE, "target": np.array([0.5, -0.5], dtype=np.float32), "power": np.float32(0.0)},
        "blue-1": sample_action(),
        "red-0": sample_action(),
    }
    merged = env._merge_team_actions(actions)
    assert merged["blue"]["action_type"][0] == ACTION_MOVE
    assert merged["blue"]["action_type"][1] == ACTION_NOOP
    assert np.allclose(merged["blue"]["target"][0], [0.5, -0.5])
