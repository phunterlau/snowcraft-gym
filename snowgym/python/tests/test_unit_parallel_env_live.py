"""M8-S2: simulator-backed checks of `SnowGymUnitParallelEnv` (declaration §3).

Each test runs against a real `snowgym/server/main.ts` subprocess, unlike
`test_unit_parallel_env.py`, whose fake client cannot establish physics parity,
real death/removal, or timeout behavior."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest

from snowgym_client.encoding import ACTION_MOVE, ACTION_NOOP, ACTION_THROW
from snowgym_client.parallel_env import SnowGymParallelEnv
from snowgym_client.unit_parallel_env import SnowGymUnitParallelEnv

REPOSITORY = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run the simulator")


def start_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        ["node", "--import", "tsx", "snowgym/server/main.ts", "--port", str(port)],
        cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    assert "listening" in (process.stdout.readline() if process.stdout else "")
    return process, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def server_url():
    process, url = start_server()
    yield url
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def second_server_url():
    process, url = start_server()
    yield url
    process.terminate()
    process.wait(timeout=10)


def noop():
    return {"action_type": ACTION_NOOP, "target": np.zeros(2, dtype=np.float32), "power": np.float32(0.0)}


def scripted_unit_action(agent_index: int, step: int) -> dict:
    """A pure function of (agent_index, step): a fixed mix of NOOP/MOVE/THROW."""
    kind = (step + agent_index) % 3
    target = np.array([np.sin(step + agent_index), np.cos(step - agent_index)], dtype=np.float32) * 0.8
    return {"action_type": [ACTION_NOOP, ACTION_MOVE, ACTION_THROW][kind], "target": target,
            "power": np.float32(0.25 * (1 + agent_index % 3))}


def focus_fire_red_zero(env: SnowGymUnitParallelEnv, agent: str) -> dict:
    """Blue units throw at the lowest-id living enemy; red does nothing. A deterministic
    script under which red-0 dies while red-1 and red-2 are still alive."""
    team, slot = agent.rsplit("-", 1)
    if team == "red":
        return noop()
    raw = env.environment.raw_observations["blue"]
    me = raw["allies"][int(slot)]
    foes = [e for e in raw["enemies"] if e["alive"]]
    if not foes:
        return noop()
    foe = min(foes, key=lambda e: e["id"])
    half_w, half_h = raw["arena"]["width"] / 2, raw["arena"]["height"] / 2
    distance = ((foe["x"] - me["x"]) ** 2 + (foe["y"] - me["y"]) ** 2) ** 0.5
    target = np.array([np.clip(foe["x"] / half_w, -1, 1), np.clip(foe["y"] / half_h, -1, 1)], dtype=np.float32)
    return {"action_type": ACTION_THROW if distance < 14 else ACTION_MOVE, "target": target,
            "power": np.float32(1.0)}


def team_action(team: str, step: int, capacity: int, roster: int) -> dict:
    """The same scripted actions, laid out independently of the unit wrapper's merge."""
    action_type = np.zeros(capacity, dtype=np.int64)
    target = np.zeros((capacity, 2), dtype=np.float32)
    power = np.zeros(capacity, dtype=np.float32)
    for slot in range(roster):
        index = slot + (0 if team == "blue" else roster)
        unit = scripted_unit_action(index, step)
        action_type[slot], target[slot], power[slot] = unit["action_type"], unit["target"], unit["power"]
    return {"action_type": action_type, "target": target, "power": power}


def test_same_actions_give_identical_state_hashes_and_observations_as_the_team_env(server_url, second_server_url):
    team_env = SnowGymParallelEnv(server_url=server_url, blue_units=3, red_units=3, max_ticks=1800)
    unit_env = SnowGymUnitParallelEnv(server_url=second_server_url, blue_units=3, red_units=3, max_ticks=1800)
    team_obs, team_info = team_env.reset(seed=11)
    unit_obs, unit_info = unit_env.reset(seed=11)
    assert team_info["blue"]["stateHash"] == unit_info["blue-0"]["stateHash"]
    for step in range(40):
        team_obs, _, _, _, team_info = team_env.step(
            {team: team_action(team, step, team_env.max_team_units, 3) for team in ("blue", "red")})
        unit_obs, _, _, _, unit_info = unit_env.step(
            {f"{team}-{i}": scripted_unit_action(i + (0 if team == "blue" else 3), step)
             for team in ("blue", "red") for i in range(3)})
        assert unit_info["blue-0"]["stateHash"] == team_info["blue"]["stateHash"], f"diverged at step {step}"
        for agent, observation in unit_obs.items():
            for key, value in team_obs[agent.rsplit("-", 1)[0]].items():
                assert np.array_equal(observation[key], value), (step, agent, key)


def test_same_seed_and_actions_reproduce_the_state_hash_sequence(server_url):
    def run() -> list[str]:
        env = SnowGymUnitParallelEnv(server_url=server_url, blue_units=3, red_units=3, max_ticks=1800)
        env.reset(seed=5)
        hashes = []
        for step in range(25):
            _, _, _, _, infos = env.step({a: scripted_unit_action(env.possible_agents.index(a), step)
                                          for a in env.agents})
            hashes.append(infos["blue-0"]["stateHash"])
        return hashes

    assert run() == run()


def test_a_pure_timeout_truncates_every_survivor_without_terminating_any(server_url):
    env = SnowGymUnitParallelEnv(server_url=server_url, blue_units=2, red_units=2, max_ticks=60)
    env.reset(seed=3)
    for step in range(10):
        _, _, terminations, truncations, infos = env.step({agent: noop() for agent in env.agents})
        if truncations["blue-0"]:
            break
    assert step == 9, "60 ticks at 6 ticks per decision is ten steps"
    assert not any(terminations.values())
    assert all(truncations.values())
    assert env.agents == []
    assert infos["blue-0"]["snowgym_unit"]["team_terminated"] is False
    assert infos["blue-0"]["truncated"] is True


def test_a_real_mid_battle_death_removes_only_that_unit_and_keeps_slot_identity(server_url):
    env = SnowGymUnitParallelEnv(server_url=server_url, blue_units=3, red_units=3, max_ticks=1800)
    # seed 0 found by running this script over seeds 0-2: red-0 dies at decision 30 with
    # red-1/red-2 alive (a symmetric nearest-enemy script killed all of red in one step).
    env.reset(seed=0)
    ids_before = [u["id"] for u in env.environment.raw_observations["red"]["allies"]]
    death_step = None
    for step in range(200):
        actions = {agent: focus_fire_red_zero(env, agent) for agent in env.agents}
        before = list(env.agents)
        _, _, terminations, truncations, infos = env.step(actions)
        if terminations.get("red-0"):
            death_step = step
            break
    assert death_step is not None, "the scripted focus-fire never killed red-0"
    assert [a for a, done in terminations.items() if done] == ["red-0"]
    assert not any(truncations.values())
    assert infos["red-0"]["snowgym_unit"] == {
        "team_terminated": False, "team_truncated": False, "unit_alive": False, "unit_died": True}
    assert set(env.agents) == set(before) - {"red-0"}
    # Stable slots: the dead unit stays in its slot, flagged not alive, never reindexed.
    red_allies = env.environment.raw_observations["red"]["allies"]
    assert [u["id"] for u in red_allies] == ids_before
    assert [u["alive"] for u in red_allies] == [False, True, True]

    # The battle continues without red-0 needing (or being allowed) an action.
    _, _, terminations, _, _ = env.step({agent: focus_fire_red_zero(env, agent) for agent in env.agents})
    assert "red-0" not in terminations
    while env.agents:
        _, _, terminations, truncations, infos = env.step(
            {agent: focus_fire_red_zero(env, agent) for agent in env.agents})
    assert all(terminations.values()) and not any(truncations.values())
    assert all(info["snowgym_unit"]["team_terminated"] for info in infos.values())


def test_a_malformed_action_never_reaches_the_simulator(server_url):
    env = SnowGymUnitParallelEnv(server_url=server_url, blue_units=1, red_units=1, max_ticks=1800)
    env.reset(seed=2)
    bad = {**noop(), "action_type": 2.7, "target": np.float32(0.25)}
    with pytest.raises(ValueError, match="blue-0"):
        env.step({"blue-0": bad, "red-0": noop()})
    _, _, _, _, infos = env.step({"blue-0": noop(), "red-0": noop()})
    assert infos["blue-0"]["tick"] == 6, "the rejected step must not have advanced the simulation"
