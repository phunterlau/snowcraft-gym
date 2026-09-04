from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from snowgym_client.state_hash import LEGACY_STATE_HASH_VERSION, hash_observation


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "state-hash-v1.json"
V2_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "state-hash-v2.json"


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_matches_shared_typescript_python_golden_fixture() -> None:
    fixture = load_fixture()

    assert (
        hash_observation(fixture["observation"], LEGACY_STATE_HASH_VERSION)
        == fixture["expected"]
    )


def test_hash_is_independent_of_entity_array_order() -> None:
    fixture = load_fixture()
    observation = copy.deepcopy(fixture["observation"])
    for key in ("allies", "enemies", "projectiles", "obstacles"):
        observation[key].reverse()

    assert hash_observation(observation, LEGACY_STATE_HASH_VERSION) == fixture["expected"]


def test_hash_changes_with_public_state() -> None:
    fixture = load_fixture()
    observation = copy.deepcopy(fixture["observation"])
    observation["allies"][0]["health"] -= 1

    assert hash_observation(observation, LEGACY_STATE_HASH_VERSION) != fixture["expected"]


def test_current_hash_includes_persistent_controller_state() -> None:
    observation = load_fixture()["observation"]
    first = hash_observation(observation)
    observation["allies"][0]["moveTarget"] = {"x": 4, "y": -2}
    observation["allies"][0]["steeringTarget"] = {"x": 1, "y": -1}

    assert hash_observation(observation) != first


def test_matches_shared_actuator_complete_v2_fixture() -> None:
    fixture = json.loads(V2_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert hash_observation(fixture["observation"]) == fixture["expected"]
