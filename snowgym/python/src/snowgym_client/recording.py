"""Versioned visual recording artifacts for SnowGym episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPLAY_FORMAT = "snowgym.replay.v0"
JsonObject = dict[str, Any]


class ReplayRecorder:
    """Collect detached server observations and semantic actions."""

    def __init__(self, observation: JsonObject, status: JsonObject):
        self._frames = [observation]
        self._actions: list[JsonObject] = []
        self._status = status

    def append(self, observation: JsonObject, info: JsonObject) -> None:
        action = info.get("action")
        if not isinstance(action, dict):
            action = {"actions": []}
        self._actions.append(action)
        self._frames.append(observation)
        self._status = info

    def finish(self, decisions: int) -> JsonObject:
        status = self._status
        replay = {
            "format": REPLAY_FORMAT,
            "apiVersion": required_string(status, "apiVersion"),
            "scenario": required_string(status, "scenario"),
            "seed": required_integer(status, "seed"),
            "simulationHz": required_integer(status, "simulationHz"),
            "decisionHz": required_integer(status, "decisionHz"),
            "ticksPerDecision": required_integer(status, "ticksPerDecision"),
            "frames": self._frames,
            "actions": self._actions,
            "outcome": {
                "decisions": decisions,
                "terminated": bool(status.get("terminated", False)),
                "truncated": bool(status.get("truncated", False)),
                "winner": status.get("winner"),
                "blueAlive": required_integer(status, "blueAlive"),
                "redAlive": required_integer(status, "redAlive"),
                "finalTick": required_integer(status, "tick"),
            },
        }
        configuration = status.get("configuration")
        if isinstance(configuration, dict):
            replay["configuration"] = configuration
        return replay


def write_replay(path: str | Path, replay: JsonObject) -> Path:
    """Write a portable, human-inspectable replay JSON file."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    return destination


def required_integer(record: JsonObject, key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"recording status is missing integer {key}")
    return value


def required_string(record: JsonObject, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"recording status is missing string {key}")
    return value
