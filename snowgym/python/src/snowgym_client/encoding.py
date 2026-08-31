"""Numeric Gymnasium spaces and SnowGym JSON encoders."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

JsonObject = dict[str, Any]
GymAction = dict[str, np.ndarray]
GymObservation = dict[str, np.ndarray]

MAX_TEAM_UNITS = 3
MAX_CONFIGURABLE_TEAM_UNITS = 8
MAX_PROJECTILES = 64
MAX_OBSTACLES = 64
UNIT_FEATURES = 10
PROJECTILE_FEATURES = 8
OBSTACLE_FEATURES = 9
VELOCITY_SCALE = 20.0
HEIGHT_SCALE = 3.0
HEIGHT_VELOCITY_SCALE = 10.0

ACTION_NOOP = 0
ACTION_MOVE = 1
ACTION_THROW = 2

STATE_INDEX = {
    "idle": 0,
    "moving": 1,
    "preparingThrow": 2,
    "throwing": 3,
    "recovering": 4,
    "hit": 5,
    "frozen": 6,
    "defeated": 7,
}

OBSTACLE_TYPE_INDEX = {
    "tree": 0,
    "rock": 1,
    "fort": 2,
    "fence": 3,
    "prop": 4,
}


def make_action_space(max_team_units: int = MAX_TEAM_UNITS) -> spaces.Dict:
    max_team_units = validate_team_capacity(max_team_units)
    return spaces.Dict(
        {
            "action_type": spaces.MultiDiscrete(np.full(max_team_units, 3, dtype=np.int64)),
            "target": spaces.Box(-1.0, 1.0, shape=(max_team_units, 2), dtype=np.float32),
            "power": spaces.Box(0.0, 1.0, shape=(max_team_units,), dtype=np.float32),
        }
    )


def make_observation_space(
    max_team_units: int = MAX_TEAM_UNITS,
    *,
    include_unit_masks: bool = False,
) -> spaces.Dict:
    max_team_units = validate_team_capacity(max_team_units)
    definitions: dict[str, spaces.Space] = {
            "allies": spaces.Box(-1.0, 1.0, shape=(max_team_units, UNIT_FEATURES), dtype=np.float32),
            "enemies": spaces.Box(-1.0, 1.0, shape=(max_team_units, UNIT_FEATURES), dtype=np.float32),
            "projectiles": spaces.Box(
                -1.0,
                1.0,
                shape=(MAX_PROJECTILES, PROJECTILE_FEATURES),
                dtype=np.float32,
            ),
            "projectile_mask": spaces.Box(0, 1, shape=(MAX_PROJECTILES,), dtype=np.int8),
            "unit_action_mask": spaces.Box(
                0,
                1,
                shape=(max_team_units, 3),
                dtype=np.int8,
            ),
            "tick": spaces.Box(0, np.iinfo(np.int64).max, shape=(1,), dtype=np.int64),
            "team_alive": spaces.Box(0, max_team_units, shape=(2,), dtype=np.int32),
            "obstacles": spaces.Box(
                -1.0,
                1.0,
                shape=(MAX_OBSTACLES, OBSTACLE_FEATURES),
                dtype=np.float32,
            ),
            "obstacle_mask": spaces.Box(0, 1, shape=(MAX_OBSTACLES,), dtype=np.int8),
    }
    if include_unit_masks:
        definitions["ally_mask"] = spaces.Box(0, 1, shape=(max_team_units,), dtype=np.int8)
        definitions["enemy_mask"] = spaces.Box(0, 1, shape=(max_team_units,), dtype=np.int8)
    return spaces.Dict(definitions)


def encode_observation(
    raw: JsonObject,
    max_team_units: int = MAX_TEAM_UNITS,
    *,
    include_unit_masks: bool = False,
) -> GymObservation:
    max_team_units = validate_team_capacity(max_team_units)
    arena = require_dict(raw, "arena")
    width = positive_number(arena, "width")
    height = positive_number(arena, "height")
    allies_raw = require_list(raw, "allies")
    enemies_raw = require_list(raw, "enemies")
    projectiles_raw = require_list(raw, "projectiles")
    obstacles_raw = require_list(raw, "obstacles")
    match = require_dict(raw, "match")

    if len(allies_raw) > max_team_units or len(enemies_raw) > max_team_units:
        raise ValueError(
            f"server roster exceeds Gym capacity {max_team_units}: "
            f"allies={len(allies_raw)} enemies={len(enemies_raw)}"
        )

    allies = np.zeros((max_team_units, UNIT_FEATURES), dtype=np.float32)
    enemies = np.zeros((max_team_units, UNIT_FEATURES), dtype=np.float32)
    projectiles = np.zeros((MAX_PROJECTILES, PROJECTILE_FEATURES), dtype=np.float32)
    projectile_mask = np.zeros(MAX_PROJECTILES, dtype=np.int8)
    obstacles = np.zeros((MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32)
    obstacle_mask = np.zeros(MAX_OBSTACLES, dtype=np.int8)
    action_mask = np.zeros((max_team_units, 3), dtype=np.int8)
    ally_mask = np.zeros(max_team_units, dtype=np.int8)
    enemy_mask = np.zeros(max_team_units, dtype=np.int8)

    for index, unit in enumerate(allies_raw):
        unit_data = require_object(unit, "ally")
        allies[index] = encode_unit(unit_data, width, height)
        action_mask[index] = encode_action_mask(unit_data)
        ally_mask[index] = 1
    for index, unit in enumerate(enemies_raw):
        enemies[index] = encode_unit(require_object(unit, "enemy"), width, height)
        enemy_mask[index] = 1
    for index, projectile in enumerate(projectiles_raw[:MAX_PROJECTILES]):
        projectiles[index] = encode_projectile(
            require_object(projectile, "projectile"), width, height
        )
        projectile_mask[index] = 1
    for index, obstacle in enumerate(obstacles_raw[:MAX_OBSTACLES]):
        obstacles[index] = encode_obstacle(
            require_object(obstacle, "obstacle"), width, height
        )
        obstacle_mask[index] = 1

    encoded = {
        "allies": allies,
        "enemies": enemies,
        "projectiles": projectiles,
        "projectile_mask": projectile_mask,
        "unit_action_mask": action_mask,
        "tick": np.asarray([integer(raw, "tick")], dtype=np.int64),
        "team_alive": np.asarray(
            [integer(match, "blueAlive"), integer(match, "redAlive")], dtype=np.int32
        ),
        "obstacles": obstacles,
        "obstacle_mask": obstacle_mask,
    }
    if include_unit_masks:
        encoded["ally_mask"] = ally_mask
        encoded["enemy_mask"] = enemy_mask
    return encoded


def encode_action(
    action: GymAction,
    raw_observation: JsonObject,
    max_team_units: int = MAX_TEAM_UNITS,
) -> JsonObject:
    max_team_units = validate_team_capacity(max_team_units)
    arena = require_dict(raw_observation, "arena")
    half_width = positive_number(arena, "width") / 2.0
    half_height = positive_number(arena, "height") / 2.0
    allies = require_list(raw_observation, "allies")
    action_types = np.asarray(action["action_type"], dtype=np.int64)
    targets = np.asarray(action["target"], dtype=np.float32)
    powers = np.asarray(action["power"], dtype=np.float32)
    semantic_actions: list[JsonObject] = []

    if len(allies) > max_team_units:
        raise ValueError(f"server ally roster exceeds Gym capacity {max_team_units}")
    for index, raw_unit in enumerate(allies):
        unit = require_object(raw_unit, "ally")
        unit_id = integer(unit, "id")
        action_type = int(action_types[index])
        if not bool(unit.get("alive", False)) or action_type == ACTION_NOOP:
            semantic_actions.append({"type": "noop", "unitId": unit_id})
            continue

        x = float(np.clip(targets[index, 0], -1.0, 1.0) * half_width)
        y = float(np.clip(targets[index, 1], -1.0, 1.0) * half_height)
        if action_type == ACTION_MOVE:
            semantic_actions.append({"type": "move", "unitId": unit_id, "x": x, "y": y})
        elif action_type == ACTION_THROW:
            semantic_actions.append(
                {
                    "type": "throw",
                    "unitId": unit_id,
                    "x": x,
                    "y": y,
                    "power": float(np.clip(powers[index], 0.0, 1.0)),
                }
            )
        else:
            raise ValueError(f"unknown action type {action_type}")

    return {"actions": semantic_actions}


def encode_unit(unit: JsonObject, width: float, height: float) -> np.ndarray:
    max_health = max(number(unit, "maxHealth"), 1.0)
    state = unit.get("state")
    if state not in STATE_INDEX:
        raise ValueError(f"unknown unit state {state!r}")
    return np.asarray(
        [
            1.0,
            1.0 if bool(unit.get("alive", False)) else 0.0,
            normalize(number(unit, "x"), width / 2.0),
            normalize(number(unit, "y"), height / 2.0),
            normalize(number(unit, "vx"), VELOCITY_SCALE),
            normalize(number(unit, "vy"), VELOCITY_SCALE),
            float(np.clip(number(unit, "health") / max_health, 0.0, 1.0)),
            float(np.clip(number(unit, "throwCooldown"), 0.0, 1.0)),
            float(np.clip(number(unit, "charge"), 0.0, 1.0)),
            STATE_INDEX[str(state)] / max(len(STATE_INDEX) - 1, 1),
        ],
        dtype=np.float32,
    )


def encode_projectile(projectile: JsonObject, width: float, height: float) -> np.ndarray:
    team = projectile.get("team")
    if team not in {"blue", "red"}:
        raise ValueError(f"unknown projectile team {team!r}")
    return np.asarray(
        [
            1.0,
            -1.0 if team == "blue" else 1.0,
            normalize(number(projectile, "x"), width / 2.0),
            normalize(number(projectile, "y"), height / 2.0),
            normalize(number(projectile, "vx"), VELOCITY_SCALE),
            normalize(number(projectile, "vy"), VELOCITY_SCALE),
            normalize(number(projectile, "height"), HEIGHT_SCALE),
            normalize(number(projectile, "heightVelocity"), HEIGHT_VELOCITY_SCALE),
        ],
        dtype=np.float32,
    )


def encode_action_mask(unit: JsonObject) -> np.ndarray:
    state = unit.get("state")
    alive = bool(unit.get("alive", False))
    cooldown_ready = number(unit, "throwCooldown") <= 0.0
    can_move = alive and state in {"idle", "moving", "recovering"}
    can_throw = alive and cooldown_ready and state in {"idle", "moving", "preparingThrow"}
    return np.asarray([1, int(can_move), int(can_throw)], dtype=np.int8)


def encode_obstacle(obstacle: JsonObject, width: float, height: float) -> np.ndarray:
    obstacle_type = obstacle.get("type")
    if obstacle_type not in OBSTACLE_TYPE_INDEX:
        raise ValueError(f"unknown obstacle type {obstacle_type!r}")
    return np.asarray(
        [
            OBSTACLE_TYPE_INDEX[obstacle_type] / (len(OBSTACLE_TYPE_INDEX) - 1),
            normalize(number(obstacle, "x"), width / 2.0),
            normalize(number(obstacle, "y"), height / 2.0),
            normalize(number(obstacle, "halfWidth"), width / 2.0),
            normalize(number(obstacle, "halfHeight"), height / 2.0),
            float(bool(obstacle.get("blocksSight", False))),
            float(bool(obstacle.get("blocksProjectiles", False))),
            float(bool(obstacle.get("blocksMovement", False))),
            1.0,  # presence flag (padding rows are all-zero)
        ],
        dtype=np.float32,
    )


def normalize(value: float, scale: float) -> float:
    return float(np.clip(value / max(scale, 1e-9), -1.0, 1.0))


def require_dict(record: JsonObject, key: str) -> JsonObject:
    return require_object(record.get(key), key)


def require_object(value: Any, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_list(record: JsonObject, key: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def number(record: JsonObject, key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def positive_number(record: JsonObject, key: str) -> float:
    value = number(record, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def integer(record: JsonObject, key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def validate_team_capacity(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("max_team_units must be an integer")
    if value < 1 or value > MAX_CONFIGURABLE_TEAM_UNITS:
        raise ValueError(
            f"max_team_units must be in [1, {MAX_CONFIGURABLE_TEAM_UNITS}]"
        )
    return value
