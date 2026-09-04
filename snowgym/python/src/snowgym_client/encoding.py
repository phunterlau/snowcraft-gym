"""Numeric Gymnasium spaces and SnowGym JSON encoders."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

JsonObject = dict[str, Any]
GymAction = dict[str, np.ndarray]
GymObservation = dict[str, np.ndarray]

MAX_TEAM_UNITS = 3
MAX_CONFIGURABLE_TEAM_UNITS = 10
MAX_PROJECTILES = 64
MAX_OBSTACLES = 64
UNIT_FEATURES = 10
UNIT_FEATURES_V3 = 21
PROJECTILE_FEATURES = 8
PROJECTILE_FEATURES_V3 = 9
OBSTACLE_FEATURES = 9
VELOCITY_SCALE = 20.0
HEIGHT_SCALE = 3.0
HEIGHT_VELOCITY_SCALE = 10.0

ACTION_NOOP = 0
ACTION_MOVE = 1
ACTION_THROW = 2
ACTION_HOLD = 3
ACTION_TYPE_COUNT = 4

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
            "action_type": spaces.MultiDiscrete(
                np.full(max_team_units, ACTION_TYPE_COUNT, dtype=np.int64)
            ),
            "target": spaces.Box(-1.0, 1.0, shape=(max_team_units, 2), dtype=np.float32),
            "power": spaces.Box(0.0, 1.0, shape=(max_team_units,), dtype=np.float32),
        }
    )


def make_observation_space(
    max_team_units: int = MAX_TEAM_UNITS,
    *,
    include_unit_masks: bool = False,
    observation_version: int = 2,
) -> spaces.Dict:
    max_team_units = validate_team_capacity(max_team_units)
    validate_observation_version(observation_version)
    unit_features = UNIT_FEATURES_V3 if observation_version == 3 else UNIT_FEATURES
    projectile_features = (
        PROJECTILE_FEATURES_V3 if observation_version == 3 else PROJECTILE_FEATURES
    )
    definitions: dict[str, spaces.Space] = {
            "allies": spaces.Box(-1.0, 1.0, shape=(max_team_units, unit_features), dtype=np.float32),
            "enemies": spaces.Box(-1.0, 1.0, shape=(max_team_units, unit_features), dtype=np.float32),
            "projectiles": spaces.Box(
                -1.0,
                1.0,
                shape=(MAX_PROJECTILES, projectile_features),
                dtype=np.float32,
            ),
            "projectile_mask": spaces.Box(0, 1, shape=(MAX_PROJECTILES,), dtype=np.int8),
            "unit_action_mask": spaces.Box(
                0,
                1,
                shape=(max_team_units, ACTION_TYPE_COUNT),
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
    if observation_version == 3:
        definitions.update(
            {
                "decision_hz": spaces.Box(1, 60, shape=(1,), dtype=np.int32),
                "decision_dt": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "max_ticks": spaces.Box(
                    1, np.iinfo(np.int64).max, shape=(1,), dtype=np.int64
                ),
                "remaining_fraction": spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
    return spaces.Dict(definitions)


def encode_observation(
    raw: JsonObject,
    max_team_units: int = MAX_TEAM_UNITS,
    *,
    include_unit_masks: bool = False,
    observation_version: int = 2,
) -> GymObservation:
    max_team_units = validate_team_capacity(max_team_units)
    validate_observation_version(observation_version)
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

    unit_features = UNIT_FEATURES_V3 if observation_version == 3 else UNIT_FEATURES
    projectile_features = (
        PROJECTILE_FEATURES_V3 if observation_version == 3 else PROJECTILE_FEATURES
    )
    allies = np.zeros((max_team_units, unit_features), dtype=np.float32)
    enemies = np.zeros((max_team_units, unit_features), dtype=np.float32)
    projectiles = np.zeros((MAX_PROJECTILES, projectile_features), dtype=np.float32)
    projectile_mask = np.zeros(MAX_PROJECTILES, dtype=np.int8)
    obstacles = np.zeros((MAX_OBSTACLES, OBSTACLE_FEATURES), dtype=np.float32)
    obstacle_mask = np.zeros(MAX_OBSTACLES, dtype=np.int8)
    action_mask = np.zeros((max_team_units, ACTION_TYPE_COUNT), dtype=np.int8)
    ally_mask = np.zeros(max_team_units, dtype=np.int8)
    enemy_mask = np.zeros(max_team_units, dtype=np.int8)

    for index, unit in enumerate(allies_raw):
        unit_data = require_object(unit, "ally")
        allies[index] = encode_unit(unit_data, width, height, observation_version)
        action_mask[index] = encode_action_mask(unit_data)
        ally_mask[index] = 1
    for index, unit in enumerate(enemies_raw):
        enemies[index] = encode_unit(
            require_object(unit, "enemy"), width, height, observation_version
        )
        enemy_mask[index] = 1
    for index, projectile in enumerate(projectiles_raw[:MAX_PROJECTILES]):
        projectiles[index] = encode_projectile(
            require_object(projectile, "projectile"), width, height, observation_version
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
    if observation_version == 3:
        if raw.get("observationVersion") != "snowgym.observation.v1":
            raise ValueError("v3 encoding requires snowgym.observation.v1")
        decision = require_dict(raw, "decision")
        encoded.update(
            {
                "decision_hz": np.asarray([integer(decision, "hz")], dtype=np.int32),
                "decision_dt": np.asarray([number(decision, "dt")], dtype=np.float32),
                "max_ticks": np.asarray([integer(decision, "maxTicks")], dtype=np.int64),
                "remaining_fraction": np.asarray(
                    [number(decision, "remainingFraction")], dtype=np.float32
                ),
            }
        )
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
        if action_type == ACTION_HOLD:
            semantic_actions.append({"type": "hold", "unitId": unit_id})
        elif action_type == ACTION_MOVE:
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


def decode_action(
    semantic_action: JsonObject,
    raw_observation: JsonObject,
    max_team_units: int = MAX_TEAM_UNITS,
) -> GymAction:
    """Convert a semantic team action into the fixed Gym tensor action.

    Missing live units are represented as no-ops. Unknown, duplicate, or
    wrong-team unit ids are rejected so teacher labels cannot silently drift.
    """
    max_team_units = validate_team_capacity(max_team_units)
    arena = require_dict(raw_observation, "arena")
    half_width = positive_number(arena, "width") / 2.0
    half_height = positive_number(arena, "height") / 2.0
    allies_raw = require_list(raw_observation, "allies")
    actions_raw = require_list(semantic_action, "actions")
    if len(allies_raw) > max_team_units:
        raise ValueError(f"server ally roster exceeds Gym capacity {max_team_units}")

    ally_indices: dict[int, int] = {}
    ally_alive: dict[int, bool] = {}
    for index, raw_unit in enumerate(allies_raw):
        unit = require_object(raw_unit, "ally")
        unit_id = integer(unit, "id")
        if unit_id in ally_indices:
            raise ValueError(f"duplicate ally unit id {unit_id}")
        ally_indices[unit_id] = index
        ally_alive[unit_id] = bool(unit.get("alive", False))

    action_types = np.full(max_team_units, ACTION_NOOP, dtype=np.int64)
    targets = np.zeros((max_team_units, 2), dtype=np.float32)
    powers = np.zeros(max_team_units, dtype=np.float32)
    seen: set[int] = set()
    allowed_fields = {
        "noop": {"type", "unitId"},
        "hold": {"type", "unitId"},
        "move": {"type", "unitId", "x", "y"},
        "throw": {"type", "unitId", "x", "y", "power"},
    }
    type_indices = {
        "noop": ACTION_NOOP,
        "move": ACTION_MOVE,
        "throw": ACTION_THROW,
        "hold": ACTION_HOLD,
    }
    for raw_action in actions_raw:
        action = require_object(raw_action, "action")
        action_type = action.get("type")
        if action_type not in allowed_fields:
            raise ValueError(f"unknown semantic action type {action_type!r}")
        unknown = set(action) - allowed_fields[str(action_type)]
        missing = allowed_fields[str(action_type)] - set(action)
        if unknown or missing:
            raise ValueError(
                f"invalid {action_type} fields: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        unit_id = integer(action, "unitId")
        if unit_id not in ally_indices:
            raise ValueError(f"semantic action references non-ally unit {unit_id}")
        if unit_id in seen:
            raise ValueError(f"duplicate semantic action for unit {unit_id}")
        seen.add(unit_id)
        if not ally_alive[unit_id] and action_type != "noop":
            raise ValueError(f"dead unit {unit_id} must receive noop")
        index = ally_indices[unit_id]
        action_types[index] = type_indices[str(action_type)]
        if action_type in {"move", "throw"}:
            targets[index, 0] = np.float32(
                np.clip(number(action, "x") / half_width, -1.0, 1.0)
            )
            targets[index, 1] = np.float32(
                np.clip(number(action, "y") / half_height, -1.0, 1.0)
            )
        if action_type == "throw":
            powers[index] = np.float32(np.clip(number(action, "power"), 0.0, 1.0))
    return {"action_type": action_types, "target": targets, "power": powers}


def encode_unit(
    unit: JsonObject, width: float, height: float, observation_version: int = 2
) -> np.ndarray:
    max_health = max(number(unit, "maxHealth"), 1.0)
    state = unit.get("state")
    if state not in STATE_INDEX:
        raise ValueError(f"unknown unit state {state!r}")
    values = [
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
        ]
    if observation_version == 3:
        move_target = optional_point(unit.get("moveTarget"), "moveTarget")
        steering_target = optional_point(unit.get("steeringTarget"), "steeringTarget")
        aim = require_object(unit.get("aimDirection"), "aimDirection")
        values.extend(
            [
                1.0 if move_target is not None else 0.0,
                normalize(move_target[0], width / 2.0) if move_target else 0.0,
                normalize(move_target[1], height / 2.0) if move_target else 0.0,
                normalize(steering_target[0], width / 2.0) if steering_target else 0.0,
                normalize(steering_target[1], height / 2.0) if steering_target else 0.0,
                normalize(number(aim, "x"), 1.0),
                normalize(number(aim, "y"), 1.0),
                normalize(number(unit, "stunRemaining"), 1.0),
                normalize(number(unit, "throwPhaseRemaining"), 1.0),
                normalize(number(unit, "immunityRemaining"), 5.0),
                normalize(number(unit, "speedRemaining"), 6.0),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def encode_projectile(
    projectile: JsonObject,
    width: float,
    height: float,
    observation_version: int = 2,
) -> np.ndarray:
    team = projectile.get("team")
    if team not in {"blue", "red"}:
        raise ValueError(f"unknown projectile team {team!r}")
    values = [
            1.0,
            -1.0 if team == "blue" else 1.0,
            normalize(number(projectile, "x"), width / 2.0),
            normalize(number(projectile, "y"), height / 2.0),
            normalize(number(projectile, "vx"), VELOCITY_SCALE),
            normalize(number(projectile, "vy"), VELOCITY_SCALE),
            normalize(number(projectile, "height"), HEIGHT_SCALE),
            normalize(number(projectile, "heightVelocity"), HEIGHT_VELOCITY_SCALE),
        ]
    if observation_version == 3:
        values.append(normalize(number(projectile, "age"), 5.0))
    return np.asarray(values, dtype=np.float32)


def encode_action_mask(unit: JsonObject) -> np.ndarray:
    state = unit.get("state")
    alive = bool(unit.get("alive", False))
    cooldown_ready = number(unit, "throwCooldown") <= 0.0
    can_move = alive and state in {"idle", "moving", "recovering"}
    can_throw = alive and cooldown_ready and state in {"idle", "moving", "preparingThrow"}
    can_hold = alive and state in {"idle", "moving", "recovering"}
    return np.asarray([1, int(can_move), int(can_throw), int(can_hold)], dtype=np.int8)


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


def optional_point(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    point = require_object(value, name)
    return number(point, "x"), number(point, "y")


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


def validate_observation_version(value: int) -> int:
    if value not in (2, 3) or isinstance(value, bool):
        raise ValueError("observation_version must be 2 or 3")
    return value
