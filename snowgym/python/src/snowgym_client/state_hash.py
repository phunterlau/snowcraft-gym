"""Cross-language canonical hashing for detached SnowGym observations."""

from __future__ import annotations

import math
from typing import Any

LEGACY_STATE_HASH_VERSION = "snowgym.state.v1"
STATE_HASH_VERSION = "snowgym.state.v2"
SCALE = 1_000_000_000
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
UINT64_MASK = 0xFFFFFFFFFFFFFFFF

JsonObject = dict[str, Any]


def hash_observation(
    observation: JsonObject, version: str = STATE_HASH_VERSION
) -> str:
    """Return the same public-state regression checksum as TypeScript."""
    value = FNV_OFFSET
    for byte in canonical_observation(observation, version).encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & UINT64_MASK
    return f"fnv1a64:{value:016x}"


def canonical_observation(
    observation: JsonObject, version: str = STATE_HASH_VERSION
) -> str:
    if version == LEGACY_STATE_HASH_VERSION:
        return canonical_observation_v1(observation)
    if version != STATE_HASH_VERSION:
        raise ValueError(f"unsupported state hash version {version!r}")
    arena = require_object(observation.get("arena"), "arena")
    simulation_hz = require_integer(observation.get("simulationHz"), "simulationHz")
    decision = observation.get("decision")
    if decision is None:
        decision = {
            "hz": simulation_hz,
            "dt": 1.0 / simulation_hz,
            "maxTicks": 2**53 - 1,
            "remainingFraction": 1.0,
        }
    decision = require_object(decision, "decision")
    tokens = [
        STATE_HASH_VERSION,
        str(observation.get("observationVersion", "legacy")),
        integer(observation.get("tick"), "tick"),
        required_string(observation.get("selfTeam"), "selfTeam"),
        str(simulation_hz),
        quantized(arena.get("width"), "arena.width"),
        quantized(arena.get("height"), "arena.height"),
        "decision",
        quantized(decision.get("hz"), "decision.hz"),
        quantized(decision.get("dt"), "decision.dt"),
        integer(decision.get("maxTicks"), "decision.maxTicks"),
        quantized(decision.get("remainingFraction"), "decision.remainingFraction"),
    ]
    append_units(
        tokens, "allies", require_list(observation.get("allies"), "allies"), True
    )
    append_units(
        tokens, "enemies", require_list(observation.get("enemies"), "enemies"), True
    )
    append_projectiles(
        tokens, require_list(observation.get("projectiles"), "projectiles"), True
    )
    append_obstacles(tokens, observation.get("obstacles", []))
    match = require_object(observation.get("match"), "match")
    tokens.extend(
        [
            "match",
            integer(match.get("blueAlive"), "match.blueAlive"),
            integer(match.get("redAlive"), "match.redAlive"),
        ]
    )
    return "|".join(tokens)


def canonical_observation_v1(observation: JsonObject) -> str:
    arena = require_object(observation.get("arena"), "arena")
    tokens = [
        LEGACY_STATE_HASH_VERSION,
        integer(observation.get("tick"), "tick"),
        required_string(observation.get("selfTeam"), "selfTeam"),
        integer(observation.get("simulationHz"), "simulationHz"),
        quantized(arena.get("width"), "arena.width"),
        quantized(arena.get("height"), "arena.height"),
    ]
    append_units(
        tokens, "allies", require_list(observation.get("allies"), "allies"), False
    )
    append_units(
        tokens, "enemies", require_list(observation.get("enemies"), "enemies"), False
    )
    append_projectiles(
        tokens, require_list(observation.get("projectiles"), "projectiles"), False
    )
    append_obstacles(tokens, observation.get("obstacles", []))
    match = require_object(observation.get("match"), "match")
    tokens.extend(
        [
            "match",
            integer(match.get("blueAlive"), "match.blueAlive"),
            integer(match.get("redAlive"), "match.redAlive"),
        ]
    )
    return "|".join(tokens)


def append_units(
    tokens: list[str], label: str, values: list[Any], include_controller: bool
) -> None:
    units = sorted(
        (require_object(value, label) for value in values),
        key=lambda unit: require_integer(unit.get("id"), f"{label}.id"),
    )
    tokens.extend([label, str(len(units))])
    for unit in units:
        tokens.extend(
            [
                integer(unit.get("id"), f"{label}.id"),
                required_string(unit.get("team"), f"{label}.team"),
                quantized(unit.get("x"), f"{label}.x"),
                quantized(unit.get("y"), f"{label}.y"),
                quantized(unit.get("vx"), f"{label}.vx"),
                quantized(unit.get("vy"), f"{label}.vy"),
                quantized(unit.get("health"), f"{label}.health"),
                quantized(unit.get("maxHealth"), f"{label}.maxHealth"),
                "1" if bool(unit.get("alive", False)) else "0",
                required_string(unit.get("state"), f"{label}.state"),
                quantized(unit.get("throwCooldown"), f"{label}.throwCooldown"),
                quantized(unit.get("charge"), f"{label}.charge"),
            ]
        )
        if include_controller:
            append_point(tokens, "moveTarget", unit.get("moveTarget"))
            append_point(tokens, "steeringTarget", unit.get("steeringTarget"))
            append_point(tokens, "aimDirection", unit.get("aimDirection"))
            tokens.extend(
                [
                    quantized(unit.get("stunRemaining", 0), f"{label}.stunRemaining"),
                    quantized(
                        unit.get("throwPhaseRemaining", 0),
                        f"{label}.throwPhaseRemaining",
                    ),
                    quantized(
                        unit.get("immunityRemaining", 0),
                        f"{label}.immunityRemaining",
                    ),
                    quantized(unit.get("speedRemaining", 0), f"{label}.speedRemaining"),
                ]
            )


def append_projectiles(
    tokens: list[str], values: list[Any], include_age: bool
) -> None:
    projectiles = sorted(
        (require_object(value, "projectile") for value in values),
        key=lambda projectile: require_integer(projectile.get("id"), "projectile.id"),
    )
    tokens.extend(["projectiles", str(len(projectiles))])
    for projectile in projectiles:
        tokens.extend(
            [
                integer(projectile.get("id"), "projectile.id"),
                integer(projectile.get("ownerId"), "projectile.ownerId"),
                required_string(projectile.get("team"), "projectile.team"),
                quantized(projectile.get("x"), "projectile.x"),
                quantized(projectile.get("y"), "projectile.y"),
                quantized(projectile.get("vx"), "projectile.vx"),
                quantized(projectile.get("vy"), "projectile.vy"),
                quantized(projectile.get("height"), "projectile.height"),
                quantized(projectile.get("heightVelocity"), "projectile.heightVelocity"),
            ]
        )
        if include_age:
            tokens.append(quantized(projectile.get("age", 0), "projectile.age"))


def append_point(tokens: list[str], label: str, value: Any) -> None:
    if value is None:
        tokens.extend([label, "0"])
        return
    point = require_object(value, label)
    tokens.extend(
        [
            label,
            "1",
            quantized(point.get("x"), f"{label}.x"),
            quantized(point.get("y"), f"{label}.y"),
        ]
    )


def append_obstacles(tokens: list[str], values: Any) -> None:
    obstacles = sorted(
        (require_object(value, "obstacle") for value in require_list(values, "obstacles")),
        key=lambda obstacle: require_integer(obstacle.get("id"), "obstacle.id"),
    )
    tokens.extend(["obstacles", str(len(obstacles))])
    for obstacle in obstacles:
        tokens.extend(
            [
                integer(obstacle.get("id"), "obstacle.id"),
                required_string(obstacle.get("type"), "obstacle.type"),
                quantized(obstacle.get("x"), "obstacle.x"),
                quantized(obstacle.get("y"), "obstacle.y"),
                quantized(obstacle.get("halfWidth"), "obstacle.halfWidth"),
                quantized(obstacle.get("halfHeight"), "obstacle.halfHeight"),
                "1" if bool(obstacle.get("blocksSight", False)) else "0",
                "1" if bool(obstacle.get("blocksProjectiles", False)) else "0",
                "1" if bool(obstacle.get("blocksMovement", False)) else "0",
            ]
        )


def quantized(value: Any, name: str) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return str(math.floor(number * SCALE + 0.5))


def integer(value: Any, name: str) -> str:
    return str(require_integer(value, name))


def require_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_object(value: Any, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value
