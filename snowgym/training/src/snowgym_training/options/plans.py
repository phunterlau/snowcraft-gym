"""Frozen production-teacher plans used to establish option achievability."""

from __future__ import annotations

from typing import Any

from .definitions import FROZEN_OPTION_SPECS, OptionSpec


def teacher_option_plan(name: str) -> tuple[dict[str, Any], OptionSpec]:
    if name not in FROZEN_OPTION_SPECS:
        raise ValueError(f"unknown teacher option {name!r}")
    engagement = {
        "posture": "balanced",
        "fire": name if name in {"focus", "distributed"} else "opportunistic",
        "preferredRange": "medium",
        "cohesion": "normal",
    }
    if name == "engage" or name in {"focus", "distributed", "flank"}:
        order = {
            "mission": "engage",
            "objective": {"kind": "enemy_cluster", "select": "nearest"},
            "approach": "left_flank" if name == "flank" else "direct",
            "engagement": engagement,
        }
    elif name == "advance":
        order = {
            "mission": "advance",
            "objective": {"kind": "region", "region": "enemy_backfield"},
            "approach": "direct",
            "engagement": engagement,
        }
    elif name == "hold":
        order = {
            "mission": "hold",
            "objective": {"kind": "current_position"},
            "approach": "direct",
            "engagement": engagement,
        }
    elif name == "withdraw":
        order = {
            "mission": "withdraw",
            "objective": {"kind": "region", "region": "own_backfield"},
            "approach": "direct",
            "engagement": {**engagement, "posture": "conservative"},
        }
    else:
        main = {
            "role": "main",
            "allocationWeight": 4,
            "selection": "frontline",
            "order": {
                "mission": "hold",
                "objective": {"kind": "current_position"},
                "approach": "direct",
                "engagement": engagement,
            },
        }
        reserve = {
            "role": "reserve",
            "allocationWeight": 1,
            "selection": "rearline",
            "order": {
                "mission": "support",
                "objective": {"kind": "ally_group", "role": "main"},
                "approach": "avoid_center",
                "engagement": {**engagement, "posture": "conservative"},
            },
        }
        return (
            {
                "schemaVersion": "snowgym.command-plan.v0",
                "intentSummary": "Hold the main line while reserve supports it.",
                "groups": [main, reserve],
            },
            OptionSpec("support", 300, role="reserve"),
        )
    return (
        {
            "schemaVersion": "snowgym.command-plan.v0",
            "intentSummary": f"Fixed {name} option teacher proof.",
            "groups": [
                {
                    "role": "main",
                    "allocationWeight": 1,
                    "selection": "balanced",
                    "order": order,
                }
            ],
        },
        FROZEN_OPTION_SPECS[name],
    )


def teacher_option_scenario(name: str | None = None) -> dict[str, Any]:
    return {
        "blueUnits": 5,
        "redUnits": 5,
        "arenaWidth": 30 if name == "support" else 100,
        "arenaHeight": 20 if name == "support" else 80,
        "maxTicks": 1800,
        "decisionHz": 10,
        "redDifficulty": "easy",
        "redController": "random",
    }
