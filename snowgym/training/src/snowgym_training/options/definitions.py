"""Predeclared M7b option horizons and success thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionSpec:
    name: str
    horizon: int
    role: str = "main"

    def __post_init__(self) -> None:
        if self.name not in FROZEN_OPTION_NAMES:
            raise ValueError(f"unknown fixed option {self.name!r}")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError("option horizon must be a positive integer")
        if self.role not in {"main", "maneuver", "reserve"}:
            raise ValueError("option role is invalid")


FROZEN_OPTION_NAMES = (
    "engage",
    "advance",
    "hold",
    "withdraw",
    "flank",
    "focus",
    "distributed",
    "support",
)

FROZEN_OPTION_SPECS = {
    "engage": OptionSpec("engage", 200),
    "advance": OptionSpec("advance", 150),
    "hold": OptionSpec("hold", 150),
    "withdraw": OptionSpec("withdraw", 200),
    "flank": OptionSpec("flank", 200),
    "focus": OptionSpec("focus", 200),
    "distributed": OptionSpec("distributed", 200),
    "support": OptionSpec("support", 300),
}
