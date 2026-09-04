"""Frozen, seed-disjoint M7b experiment protocol loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .definitions import FROZEN_OPTION_SPECS

PROTOCOL_FORMAT = "snowgym.m7-option-protocol.v0"
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "m7_option_protocol_v0.json"


def load_option_protocol(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_PROTOCOL
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load M7 option protocol: {error}") from error
    if not isinstance(value, dict) or value.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"M7 option protocol format must be {PROTOCOL_FORMAT}")
    options = value.get("options")
    if not isinstance(options, dict) or set(options) != set(FROZEN_OPTION_SPECS):
        raise ValueError("M7 option protocol must define every frozen option exactly once")
    for name, spec in FROZEN_OPTION_SPECS.items():
        if not isinstance(options[name], dict) or options[name].get("horizon") != spec.horizon:
            raise ValueError(f"M7 option protocol {name} horizon changed")
    seeds = value.get("seeds")
    required_splits = {
        "teacherProof", "training", "development", "qualification", "planGeneration", "sealedMaps"
    }
    if not isinstance(seeds, dict) or set(seeds) != required_splits:
        raise ValueError("M7 option protocol seed splits are invalid")
    ranges = []
    for name, interval in seeds.items():
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in interval)
            or interval[0] > interval[1]
        ):
            raise ValueError(f"M7 option protocol {name} seed range is invalid")
        ranges.append((interval[0], interval[1], name))
    ordered = sorted(ranges)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left[1] >= right[0]:
            raise ValueError(
                f"M7 option protocol seed splits overlap: {left[2]} and {right[2]}"
            )
    paired = value.get("pairedSeedsPerMission")
    if paired != {"development": 40, "qualification": 100}:
        raise ValueError("M7 option paired seed counts changed")
    return value
