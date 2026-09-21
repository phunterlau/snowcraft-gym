"""M8-S7 post-hoc correction (`reviews/m8_s7_results.md`, section on the action-share defect).

`roster_trace.episode_metrics` computed `throwActionShare` / `moveActionShare` over every blue
unit's action entry, including entries for dead units, which deflates the shares of policies that
lose units. Those two measures were not declared measures or predictions and are left as archived
(the sealed run pins the module's digest). This module recomputes them over living units only,
from the archived traces."""

from __future__ import annotations

from . import roster_trace as rt


def live_action_shares(traces):
    """Post-contact action-type shares over living blue unit-decisions, pooled across episodes."""
    counts = {"throw": 0, "move": 0, "noop": 0, "other": 0}
    total = 0
    for trace in traces:
        hit = rt.first_hit_decision(trace["states"])
        if hit is None:
            continue
        for t in range(hit - 1, len(trace["acts"])):
            blue = trace["states"][t]["blue"]
            for unit_id, kind, _accepted in trace["acts"][t]:
                if not blue[unit_id - 1][0]:
                    continue
                counts[kind if kind in counts else "other"] += 1
                total += 1
    return {kind: (count / total if total else None) for kind, count in counts.items()} | {"unitDecisions": total}
