"""Headless production-teacher achievability runner for frozen options."""

from __future__ import annotations

from typing import Any

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from .environment import FixedPlanOptionBatchEnv
from .plans import teacher_option_plan, teacher_option_scenario


def evaluate_teacher_option(
    name: str,
    *,
    seed: int,
    gamma: float = 0.9976921765,
    client: SnowGymBatchClient | None = None,
) -> dict[str, Any]:
    owned = client is None
    batch_client = client or SnowGymBatchClient()
    plan, spec = teacher_option_plan(name)
    scenario = teacher_option_scenario(name)
    environment = FixedPlanOptionBatchEnv(
        SnowGymBatchEnv(1, client=batch_client, observation_version=3), gamma=gamma
    )
    try:
        environment.reset([seed], [scenario], [f"teacher-{name}-{seed}"], [plan], [spec])
        final = None
        rejected = 0
        for _ in range(spec.horizon):
            _, _, terminated, truncated, infos = environment.step(teacher=True)
            final = infos[0]["option"]
            rejected += sum(
                result.get("accepted") is False
                for result in infos[0].get("actionResults", [])
            )
            if bool(terminated[0] or truncated[0]):
                break
        if final is None:
            raise RuntimeError("teacher option evaluation produced no decisions")
        return {
            "option": name,
            "seed": seed,
            "success": final["success"],
            "failed": final["failed"],
            "timedOut": final["timedOut"],
            "decisions": final["decision"],
            "progress": final["progress"],
            "rejectedActions": rejected,
            "metrics": final["metrics"],
        }
    finally:
        if owned:
            batch_client.close()
