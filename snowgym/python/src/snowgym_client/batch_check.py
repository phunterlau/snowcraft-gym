"""Live golden parity check between HTTP and persistent batch transports."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .batch import SnowGymBatchClient
from .client import SnowGymHttpClient


def run_batch_parity(
    server_url: str, *, worlds: int = 8, seed: int = 7000
) -> dict[str, Any]:
    if worlds <= 0:
        raise ValueError("worlds must be positive")
    http = SnowGymHttpClient(server_url)
    capabilities = http.capabilities()
    scenario = {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": 1200,
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }
    checked: list[dict[str, Any]] = []
    with SnowGymBatchClient() as batch:
        batch_api = batch.capabilities["capabilities"]["apiVersion"]
        if batch_api != capabilities.get("apiVersion"):
            raise RuntimeError("HTTP and batch capability versions differ")
        for index in range(worlds):
            world_id = f"parity-{index}"
            world_seed = seed + index
            reset_key = f"parity-reset-{index}"
            http_reset = http.reset(
                world_seed, scenario, idempotency_key=reset_key
            )
            batch_reset = batch.request(
                "reset",
                [
                    {
                        "worldId": world_id,
                        "body": {
                            "seed": world_seed,
                            "scenario": scenario,
                            "idempotencyKey": reset_key,
                        },
                    }
                ],
            )[0]
            if batch_reset.get("status") != 200 or batch_reset.get("body") != http_reset:
                raise RuntimeError(f"reset parity failed for {world_id}")
            state_hash = http_reset["status"]["stateHash"]
            action = {
                "actions": [
                    {"type": "hold", "unitId": http_reset["observation"]["allies"][0]["id"]}
                ]
            }
            step_key = f"parity-step-{index}"
            http_step = http.step(
                action,
                expected_state_hash=state_hash,
                idempotency_key=step_key,
            )
            batch_step = batch.request(
                "step",
                [
                    {
                        "worldId": world_id,
                        "body": {
                            "action": action,
                            "expectedStateHash": state_hash,
                            "idempotencyKey": step_key,
                        },
                    }
                ],
            )[0]
            if batch_step.get("status") != 200 or batch_step.get("body") != http_step:
                raise RuntimeError(f"step parity failed for {world_id}")
            checked.append(
                {
                    "worldId": world_id,
                    "seed": world_seed,
                    "resetStateHash": state_hash,
                    "stepStateHash": http_step["info"]["stateHash"],
                }
            )
    return {
        "ok": True,
        "format": "snowgym.batch-parity.v0",
        "apiVersion": capabilities["apiVersion"],
        "worlds": worlds,
        "checked": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HTTP/batch SnowGym parity")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_batch_parity(args.server, worlds=args.worlds, seed=args.seed)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
