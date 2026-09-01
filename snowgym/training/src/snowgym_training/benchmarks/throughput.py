"""Persistent batch-host throughput benchmark with serialization accounting."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

BENCHMARK_FORMAT = "snowgym.batch-throughput.v0"


def run_throughput_benchmark(
    world_counts: list[int], *, decisions: int = 20, seed: int = 9000
) -> dict[str, Any]:
    if not world_counts or any(count <= 0 for count in world_counts):
        raise ValueError("world counts must be positive")
    if decisions <= 0:
        raise ValueError("decisions must be positive")
    results = [benchmark_worlds(count, decisions=decisions, seed=seed) for count in world_counts]
    return {
        "format": BENCHMARK_FORMAT,
        "decisionsPerWorld": decisions,
        "results": results,
    }


def benchmark_worlds(worlds: int, *, decisions: int, seed: int) -> dict[str, Any]:
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parent_cpu_before = time.process_time()
    wall_started = time.perf_counter()
    client = SnowGymBatchClient()
    environment = SnowGymBatchEnv(worlds, client=client)
    scenario = {
        "blueUnits": 1,
        "redUnits": 1,
        "arenaWidth": 40,
        "arenaHeight": 30,
        "maxTicks": max(1200, decisions * 6 + 6),
        "decisionHz": 10,
        "redDifficulty": "normal",
        "redController": "random",
    }
    try:
        environment.reset(
            [seed + index for index in range(worlds)],
            [dict(scenario) for _ in range(worlds)],
        )
        actions = {
            "action_type": np.zeros((worlds, 10), dtype=np.int64),
            "target": np.zeros((worlds, 10, 2), dtype=np.float32),
            "power": np.zeros((worlds, 10), dtype=np.float32),
        }
        ticks = 0
        for _ in range(decisions):
            _, _, terminated, truncated, infos = environment.step(actions)
            if terminated.any() or truncated.any():
                raise RuntimeError("benchmark world ended before the decision budget")
            ticks += sum(int(info["ticksPerDecision"]) for info in infos)
    finally:
        client.close()
    elapsed = time.perf_counter() - wall_started
    parent_cpu = time.process_time() - parent_cpu_before
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime
        + child_after.ru_stime
        - child_before.ru_utime
        - child_before.ru_stime
    )
    total_decisions = worlds * decisions
    serialization = client.serialization_seconds
    return {
        "worlds": worlds,
        "decisions": total_decisions,
        "ticks": ticks,
        "elapsedSeconds": elapsed,
        "decisionsPerSecond": total_decisions / elapsed,
        "simulationTicksPerSecond": ticks / elapsed,
        "realTimeFactor": (ticks / 60.0) / elapsed,
        "cpuUtilization": (parent_cpu + child_cpu) / elapsed,
        "payloadBytes": client.payload_bytes,
        "serializationSeconds": serialization,
        "serializationShare": serialization / elapsed,
        "transportAndSimulationSeconds": client.io_seconds,
    }


def write_benchmark(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark persistent SnowGym worlds")
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_throughput_benchmark(
            args.worlds, decisions=args.decisions, seed=args.seed
        )
        if args.output is not None:
            write_benchmark(args.output, result)
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("SnowGym persistent batch throughput")
        for item in result["results"]:
            print(
                f"  {item['worlds']:>2} worlds: "
                f"{item['decisionsPerSecond']:.1f} decisions/s"
            )


if __name__ == "__main__":
    main()
