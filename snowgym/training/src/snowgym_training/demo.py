"""Run the committed learned blue policy and record a visual replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from snowgym_client.client import SnowGymClient, SnowGymHttpClient
from snowgym_client.env import SnowGymEnv
from snowgym_client.recording import ReplayRecorder, write_replay

from .policy import TorchPolicy


def default_checkpoint_path() -> Path:
    return Path(__file__).parents[2] / "checkpoints" / "bc_1v1_v0"


def run_learned_demo(
    *,
    output: str | Path,
    checkpoint: str | Path | None = None,
    server_url: str = "http://127.0.0.1:8787",
    seed: int = 42,
    max_decisions: int = 400,
    client: SnowGymClient | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    policy = TorchPolicy(checkpoint or default_checkpoint_path())
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
    environment = SnowGymEnv(
        client=client or SnowGymHttpClient(server_url),
        max_team_units=10,
        configurable=True,
        blue_units=1,
        red_units=1,
    )
    observation, info = environment.reset(
        seed=seed, options={"scenario": scenario}
    )
    raw = environment.raw_observation
    if raw is None:
        raise RuntimeError("server reset did not return a raw observation")
    recorder = ReplayRecorder(raw, info)
    decisions = rejected = 0
    terminated = truncated = False
    try:
        while not (terminated or truncated) and decisions < max_decisions:
            action = policy.act(observation)
            observation, _, terminated, truncated, info = environment.step(action)
            results = info.get("actionResults")
            if isinstance(results, list):
                rejected += sum(
                    isinstance(result, dict) and result.get("accepted") is False
                    for result in results
                )
            decisions += 1
            raw = environment.raw_observation
            if raw is None:
                raise RuntimeError("server step did not return a raw observation")
            recorder.append(raw, info)
    finally:
        environment.close()
    if not (terminated or truncated):
        raise RuntimeError("learned demo stopped before the episode completed")
    replay_path = write_replay(destination, recorder.finish(decisions))
    return {
        "ok": True,
        "format": "snowgym.learned-demo.v0",
        "checkpointDigest": policy.metadata["checkpointDigest"],
        "seed": seed,
        "decisions": decisions,
        "tick": info["tick"],
        "winner": info.get("winner"),
        "blueAlive": info["blueAlive"],
        "redAlive": info["redAlive"],
        "rejectedActions": rejected,
        "stateHash": info["stateHash"],
        "replay": str(replay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the learned SnowGym blue policy and record a replay"
    )
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_learned_demo(
            output=args.record,
            checkpoint=args.checkpoint,
            server_url=args.server,
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("SnowGym learned blue-team demo")
        print(f"  seed:       {result['seed']}")
        print(f"  decisions:  {result['decisions']}")
        print(f"  winner:     {result['winner']}")
        print(f"  survivors:  blue={result['blueAlive']} red={result['redAlive']}")
        print(f"  rejections: {result['rejectedActions']}")
        print(f"  replay:     {result['replay']}")


if __name__ == "__main__":
    main()
