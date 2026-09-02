"""Record a deterministic PPO blue-team episode through the headless batch host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_client.recording import ReplayRecorder, write_replay

from .curriculum import load_curriculum
from .model import model_config
from .ppo import HybridActorCritic
from .ppo_checkpoint import load_ppo_checkpoint
from .ppo_collect import numpy_actions, tensor_dict
from .trajectory import json_digest


def run_ppo_demo(
    *,
    checkpoint: str | Path,
    output: str | Path,
    gate_id: str = "1v1-random",
    seed: int = 3101,
    max_decisions: int = 400,
    curriculum_path: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(max_decisions, int) or isinstance(max_decisions, bool) or max_decisions <= 0:
        raise ValueError("max_decisions must be a positive integer")
    curriculum = load_curriculum(curriculum_path)
    gate = next((item for item in curriculum["gates"] if item["id"] == gate_id), None)
    if gate is None:
        raise ValueError(f"unknown PPO curriculum gate {gate_id!r}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    if metadata["curriculumDigest"] != json_digest(curriculum):
        raise ValueError("PPO checkpoint curriculum does not match replay curriculum")
    if metadata["collectorConfig"]["gateId"] != gate_id:
        raise ValueError("PPO checkpoint gate does not match replay gate")
    model = HybridActorCritic(model_config(metadata["architecture"])).cpu()
    model.load_state_dict(state["model"])
    model.eval()

    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(1, client=client)
        observation, infos = environment.reset([seed], [dict(gate["scenario"])])
        raw = environment.raw_observations[0]
        if raw is None:
            raise RuntimeError("batch reset did not return a raw observation")
        recorder = ReplayRecorder(raw, infos[0])
        decisions = rejected = 0
        terminated = truncated = False
        info = infos[0]
        while not (terminated or truncated) and decisions < max_decisions:
            with torch.no_grad():
                action, _, _ = model.act(tensor_dict(observation), deterministic=True)
            observation, _, terminated_values, truncated_values, infos = environment.step(
                numpy_actions(action)
            )
            terminated = bool(terminated_values[0])
            truncated = bool(truncated_values[0])
            info = infos[0]
            rejected += sum(
                isinstance(item, dict) and item.get("accepted") is False
                for item in info.get("actionResults", [])
            )
            decisions += 1
            raw = environment.raw_observations[0]
            if raw is None:
                raise RuntimeError("batch step did not return a raw observation")
            recorder.append(raw, info)
    if not (terminated or truncated):
        raise RuntimeError("PPO replay stopped before the episode completed")
    replay_path = write_replay(destination, recorder.finish(decisions))
    return {
        "ok": True,
        "format": "snowgym.ppo-demo.v0",
        "checkpointDigest": metadata["checkpointDigest"],
        "gate": gate_id,
        "seed": seed,
        "decisions": decisions,
        "winner": info.get("winner"),
        "blueAlive": info["blueAlive"],
        "redAlive": info["redAlive"],
        "rejectedActions": rejected,
        "stateHash": info["stateHash"],
        "replay": str(replay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a SnowGym PPO blue-team replay")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--gate", default="1v1-random")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_ppo_demo(
            checkpoint=args.checkpoint,
            output=args.record,
            gate_id=args.gate,
            seed=args.seed,
            max_decisions=args.max_decisions,
            curriculum_path=args.curriculum,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
