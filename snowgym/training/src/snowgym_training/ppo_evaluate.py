"""Held-out headless evaluation for SnowGym PPO checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from snowgym_client.opponents import masked_random_action

from .curriculum import load_curriculum
from .executor import model_config
from .ppo import HybridActorCritic
from .ppo_checkpoint import load_ppo_checkpoint
from .ppo_collect import numpy_actions, tensor_dict
from .trajectory import json_digest

PPO_EVALUATION_FORMAT = "snowgym.ppo-evaluation.v0"
POLICIES = ("ppo", "masked_random", "scripted_teacher")


def evaluate_ppo_checkpoint(
    *,
    checkpoint: str | Path,
    gate_id: str = "1v1-random",
    curriculum_path: str | Path | None = None,
    max_decisions: int = 400,
) -> dict[str, Any]:
    if not isinstance(max_decisions, int) or isinstance(max_decisions, bool) or max_decisions <= 0:
        raise ValueError("max_decisions must be a positive integer")
    curriculum = load_curriculum(curriculum_path)
    gate = next((item for item in curriculum["gates"] if item["id"] == gate_id), None)
    if gate is None:
        raise ValueError(f"unknown PPO curriculum gate {gate_id!r}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    if metadata["curriculumDigest"] != json_digest(curriculum):
        raise ValueError("PPO checkpoint curriculum does not match evaluation curriculum")
    if metadata["collectorConfig"]["gateId"] != gate_id:
        raise ValueError("PPO checkpoint gate does not match evaluation gate")
    model = HybridActorCritic(model_config(metadata["architecture"])).cpu()
    model.load_state_dict(state["model"])
    model.eval()

    results: list[dict[str, Any]] = []
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(1, client=client)
        for policy in POLICIES:
            for seed in gate["evaluationSeeds"]:
                observation, _ = environment.reset([seed], [dict(gate["scenario"])])
                generator = np.random.default_rng(
                    np.random.SeedSequence([seed, 0x50504F, POLICIES.index(policy)])
                )
                decisions = 0
                rejected = 0
                terminated = truncated = False
                reward = 0.0
                info: dict[str, Any] = {}
                while not (terminated or truncated) and decisions < max_decisions:
                    if policy == "scripted_teacher":
                        transition = environment.step_scripted()
                    elif policy == "masked_random":
                        single = {name: values[0] for name, values in observation.items()}
                        action = masked_random_action(single, generator)
                        transition = environment.step(
                            {name: values[None, ...] for name, values in action.items()}
                        )
                    else:
                        with torch.no_grad():
                            action, _, _ = model.act(tensor_dict(observation), deterministic=True)
                        transition = environment.step(numpy_actions(action))
                    observation, rewards, terminated_values, truncated_values, infos = transition
                    reward = float(rewards[0])
                    terminated = bool(terminated_values[0])
                    truncated = bool(truncated_values[0])
                    info = infos[0]
                    rejected += sum(
                        isinstance(item, dict) and item.get("accepted") is False
                        for item in info.get("actionResults", [])
                    )
                    decisions += 1
                if reward not in (-1.0, 0.0, 1.0):
                    raise RuntimeError("evaluation received non-canonical terminal reward")
                results.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "decisions": decisions,
                        "tick": info.get("tick", 0),
                        "winner": info.get("winner"),
                        "canonicalReturn": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "decisionLimited": not (terminated or truncated),
                        "blueAlive": info.get("blueAlive"),
                        "redAlive": info.get("redAlive"),
                        "rejectedActions": rejected,
                    }
                )
    summary = {
        policy: summarize([item for item in results if item["policy"] == policy])
        for policy in POLICIES
    }
    learned = summary["ppo"]
    random = summary["masked_random"]
    threshold = {
        "minimumWinRate": gate["minimumWinRate"],
        "minimumImprovementOverMaskedRandom": gate[
            "minimumImprovementOverMaskedRandom"
        ],
    }
    threshold["passed"] = (
        learned["winRate"] >= threshold["minimumWinRate"]
        and learned["winRate"] - random["winRate"]
        >= threshold["minimumImprovementOverMaskedRandom"]
    )
    value: dict[str, Any] = {
        "format": PPO_EVALUATION_FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "curriculumDigest": metadata["curriculumDigest"],
        "gate": gate,
        "maxDecisions": max_decisions,
        "results": results,
        "summary": summary,
        "threshold": threshold,
    }
    value["resultDigest"] = json_digest(value)
    return value


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty PPO evaluation")
    episodes = len(results)
    blue_wins = sum(item["winner"] == "blue" for item in results)
    return {
        "episodes": episodes,
        "blueWins": blue_wins,
        "redWins": sum(item["winner"] == "red" for item in results),
        "draws": sum(item["winner"] not in ("blue", "red") for item in results),
        "winRate": blue_wins / episodes,
        "meanCanonicalReturn": sum(item["canonicalReturn"] for item in results) / episodes,
        "meanDecisions": sum(item["decisions"] for item in results) / episodes,
        "rejectedActions": sum(item["rejectedActions"] for item in results),
    }


def write_evaluation(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SnowGym PPO checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", default="1v1-random")
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_ppo_checkpoint(
            checkpoint=args.checkpoint,
            gate_id=args.gate,
            curriculum_path=args.curriculum,
            max_decisions=args.max_decisions,
        )
        if args.output is not None:
            write_evaluation(args.output, result)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "gate": result["gate"]["id"],
        "checkpointDigest": result["checkpointDigest"],
        "threshold": result["threshold"],
        "resultDigest": result["resultDigest"],
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
