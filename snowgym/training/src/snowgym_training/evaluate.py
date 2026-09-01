"""Held-out closed-loop evaluation for learned SnowGym checkpoints."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.client import SnowGymClient, SnowGymHttpClient
from snowgym_client.env import SnowGymEnv
from snowgym_client.recording import ReplayRecorder, write_replay

from .checkpoint import load_checkpoint
from .export_scripted import default_spec_path
from .policy import TorchPolicy
from .trajectory import json_digest, load_export_spec

EVALUATION_FORMAT = "snowgym.learned-evaluation.v0"


def default_baseline_path() -> Path:
    packaged = files("snowgym_training").joinpath(
        "baselines/teacher_1v1_v0.json"
    )
    if packaged.is_file():
        return Path(str(packaged))
    return Path(__file__).parents[2] / "baselines" / "teacher_1v1_v0.json"


def run_checkpoint_evaluation(
    *,
    checkpoint: str | Path,
    spec_path: str | Path | None = None,
    split: str = "evaluation",
    server_url: str = "http://127.0.0.1:8787",
    max_decisions: int = 400,
    replay_directory: str | Path | None = None,
    client: SnowGymClient | None = None,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    spec = load_export_spec(spec_path or default_spec_path())
    if split not in spec["splits"]:
        raise ValueError(f"unknown split {split!r}")
    metadata, _ = load_checkpoint(checkpoint)
    if metadata["evaluationSuite"] != f"teacher_1v1_v0/{split}":
        raise ValueError("checkpoint evaluation suite does not match requested split")
    policy = TorchPolicy(checkpoint)
    initial_scenario = spec["splits"][split][0]["scenario"]
    environment = SnowGymEnv(
        client=client or SnowGymHttpClient(server_url),
        max_team_units=int(spec["maxTeamUnits"]),
        configurable=True,
        blue_units=int(initial_scenario["blueUnits"]),
        red_units=int(initial_scenario["redUnits"]),
    )
    replay_root = Path(replay_directory) if replay_directory is not None else None
    if replay_root is not None and replay_root.exists():
        raise FileExistsError(f"refusing to overwrite replay directory {replay_root}")
    if replay_root is not None:
        replay_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    replay_paths: list[str] = []
    try:
        for episode in spec["splits"][split]:
            observation, info = environment.reset(
                seed=int(episode["seed"]), options={"scenario": episode["scenario"]}
            )
            raw = environment.raw_observation
            if raw is None:
                raise ValueError("server reset did not return raw observation")
            recorder = ReplayRecorder(raw, info) if replay_root is not None else None
            initial_blue_health = health_sum(observation, "allies", "ally_mask")
            initial_red_health = health_sum(observation, "enemies", "enemy_mask")
            decisions = rejected = 0
            terminated = truncated = False
            reward = 0.0
            while not (terminated or truncated) and decisions < max_decisions:
                action = policy.act(observation)
                observation, reward, terminated, truncated, info = environment.step(action)
                rejected += rejected_count(info.get("actionResults"))
                decisions += 1
                raw = environment.raw_observation
                if recorder is not None and raw is not None:
                    recorder.append(raw, info)
            if recorder is not None and replay_root is not None:
                replay_path = replay_root / f"learned-seed-{episode['seed']}.json"
                write_replay(replay_path, recorder.finish(decisions))
                replay_paths.append(replay_path.name)
            results.append(
                {
                    "policy": "learned",
                    "seed": episode["seed"],
                    "scenario": episode["scenario"],
                    "decisions": decisions,
                    "tick": info["tick"],
                    "stateHash": info["stateHash"],
                    "winner": info.get("winner"),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "decisionLimited": not (terminated or truncated),
                    "blueAlive": info["blueAlive"],
                    "redAlive": info["redAlive"],
                    "blueHealthLost": initial_blue_health
                    - health_sum(observation, "allies", "ally_mask"),
                    "redHealthDealt": initial_red_health
                    - health_sum(observation, "enemies", "enemy_mask"),
                    "rejectedActions": rejected,
                }
            )
    finally:
        environment.close()
    baseline = load_baseline(
        baseline_path or default_baseline_path(), spec, split, max_decisions
    )
    value = {
        "format": EVALUATION_FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "sourceSpecDigest": json_digest(spec),
        "split": split,
        "maxDecisions": max_decisions,
        "results": results,
        "summary": {
            "scripted_teacher": baseline["summary"]["scripted_teacher"],
            "masked_random": baseline["summary"]["masked_random"],
            "learned": summarize(results),
        },
        "replays": replay_paths,
    }
    value["resultDigest"] = json_digest(value)
    return value


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(results)
    if episodes == 0:
        raise ValueError("cannot summarize an empty evaluation")
    return {
        "episodes": episodes,
        "blueWins": sum(result["winner"] == "blue" for result in results),
        "redWins": sum(result["winner"] == "red" for result in results),
        "draws": sum(result["winner"] is None for result in results),
        "meanDecisions": sum(result["decisions"] for result in results) / episodes,
        "meanBlueHealthLost": sum(result["blueHealthLost"] for result in results)
        / episodes,
        "meanRedHealthDealt": sum(result["redHealthDealt"] for result in results)
        / episodes,
        "rejectedActions": sum(result["rejectedActions"] for result in results),
    }


def health_sum(observation: dict[str, np.ndarray], units: str, mask: str) -> float:
    values = np.asarray(observation[units], dtype=np.float32)
    present = np.asarray(observation[mask], dtype=np.int8).astype(bool)
    return float(values[present, 6].sum())


def rejected_count(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return sum(
        isinstance(result, dict) and result.get("accepted") is False for result in value
    )


def load_baseline(
    path: str | Path,
    spec: dict[str, Any],
    split: str,
    max_decisions: int,
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load teacher baseline: {error}") from error
    claimed = value.get("resultDigest")
    source = {key: item for key, item in value.items() if key != "resultDigest"}
    if claimed != json_digest(source):
        raise ValueError("teacher baseline digest mismatch")
    if value.get("sourceSpecDigest") != json_digest(spec) or value.get("split") != split:
        raise ValueError("teacher baseline does not match evaluation specification")
    if value.get("maxDecisions") != max_decisions:
        raise ValueError("teacher baseline maxDecisions does not match evaluation")
    return value


def write_evaluation(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SnowGym BC checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_checkpoint_evaluation(
            checkpoint=args.checkpoint,
            spec_path=args.spec,
            split=args.split,
            server_url=args.server,
            max_decisions=args.max_decisions,
            replay_directory=args.record_dir,
            baseline_path=args.baseline,
        )
        if args.output is not None:
            write_evaluation(args.output, result)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("SnowGym learned checkpoint evaluation")
        for policy, summary in result["summary"].items():
            print(f"  {policy}: wins={summary['blueWins']}/{summary['episodes']}")


if __name__ == "__main__":
    main()
