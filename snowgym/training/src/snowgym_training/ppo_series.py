"""Checkpoint-series orchestration for auditable SnowGym PPO runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .model import ModelConfig
from .ppo import PPOConfig
from .ppo_evaluate import evaluate_ppo_checkpoint
from .ppo_train import train_ppo
from .trainer import resolve_git_commit
from .trajectory import json_digest

PPO_SERIES_FORMAT = "snowgym.ppo-series.v0"


def run_ppo_series(
    *,
    output: str | Path,
    checkpoints: list[int],
    gate_id: str = "1v1-random",
    worlds: int = 8,
    rollout_steps: int = 32,
    training_seed: int = 73,
    reward_mode: str = "canonical",
    curriculum_path: str | Path | None = None,
    max_decisions: int = 400,
    qualifying: bool = False,
    model_config: ModelConfig | None = None,
    ppo_config: PPOConfig | None = None,
    git_commit: str | None = None,
    warm_start: str | Path | None = None,
    series_config_digest: str | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite PPO series {destination}")
    if (
        not checkpoints
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("checkpoints must be unique, strictly increasing positive integers")
    commit = git_commit or resolve_git_commit()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    result_root = temporary / destination.name
    result_root.mkdir()
    entries: list[dict[str, Any]] = []
    learning_curve: list[dict[str, Any]] = []
    previous_checkpoint: Path | None = None
    curriculum_digest: str | None = None
    gate: dict[str, Any] | None = None
    try:
        for target in checkpoints:
            relative_run = Path("checkpoints") / f"update-{target:06d}"
            run = train_ppo(
                output=result_root / relative_run,
                gate_id=gate_id,
                worlds=worlds,
                rollout_steps=rollout_steps,
                target_updates=target,
                training_seed=training_seed,
                curriculum_path=curriculum_path,
                resume=previous_checkpoint,
                model_config=model_config,
                ppo_config=ppo_config,
                git_commit=commit,
                reward_mode=reward_mode,
                mode="qualification-candidate" if qualifying else "development-series",
                warm_start=warm_start if previous_checkpoint is None else None,
            )
            checkpoint = result_root / relative_run / "checkpoint"
            evaluation = evaluate_ppo_checkpoint(
                checkpoint=checkpoint,
                gate_id=gate_id,
                curriculum_path=curriculum_path,
                max_decisions=max_decisions,
            )
            evaluation_path = result_root / "evaluations" / f"update-{target:06d}.json"
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            evaluation_path.write_text(
                json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            learning_curve.extend(run["updates"])
            curriculum_digest = run["curriculumDigest"]
            gate = run["gate"]
            entries.append(
                {
                    "update": target,
                    "environmentSteps": run["environmentSteps"],
                    "runPath": str(relative_run),
                    "runDigest": run["runDigest"],
                    "checkpointDigest": run["checkpoint"]["checkpointDigest"],
                    "evaluationPath": str(evaluation_path.relative_to(result_root)),
                    "evaluationDigest": evaluation["resultDigest"],
                    "thresholdPassed": evaluation["threshold"]["passed"],
                }
            )
            previous_checkpoint = checkpoint
        manifest: dict[str, Any] = {
            "format": PPO_SERIES_FORMAT,
            "mode": "qualifying" if qualifying else "development",
            "gitCommit": commit,
            "seriesConfigDigest": series_config_digest,
            "curriculumDigest": curriculum_digest,
            "gateId": gate_id,
            "gate": gate,
            "checkpointUpdates": checkpoints,
            "worlds": worlds,
            "rolloutSteps": rollout_steps,
            "trainingSeed": training_seed,
            "rewardMode": reward_mode,
            "initialization": entries and run["initialization"],
            "maxEvaluationDecisions": max_decisions,
            "checkpoints": entries,
            "learningCurve": learning_curve,
            "finalThresholdPassed": entries[-1]["thresholdPassed"],
        }
        manifest["seriesDigest"] = json_digest(manifest)
        (result_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(result_root, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a SnowGym PPO series")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--gate", default="1v1-random")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--training-seed", type=int, default=73)
    parser.add_argument("--reward-mode", choices=("canonical", "health-potential"), default="canonical")
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--qualifying", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_ppo_series(
            output=args.output,
            checkpoints=args.checkpoints,
            gate_id=args.gate,
            worlds=args.worlds,
            rollout_steps=args.rollout_steps,
            training_seed=args.training_seed,
            reward_mode=args.reward_mode,
            curriculum_path=args.curriculum,
            max_decisions=args.max_decisions,
            qualifying=args.qualifying,
            warm_start=args.warm_start,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "mode": result["mode"],
        "gate": result["gateId"],
        "checkpoints": result["checkpointUpdates"],
        "finalThresholdPassed": result["finalThresholdPassed"],
        "seriesDigest": result["seriesDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
