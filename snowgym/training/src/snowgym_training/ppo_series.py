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
from .ppo_checkpoint import load_ppo_checkpoint
from .ppo_train import train_ppo
from .trainer import resolve_git_commit
from .trajectory import json_digest

PPO_SERIES_FORMAT = "snowgym.ppo-series.v0"


def audit_ppo_series(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load PPO series manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != PPO_SERIES_FORMAT:
        raise ValueError(f"PPO series format must be {PPO_SERIES_FORMAT}")
    claimed = manifest.get("seriesDigest")
    source = {name: value for name, value in manifest.items() if name != "seriesDigest"}
    if claimed != json_digest(source):
        raise ValueError("PPO series manifest digest mismatch")
    entries = manifest.get("checkpoints")
    updates = manifest.get("checkpointUpdates")
    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
        or not isinstance(updates, list)
    ):
        raise ValueError("PPO series checkpoint list is invalid")
    if [entry.get("update") for entry in entries if isinstance(entry, dict)] != updates:
        raise ValueError("PPO series checkpoint schedule mismatch")
    rebuilt_curve: list[dict[str, Any]] = []
    for entry in entries:
        run_root = confined_path(root, entry.get("runPath"), "runPath")
        evaluation_path = confined_path(root, entry.get("evaluationPath"), "evaluationPath")
        try:
            run = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load PPO series child artifact: {error}") from error
        run_source = {name: value for name, value in run.items() if name != "runDigest"}
        if run.get("runDigest") != json_digest(run_source) or run["runDigest"] != entry.get("runDigest"):
            raise ValueError("PPO series child run digest mismatch")
        checkpoint_metadata, _ = load_ppo_checkpoint(run_root / "checkpoint")
        if checkpoint_metadata["checkpointDigest"] != entry.get("checkpointDigest"):
            raise ValueError("PPO series checkpoint digest mismatch")
        evaluation_source = {
            name: value for name, value in evaluation.items() if name != "resultDigest"
        }
        if (
            evaluation.get("resultDigest") != json_digest(evaluation_source)
            or evaluation["resultDigest"] != entry.get("evaluationDigest")
            or evaluation.get("checkpointDigest") != entry.get("checkpointDigest")
        ):
            raise ValueError("PPO series evaluation digest mismatch")
        if evaluation.get("threshold", {}).get("passed") != entry.get("thresholdPassed"):
            raise ValueError("PPO series threshold result mismatch")
        if run.get("targetUpdate") != entry.get("update"):
            raise ValueError("PPO series run update mismatch")
        child_updates = run.get("updates")
        if not isinstance(child_updates, list):
            raise ValueError("PPO series child updates are invalid")
        rebuilt_curve.extend(child_updates)
    if rebuilt_curve != manifest.get("learningCurve"):
        raise ValueError("PPO series learning curve does not match child runs")
    if [item.get("updateIndex") for item in rebuilt_curve] != list(range(updates[-1])):
        raise ValueError("PPO series learning curve is not contiguous")
    if manifest.get("finalThresholdPassed") != entries[-1].get("thresholdPassed"):
        raise ValueError("PPO series final threshold does not match final checkpoint")
    return {
        "ok": True,
        "format": PPO_SERIES_FORMAT,
        "mode": manifest.get("mode"),
        "gate": manifest.get("gateId"),
        "updates": updates,
        "finalThresholdPassed": manifest["finalThresholdPassed"],
        "seriesDigest": claimed,
    }


def confined_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"PPO series {name} is invalid")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"PPO series {name} escapes the artifact root")
    return candidate


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
    ppo_warm_start: str | Path | None = None,
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
                ppo_warm_start=ppo_warm_start if previous_checkpoint is None else None,
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
    parser.add_argument("--ppo-warm-start", type=Path)
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
            ppo_warm_start=args.ppo_warm_start,
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


def audit_main() -> None:
    parser = argparse.ArgumentParser(description="Audit a SnowGym PPO checkpoint series")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_ppo_series(args.path)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
