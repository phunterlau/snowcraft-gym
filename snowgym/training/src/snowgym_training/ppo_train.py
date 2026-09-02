"""Deterministic end-to-end PPO smoke trainer for SnowGym."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from .curriculum import load_curriculum
from .model import ModelConfig
from .ppo import HybridActorCritic, PPOConfig, ppo_update
from .ppo_checkpoint import restore_ppo_checkpoint, save_ppo_checkpoint
from .ppo_collect import SeedSchedule, collect_rollout
from .trainer import resolve_git_commit
from .trajectory import json_digest

PPO_RUN_FORMAT = "snowgym.ppo-run.v0"


def train_ppo(
    *,
    output: str | Path,
    gate_id: str = "1v1-random",
    worlds: int = 8,
    rollout_steps: int = 32,
    target_updates: int = 1,
    training_seed: int = 73,
    curriculum_path: str | Path | None = None,
    resume: str | Path | None = None,
    model_config: ModelConfig | None = None,
    ppo_config: PPOConfig | None = None,
    git_commit: str | None = None,
    reward_mode: str = "canonical",
) -> dict[str, Any]:
    """Train through a frozen gate and atomically write one final checkpoint."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite PPO run {destination}")
    for name, value in {
        "worlds": worlds,
        "rollout_steps": rollout_steps,
        "target_updates": target_updates,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise ValueError("training_seed must be an integer")

    curriculum = load_curriculum(curriculum_path)
    gate = next((item for item in curriculum["gates"] if item["id"] == gate_id), None)
    if gate is None:
        raise ValueError(f"unknown PPO curriculum gate {gate_id!r}")
    seed_minimum, seed_maximum = curriculum["trainingSeedRanges"][gate_id]
    curriculum_digest = json_digest(curriculum)
    architecture = model_config or ModelConfig()
    config = ppo_config or PPOConfig(minibatch_size=worlds * rollout_steps)
    collector_config = {
        "gateId": gate_id,
        "worlds": worlds,
        "rolloutSteps": rollout_steps,
        "rewardMode": reward_mode,
    }

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(training_seed)
    model = HybridActorCritic(architecture).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_update = 0
    environment_steps = 0
    if resume is None:
        schedule = SeedSchedule(seed_minimum, seed_maximum)
    else:
        restored = restore_ppo_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            config=config,
            curriculum_digest=curriculum_digest,
            training_seed=training_seed,
            collector_config=collector_config,
        )
        schedule_state = restored["seedSchedule"]
        if (schedule_state["minimum"], schedule_state["maximum"]) != (
            seed_minimum,
            seed_maximum,
        ):
            raise ValueError("PPO checkpoint seed schedule does not match curriculum gate")
        schedule = SeedSchedule(
            schedule_state["minimum"],
            schedule_state["maximum"],
            schedule_state["nextSeed"],
        )
        start_update = int(restored["updateIndex"])
        environment_steps = int(restored["environmentSteps"])
    if start_update >= target_updates:
        raise ValueError("target_updates must exceed the resumed update index")

    updates: list[dict[str, Any]] = []
    with SnowGymBatchClient() as client:
        environment = SnowGymBatchEnv(worlds, client=client)
        for update_index in range(start_update, target_updates):
            collection = collect_rollout(
                environment,
                model,
                scenario=gate["scenario"],
                seed_schedule=schedule,
                rollout_steps=rollout_steps,
                config=config,
                reward_mode=reward_mode,
            )
            metrics = ppo_update(
                model,
                optimizer,
                collection.rollout,
                config,
                training_seed=training_seed,
                update_index=update_index,
            )
            environment_steps += worlds * rollout_steps
            updates.append(
                {
                    "updateIndex": update_index,
                    "environmentSteps": environment_steps,
                    "episodeSeeds": list(collection.episode_seeds),
                    "completedEpisodes": collection.completed_episodes,
                    "boundaryTruncations": collection.boundary_truncations,
                    "rejectedActions": collection.rejected_actions,
                    "canonicalRewardSum": collection.canonical_reward_sum,
                    "trainingRewardSum": collection.training_reward_sum,
                    "metrics": metrics,
                }
            )

    commit = git_commit or resolve_git_commit()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    result_root = temporary / destination.name
    try:
        checkpoint = result_root / "checkpoint"
        checkpoint_metadata = save_ppo_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            curriculum_digest=curriculum_digest,
            training_seed=training_seed,
            update_index=target_updates,
            environment_steps=environment_steps,
            git_commit=commit,
            seed_schedule=schedule.state(),
            collector_config=collector_config,
        )
        manifest: dict[str, Any] = {
            "format": PPO_RUN_FORMAT,
            "mode": "infrastructure-smoke",
            "gitCommit": commit,
            "curriculumDigest": curriculum_digest,
            "gate": gate,
            "architecture": architecture.as_dict(),
            "ppoConfig": asdict(config),
            "trainingSeed": training_seed,
            "worlds": worlds,
            "rolloutSteps": rollout_steps,
            "rewardMode": reward_mode,
            "startUpdate": start_update,
            "targetUpdate": target_updates,
            "environmentSteps": environment_steps,
            "seedSchedule": schedule.state(),
            "updates": updates,
            "checkpoint": {
                "path": "checkpoint",
                "checkpointDigest": checkpoint_metadata["checkpointDigest"],
                "stateDigest": checkpoint_metadata["stateDigest"],
            },
        }
        manifest["runDigest"] = json_digest(manifest)
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(result_root, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic SnowGym PPO training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", default="1v1-random")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--target-updates", type=int, default=1)
    parser.add_argument("--training-seed", type=int, default=73)
    parser.add_argument(
        "--reward-mode", choices=("canonical", "health-potential"), default="canonical"
    )
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = train_ppo(
            output=args.output,
            gate_id=args.gate,
            worlds=args.worlds,
            rollout_steps=args.rollout_steps,
            target_updates=args.target_updates,
            training_seed=args.training_seed,
            curriculum_path=args.curriculum,
            resume=args.resume,
            reward_mode=args.reward_mode,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "gate": result["gate"]["id"],
        "targetUpdate": result["targetUpdate"],
        "environmentSteps": result["environmentSteps"],
        "runDigest": result["runDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
