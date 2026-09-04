"""Audited fixed-option PPO runner using the accepted target-only initializer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..checkpoint import load_checkpoint
from ..executor import model_config
from ..loss import LossConfig
from ..plan_ppo import (
    freeze_initializer,
    initialize_plan_ppo_policy,
    plan_ppo_parameter_groups,
    plan_ppo_update,
    target_only_plan_ppo_config,
)
from ..ppo import HybridActorCritic, PPOConfig, discount_manifest
from ..ppo_checkpoint import (
    load_ppo_checkpoint,
    normalized_ppo_config,
    restore_ppo_checkpoint,
    save_ppo_checkpoint,
)
from ..ppo_collect import SeedSchedule
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .collection import OptionEntry, OptionSchedule, collect_option_rollout
from .environment import FixedPlanOptionBatchEnv
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol

OPTION_PPO_RUN_FORMAT = "snowgym.option-ppo-run.v0"
OPTION_ORDER = (
    "engage", "advance", "hold", "withdraw", "flank", "focus", "distributed", "support"
)
DEFAULT_INITIALIZER = (
    Path(__file__).resolve().parents[3]
    / "runs"
    / "plan_bc_ablation_qual_v1"
    / "plan-conditioned"
)


def train_option_ppo(
    *,
    output: str | Path,
    option: str,
    worlds: int = 8,
    rollout_steps: int | None = None,
    target_updates: int = 1,
    anchor_total_updates: int = 100,
    stage: int = 1,
    training_seed: int = 91_001,
    initializer_path: str | Path = DEFAULT_INITIALIZER,
    resume: str | Path | None = None,
    ppo_warm_start: str | Path | None = None,
    ppo_config: PPOConfig | None = None,
    loss_config: LossConfig | None = None,
    git_commit: str | None = None,
    physical_gate_passed: bool = False,
    plan_gate_passed: bool = False,
    infrastructure_smoke: bool = False,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite option PPO run {destination}")
    if option not in OPTION_ORDER:
        raise ValueError(f"unknown M7b option {option!r}")
    for name, value in {
        "worlds": worlds,
        "target_updates": target_updates,
        "anchor_total_updates": anchor_total_updates,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if target_updates > anchor_total_updates:
        raise ValueError("target_updates exceeds the frozen anchor schedule")
    if resume is not None and ppo_warm_start is not None:
        raise ValueError("exact resume and staged PPO transfer are mutually exclusive")
    protocol = load_option_protocol()
    protocol_digest = json_digest(protocol)
    seed_start = int(protocol["seeds"]["training"][0]) + OPTION_ORDER.index(option) * 10_000
    seed_end = seed_start + 9_999
    plan, spec = teacher_option_plan(option)
    if rollout_steps is None:
        rollout_steps = spec.horizon
    if (
        not isinstance(rollout_steps, int)
        or isinstance(rollout_steps, bool)
        or rollout_steps <= 0
    ):
        raise ValueError("rollout_steps must be a positive integer")
    if rollout_steps < spec.horizon and not infrastructure_smoke:
        raise ValueError(
            "research option rollouts must span the full option horizon; "
            "use infrastructure_smoke only for plumbing tests"
        )
    entries = tuple(OptionEntry(plan, spec) for _ in range(seed_end - seed_start + 1))
    source_metadata, source_state = load_checkpoint(initializer_path)
    source_config = model_config(source_metadata["architecture"])
    architecture = target_only_plan_ppo_config(source_config)
    config = ppo_config or PPOConfig(minibatch_size=worlds * rollout_steps)
    bc_config = loss_config or LossConfig(
        action_weight=1,
        target_weight=5,
        power_weight=0.5,
        throw_action_weight=5,
    )
    collector_config = {
        "gateId": f"m7b-{option}-stage{stage}",
        "worlds": worlds,
        "rolloutSteps": rollout_steps,
        "rewardMode": "executor",
        "option": option,
        "stage": stage,
        "anchorTotalUpdates": anchor_total_updates,
        "protocolDigest": protocol_digest,
    }
    initialization = {
        "type": "behavior-clone",
        "checkpointDigest": source_metadata["checkpointDigest"],
        "stateDigest": source_metadata["stateDigest"],
        "datasetManifestHash": source_metadata["datasetManifestHash"],
    }
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(training_seed)
    model = HybridActorCritic(
        architecture,
        initial_target_log_std=config.initial_target_log_std,
        initial_power_log_std=config.initial_power_log_std,
    ).cpu()
    migration = initialize_plan_ppo_policy(model, source_state["model"])
    initializer = freeze_initializer(model)
    start_update = 0
    environment_steps = 0
    schedule = SeedSchedule(seed_start, seed_end)
    option_schedule = OptionSchedule(entries, prefix=f"m7b-{option}")
    if ppo_warm_start is not None:
        transfer_metadata, transfer_state = load_ppo_checkpoint(ppo_warm_start)
        if transfer_metadata["architecture"] != architecture.as_dict():
            raise ValueError("staged PPO transfer architecture changed")
        if normalized_ppo_config(transfer_metadata["ppoConfig"]) != asdict(config):
            raise ValueError("staged PPO transfer PPO configuration changed")
        if transfer_metadata["curriculumDigest"] != protocol_digest:
            raise ValueError("staged PPO transfer protocol changed")
        source_collector = transfer_metadata["collectorConfig"]
        model.load_state_dict(transfer_state["model"])
        if source_collector.get("option") == option:
            schedule_state = transfer_metadata["seedSchedule"]
            schedule = SeedSchedule(
                schedule_state["minimum"],
                schedule_state["maximum"],
                schedule_state["nextSeed"],
            )
            option_state = transfer_metadata.get("optionSchedule")
            if not isinstance(option_state, dict):
                raise ValueError("staged PPO transfer has no option schedule")
            option_schedule = OptionSchedule.restore(entries, option_state)
        start_update = int(transfer_metadata["updateIndex"])
        environment_steps = int(transfer_metadata["environmentSteps"])
        initialization = {
            "type": "ppo-transfer",
            "checkpointDigest": transfer_metadata["checkpointDigest"],
            "stateDigest": transfer_metadata["stateDigest"],
            "curriculumDigest": transfer_metadata["curriculumDigest"],
            "sourceGate": source_collector["gateId"],
            "updateIndex": transfer_metadata["updateIndex"],
        }
    parameter_groups = plan_ppo_parameter_groups(
        model,
        stage,
        new_module_learning_rate=max(config.learning_rate, 1e-4),
        physical_gate_passed=physical_gate_passed,
        plan_gate_passed=plan_gate_passed,
    )
    optimizer = torch.optim.Adam(parameter_groups)
    if resume is not None:
        restored = restore_ppo_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            config=config,
            curriculum_digest=protocol_digest,
            training_seed=training_seed,
            collector_config=collector_config,
        )
        initialization = dict(restored["initialization"])
        schedule_state = restored["seedSchedule"]
        schedule = SeedSchedule(
            schedule_state["minimum"],
            schedule_state["maximum"],
            schedule_state["nextSeed"],
        )
        option_state = restored.get("optionSchedule")
        if not isinstance(option_state, dict):
            raise ValueError("option PPO resume checkpoint has no option schedule")
        option_schedule = OptionSchedule.restore(entries, option_state)
        start_update = int(restored["updateIndex"])
        environment_steps = int(restored["environmentSteps"])
    if start_update >= target_updates:
        raise ValueError("target_updates must exceed the resumed update index")

    updates: list[dict[str, Any]] = []
    scenario = teacher_option_scenario(option)
    with SnowGymBatchClient() as client:
        wrapped = FixedPlanOptionBatchEnv(
            SnowGymBatchEnv(worlds, client=client, observation_version=3),
            gamma=config.gamma,
        )
        for update_index in range(start_update, target_updates):
            collection = collect_option_rollout(
                wrapped,
                model,
                scenario=scenario,
                seed_schedule=schedule,
                option_schedule=option_schedule,
                rollout_steps=rollout_steps,
                config=config,
            )
            metrics = plan_ppo_update(
                model,
                optimizer,
                collection.rollout,
                collection.teacher_actions,
                initializer,
                config,
                loss_config=bc_config,
                training_seed=training_seed,
                update_index=update_index,
                total_updates=anchor_total_updates,
            )
            environment_steps += worlds * rollout_steps
            updates.append(
                {
                    "updateIndex": update_index,
                    "environmentSteps": environment_steps,
                    "episodeSeeds": list(collection.episode_seeds),
                    "episodeOptionIds": list(collection.episode_option_ids),
                    "completedOptions": collection.completed_options,
                    "successfulOptions": collection.successful_options,
                    "boundaryTruncations": collection.boundary_truncations,
                    "rejectedActions": collection.rejected_actions,
                    "rewardSums": collection.reward_sums,
                    "actionCounts": collection.action_counts,
                    "teacherActionCounts": collection.teacher_action_counts,
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
            curriculum_digest=protocol_digest,
            training_seed=training_seed,
            update_index=target_updates,
            environment_steps=environment_steps,
            git_commit=commit,
            seed_schedule=schedule.state(),
            collector_config=collector_config,
            initialization=initialization,
            option_schedule=option_schedule.state(),
        )
        manifest = {
            "format": OPTION_PPO_RUN_FORMAT,
            "mode": "infrastructure-smoke" if infrastructure_smoke else "development",
            "gitCommit": commit,
            "protocolDigest": protocol_digest,
            "option": option,
            "optionSpec": asdict(spec),
            "scenario": scenario,
            "architecture": architecture.as_dict(),
            "ppoConfig": asdict(config),
            "bcLossConfig": bc_config.as_dict(),
            "discounting": discount_manifest(config, 10),
            "stage": stage,
            "learningRates": {
                group["name"]: group["lr"] for group in parameter_groups
            },
            "anchorTotalUpdates": anchor_total_updates,
            "trainingSeed": training_seed,
            "initialization": initialization,
            "rootInitializer": {
                "checkpointDigest": source_metadata["checkpointDigest"],
                "stateDigest": source_metadata["stateDigest"],
                "datasetManifestHash": source_metadata["datasetManifestHash"],
            },
            "migration": migration,
            "startUpdate": start_update,
            "targetUpdates": target_updates,
            "environmentSteps": environment_steps,
            "seedSchedule": schedule.state(),
            "optionSchedule": option_schedule.state(),
            "updates": updates,
            "checkpoint": checkpoint_metadata,
        }
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result_root.replace(destination)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--option", required=True, choices=OPTION_ORDER)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--target-updates", type=int, default=1)
    parser.add_argument("--anchor-total-updates", type=int, default=100)
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--training-seed", type=int, default=91_001)
    parser.add_argument("--initializer", default=str(DEFAULT_INITIALIZER))
    parser.add_argument("--resume")
    parser.add_argument("--ppo-warm-start")
    parser.add_argument("--physical-gate-passed", action="store_true")
    parser.add_argument("--plan-gate-passed", action="store_true")
    parser.add_argument("--infrastructure-smoke", action="store_true")
    args = parser.parse_args()
    result = train_option_ppo(
        output=args.output,
        option=args.option,
        worlds=args.worlds,
        rollout_steps=args.rollout_steps,
        target_updates=args.target_updates,
        anchor_total_updates=args.anchor_total_updates,
        stage=args.stage,
        training_seed=args.training_seed,
        initializer_path=args.initializer,
        resume=args.resume,
        ppo_warm_start=args.ppo_warm_start,
        physical_gate_passed=args.physical_gate_passed,
        plan_gate_passed=args.plan_gate_passed,
        infrastructure_smoke=args.infrastructure_smoke,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
