"""Paired closed-loop evaluation of fixed-option PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..checkpoint import load_checkpoint
from ..executor import model_config
from ..plan_ppo import freeze_initializer, initialize_plan_ppo_policy
from ..ppo import HybridActorCritic
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import merge_observations, numpy_actions, tensor_dict
from ..trajectory import json_digest
from .definitions import FROZEN_OPTION_SPECS
from .environment import FixedPlanOptionBatchEnv
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol
from .train import DEFAULT_INITIALIZER, OPTION_ORDER
from .interventions import contact_distance, require_raw, team_health

CONDITIONS = ("correct", "shuffled", "initializer")
SHUFFLED_OPTION = {
    "engage": "hold",
    "advance": "withdraw",
    "hold": "advance",
    "withdraw": "advance",
    "flank": "engage",
    "focus": "distributed",
    "distributed": "focus",
    "support": "engage",
}


def evaluate_option_episode(
    model: HybridActorCritic,
    initializer: HybridActorCritic,
    *,
    option: str,
    seed: int,
    condition: str,
    client: SnowGymBatchClient,
) -> dict[str, Any]:
    if option not in OPTION_ORDER or condition not in CONDITIONS:
        raise ValueError("option evaluation condition is invalid")
    correct_plan, spec = teacher_option_plan(option)
    scenario = teacher_option_scenario(option)
    base = SnowGymBatchEnv(1, client=client, observation_version=3)
    wrapped = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
    observation, _ = wrapped.reset(
        [seed], [scenario], [f"eval-{option}-{condition}-{seed}"], [correct_plan], [spec]
    )
    policy = initializer if condition == "initializer" else model
    alternative = teacher_option_plan(SHUFFLED_OPTION[option])[0]
    option_result = None
    option_done = False
    environment_done = False
    rejected = 0
    total_actions = 0
    final_info: dict[str, Any] = {}
    initial_raw = require_raw(base)
    plan_body = base.plan_observations()[1][0]
    assignment = next(
        item for item in plan_body["assignments"] if item["role"] == spec.role
    )
    assigned = tuple(int(value) for value in assignment["unitIds"])
    initial_enemy_health = team_health(initial_raw["enemies"])
    first_contact: int | None = None
    first_hit: int | None = None
    maximum_decisions = math.ceil(int(scenario["maxTicks"]) / 6)
    for decision in range(maximum_decisions):
        option_was_active = not option_done
        policy_observation = observation
        if condition == "shuffled":
            preview, _, _ = base.preview_plans(
                [f"shuffle-{option}-{seed}"], [alternative]
            )
            physical = {
                name: value
                for name, value in observation.items()
                if name not in preview
            }
            policy_observation = merge_observations(physical, preview)
        with torch.no_grad():
            action, _, _ = policy.act(
                tensor_dict(policy_observation), deterministic=True
            )
        if not option_done:
            observation, _, terminated, truncated, infos = wrapped.step(
                numpy_actions(action)
            )
            option_result = infos[0]["option"]
            option_done = bool(terminated[0] or truncated[0])
        else:
            physical, _, terminated, truncated, infos = base.step(numpy_actions(action))
            plan_tensors, _ = base.plan_observations()
            observation = merge_observations(physical, plan_tensors)
        final_info = infos[0]
        raw = require_raw(base)
        if (
            option_was_active
            and first_contact is None
            and contact_distance(raw, assigned) <= 9.0
        ):
            first_contact = decision + 1
        if (
            option_was_active
            and first_hit is None
            and team_health(raw["enemies"]) < initial_enemy_health
        ):
            first_hit = decision + 1
        action_results = final_info.get("actionResults", [])
        rejected += sum(
            result.get("accepted") is False
            for result in action_results
            if isinstance(result, dict)
        )
        total_actions += sum(isinstance(result, dict) for result in action_results)
        environment_done = bool(
            final_info.get("terminated", False) or final_info.get("truncated", False)
        )
        if environment_done:
            break
    if option_result is None or not option_done:
        raise RuntimeError("option evaluation did not reach an option boundary")
    if not environment_done:
        raise RuntimeError("option evaluation did not reach a battle boundary")
    return {
        "seed": seed,
        "success": bool(option_result["success"]),
        "progress": float(option_result["progress"]),
        "physicalWin": final_info.get("winner") == "blue",
        "rejectedActions": rejected,
        "totalActions": total_actions,
        "firstContactDecision": first_contact,
        "firstHitDecision": first_hit,
    }


def evaluate_m7b_checkpoint(
    checkpoint: str | Path,
    *,
    output: str | Path,
    split: str = "development",
    initializer_path: str | Path = DEFAULT_INITIALIZER,
    options: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if split not in {"development", "qualification"}:
        raise ValueError("M7b evaluation split must be development or qualification")
    selected_options = resolve_evaluation_options(split, options)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite M7b evaluation {destination}")
    protocol = load_option_protocol()
    count = int(protocol["pairedSeedsPerMission"][split])
    start = int(protocol["seeds"][split][0])
    checkpoint_metadata, checkpoint_state = load_ppo_checkpoint(checkpoint)
    source_metadata, source_state = load_checkpoint(initializer_path)
    architecture = model_config(checkpoint_metadata["architecture"])
    model = HybridActorCritic(
        architecture,
        initial_target_log_std=checkpoint_metadata["ppoConfig"]["initial_target_log_std"],
        initial_power_log_std=checkpoint_metadata["ppoConfig"]["initial_power_log_std"],
    ).eval()
    model.load_state_dict(checkpoint_state["model"])
    source_config = model_config(source_metadata["architecture"])
    initializer = HybridActorCritic(
        architecture,
        initial_target_log_std=checkpoint_metadata["ppoConfig"]["initial_target_log_std"],
        initial_power_log_std=checkpoint_metadata["ppoConfig"]["initial_power_log_std"],
    ).eval()
    initialize_plan_ppo_policy(initializer, source_state["model"])
    initializer = freeze_initializer(initializer)
    missions: dict[str, Any] = {}
    with SnowGymBatchClient() as client:
        for option in selected_options:
            mission_index = OPTION_ORDER.index(option)
            seeds = [start + mission_index * count + offset for offset in range(count)]
            missions[option] = {
                condition: [
                    evaluate_option_episode(
                        model,
                        initializer,
                        option=option,
                        seed=seed,
                        condition=condition,
                        client=client,
                    )
                    for seed in seeds
                ]
                for condition in CONDITIONS
            }
    difference = 0.0
    initializer_state = initializer.policy.state_dict()
    for name, value in model.policy.state_dict().items():
        difference += float((value.detach() - initializer_state[name]).square().sum())
    stage = int(checkpoint_metadata["collectorConfig"].get("stage", 0))
    new_learning_rate = max(float(checkpoint_metadata["ppoConfig"]["learning_rate"]), 1e-4)
    value = {
        "format": (
            "snowgym.m7b-evaluation.v0"
            if split == "qualification"
            else "snowgym.m7b-development-evaluation.v0"
        ),
        "split": split,
        "checkpointDigest": checkpoint_metadata["checkpointDigest"],
        "sourceDigest": source_metadata["checkpointDigest"],
        "protocolDigest": json_digest(protocol),
        "inheritedHeadLearningRate": new_learning_rate / 10 if stage >= 2 else 0.0,
        "newModuleLearningRate": new_learning_rate,
        "parameterL2Change": math.sqrt(difference),
        "evaluatedOptions": list(selected_options),
        "missions": missions,
    }
    value["evaluationDigest"] = json_digest(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("development", "qualification"), default="development")
    parser.add_argument("--initializer", default=str(DEFAULT_INITIALIZER))
    parser.add_argument(
        "--option",
        action="append",
        choices=OPTION_ORDER,
        help="development-only mission subset; repeat to select multiple",
    )
    args = parser.parse_args()
    result = evaluate_m7b_checkpoint(
        args.checkpoint,
        output=args.output,
        split=args.split,
        initializer_path=args.initializer,
        options=tuple(args.option) if args.option else None,
    )
    print(json.dumps(result, sort_keys=True))


def resolve_evaluation_options(
    split: str, options: tuple[str, ...] | None
) -> tuple[str, ...]:
    selected = OPTION_ORDER if options is None else options
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("M7b evaluation options must be non-empty and unique")
    if any(option not in OPTION_ORDER for option in selected):
        raise ValueError("M7b evaluation contains an unknown option")
    canonical = tuple(option for option in OPTION_ORDER if option in selected)
    if split == "qualification" and canonical != OPTION_ORDER:
        raise ValueError("M7b qualification evaluation requires every frozen mission")
    return canonical


if __name__ == "__main__":
    main()
