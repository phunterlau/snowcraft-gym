"""Reproducible expanded initializer identity and optimizer-derived evidence."""

from __future__ import annotations

import math
from typing import Any

import torch

from ..checkpoint import semantic_state_digest
from ..executor import model_config
from ..plan_ppo import INHERITED_HEAD_PREFIXES, POLICY_NEW_PREFIXES, freeze_initializer, initialize_plan_ppo_policy
from ..ppo import HybridActorCritic


def checkpoint_model(metadata: dict[str, Any]) -> HybridActorCritic:
    return HybridActorCritic(
        model_config(metadata["architecture"]),
        initial_target_log_std=metadata["ppoConfig"].get("initial_target_log_std", -1.0),
        initial_power_log_std=metadata["ppoConfig"].get("initial_power_log_std", -1.0),
    ).eval()


def recover_initializer(
    metadata: dict[str, Any], state: dict[str, Any],
    source_metadata: dict[str, Any], source_state: dict[str, Any],
) -> tuple[HybridActorCritic, dict[str, Any]]:
    """Prefer stored weights; reconstruct legacy option runs without consuming RNG."""
    source_digest = source_metadata["checkpointDigest"]
    expected = metadata.get("initializerSourceDigest")
    if expected is None and metadata["initialization"].get("type") == "behavior-clone":
        expected = metadata["initialization"]["checkpointDigest"]
    if expected is not None and expected != source_digest:
        raise ValueError("expanded initializer source checkpoint mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(metadata["trainingSeed"])
        initializer = checkpoint_model(metadata)
        if "initializerModel" in state:
            initializer.load_state_dict(state["initializerModel"])
            method = "stored-expanded-state"
        else:
            initialize_plan_ppo_policy(initializer, source_state["model"])
            method = "legacy-seeded-reconstruction"
    digest = semantic_state_digest(initializer.state_dict())
    if "initializerDigest" in metadata and digest != metadata["initializerDigest"]:
        raise ValueError("expanded initializer state digest mismatch")
    return freeze_initializer(initializer), {
        "method": method,
        "stateDigest": digest,
        "sourceCheckpointDigest": source_digest,
        "trainingSeed": metadata["trainingSeed"],
        "sourceIdentityVerified": expected is not None,
    }


def parameter_changes(model: HybridActorCritic, initializer: HybridActorCritic) -> dict[str, float]:
    squares = {"inheritedHeads": 0.0, "newActor": 0.0, "otherInheritedActor": 0.0, "critic": 0.0}
    initial = initializer.state_dict()
    for name, value in model.state_dict().items():
        if name.startswith(INHERITED_HEAD_PREFIXES):
            group = "inheritedHeads"
        elif name.startswith(tuple(f"policy.{prefix}" for prefix in POLICY_NEW_PREFIXES)):
            group = "newActor"
        elif name.startswith("policy."):
            group = "otherInheritedActor"
        elif name.startswith("role_aware_critic."):
            group = "critic"
        else:
            continue
        squares[group] += float((value.detach() - initial[name]).square().sum())
    return {**{name: math.sqrt(value) for name, value in squares.items()},
            "actorTotal": math.sqrt(sum(value for name, value in squares.items() if name != "critic"))}


def optimizer_learning_rates(state: dict[str, Any]) -> dict[str, float]:
    result = {}
    for group in state["optimizer"]["param_groups"]:
        name, rate = group.get("name"), group["lr"]
        if name not in {"new", "heads", "encoder-final"} or name in result:
            raise ValueError("option optimizer groups must have unique audited names")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
            raise ValueError("option optimizer learning rate is invalid")
        result[name] = float(rate)
    if "new" not in result:
        raise ValueError("option optimizer has no new-module group")
    return result
