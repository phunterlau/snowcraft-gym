"""Read-only loss-component gradient diagnostics for an Engage checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW

from ..checkpoint import load_checkpoint, semantic_state_digest
from ..executor import model_config
from ..plan_ppo import freeze_initializer, initialize_plan_ppo_policy
from ..ppo import HybridActorCritic, PPOConfig, living_unit_mask, ppo_loss
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from .train import DEFAULT_INITIALIZER

FORMAT = "snowgym.engage.gradient-diagnostics.v0"
GROUP_PREFIXES = {
    "plan-residual": ("policy.plan_ppo_residual.",),
    "action-head": ("policy.action_head.",),
    "move-head": ("policy.move_target_head.",),
    "throw-head": ("policy.throw_target_head.",),
    "power-head": ("policy.power_head.",),
    "controller-extension": (
        "policy.ally_v3_adapter.",
        "policy.enemy_v3_adapter.",
        "policy.projectile_v3_adapter.",
        "policy.v3_scalar_adapter.",
    ),
    "critic": ("role_aware_critic.",),
}


def run_gradient_diagnostics(
    checkpoint: str | Path,
    dataset: str | Path,
    *,
    output: str | Path,
    initializer_path: str | Path = DEFAULT_INITIALIZER,
    sample_count: int = 256,
) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite gradient diagnostics {destination}")
    metadata, state = load_ppo_checkpoint(checkpoint)
    source_metadata, source_state = load_checkpoint(initializer_path)
    architecture = model_config(metadata["architecture"])
    config = PPOConfig(**metadata["ppoConfig"])
    model = HybridActorCritic(
        architecture,
        initial_target_log_std=config.initial_target_log_std,
        initial_power_log_std=config.initial_power_log_std,
    ).eval()
    model.load_state_dict(state["model"])
    initializer = HybridActorCritic(
        architecture,
        initial_target_log_std=config.initial_target_log_std,
        initial_power_log_std=config.initial_power_log_std,
    ).eval()
    initialize_plan_ppo_policy(initializer, source_state["model"])
    initializer = freeze_initializer(initializer)
    before = semantic_state_digest({"model": model.state_dict()})
    arrays = np.load(dataset, allow_pickle=False)
    total = int(arrays["decision"].shape[0])
    count = min(sample_count, total)
    if count <= 0:
        raise ValueError("gradient diagnostic dataset is empty")
    observation = {
        name.removeprefix("observation__"): torch.from_numpy(arrays[name][:count])
        for name in arrays.files
        if name.startswith("observation__")
    }
    actions = {
        "action_type": torch.from_numpy(arrays["learner_action_type"][:count]),
        "target": torch.from_numpy(arrays["learner_target"][:count]),
        "power": torch.from_numpy(arrays["learner_power"][:count]),
    }
    teacher = {
        "action_type": torch.from_numpy(arrays["teacher_action_type"][:count]),
        "target": torch.from_numpy(arrays["teacher_target"][:count]),
        "power": torch.from_numpy(arrays["teacher_power"][:count]),
    }
    returns = torch.from_numpy(arrays["return_to_go"][:count]).float()
    prediction = model(observation)
    with torch.no_grad():
        initial_prediction = initializer(observation)
        old_log_probability, _ = model.evaluate_actions(observation, actions)
    new_log_probability, entropy = model.evaluate_actions(
        observation, actions, prediction=prediction
    )
    advantages = returns - prediction["value"].detach()
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-8)
    base = ppo_loss(
        new_log_probability,
        old_log_probability,
        advantages,
        prediction["value"],
        returns,
        entropy,
        config,
        normalize_advantages=False,
        active_mask=living_unit_mask(observation),
    )
    losses = component_losses(
        model, prediction, initial_prediction, observation, teacher, base, config
    )
    named = list(model.named_parameters())
    vectors: dict[str, list[Tensor | None]] = {}
    for component, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            [parameter for _, parameter in named],
            retain_graph=True,
            allow_unused=True,
        )
        vectors[component] = [
            None if gradient is None else gradient.detach().clone()
            for gradient in gradients
        ]
    after = semantic_state_digest({"model": model.state_dict()})
    if after != before:
        raise RuntimeError("read-only gradient diagnostic changed model state")
    norms = gradient_norm_rows(named, vectors)
    cosines = gradient_cosine_rows(named, vectors)
    global_norm = math.sqrt(sum(
        float(gradient.square().sum())
        for gradient in vectors["ppo-actor"]
        if gradient is not None
    ) + sum(
        float(gradient.square().sum())
        for gradient in vectors["critic-value"]
        if gradient is not None
    ))
    actor_norm = group_norm(named, vectors["ppo-actor"], exclude_prefix="role_aware_critic.")
    critic_norm = group_norm(named, vectors["critic-value"], include_prefix="role_aware_critic.")
    destination.mkdir(parents=True)
    components = {
        "format": FORMAT,
        "checkpointDigest": metadata["checkpointDigest"],
        "implementationGitCommit": resolve_git_commit(),
        "initializerDigest": source_metadata["checkpointDigest"],
        "datasetDigest": file_digest(Path(dataset)),
        "samples": count,
        "stateDigestBefore": before,
        "stateDigestAfter": after,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "globalNormBeforeCurrentClip": global_norm,
        "actorOnlyNorm": actor_norm,
        "criticOnlyNorm": critic_norm,
        "currentGlobalClipScale": min(
            1.0, config.max_grad_norm / max(global_norm, 1e-12)
        ),
    }
    components["artifactDigest"] = json_digest(components)
    (destination / "gradient_components.json").write_text(
        json.dumps(components, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(destination / "gradient_norms.csv", norms)
    write_csv(destination / "gradient_cosines.csv", cosines)
    manifest = {
        "format": "snowgym.engage.gradient-diagnostics-manifest.v0",
        "implementationGitCommit": components["implementationGitCommit"],
        "artifacts": {
            name: file_digest(destination / name)
            for name in (
                "gradient_components.json", "gradient_norms.csv", "gradient_cosines.csv"
            )
        },
    }
    manifest["manifestDigest"] = json_digest(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"components": components, "manifest": manifest}


def component_losses(
    model: HybridActorCritic,
    prediction: dict[str, Tensor],
    initial: dict[str, Tensor],
    observation: dict[str, Tensor],
    teacher: dict[str, Tensor],
    base: dict[str, Tensor],
    config: PPOConfig,
) -> dict[str, Tensor]:
    active = living_unit_mask(observation)
    teacher_type = teacher["action_type"].long()
    action_bc = F.cross_entropy(
        prediction["action_logits"][active], teacher_type[active]
    )
    move = active & (teacher_type == ACTION_MOVE)
    throw = active & (teacher_type == ACTION_THROW)
    move_bc = conditional_mse(
        prediction["supervised_target_by_action"][..., ACTION_MOVE, :],
        teacher["target"].float(),
        move,
    )
    throw_bc = conditional_mse(
        prediction["supervised_target_by_action"][..., ACTION_THROW, :],
        teacher["target"].float(),
        throw,
    )
    power_bc = conditional_mse(
        prediction["power"], teacher["power"].float(), throw
    )
    current_log = F.log_softmax(prediction["action_logits"], dim=-1)
    initial_log = F.log_softmax(initial["action_logits"], dim=-1)
    categorical_kl = masked_mean(
        (current_log.exp() * (current_log - initial_log)).sum(dim=-1), active
    )
    target_variance = model.target_log_std.exp().square().view(1, 1, 2)
    target_anchor = (
        prediction["target_raw_by_action"] - initial["target_raw_by_action"]
    ).square() / (2 * target_variance.unsqueeze(-2))
    move_anchor = masked_mean(target_anchor[..., ACTION_MOVE, :].sum(dim=-1), active)
    throw_anchor = masked_mean(target_anchor[..., ACTION_THROW, :].sum(dim=-1), active)
    power_anchor = masked_mean(
        (prediction["power_raw"] - initial["power_raw"]).square()
        / (2 * model.power_log_std.exp().square()),
        active,
    )
    return {
        "ppo-actor": base["policy"] - config.entropy_weight * base["entropy"],
        "critic-value": config.value_weight * base["value"],
        "action-bc": action_bc,
        "move-target-bc": move_bc,
        "throw-target-bc": throw_bc,
        "power-bc": power_bc,
        "categorical-initializer-kl": categorical_kl,
        "move-initializer-anchor": move_anchor,
        "throw-initializer-anchor": throw_anchor,
        "power-initializer-anchor": power_anchor,
    }


def conditional_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any()):
        return (prediction[mask] - target[mask]).square().mean()
    return prediction.sum() * 0


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1)


def gradient_norm_rows(
    named: list[tuple[str, Tensor]], vectors: dict[str, list[Tensor | None]]
) -> list[dict[str, Any]]:
    rows = []
    for component, gradients in vectors.items():
        for group, prefixes in GROUP_PREFIXES.items():
            selected = [
                gradient
                for (name, _), gradient in zip(named, gradients, strict=True)
                if name.startswith(prefixes) and gradient is not None
            ]
            rows.append(
                {
                    "component": component,
                    "parameterGroup": group,
                    "gradientNorm": math.sqrt(sum(float(value.square().sum()) for value in selected)),
                    "gradientTensors": len(selected),
                }
            )
    return rows


def gradient_cosine_rows(
    named: list[tuple[str, Tensor]], vectors: dict[str, list[Tensor | None]]
) -> list[dict[str, Any]]:
    rows = []
    components = list(vectors)
    for group, prefixes in GROUP_PREFIXES.items():
        indices = [index for index, (name, _) in enumerate(named) if name.startswith(prefixes)]
        for first_index, first in enumerate(components):
            for second in components[first_index + 1 :]:
                left = flatten_gradients(vectors[first], named, indices)
                right = flatten_gradients(vectors[second], named, indices)
                denominator = float(left.norm() * right.norm())
                rows.append(
                    {
                        "parameterGroup": group,
                        "firstComponent": first,
                        "secondComponent": second,
                        "cosine": None if denominator <= 1e-12 else float(torch.dot(left, right) / denominator),
                    }
                )
    return rows


def flatten_gradients(
    gradients: list[Tensor | None],
    named: list[tuple[str, Tensor]],
    indices: list[int],
) -> Tensor:
    if not indices:
        return torch.zeros(1)
    return torch.cat([
        (
            torch.zeros_like(named[index][1]).flatten()
            if gradients[index] is None
            else gradients[index].flatten()
        )
        for index in indices
    ])


def group_norm(
    named: list[tuple[str, Tensor]],
    gradients: list[Tensor | None],
    *,
    include_prefix: str | None = None,
    exclude_prefix: str | None = None,
) -> float:
    return math.sqrt(sum(
        float(gradient.square().sum())
        for (name, _), gradient in zip(named, gradients, strict=True)
        if gradient is not None
        and (include_prefix is None or name.startswith(include_prefix))
        and (exclude_prefix is None or not name.startswith(exclude_prefix))
    ))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initializer", default=str(DEFAULT_INITIALIZER))
    parser.add_argument("--sample-count", type=int, default=256)
    args = parser.parse_args()
    result = run_gradient_diagnostics(
        args.checkpoint,
        args.dataset,
        output=args.output,
        initializer_path=args.initializer,
        sample_count=args.sample_count,
    )
    print(json.dumps(result["components"], sort_keys=True))


if __name__ == "__main__":
    main()
