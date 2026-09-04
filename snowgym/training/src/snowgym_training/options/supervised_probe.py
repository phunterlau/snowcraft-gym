"""Bounded R1f teacher-only fitting probe; no PPO or critic optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient

from ..checkpoint import load_checkpoint
from ..loss import LossConfig, behavior_clone_loss
from ..plan_ppo import freeze_initializer, plan_ppo_parameter_groups
from ..ppo import HybridActorCritic, PPOConfig
from ..ppo_checkpoint import load_ppo_checkpoint, save_ppo_checkpoint
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .bootstrap import engage_bootstrap_report
from .evaluate import evaluate_m7b_checkpoint, evaluate_option_episode
from .identity import checkpoint_model, parameter_changes, recover_initializer
from .interventions import require_capabilities
from .probe_metrics import teacher_agreement
from .protocol import load_option_protocol
from .r1b import evaluation_summary
from .reservoir import TeacherBcReservoir, file_digest, load_teacher_bc_reservoir
from .train import DEFAULT_INITIALIZER

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1f_supervised_probe_v0.json"


def load_probe_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = {
        "format": "snowgym.engage-supervised-probe-config.v0",
        "sourceCheckpointDigest": "sha256:5e4feb4e1f97d2ba3a5521bcb6b4a5df8f9194ed02d4badf4a1b96304196c80c",
        "reservoirDigest": "sha256:a1410c32a718c53664b91878852a2203247454ae0bf2dcb4caeb904b0ac334a6",
        "stage": 1, "epochs": 20, "retainedEpochs": [0, 10, 20],
        "minibatchSize": 256, "trainingSeed": 91001, "learningRate": 0.0003,
        "maxGradNorm": 0.5,
        "bcLossConfig": {"action_weight": 1.0, "target_weight": 5.0, "power_weight": 0.5, "throw_action_weight": 5.0},
        "trainingEvaluationSeeds": [100000, 100039],
        "developmentEvaluationSeeds": [200000, 200039],
        "selectionPolicy": "final-epoch-only",
    }
    if value != frozen:
        raise ValueError("supervised probe configuration differs from the frozen R1f design")
    return value


def supervised_optimizer(model: HybridActorCritic, *, learning_rate: float) -> torch.optim.Optimizer:
    groups = plan_ppo_parameter_groups(model, 1, new_module_learning_rate=learning_rate)
    for parameter in model.role_aware_critic.parameters():
        parameter.requires_grad_(False)
    for group in groups:
        group["params"] = [parameter for parameter in group["params"] if parameter.requires_grad]
    return torch.optim.Adam(groups)


def supervised_epoch(
    model: HybridActorCritic, optimizer: torch.optim.Optimizer, reservoir: TeacherBcReservoir,
    *, epoch: int, seed: int, minibatch_size: int, loss_config: LossConfig, max_grad_norm: float,
) -> dict[str, Any]:
    if epoch < 1 or minibatch_size < 1 or max_grad_norm <= 0:
        raise ValueError("supervised epoch arguments are invalid")
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
    order = torch.randperm(reservoir.size, generator=generator)
    totals = {name: 0.0 for name in ("total", "action", "target", "power")}
    seen = steps = 0
    maximum_norm = 0.0
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for start in range(0, reservoir.size, minibatch_size):
        indices = order[start : start + minibatch_size]
        observation, teacher = reservoir.batch(indices)
        losses = behavior_clone_loss(model(observation), teacher, observation, loss_config)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise ValueError("non-finite supervised loss")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        maximum_norm = max(maximum_norm, float(norm))
        optimizer.step()
        for name, value in losses.items():
            totals[name] += float(value.detach()) * len(indices)
        seen += len(indices)
        steps += 1
    return {"epoch": epoch, "samples": seen, "optimizerSteps": steps,
            "maximumActorGradientNormBeforeClip": maximum_norm,
            "loss": {name: total / seen for name, total in totals.items()}}


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite probe artifact {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_inputs(config: dict[str, Any], source: dict[str, Any], reservoir: TeacherBcReservoir) -> list[int]:
    if source["checkpointDigest"] != config["sourceCheckpointDigest"]:
        raise ValueError("probe source checkpoint digest mismatch")
    if reservoir.metadata["digest"] != config["reservoirDigest"]:
        raise ValueError("probe reservoir digest mismatch")
    manifest = json.loads(Path(reservoir.metadata["manifestPath"]).read_text(encoding="utf-8"))
    claimed = manifest.pop("manifestDigest", None)
    if json_digest(manifest) != claimed:
        raise ValueError("probe reservoir manifest digest mismatch")
    protocol = load_option_protocol()
    seeds = list(range(config["trainingEvaluationSeeds"][0], config["trainingEvaluationSeeds"][1] + 1))
    if (manifest.get("seedPartition") != "training" or manifest.get("seeds") != seeds
        or manifest.get("protocolDigest") != json_digest(protocol)
        or reservoir.metadata["samples"] != 5367 or reservoir.metadata["episodes"] != 40):
        raise ValueError("probe reservoir training seed or protocol provenance mismatch")
    if source["curriculumDigest"] != json_digest(protocol):
        raise ValueError("probe checkpoint option protocol mismatch")
    if config["developmentEvaluationSeeds"] != [protocol["seeds"]["development"][0], protocol["seeds"]["development"][0] + 39]:
        raise ValueError("probe development seeds changed")
    return seeds


def run_supervised_probe(
    *, source_checkpoint: str | Path, reservoir_path: str | Path, output: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_probe_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite probe {destination}")
    source, state = load_ppo_checkpoint(source_checkpoint)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    seeds = validate_inputs(config, source, reservoir)
    root_metadata, root_state = load_checkpoint(DEFAULT_INITIALIZER)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(config["trainingSeed"])
    model = checkpoint_model(source)
    model.load_state_dict(state["model"])
    warm_start = freeze_initializer(model)
    initializer, identity = recover_initializer(source, state, root_metadata, root_state)
    optimizer = supervised_optimizer(model, learning_rate=config["learningRate"])
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    commit = resolve_git_commit()
    protocol = load_option_protocol()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        write_json(root / "config.json", config)
        checkpoints = {}
        agreement = {}
        training_evaluation = {}
        epochs = []
        capabilities = None
        for epoch in range(config["epochs"] + 1):
            if epoch:
                metrics = supervised_epoch(model, optimizer, reservoir, epoch=epoch,
                    seed=config["trainingSeed"], minibatch_size=config["minibatchSize"],
                    loss_config=LossConfig(**config["bcLossConfig"]), max_grad_norm=config["maxGradNorm"])
                epochs.append(metrics)
                print(json.dumps({"phase": "supervised-training", **metrics}), flush=True)
            if epoch not in config["retainedEpochs"]:
                continue
            # The shared checkpoint container retains optimizer/model provenance;
            # gateId identifies BC-only training and prevents option-PPO exact resume.
            checkpoint = root / f"epoch-{epoch:03d}"
            checkpoints[str(epoch)] = save_ppo_checkpoint(checkpoint, model=model, optimizer=optimizer,
                config=PPOConfig(**source["ppoConfig"]), curriculum_digest=json_digest(protocol),
                training_seed=config["trainingSeed"], update_index=epoch, environment_steps=0,
                git_commit=commit, seed_schedule={"minimum": seeds[0], "maximum": seeds[-1], "nextSeed": seeds[0]},
                collector_config={"gateId": "m7b-engage-supervised-probe", "worlds": 1,
                    "rolloutSteps": 200, "rewardMode": "executor", "option": "engage", "stage": 1},
                initialization={"type": "ppo-transfer", "checkpointDigest": source["checkpointDigest"],
                    "stateDigest": source["stateDigest"], "curriculumDigest": source["curriculumDigest"],
                    "sourceGate": source["collectorConfig"]["gateId"], "updateIndex": source["updateIndex"]},
                initializer=initializer, initializer_source_digest=root_metadata["checkpointDigest"])
            agreement[str(epoch)] = teacher_agreement(model, reservoir)
            write_json(root / f"teacher-agreement-epoch-{epoch:03d}.json", agreement[str(epoch)])
            print(json.dumps({"phase": "teacher-agreement", "epoch": epoch, "metrics": agreement[str(epoch)]["all"]}), flush=True)
            if epoch in (0, config["epochs"]):
                with SnowGymBatchClient() as client:
                    capabilities = require_capabilities(client)
                    if capabilities["simulationVersion"] != reservoir.metadata["simulationVersion"] or capabilities["stateHashVersion"] != reservoir.metadata["stateHashVersion"]:
                        raise ValueError("probe simulator version does not match teacher reservoir")
                    rows = []
                    for seed in seeds:
                        rows.append(evaluate_option_episode(model, initializer, option="engage", seed=seed,
                            condition="correct", client=client))
                    training_evaluation[str(epoch)] = rows
                write_json(root / f"training-evaluation-epoch-{epoch:03d}.json", rows)
                print(json.dumps({"phase": "training-closed-loop", "epoch": epoch,
                                  "successes": sum(row["success"] for row in rows)}), flush=True)
        changed_frozen = [name for name, parameter in model.named_parameters()
                          if name in frozen and not torch.equal(parameter.detach(), frozen[name])]
        if changed_frozen:
            raise RuntimeError(f"probe changed frozen parameters: {changed_frozen}")
        final_evaluation = evaluate_m7b_checkpoint(root / f"epoch-{config['epochs']:03d}",
            output=root / "development-evaluation.json", options=("engage",))
        gate = engage_bootstrap_report(final_evaluation)
        write_json(root / "bootstrap-diagnostic.json", gate)
        summary = {
            "format": "snowgym.engage-supervised-probe-report.v0",
            "trainingMethod": "supervised-only", "ppoUpdates": 0, "criticUpdates": 0,
            "trainingSummaries": {key: summarize_rows(rows) for key, rows in training_evaluation.items()},
            "development": evaluation_summary(final_evaluation),
            "parameterChangeFromR1e": parameter_changes(model, warm_start),
            "frozenParametersUnchanged": True,
            "bootstrapDiagnosticPassed": gate["passed"],
            "qualificationEligible": False,
            "interpretation": "Bounded Stage-1 supervised fitting probe; R1 promotion and PPO attribution require separate experiments.",
        }
        write_json(root / "report.json", summary)
        source_files = {str(path.relative_to(Path(__file__).resolve().parents[1])): file_digest(path)
                        for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))}
        manifest = {"format": "snowgym.engage-supervised-probe-run.v0", "gitCommit": commit,
            "config": config, "configDigest": json_digest(config), "protocolDigest": json_digest(protocol),
            "sourceCheckpoint": source, "rootInitializerIdentity": identity,
            "reservoir": reservoir.metadata, "capabilities": capabilities,
            "checkpoints": checkpoints, "epochs": epochs,
            "sourceFiles": source_files,
            "artifacts": {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*")) if path.is_file()}}
        manifest["manifestDigest"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        root.replace(destination)
        return summary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["totalActions"] for row in rows)
    return {"episodes": len(rows), "successRate": float(np.mean([row["success"] for row in rows])),
        "contactRate": float(np.mean([row["firstContactDecision"] is not None for row in rows])),
        "hitRate": float(np.mean([row["firstHitDecision"] is not None for row in rows])),
        "meanProgress": float(np.mean([row["progress"] for row in rows])),
        "physicalWinRate": float(np.mean([row["physicalWin"] for row in rows])),
        "rejectedActionRate": sum(row["rejectedActions"] for row in rows) / total if total else 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(run_supervised_probe(source_checkpoint=args.source_checkpoint,
        reservoir_path=args.reservoir, output=args.output, config_path=args.config), sort_keys=True))


if __name__ == "__main__":
    main()
