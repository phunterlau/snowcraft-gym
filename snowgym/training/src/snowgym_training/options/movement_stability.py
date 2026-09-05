"""R1m-S1: independent critic scheduling, with a frozen matched budget."""

import argparse
import copy
import json
from pathlib import Path

import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.movement_ppo import AssistedMovementPolicy, movement_loss
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import SeedSchedule
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .corrective_data import paired_effect
from .identity import checkpoint_model
from .interventions import require_capabilities
from .movement_collect import ASSIST_FIELDS, MovementCollector
from .movement_checkpoint import load_movement
from .movement_train import (TRAINING, REFERENCE, load_config, make_wrapper,
                             ppo_update, train_run, evaluate)
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json

ARMS = ("coupled", "independent-after-actor-stop")


def configuration(arm):
    if arm not in ARMS:
        raise ValueError("unknown critic schedule")
    return {**load_config(), "format": "snowgym.movement-stability-config.v0",
            "updates": 20, "criticSchedule": arm}


def independent_update(model, optimizer, rollout, config):
    size = len(rollout["advantage"])
    traces, stop_kl, stop_rng = [], None, None
    actor_steps = 0
    try:
        for epoch in range(config["epochs"]):
            for indices in torch.randperm(size).split(config["minibatchSize"]):
                obs = {k: v[indices] for k, v in rollout["observation"].items()}
                logp, prediction = model.evaluate_latents(obs, rollout["action_type"][indices], rollout["latent"][indices])
                losses = movement_loss(logp, rollout["logp"][indices], rollout["advantage"][indices],
                    prediction["value"], rollout["returns"][indices], prediction, clip_ratio=config["clipRatio"])
                if not all(torch.isfinite(v) for v in losses.values()):
                    raise ValueError("non-finite movement PPO loss")
                if stop_rng is None and float(losses["meanMovementKl"].detach()) > config["movementKlStop"]:
                    stop_kl = float(losses["meanMovementKl"].detach())
                    stop_rng = torch.get_rng_state().clone()
                optimizer.zero_grad(set_to_none=True)
                # Independent critic: no actor graph or Adam momentum update after stop.
                (losses["total"] if stop_rng is None else .5 * losses["value"]).backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), config["actorGradClip"], error_if_nonfinite=True)
                critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), config["criticGradClip"], error_if_nonfinite=True)
                optimizer.step()
                actor_steps += int(stop_rng is None)
                traces.append({"epoch": epoch, **{k: float(v.detach()) for k, v in losses.items()},
                    "actorGradientNorm": float(actor_norm), "criticGradientNorm": float(critic_norm),
                    "criticOnly": stop_rng is not None})
    finally:
        if stop_rng is not None:
            torch.set_rng_state(stop_rng)
    return {"optimizerSteps": len(traces), "actorOptimizerSteps": actor_steps,
        "criticOptimizerSteps": len(traces), "klStopped": stop_rng is not None,
        "stopMeanMovementKl": stop_kl, "minibatches": traces,
        "meanReward": float(rollout["reward"].mean()),
        "movementFraction": float(((rollout["action_type"] == 1) &
            (rollout["observation"]["allies"][..., 1] > .5)).float().mean())}


def actor_identity(model, optimizer):
    return semantic_state_digest({n: {"parameter": p.detach(), "optimizer": optimizer.state.get(p, {})}
        for n, p in model.named_parameters() if p.requires_grad and not n.startswith("critic.")})


def common_probe(source, client, seed):
    config = configuration(ARMS[0])
    torch.manual_seed(seed)
    model = AssistedMovementPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
    collector = MovementCollector(make_wrapper(client, config["batchSize"], config["gamma"]),
                                 model, SeedSchedule(*config["trainingEpisodeSeeds"]))
    collector.start(config["rolloutDecisions"])
    collector.advance()
    rollout = collector.rollout(gamma=config["gamma"], gae_lambda=config["gaeLambda"])
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        before = float(((model(rollout["observation"])["value"] - rollout["returns"]) ** 2).mean())
    rows = {}
    for arm in ARMS:
        current = copy.deepcopy(model)
        optimizer = torch.optim.Adam([p for p in current.parameters() if p.requires_grad], lr=config["learningRate"])
        torch.set_rng_state(rng)
        trace = ppo_update(current, optimizer, rollout, configuration(arm))
        with torch.no_grad():
            mse = float(((current(rollout["observation"])["value"] - rollout["returns"]) ** 2).mean())
        rows[arm] = {"actorIdentity": actor_identity(current, optimizer),
            "rngDigest": semantic_state_digest({"rng": torch.get_rng_state()}),
            "criticDigest": semantic_state_digest(current.critic.state_dict()),
            "fixedReturnMseBefore": before, "fixedReturnMseAfter": mse,
            "optimizerSteps": trace["optimizerSteps"], "klStopped": trace["klStopped"],
            "updateTrace": trace}
    control, candidate = (rows[a] for a in ARMS)
    if any(control[k] != candidate[k] for k in ("actorIdentity", "rngDigest")):
        raise ValueError("common rollout actor/optimizer/RNG parity failed")
    return {"seed": seed, **ASSIST_FIELDS, "rolloutDigest": semantic_state_digest(rollout),
            "exactActorOptimizerRngParity": True, "arms": rows}


def decision(comparisons):
    fresh = [r["replicationDevelopment"] for r in comparisons]
    gains = [r["effect"]["success"]["mean"] for r in fresh]
    checks = {"allFreshGainsNonnegative": all(g >= 0 for g in gains),
        "meanFreshGainAtLeastTenPoints": sum(gains)/len(gains) >= .1-1e-12,
        "onePositiveFreshLowerBound": any(r["effect"]["success"]["ci95"][0] > 0 for r in fresh),
        "rejections": all(v["rejectedActionRate"] < .001 for row in comparisons
                           for k, v in row.items() if k != "seed")}
    return {"promising": all(checks.values()), "criteria": checks, **ASSIST_FIELDS}


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    archive = TRAINING / "runs/m7b_engage_r1m_movement_v0"
    audited = audit_artifact_manifest(archive, "manifest.json")
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata["checkpointDigest"] != configuration(ARMS[0])["checkpointDigest"]:
        raise ValueError("source checkpoint mismatch")
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    source_digest = semantic_state_digest(source.state_dict())
    repository = TRAINING.parents[1]
    files = sorted(p for base in (repository / "snowgym", repository / "src")
                   for p in base.rglob("*") if p.suffix in (".py", ".ts")
                   and not any(part in (".venv", "node_modules", "runs", "__pycache__") for part in p.parts))
    source_files = {str(p.relative_to(repository)): file_digest(p) for p in files}
    declaration = TRAINING / "reviews/m7b_r1m_s1_declaration.md"
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"configs": {a: configuration(a) for a in ARMS},
        "declarationDigest": file_digest(declaration), "sourceCheckpoint": metadata["checkpointDigest"],
        "archiveManifestDigest": file_digest(archive / "manifest.json"), "gitCommit": resolve_git_commit(),
        "sourceFiles": source_files, **ASSIST_FIELDS})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    comparisons = []
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        for seed in configuration(ARMS[0])["trainingRngs"]:
            probe = common_probe(source, client, seed)
            historical = json.loads((archive / str(seed) / "training.json").read_text())["trace"][0]
            if any(historical[k] != v for k, v in probe["arms"][ARMS[0]]["updateTrace"].items()):
                raise ValueError("first update differs from archived R1m control")
            probe["archivedFirstUpdateParity"] = True
            write_json(root / f"common-probe-{seed}.json", probe)
        for seed in configuration(ARMS[0])["trainingRngs"]:
            evaluations = {}
            for arm in ARMS:
                directory = root / str(seed) / arm
                directory.mkdir(parents=True)
                config = configuration(arm)
                _, training = train_run(source, metadata, client, config, directory, seed)
                model, _, _, _ = load_movement(directory / "final")
                rows = {}
                for label, key in (("development", "developmentSeeds"),
                                   ("replicationDevelopment", "replicationDevelopmentSeeds")):
                    lo, hi = config[key]
                    rows[label] = evaluate(model, client, range(lo, hi+1), config)
                write_json(directory / "evaluation.json", rows)
                evaluations[arm] = rows
            comparison = {"seed": seed}
            for label in ("development", "replicationDevelopment"):
                control, candidate = (evaluations[a][label] for a in ARMS)
                comparison[label] = {"effect": paired_effect(candidate, control, config),
                    "controlSuccesses": sum(r["success"] for r in control),
                    "candidateSuccesses": sum(r["success"] for r in candidate),
                    "rejectedActionRate": sum(r["rejectedActions"] for r in candidate)/max(1, sum(r["totalActions"] for r in candidate))}
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    if semantic_state_digest(source.state_dict()) != source_digest:
        raise ValueError("source changed")
    if audit_artifact_manifest(archive, "manifest.json") != audited:
        raise ValueError("archive changed")
    if any(file_digest(repository / p) != d for p, d in source_files.items()):
        raise ValueError("implementation changed during experiment")
    report = {"format": "snowgym.movement-stability-report.v0", "comparisons": comparisons,
              "decision": decision(comparisons), **ASSIST_FIELDS}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.movement-stability-manifest.v0", **ASSIST_FIELDS,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)), flush=True)
