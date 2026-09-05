"""Bounded R1m reward-only movement experiment with frozen corrected shots."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv
from ..checkpoint import semantic_state_digest
from ..executor.movement_ppo import AssistedMovementPolicy, movement_loss
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import SeedSchedule, numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .control_channels import evaluate_episode as historical_episode
from .corrective_data import paired_effect
from .engage_v1 import EngageOptionBatchV1, OPTION_STATE_VERSION
from .identity import checkpoint_model
from .interventions import require_capabilities
from .movement_checkpoint import load_movement, save_movement
from .movement_collect import ASSIST_FIELDS, MovementCollector, corrected_shots
from .opportunity_audit import plain, write_jsonl
from .plans import teacher_option_plan, teacher_option_scenario
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json

TRAINING = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1m_movement_v0.json"
REFERENCE = TRAINING / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020"
HISTORICAL = TRAINING / "runs/m7b_engage_r1h_control_channels_v0"


def load_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text())
    expected = {"format": "snowgym.assisted-movement-config.v0",
        "checkpointDigest": "sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697",
        "trainingRngs": [94001, 94002, 94003], "trainingEpisodeSeeds": [100000, 107999],
        "developmentSeeds": [200000, 200039], "replicationDevelopmentSeeds": [210000, 210039],
        "batchSize": 8, "rolloutDecisions": 200, "updates": 100, "epochs": 4, "minibatchSize": 400,
        "learningRate": 3e-4, "clipRatio": .2, "actorGradClip": .5, "criticGradClip": .5,
        "movementKlStop": .01, "latentStd": .02, "gamma": .9976921765, "gaeLambda": .9885140204,
        "returnHalfLifeSeconds": 30, "traceHalfLifeSeconds": 5, "decisionHz": 10,
        "bcCoefficient": 0, "entropyCoefficient": 0, "bootstrapSeed": 950001,
        "bootstrapSamples": 10000, "calibrationSeed": 100000, "calibrationDecisions": 200,
        "selection": "final-update-only"}
    if value != expected:
        raise ValueError("movement configuration differs from the frozen bounded experiment")
    return value


def make_wrapper(client, count, gamma):
    return EngageOptionBatchV1(SnowGymBatchEnv(count, client=client, observation_version=3), gamma=gamma)


def ppo_update(model, optimizer, rollout, config):
    size = len(rollout["advantage"])
    traces = []
    stopped = False
    stop_kl = None
    for epoch in range(config["epochs"]):
        order = torch.randperm(size)
        for indices in order.split(config["minibatchSize"]):
            obs = {k: v[indices] for k, v in rollout["observation"].items()}
            logp, prediction = model.evaluate_latents(obs, rollout["action_type"][indices], rollout["latent"][indices])
            losses = movement_loss(logp, rollout["logp"][indices], rollout["advantage"][indices],
                prediction["value"], rollout["returns"][indices], prediction, clip_ratio=config["clipRatio"])
            if not all(torch.isfinite(value) for value in losses.values()):
                raise ValueError("non-finite movement PPO loss")
            if float(losses["meanMovementKl"].detach()) > config["movementKlStop"]:
                stopped = True
                stop_kl = float(losses["meanMovementKl"].detach())
                break
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), config["actorGradClip"], error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), config["criticGradClip"], error_if_nonfinite=True)
            optimizer.step()
            traces.append({"epoch": epoch, **{k: float(v.detach()) for k, v in losses.items()},
                           "actorGradientNorm": float(actor_norm), "criticGradientNorm": float(critic_norm)})
        if stopped:
            break
    return {"optimizerSteps": len(traces), "klStopped": stopped, "stopMeanMovementKl": stop_kl, "minibatches": traces,
            "meanReward": float(rollout["reward"].mean()),
            "movementFraction": float(((rollout["action_type"] == 1) &
                (rollout["observation"]["allies"][..., 1] > .5)).float().mean())}


def evaluate(model, client, seeds, config):
    wrapper = make_wrapper(client, 1, config["gamma"])
    plan, spec = teacher_option_plan("engage")
    rows = []
    for seed in seeds:
        obs, _ = wrapper.reset([seed], [teacher_option_scenario("engage")], [f"movement-{seed}"], [plan], [spec])
        hashes, actions = [wrapper.environment.state_hashes[0]], []
        rejected = total = 0
        for _ in range(spec.horizon):
            with torch.no_grad():
                action, _, _, _ = model.act(tensor_dict(obs), deterministic=True)
            executed = corrected_shots(numpy_actions(action), wrapper.environment.raw_observations)
            actions.append(plain(executed))
            obs, _, terminated, truncated, infos = wrapper.step(executed)
            hashes.append(wrapper.environment.state_hashes[0])
            results = infos[0].get("actionResults", [])
            rejected += sum(r.get("accepted") is False for r in results)
            total += len(results)
            if terminated[0] or truncated[0]:
                break
        final = infos[0]["option"]
        rows.append({"seed": seed, **ASSIST_FIELDS, "success": final["success"], "progress": final["progress"],
            "physicalWin": not any(u["alive"] for u in wrapper.environment.raw_observations[0]["enemies"]),
            "decisions": final["decision"], "rejectedActions": rejected, "totalActions": total,
            "stateHashes": hashes, "actionsDigest": json_digest(actions)})
    return rows


def calibrate(model, client, config):
    # Fixed-noise draws are diagnostic; continuation follows the zero-noise baseline.
    wrapper = make_wrapper(client, 1, config["gamma"])
    plan, spec = teacher_option_plan("engage")
    seed = config["calibrationSeed"]
    obs, _ = wrapper.reset([seed], [teacher_option_scenario("engage")], [f"movement-{seed}"], [plan], [spec])
    errors, saturated, coordinates, parity = [], 0, 0, True
    with torch.random.fork_rng():
        torch.manual_seed(950002)
        for _ in range(config["calibrationDecisions"]):
            tensors = tensor_dict(obs)
            with torch.no_grad():
                baseline, _, _, _ = model.act(tensors, deterministic=True)
                source = model.geometry.source.act(tensors, deterministic=True)[0]
                parity &= all(torch.equal(baseline[k], source[k]) for k in baseline)
                sampled, _, _, _ = model.act(tensors)
            live = tensors["allies"][..., 1] > .5
            moves = (baseline["action_type"] == 1) & live
            arena = wrapper.environment.raw_observations[0]["arena"]
            scale = torch.tensor([arena["width"]/2, arena["height"]/2])
            errors.extend(torch.linalg.vector_norm((sampled["target"]-baseline["target"])*scale, dim=-1)[moves].tolist())
            saturated += int((sampled["target"][moves].abs() >= .99).sum())
            coordinates += int(moves.sum())*2
            obs, _, terminated, truncated, _ = wrapper.step(corrected_shots(numpy_actions(baseline), wrapper.environment.raw_observations))
            if terminated[0] or truncated[0]:
                break
    if not parity:
        raise ValueError("zero residual does not reproduce the assisted source")
    return {**ASSIST_FIELDS, "seed": seed, "zeroNoiseSourceParity": parity,
            "moveOpportunities": len(errors), "worldDisplacementMean": float(np.mean(errors)),
            "worldDisplacementP90": float(np.quantile(errors, .9)),
            "saturatedCoordinateFraction": saturated/max(coordinates, 1)}


def result_gate(final, baseline, parameter_change, config):
    effect = paired_effect(final, baseline, config)
    rejection = sum(r["rejectedActions"] for r in final)/max(sum(r["totalActions"] for r in final), 1)
    criteria = {"successGain": effect["success"]["mean"] >= .2-1e-12,
                "positivePairedInterval": effect["success"]["ci95"][0] > 0,
                "parameterChange": parameter_change > 0, "rejections": rejection < .001}
    return {"passed": all(criteria.values()), "criteria": criteria, "effect": effect,
            "rejectedActionRate": rejection, **ASSIST_FIELDS}


def train_run(source, source_metadata, client, config, root, seed, *, resume=None, pause_after=None):
    torch.manual_seed(seed)
    model = AssistedMovementPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learningRate"])
    schedule = SeedSchedule(*config["trainingEpisodeSeeds"])
    first_update, partial, resume_digest = 0, None, None
    if resume:
        model, optimizer, metadata, partial = load_movement(resume)
        if metadata["config"] != config or metadata["trainingSeed"] != seed or metadata["source"] != source_metadata:
            raise ValueError("resume experiment/source identity mismatch")
        first_update, resume_digest = metadata["updateIndex"], metadata["checkpointDigest"]
        cursor = metadata["seedSchedule"]
        schedule = SeedSchedule(cursor["minimum"], cursor["maximum"], cursor["nextSeed"])
    collector = MovementCollector(make_wrapper(client, config["batchSize"], config["gamma"]), model, schedule)
    traces = []
    for update in range(first_update, config["updates"]):
        if partial is None:
            collector.start(config["rolloutDecisions"])
        else:
            collector.restore(partial)
            partial = None
        if pause_after is not None:
            collector.advance(pause_after)
            saved = save_movement(root / "partial", model, optimizer, source=source_metadata, config=config,
                seed=seed, update=update, schedule=collector.schedule.state(), collection=collector.snapshot())
            return model, {"paused": True, "checkpointDigest": saved["checkpointDigest"], **ASSIST_FIELDS}
        collector.advance()
        rollout = collector.rollout(gamma=config["gamma"], gae_lambda=config["gaeLambda"])
        trace = ppo_update(model, optimizer, rollout, config)
        trace.update({"update": update+1, "completedOptions": sum(e["option"]["success"] or e["option"]["failed"] for e in collector.events),
                      "successes": sum(e["option"]["success"] for e in collector.events)})
        traces.append(trace)
        write_jsonl(root / f"events-{update+1:03d}.jsonl.gz", ({**ASSIST_FIELDS, **event} for event in collector.events))
        print(json.dumps({"phase": "movement-ppo", "seed": seed, "update": update+1,
                          "optimizerSteps": trace["optimizerSteps"], "successes": trace["successes"]}), flush=True)
    if semantic_state_digest(model.geometry.source.state_dict()) != semantic_state_digest(source.state_dict()):
        raise RuntimeError("movement training changed its frozen source")
    saved = save_movement(root / "final", model, optimizer, source=source_metadata, config=config,
        seed=seed, update=config["updates"], schedule=collector.schedule.state())
    change = sum(float((p.detach()-initial[n]).square().sum()) for n, p in model.named_parameters()
                 if n in initial and not n.startswith("critic."))**.5
    report = {**ASSIST_FIELDS, "paused": False, "trace": traces, "actorParameterL2Change": change,
              "checkpointDigest": saved["checkpointDigest"], "resumedFrom": resume_digest,
              "completedPriorUpdates": first_update, "sourceUnchanged": True, "seedSchedule": collector.schedule.state()}
    write_json(root / "training.json", report)
    return model, report


def run_experiment(*, output, config_path=DEFAULT_CONFIG, resume=None, pause_after=None):
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite movement experiment {destination}")
    source_metadata, source_state = load_ppo_checkpoint(REFERENCE)
    if source_metadata["checkpointDigest"] != config["checkpointDigest"]:
        raise ValueError("R1m must use the historical R1h source")
    audit_artifact_manifest(HISTORICAL, "manifest.json")
    source = checkpoint_model(source_metadata)
    source.load_state_dict(source_state["model"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    module = Path(__file__).resolve().parents[1]
    repository = TRAINING.parents[1]
    paths = sorted([*module.rglob("*.py"), *(repository / "snowgym/server").rglob("*.ts"),
                    *(repository / "snowgym/python/src").rglob("*.py"), *(repository / "snowgym/core").rglob("*.ts"),
                    *(repository / "snowgym/orchestration").rglob("*.ts"), *(repository / "src").rglob("*.ts")])
    source_files = {str(p.relative_to(repository)): file_digest(p) for p in paths}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        with SnowGymBatchClient() as client:
            capabilities = require_capabilities(client)
            archived = json.loads((HISTORICAL / "shot-only.json").read_text())
            reproduced = [historical_episode(source, seed=row["seed"], arm="shot-only", client=client) for row in archived]
            if reproduced != archived:
                raise ValueError("R1h corrected-shot historical trajectories did not reproduce exactly")
            write_json(root / "historical-reproduction.json", {**ASSIST_FIELDS, "exact": True, "episodes": reproduced})
            print(json.dumps({"phase": "historical-baseline", "exact": True,
                              "successes": sum(r["success"] for r in reproduced)}), flush=True)
            with torch.random.fork_rng():
                torch.manual_seed(config["trainingRngs"][0])
                baseline = AssistedMovementPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
            write_json(root / "calibration.json", calibrate(baseline, client, config))
            splits = {name: range(config[key][0], config[key][1]+1) for name, key in
                      (("historical", "developmentSeeds"), ("replication", "replicationDevelopmentSeeds"))}
            baselines = {name: evaluate(baseline, client, seeds, config) for name, seeds in splits.items()}
            write_json(root / "assisted-initialization.json", baselines)
            reports = {}
            for i, seed in enumerate(config["trainingRngs"]):
                if i and not reports[str(config["trainingRngs"][0])]["gates"]["historical"]["passed"]:
                    break
                arm_root = root / str(seed)
                arm_root.mkdir()
                model, trained = train_run(source, source_metadata, client, config, arm_root, seed,
                    resume=resume if i == 0 else None, pause_after=pause_after)
                if trained["paused"]:
                    reports[str(seed)] = trained
                    break
                restored, _, _, _ = load_movement(arm_root / "final")
                evaluations = {name: evaluate(restored, client, seeds, config) for name, seeds in splits.items()}
                write_json(arm_root / "evaluation.json", evaluations)
                reports[str(seed)] = {"actorParameterL2Change": trained["actorParameterL2Change"],
                    "gates": {name: result_gate(rows, baselines[name], trained["actorParameterL2Change"], config)
                              for name, rows in evaluations.items()},
                    "successes": {name: sum(r["success"] for r in rows) for name, rows in evaluations.items()}}
            report = {"format": "snowgym.assisted-movement-report.v0", **ASSIST_FIELDS,
                "historicalBaselineExact": True, "baselineSuccesses": {name: sum(r["success"] for r in rows) for name, rows in baselines.items()},
                "runs": reports, "replicated": str(config["trainingRngs"][-1]) in reports,
                "selection": "final-update-only", "pairedIntervalsUnit": "environment-seed; optimizer runs kept separate"}
            write_json(root / "report.json", report)
        if any(file_digest(repository / path) != digest for path, digest in source_files.items()):
            raise RuntimeError("source files changed during the movement experiment")
        manifest = {"format": "snowgym.assisted-movement-run.v0", **ASSIST_FIELDS,
            "gitCommit": resolve_git_commit(), "config": config, "configDigest": json_digest(config),
            "sourceCheckpoint": source_metadata, "historicalManifestDigest": file_digest(HISTORICAL / "manifest.json"),
            "optionStateVersion": OPTION_STATE_VERSION, "capabilities": capabilities, "sourceFiles": source_files,
            "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
        manifest["manifestDigest"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        root.replace(destination)
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", help="Resume an immutable matching first-RNG movement checkpoint")
    parser.add_argument("--pause-after", type=int, help="Diagnostic partial-collection checkpoint after this many decisions")
    args = parser.parse_args()
    print(json.dumps(run_experiment(output=args.output, config_path=args.config, resume=args.resume, pause_after=args.pause_after)), flush=True)


if __name__ == "__main__":
    main()
