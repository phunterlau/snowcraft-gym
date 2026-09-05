"""R1m-S3: bounded reward-only short movement recovery with frozen tails."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.recovery_ppo import RecoveryPolicy, RECOVERY_INPUT
from ..ppo import generalized_advantage_estimate
from ..ppo_checkpoint import load_ppo_checkpoint
from ..ppo_collect import tensor_dict, numpy_actions
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import post_hit, recovery_checkpoint
from .identity import checkpoint_model
from .interventions import require_capabilities, team_health
from .movement_collect import ASSIST_FIELDS, corrected_shots
from .movement_train import TRAINING, REFERENCE, make_wrapper, load_config, ppo_update, result_gate
from .opportunity_audit import plain, write_jsonl
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json


def configuration():
    return {**load_config(), "format": "snowgym.short-recovery-config.v0",
        "recoveryInput": RECOVERY_INPUT, "recoveryWindow": 30, "trainingRngs": [94101, 94102, 94103],
        "trainingEpisodeSeeds": [100000, 100063], "updates": 30, "minibatchSize": 120,
        "bootstrapSeed": 970001, "criticSchedule": "coupled", "checkpointEvery": 10,
        "collectionMode": "eight-independent-prefix-restorations", "selection": "final-update-only"}


def recovery_observation(observation, offset, window):
    if window < 1 or not 0 <= offset < window:
        raise ValueError("invalid recovery window/offset")
    result = {k: v.clone() for k, v in tensor_dict(observation).items()}
    result["recovery_remaining"] = torch.full((len(result["allies"]), 1), (window-offset)/window)
    return result


def fold_tail(rewards, learned, gamma):
    if not 1 <= learned <= len(rewards):
        raise ValueError("invalid learned reward prefix")
    result = torch.tensor(rewards[:learned], dtype=torch.float32)
    tail = 0.
    for reward in reversed(rewards[learned:]):
        tail = float(reward) + gamma*tail
    result[-1] += gamma*tail
    return result, tail


def frame_from_baseline(row):
    trigger = row["trigger"]
    if trigger is None or trigger["terminal"]:
        return None
    value = {"seed": row["seed"], "trigger": trigger, "baseline": row["final"],
        "suffixHashes": row["stateHashes"][trigger["decision"]:],
        "suffixActionsDigest": json_digest([r["action"] for r in row["trace"]][trigger["decision"]:])}
    value["frameDigest"] = json_digest(value)
    return value


def validate_frames(frames, *, training):
    seeds = [f["seed"] for f in frames]
    if not frames or len(set(seeds)) != len(seeds):
        raise ValueError("empty or duplicated snapshot seeds")
    for f in frames:
        if f["frameDigest"] != json_digest({k: v for k, v in f.items() if k != "frameDigest"}):
            raise ValueError("snapshot digest mismatch")
        if training and not 100000 <= f["seed"] <= 100063:
            raise ValueError("development snapshot cannot enter training")
        if not training and not (200000 <= f["seed"] <= 200039 or 210000 <= f["seed"] <= 210039):
            raise ValueError("evaluation seed allocation mismatch")


def episode(model, source, wrapper, frame, config, *, deterministic=False):
    trigger = frame["trigger"]
    obs = post_hit.restore(wrapper, frame["seed"], trigger["prefix"], trigger["identity"])
    raw = wrapper.environment.raw_observations[0]
    start_health = (team_health(raw["allies"]), team_health(raw["enemies"]))
    records, events, rewards, hashes = [], [], [], [wrapper.environment.state_hashes[0]]
    local = None
    for offset in range(200-trigger["decision"]):
        learned = offset < config["recoveryWindow"]
        with torch.no_grad():
            if learned:
                inputs = recovery_observation(obs, offset, config["recoveryWindow"])
                action, latent, logp, value = model.act(inputs, deterministic=deterministic)
                records.append({"observation": inputs, "action_type": action["action_type"].clone(),
                    "latent": latent.clone(), "logp": logp.clone(), "value": value.clone()})
            else:
                action = source.act(tensor_dict(obs), deterministic=True)[0]
        executed = corrected_shots(numpy_actions(action), wrapper.environment.raw_observations)
        obs, reward, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]
        rewards.append(float(reward[0]))
        hashes.append(wrapper.environment.state_hashes[0])
        events.append({"offset": offset, "learnerControlled": learned, "action": plain(executed),
                       "stateHash": hashes[-1], "info": plain(info)})
        done = bool(terminated[0] or truncated[0])
        if offset == config["recoveryWindow"]-1 or done and local is None:
            local = post_hit.outcome(wrapper, info, start_health, offset+1)
        if done:
            break
    count = len(records)
    folded, tail = fold_tail(rewards, count, config["gamma"])
    values = torch.cat([r["value"] for r in records])
    next_values = torch.cat([values[1:], torch.zeros(1)])
    terminal = torch.zeros(count, dtype=torch.bool)
    terminal[-1] = True  # Actual frozen continuation has already reached a terminal.
    advantage, returns = generalized_advantage_estimate(folded[:, None], values[:, None], next_values[:, None],
        terminal[:, None], torch.zeros(count, 1, dtype=torch.bool), gamma=config["gamma"], gae_lambda=config["gaeLambda"])
    rollout = {"observation": {k: torch.cat([r["observation"][k] for r in records]) for k in records[0]["observation"]},
        **{k: torch.cat([r[k] for r in records]) for k in ("action_type", "latent", "logp", "value")},
        "reward": folded, "advantage": advantage.flatten(), "returns": returns.flatten()}
    summary = {"seed": frame["seed"], **post_hit.outcome(wrapper, info, start_health, len(events)),
        **ASSIST_FIELDS, "recoveryInput": RECOVERY_INPUT, "local": local, "stateHashes": hashes,
        "actionsDigest": json_digest([e["action"] for e in events]), "learnedDecisions": count,
        "tailDecisions": len(events)-count, "tailDiscountedReward": tail,
        "rejectedActions": sum(a.get("accepted") is False for e in events for a in e["info"]["actionResults"]),
        "totalActions": sum(len(e["info"]["actionResults"]) for e in events)}
    trajectory = {"summary": summary, "frameDigest": frame["frameDigest"], "events": events,
        "learnerLatents": plain(rollout["latent"]), "behaviorLogProbabilities": plain(rollout["logp"]),
        "foldedRewards": plain(folded), "rewards": rewards}
    return rollout, summary, trajectory


def combine(rollouts):
    return {"observation": {k: torch.cat([r["observation"][k] for r in rollouts]) for k in rollouts[0]["observation"]},
            **{k: torch.cat([r[k] for r in rollouts]) for k in rollouts[0] if k != "observation"}}


def train(source, metadata, client, frames, config, root, seed, *, resume=None, pause_after=None):
    validate_frames(frames, training=True)
    digest = json_digest(frames)
    torch.manual_seed(seed)
    model = RecoveryPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("critic.")}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learningRate"])
    sampler = np.random.default_rng(seed)
    start, history = 0, []
    if resume:
        model, optimizer, saved = recovery_checkpoint.load(resume)
        if saved["config"] != config or saved["datasetDigest"] != digest or saved["trainingSeed"] != seed or saved["source"] != metadata:
            raise ValueError("recovery resume lineage mismatch")
        sampler.bit_generator.state = saved["sampler"]
        start, history = saved["update"], saved["history"]
    wrapper = make_wrapper(client, 1, config["gamma"])
    for update in range(start, config["updates"]):
        chosen = sampler.integers(len(frames), size=config["batchSize"]).tolist()
        batches, summaries, trajectories = [], [], []
        for index in chosen:
            batch, summary, trajectory = episode(model, source, wrapper, frames[index], config)
            batches.append(batch); summaries.append(summary); trajectories.append(trajectory)
        rollout = combine(batches)
        with torch.no_grad():
            reevaluated, _ = model.evaluate_latents(rollout["observation"], rollout["action_type"], rollout["latent"])
            error = float((reevaluated-rollout["logp"]).abs().max())
        if error > .001:
            raise ValueError("stored/recomputed recovery likelihood mismatch")
        trace = ppo_update(model, optimizer, rollout, config)
        trace.update({"update": update+1, "seeds": [frames[i]["seed"] for i in chosen],
            "likelihoodMaxError": error, "successes": sum(r["success"] for r in summaries),
            "learnedDecisions": sum(r["learnedDecisions"] for r in summaries),
            "tailDecisions": sum(r["tailDecisions"] for r in summaries)})
        history.append(trace)
        write_jsonl(root / f"events-{update+1:03d}.jsonl.gz", trajectories)
        print(json.dumps({"seed": seed, "update": update+1, "successes": trace["successes"],
                          "optimizerSteps": trace["optimizerSteps"]}), flush=True)
        name = "paused" if pause_after == update+1 else "final" if update+1 == config["updates"] else f"update-{update+1:03d}"
        if name in ("paused", "final") or (update+1) % config["checkpointEvery"] == 0:
            saved = recovery_checkpoint.save(root / name, model, optimizer, source=metadata, config=config, seed=seed,
                dataset_digest=digest, update=update+1, sampler=sampler.bit_generator.state, history=history)
        if name == "paused":
            return model, {"paused": True, "checkpointDigest": saved["checkpointDigest"]}
    change = sum(float((p.detach()-initial[n]).square().sum()) for n, p in model.named_parameters() if n in initial)**.5
    if semantic_state_digest(source.state_dict()) != semantic_state_digest(model.geometry.source.state_dict()):
        raise ValueError("frozen source changed")
    report = {"history": history, "actorParameterL2Change": change, "checkpointDigest": saved["checkpointDigest"],
              "datasetDigest": digest, "sourceUnchanged": True, **ASSIST_FIELDS}
    write_json(root / "training.json", report)
    return model, report


def evaluate(model, source, client, frames, config):
    validate_frames(frames, training=False)
    wrapper = make_wrapper(client, 1, config["gamma"])
    return [episode(model, source, wrapper, f, config, deterministic=True)[1] for f in frames]


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    config = configuration()
    archives = [TRAINING / "runs/m7b_engage_r1m_movement_v0", TRAINING / "runs/m7b_engage_r1m_s2_v0"]
    audits = [audit_artifact_manifest(a, "manifest.json") for a in archives]
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata["checkpointDigest"] != config["checkpointDigest"]:
        raise ValueError("source checkpoint mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = checkpoint_model(metadata)
    source.load_state_dict(state["model"])
    source.eval().requires_grad_(False)
    repository = TRAINING.parents[1]
    files = [p for base in (repository / "snowgym", repository / "src") for p in base.rglob("*")
        if p.suffix in (".ts", ".py") and not any(x in p.parts for x in (".venv", "runs", "node_modules", "__pycache__"))]
    sources = {str(p.relative_to(repository)): file_digest(p) for p in sorted(files)}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": config, "sourceFiles": sources, "source": metadata,
        "gitCommit": resolve_git_commit(), "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1m_s3_declaration.md"),
        "archiveManifests": [file_digest(a / "manifest.json") for a in archives]})
    datasets, coverage = {}, {}
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        wrapper = make_wrapper(client, 1, config["gamma"])
        old = json.loads((archives[0] / "assisted-initialization.json").read_text())["replication"]
        for split, seeds in (("training", range(100000, 100064)), ("historical", range(200000, 200040)),
                             ("replication", range(210000, 210040))):
            frames, excluded = [], []
            for seed in seeds:
                if split == "historical":
                    baseline = json.loads(gzip.decompress((archives[1] / f"baseline-{seed}.jsonl.gz").read_bytes()))
                else:
                    baseline = post_hit.continuation(source, wrapper, post_hit.reset(wrapper, seed), seed, baseline=True)
                    if split == "replication":
                        expected = old[seed-210000]
                        if any(baseline[k] != expected[k] for k in ("stateHashes", "actionsDigest")):
                            raise ValueError("replication baseline mismatch")
                    write_jsonl(root / f"baseline-{seed}.jsonl.gz", [baseline])
                frame = frame_from_baseline(baseline)
                if frame is None:
                    excluded.append(seed)
                else:
                    frames.append(frame)
            validate_frames(frames, training=split == "training")
            datasets[split] = frames
            coverage[split] = {"eligible": len(frames), "excluded": excluded}
        if len(datasets["training"]) < 8:
            raise ValueError("fewer than eight eligible training snapshots")
        write_json(root / "snapshots.json", {"datasets": datasets, "coverage": coverage, "digest": json_digest(datasets)})
        print(json.dumps({"coverage": coverage}), flush=True)
        torch.manual_seed(config["trainingRngs"][0])
        initial = RecoveryPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
        baseline_rows = {}
        for split in ("historical", "replication"):
            baseline_rows[split] = evaluate(initial, source, client, datasets[split], config)
            for row, frame in zip(baseline_rows[split], datasets[split], strict=True):
                if row["stateHashes"] != frame["suffixHashes"] or row["actionsDigest"] != frame["suffixActionsDigest"]:
                    raise ValueError("zero-residual Keep parity failed")
        write_json(root / "initialization.json", baseline_rows)
        reports = {}
        for seed in config["trainingRngs"]:
            directory = root / str(seed)
            directory.mkdir()
            _, training = train(source, metadata, client, datasets["training"], config, directory, seed)
            model, _, _ = recovery_checkpoint.load(directory / "final")
            rows = {split: evaluate(model, source, client, datasets[split], config) for split in baseline_rows}
            write_json(directory / "evaluation.json", rows)
            reports[str(seed)] = {"actorParameterL2Change": training["actorParameterL2Change"],
                "gates": {split: result_gate(rows[split], baseline_rows[split], training["actorParameterL2Change"], config)
                          for split in rows}}
            print(json.dumps({"seed": seed, "gates": reports[str(seed)]}), flush=True)
    if any(file_digest(repository/p) != d for p, d in sources.items()) or any(
        audit_artifact_manifest(a, "manifest.json") != m for a, m in zip(archives, audits, strict=True)):
        raise ValueError("source or archive changed during run")
    report = {"format": "snowgym.short-recovery-report.v0", "coverage": coverage, "runs": reports,
        "replicated": all(g["passed"] for r in reports.values() for g in r["gates"].values()), **ASSIST_FIELDS}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.short-recovery-manifest.v0", **ASSIST_FIELDS,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)), flush=True)
