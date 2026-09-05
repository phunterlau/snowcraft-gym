"""R1m-S4: reconstruct frozen exploration and output-space score directions."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.recovery_ppo import RecoveryPolicy
from ..executor.movement_ppo import movement_loss
from ..ppo import generalized_advantage_estimate
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import recovery_train, recovery_checkpoint, post_hit
from .control_channels import recommend_movement, validate_movement_agreement
from .identity import checkpoint_model
from .interventions import require_capabilities
from .movement_collect import ASSIST_FIELDS
from .movement_train import TRAINING, REFERENCE, make_wrapper
from .opportunity_audit import plain, write_jsonl
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import write_json

ARCHIVE = TRAINING / "runs/m7b_engage_r1m_s3_v0"
UPDATES = (1, 11, 21)


def statistics(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, .9))}


def correlation(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def geometry(mean, latent, target, scale, sigma):
    """Finite geometry for one selected movement head; no simulator mutation."""
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("invalid latent standard deviation")
    mean, latent, target, scale = map(lambda x: np.asarray(x, dtype=float), (mean, latent, target, scale))
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (mean, latent, target, scale)):
        raise ValueError("geometry requires finite 2D vectors")
    base, sampled = np.tanh(mean)*scale, np.tanh(latent)*scale
    delta = target*scale-base
    gap = float(np.linalg.norm(delta))
    direction = delta/gap if gap > 1e-12 else np.zeros(2)
    mahal = float(np.linalg.norm((np.arctanh(np.clip(target, -1+1e-6, 1-1e-6))-mean)/sigma))
    shift = sampled-base
    return {"sampleWorldDisplacement": float(np.linalg.norm(shift)), "recommendationWorldGap": gap,
        "sampleTowardRecommendation": float(shift @ direction), "mahalanobis": mahal,
        "recommendationMeanShiftKl": .5*mahal**2, "withinThreeSigma": mahal <= 3,
        "saturatedCoordinates": int((np.abs(np.tanh(latent)) >= .99).sum()),
        "sampleImprovesRecommendationDistance": float(np.linalg.norm(target*scale-sampled)) < gap,
        "worldDirection": direction.tolist(), "worldJacobian": (scale*(1-np.tanh(mean)**2)).tolist()}


def score_directions(means, latents, advantages, living, moves, order, minibatch, sigma):
    """Gradient-ascent direction in independent mean outputs, at ratio one."""
    result = torch.zeros_like(means)
    normalized = torch.zeros_like(advantages)
    for indices in order.split(minibatch):
        a = advantages[indices]
        a = (a-a.mean())/(a.std(unbiased=False)+1e-8)
        normalized[indices] = a
        counts = living[indices].sum(-1).clamp_min(1)
        result[indices] = (a[:, None, None]*(latents[indices]-means[indices])/sigma**2
                          *moves[indices, :, None]/counts[:, None, None]/len(indices))
    return result, normalized


def behavior(source, root, seed, update, config):
    if update == 1:
        torch.manual_seed(seed)
        model = RecoveryPolicy(copy.deepcopy(source), standard_deviation=config["latentStd"])
        sampler = np.random.default_rng(seed)
    else:
        model, _, metadata = recovery_checkpoint.load(root / str(seed) / f"update-{update-1:03d}")
        sampler = np.random.default_rng()
        sampler.bit_generator.state = metadata["sampler"]
    return model.eval().requires_grad_(False), sampler


def replay_update(model, wrapper, frames, rows, config):
    all_means, all_latents, all_living, all_moves, all_advantages = [], [], [], [], []
    all_inputs, all_logp, all_values, all_returns = [], [], [], []
    opportunities, trajectory_rows = [], []
    observation_index = 0
    max_error = 0.
    for episode_index, row in enumerate(rows):
        seed = row["summary"]["seed"]
        frame = frames[seed]
        if row["frameDigest"] != frame["frameDigest"]:
            raise ValueError("archived trajectory snapshot mismatch")
        trigger = frame["trigger"]
        obs = post_hit.restore(wrapper, seed, trigger["prefix"], trigger["identity"])
        count = row["summary"]["learnedDecisions"]
        values = []
        for offset in range(count):
            inputs = recovery_train.recovery_observation(obs, offset, config["recoveryWindow"])
            raw = wrapper.environment.raw_observations[0]
            recommendation = recommend_movement(raw, wrapper.environment.max_team_units)
            teacher = wrapper.environment.plan_teacher_tensor_actions()
            validate_movement_agreement(teacher, recommendation)
            with torch.no_grad():
                pred = model(inputs)
                action, latent, logp, value = model.act(inputs)
            expected_latent = torch.tensor(row["learnerLatents"][offset], dtype=latent.dtype)[None]
            expected_logp = torch.tensor(row["behaviorLogProbabilities"][offset], dtype=logp.dtype)[None]
            if not torch.equal(latent, expected_latent):
                raise ValueError("sampled latent reconstruction mismatch")
            error = float((logp-expected_logp).abs().max())
            max_error = max(max_error, error)
            if error > 1e-6:
                raise ValueError("behavior log probability reconstruction mismatch")
            means = pred["mean"].detach()
            all_means.append(means); all_latents.append(latent)
            all_inputs.append(inputs); all_logp.append(logp)
            all_living.append(pred["living"]); all_moves.append(pred["move_mask"])
            values.append(value)
            scale = [raw["arena"]["width"]/2, raw["arena"]["height"]/2]
            for i, unit in enumerate(raw["allies"]):
                if not pred["move_mask"][0, i] or not recommendation["valid"][0, i]:
                    continue
                opportunities.append({"seed": seed, "episodeIndex": episode_index, "offset": offset,
                    "observationIndex": observation_index, "unitIndex": i, "unitId": unit["id"],
                    "ready": bool(recommendation["ready"][0, i]), "recommendationAvailable": True,
                    **geometry(means[0, i].numpy(), latent[0, i].numpy(), recommendation["target"][0, i],
                               scale, config["latentStd"])})
            recorded = row["events"][offset]
            executed = {k: np.asarray(v, dtype=np.int64 if k == "action_type" else np.float32)
                        for k, v in recorded["action"].items()}
            generated = recovery_train.corrected_shots(recovery_train.numpy_actions(action), wrapper.environment.raw_observations)
            if plain(generated) != recorded["action"]:
                raise ValueError("reconstructed action mismatch")
            obs, reward, terminated, truncated, _ = wrapper.step(executed)
            if wrapper.environment.state_hashes[0] != recorded["stateHash"] or float(reward[0]) != row["rewards"][offset]:
                raise ValueError("reconstructed transition mismatch")
            if (terminated[0] or truncated[0]) and offset != count-1:
                raise ValueError("premature reconstructed terminal")
            observation_index += 1
        folded, _ = recovery_train.fold_tail(row["rewards"], count, config["gamma"])
        if plain(folded) != row["foldedRewards"]:
            raise ValueError("folded reward mismatch")
        value = torch.cat(values)
        terminal = torch.zeros(count, 1, dtype=torch.bool); terminal[-1] = True
        advantage, returns = generalized_advantage_estimate(folded[:, None], value[:, None],
            torch.cat([value[1:], torch.zeros(1)])[:, None], terminal, torch.zeros_like(terminal),
            gamma=config["gamma"], gae_lambda=config["gaeLambda"])
        all_advantages.append(advantage.flatten())
        all_values.append(value); all_returns.append(returns.flatten())
        trajectory_rows.append({"seed": seed, "episodeIndex": episode_index, "values": plain(value),
            "advantages": plain(advantage.flatten()), "returns": plain(returns.flatten()),
            "learnedDecisions": count})
    means, latents, living, moves = map(torch.cat, (all_means, all_latents, all_living, all_moves))
    advantages = torch.cat(all_advantages)
    order = torch.randperm(len(advantages))  # Exact restored collection RNG; no optimization.
    first_indices = order[:config["minibatchSize"]]
    first_inputs = {k: torch.cat([o[k] for o in all_inputs])[first_indices] for k in all_inputs[0]}
    with torch.no_grad():
        logp, pred = model.evaluate_latents(first_inputs, model(first_inputs)["action_type"], latents[first_indices])
        first_loss = movement_loss(logp, torch.cat(all_logp)[first_indices], advantages[first_indices], pred["value"],
            torch.cat(all_returns)[first_indices], pred, clip_ratio=config["clipRatio"])
    scores, normalized = score_directions(means, latents, advantages, living, moves, order,
                                          config["minibatchSize"], config["latentStd"])
    first = set(order[:config["minibatchSize"]].tolist())
    for o in opportunities:
        index, unit = o["observationIndex"], o["unitIndex"]
        o.update({"advantage": float(advantages[index]), "normalizedAdvantage": float(normalized[index]),
            "firstMinibatch": index in first, "worldScoreProjection": float(
                np.asarray(o["worldDirection"]) @ (np.asarray(o["worldJacobian"])*scores[index, unit].numpy()))})
    return {"opportunities": opportunities, "trajectories": trajectory_rows,
            "firstEpochOrder": order.tolist(), "maxLogProbabilityError": max_error,
            "firstMinibatchLoss": {k: float(v) for k, v in first_loss.items()}}


def summarize(rows):
    projection = [r["worldScoreProjection"] for r in rows]
    decisions = {}
    for r in rows:
        decisions.setdefault(r["observationIndex"], []).append(r["worldScoreProjection"])
    return {"opportunities": len(rows), "uniqueSeeds": len({r["seed"] for r in rows}),
        "decisionsWithMoves": len(decisions), "trajectoriesWithMoves": len({r["episodeIndex"] for r in rows}),
        "decisionSummedScore": statistics([sum(v) for v in decisions.values()]),
        **{k: statistics([r[k] for r in rows]) for k in ("sampleWorldDisplacement", "recommendationWorldGap",
            "mahalanobis", "recommendationMeanShiftKl", "sampleTowardRecommendation", "worldScoreProjection")},
        "withinThreeSigmaFraction": float(np.mean([r["withinThreeSigma"] for r in rows])) if rows else None,
        "sampleCloserFraction": float(np.mean([r["sampleImprovesRecommendationDistance"] for r in rows])) if rows else None,
        "positiveScoreFraction": float(np.mean(np.asarray(projection)>0)) if rows else None,
        "saturatedCoordinateFraction": sum(r["saturatedCoordinates"] for r in rows)/max(1, 2*len(rows)),
        "sampleAdvantageCorrelation": correlation([r["sampleTowardRecommendation"] for r in rows],
                                                 [r["normalizedAdvantage"] for r in rows])}


def common_states(source, root, frames, config, client):
    wrapper = make_wrapper(client, 1, config["gamma"])
    cached = []
    for frame in frames:
        t = frame["trigger"]
        obs = post_hit.restore(wrapper, frame["seed"], t["prefix"], t["identity"])
        raw = wrapper.environment.raw_observations[0]
        rec = recommend_movement(raw, wrapper.environment.max_team_units)
        validate_movement_agreement(wrapper.environment.plan_teacher_tensor_actions(), rec)
        cached.append((frame["seed"], recovery_train.recovery_observation(obs, 0, 30), rec,
                       np.asarray([raw["arena"]["width"]/2, raw["arena"]["height"]/2])))
    output = []
    for seed in config["trainingRngs"]:
        initial, _ = behavior(source, root, seed, 1, config)
        with torch.no_grad():
            base = [initial(obs)["mean"].numpy() for _, obs, _, _ in cached]
        for checkpoint in ("initial", "update-010", "update-020", "final"):
            model = initial if checkpoint == "initial" else recovery_checkpoint.load(root/str(seed)/checkpoint)[0]
            before = semantic_state_digest(model.state_dict())
            per_seed = []
            for (world_seed, obs, rec, scale), old in zip(cached, base, strict=True):
                with torch.no_grad():
                    pred = model(obs)
                valid = pred["move_mask"].numpy() & rec["valid"]
                new = pred["mean"].detach().numpy()
                old_world, new_world, teacher = np.tanh(old)*scale, np.tanh(new)*scale, rec["target"]*scale
                old_gap = np.linalg.norm(teacher-old_world, axis=-1)[valid]
                new_gap = np.linalg.norm(teacher-new_world, axis=-1)[valid]
                displacement = np.linalg.norm(new_world-old_world, axis=-1)[valid]
                per_seed.append({"seed": world_seed, "moves": int(valid.sum()),
                    "worldShift": float(displacement.mean()) if len(displacement) else None,
                    "gapReduction": float((old_gap-new_gap).mean()) if len(old_gap) else None,
                    "closerFraction": float((new_gap<old_gap).mean()) if len(old_gap) else None,
                    "recommendationGap": float(new_gap.mean()) if len(new_gap) else None})
            if semantic_state_digest(model.state_dict()) != before:
                raise ValueError("common-state inference changed weights")
            output.append({"trainingRng": seed, "checkpoint": checkpoint, "snapshots": per_seed,
                "summary": {k: statistics([r[k] for r in per_seed if r[k] is not None]) for k in
                            ("worldShift", "gapReduction", "closerFraction", "recommendationGap")}})
    return output


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    manifest = audit_artifact_manifest(ARCHIVE, "manifest.json")
    declaration = json.loads((ARCHIVE/"declaration.json").read_text())
    config = declaration["config"]
    repository = TRAINING.parents[1]
    if any(file_digest(repository/p) != d for p, d in declaration["sourceFiles"].items()):
        raise ValueError("S3 implementation differs from archived source")
    dataset = json.loads((ARCHIVE/"snapshots.json").read_text())
    if dataset["digest"] != json_digest(dataset["datasets"]):
        raise ValueError("snapshot inventory digest mismatch")
    frames = dataset["datasets"]["training"]
    recovery_train.validate_frames(frames, training=True)
    if len(frames) != 57:
        raise ValueError("frozen snapshot count mismatch")
    metadata, state = load_ppo_checkpoint(REFERENCE)
    if metadata != declaration["source"]:
        raise ValueError("frozen source checkpoint mismatch")
    source = checkpoint_model(metadata); source.load_state_dict(state["model"])
    source.eval().requires_grad_(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root.mkdir(parents=True)
    write_json(root/"declaration.json", {"sourceManifest": file_digest(ARCHIVE/"manifest.json"),
        "declarationDigest": file_digest(TRAINING/"reviews/m7b_r1m_s4_declaration.md"),
        "auditSourceDigest": file_digest(Path(__file__)), "gitCommit": resolve_git_commit(),
        "updates": list(UPDATES), "trainingRngs": config["trainingRngs"], **ASSIST_FIELDS})
    reports = []
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        common = common_states(source, ARCHIVE, frames, config, client)
        write_json(root/"common-states.json", common)
        wrapper = make_wrapper(client, 1, config["gamma"])
        for seed in config["trainingRngs"]:
            for update in UPDATES:
                model, sampler = behavior(source, ARCHIVE, seed, update, config)
                before = semantic_state_digest(model.state_dict())
                with gzip.open(ARCHIVE/str(seed)/f"events-{update:03d}.jsonl.gz", "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                selected = [frames[i]["seed"] for i in sampler.integers(len(frames), size=8)]
                if selected != [r["summary"]["seed"] for r in rows]:
                    raise ValueError("snapshot sampler reconstruction mismatch")
                evidence = replay_update(model, wrapper, {f["seed"]: f for f in frames}, rows, config)
                original = json.loads((ARCHIVE/str(seed)/"training.json").read_text())["history"][update-1]["minibatches"][0]
                if any(not np.isclose(v, original[k], rtol=1e-5, atol=1e-6) for k, v in evidence["firstMinibatchLoss"].items()):
                    raise ValueError("archived first-minibatch loss mismatch")
                evidence["firstMinibatchLossMaxError"] = max(abs(v-original[k]) for k, v in evidence["firstMinibatchLoss"].items())
                if semantic_state_digest(model.state_dict()) != before:
                    raise ValueError("audit changed policy weights")
                write_jsonl(root/f"opportunities-{seed}-{update:03d}.jsonl.gz", evidence.pop("opportunities"))
                with gzip.open(root/f"opportunities-{seed}-{update:03d}.jsonl.gz", "rt") as stream:
                    opportunities = [json.loads(line) for line in stream]
                write_json(root/f"reconstruction-{seed}-{update:03d}.json", evidence)
                report = {"trainingRng": seed, "update": update, "trajectories": len(rows),
                    "decisions": sum(r["summary"]["learnedDecisions"] for r in rows),
                    "all": summarize(opportunities), "firstMinibatch": summarize([r for r in opportunities if r["firstMinibatch"]])}
                reports.append(report)
                print(json.dumps(report), flush=True)
    if audit_artifact_manifest(ARCHIVE, "manifest.json") != manifest:
        raise ValueError("audit changed archived evidence")
    report = {"format": "snowgym.recovery-audit.v0", "updates": reports,
        "common": [{k: v for k, v in r.items() if k != "snapshots"} for r in common], **ASSIST_FIELDS}
    write_json(root/"report.json", report)
    output_manifest = {"format": "snowgym.recovery-audit-manifest.v0", **ASSIST_FIELDS,
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    output_manifest["manifestDigest"] = json_digest(output_manifest)
    write_json(root/"manifest.json", output_manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
