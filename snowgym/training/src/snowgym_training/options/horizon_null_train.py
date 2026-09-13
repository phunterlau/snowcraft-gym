"""R1m-S11: null-control (permuted-advantage) and replication of S9-full.

Reviewer handoff Experiment E1. Reuses `horizon_train` (S9) machinery without
editing it: editing it would invalidate S9/S10's archived `implementationDigest`
checks. Full arm only; see `reviews/m7b_r1m_s11_declaration.md`.
"""

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
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import horizon_train as h
from .control_channels import recommend_movement
from .opportunity_audit import plain, write_jsonl
from .recovery_audit import geometry as opportunity_geometry
from .supervised_probe import write_json

PERMUTATION_TAG = 0x45314E31  # 'E1N1'; a stream distinct from row-selection and episode-sampling seeds.
STOCHASTIC_TAG = 993101
S9_ARCHIVE = h.b.TRAINING / "runs/m7b_engage_r1m_s9_v0"
S10_ARCHIVE = h.b.TRAINING / "runs/m7b_engage_r1m_s10_v0"
GROUPS = ("geometry.encoders", "geometry.move.0", "geometry.move.2", "option_move", "budget_move")


def configuration():
    base = h.configuration()
    return {**base, "format": "snowgym.horizon-null-config.v0", "arm": "full",
        "realTrainingRngs": [99302, 99303], "nullTrainingRngs": [99301, 99302, 99303],
        "reusedRealTrainingRng": 99301, "permutationTag": hex(PERMUTATION_TAG),
        "stochasticDraws": 2, "stochasticSeedBase": STOCHASTIC_TAG, "gapThreshold": 4.,
        "simulatorBudget": 450000}


def inputs():
    source, metadata, data, lineage = h.inputs()
    s9 = h.b.verified_manifest(S9_ARCHIVE)
    s10 = h.b.verified_manifest(S10_ARCHIVE)
    s9_declaration = json.loads((S9_ARCHIVE / "declaration.json").read_text())
    if s9_declaration["implementationDigest"] != h.b.file_digest(Path(h.__file__)):
        raise ValueError("S9 implementation drift")
    return source, metadata, data, {**lineage, "s9": s9, "s10": s10}


class Capture:
    """Records per-unit teacher-direction opportunities during update 1's live
    collection. `.act()` delegates through unchanged; training is not altered."""

    def __init__(self, model, wrapper, sigma):
        self.model, self.wrapper, self.sigma = model, wrapper, sigma
        self.rows = []

    def act(self, observation, *, deterministic=False):
        with torch.no_grad():
            pred = self.model(observation)
        action, latent, logp, value = self.model.act(observation, deterministic=deterministic)
        raw = self.wrapper.environment.raw_observations[0]
        rec = recommend_movement(raw, self.wrapper.environment.max_team_units)
        scale = [raw["arena"]["width"] / 2, raw["arena"]["height"] / 2]
        units = []
        for i in range(pred["mean"].shape[1]):
            available = bool(pred["move_mask"][0, i]) and bool(rec["valid"][0, i])
            entry = {"unitIndex": i, "recommendationAvailable": available}
            if available:
                g = opportunity_geometry(pred["mean"][0, i].numpy(), latent[0, i].numpy(),
                    rec["target"][0, i], scale, self.sigma)
                entry.update({"behaviorMean": pred["mean"][0, i].tolist(), "target": rec["target"][0, i].tolist(),
                    "recommendationWorldGap": g["recommendationWorldGap"], "worldDirection": g["worldDirection"]})
            units.append(entry)
        self.rows.append(units)
        return action, latent, logp, value


def permute_advantage(selected, seed, update):
    rng = np.random.default_rng([seed, update, PERMUTATION_TAG])
    order = torch.from_numpy(rng.permutation(len(selected["advantage"])))
    result = dict(selected)
    result["advantage"] = selected["advantage"][order]
    return result


def finish_capture(capture, final_model):
    if capture["observation"] is None:
        return []
    with torch.no_grad():
        final_pred = final_model(capture["observation"])
    scale = np.array([50., 40.])
    rows = []
    for k, units in enumerate(capture["units"]):
        for u in units:
            if not u["recommendationAvailable"]:
                continue
            new_mean = final_pred["mean"][k, u["unitIndex"]].numpy()
            old_mean, target = np.asarray(u["behaviorMean"]), np.asarray(u["target"])
            new_base, old_base = np.tanh(new_mean) * scale, np.tanh(old_mean) * scale
            rows.append({"unitIndex": u["unitIndex"], "recommendationWorldGap": u["recommendationWorldGap"],
                "worldDirection": u["worldDirection"], "finalWorldShift": float(np.linalg.norm(new_base - old_base)),
                "finalGapReduction": u["recommendationWorldGap"] - float(np.linalg.norm(target * scale - new_base))})
    return rows


def constant_vector_fit(rows, *, gap_threshold=4.):
    selected = [r for r in rows if r["recommendationWorldGap"] > gap_threshold]
    if len(selected) < 3:
        return {"count": len(selected), "vector": None, "magnitude": None, "r2": None,
                "shiftMean": None, "shiftStd": None}
    D = np.array([r["worldDirection"] for r in selected]); y = np.array([r["finalGapReduction"] for r in selected])
    s = np.array([r["finalWorldShift"] for r in selected])
    vector = np.linalg.solve(np.einsum("ni,nj->ij", D, D), np.einsum("ni,n->i", D, y))
    pred = np.einsum("ni,i->n", D, vector)
    total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / total if total > 1e-12 else None
    return {"count": len(selected), "vector": vector.tolist(), "magnitude": float(np.hypot(*vector)),
        "r2": float(r2) if r2 is not None else None, "shiftMean": float(s.mean()), "shiftStd": float(s.std())}


def parameter_breakdown(initial, final_state, steps, learning_rate):
    parts = {g: [0., 0] for g in GROUPS}
    for n, p in initial.items():
        g = next(g for g in GROUPS if n.startswith(g))
        parts[g][0] += float((final_state[n] - p).square().sum()); parts[g][1] += p.numel()
    total_sq = sum(v[0] for v in parts.values()); count = sum(v[1] for v in parts.values())
    total = total_sq ** .5
    return {"l2": total, "rmsPerParameter": (total_sq / count) ** .5 if count else None,
        "optimizerSteps": steps, "lrSqrtSteps": learning_rate * steps ** .5, "lrSteps": learning_rate * steps,
        "groups": {g: {"params": n, "l2": sq ** .5, "share": sq / total_sq if total_sq > 1e-18 else None,
            "rmsPerParam": (sq / n) ** .5 if n else None} for g, (sq, n) in parts.items()}}


def training_success_gain(history, deterministic_by_seed):
    def stats(block):
        stochastic = sum(x["successes"] for x in block)
        episodes = sum(len(x["seeds"]) for x in block)
        deterministic = sum(deterministic_by_seed[s] for x in block for s in x["seeds"])
        return {"stochasticSuccesses": stochastic, "episodes": episodes, "deterministicSuccesses": deterministic}
    early = stats([x for x in history if x["update"] <= 10])
    late = stats([x for x in history if x["update"] > 20])
    return {"early": early, "late": late,
        "stochasticGain": late["stochasticSuccesses"] - early["stochasticSuccesses"],
        "adjustedGain": (late["stochasticSuccesses"] - late["deterministicSuccesses"]) -
                        (early["stochasticSuccesses"] - early["deterministicSuccesses"])}


def stochastic_evaluate(model, source, client, frames, cfg, account, *, draws=2, base=STOCHASTIC_TAG):
    wrapper = h.b.make_wrapper(client, 1, cfg["gamma"])
    results = []
    for frame in frames:
        for draw in range(draws):
            seed = base * 1000003 + frame["seed"] * 10 + draw
            _, row, _ = h.episode(model, source, wrapper, frame, cfg, "full", sampling_seed=seed, deterministic=False)
            account(row["simulatorDecisions"])
            results.append({"seed": frame["seed"], "draw": draw, "success": bool(row["success"])})
    return results


def train(source, metadata, client, frames, cfg, root, seed, *, null):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    arm = "full"
    h.r.validate_frames(frames, training=True)
    full_cfg = {**cfg, "arm": arm, "null": null}
    digest = json_digest(frames)
    torch.manual_seed(seed)
    model = RecoveryPolicy(copy.deepcopy(source), standard_deviation=cfg["latentStd"])
    initial = {n: p.detach().clone() for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith("critic.")}
    initial_digest = semantic_state_digest(model.state_dict())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learningRate"])
    sampler = np.random.default_rng(seed)
    root.mkdir(parents=True)
    wrapper = h.b.make_wrapper(client, 1, cfg["gamma"])
    history, capture_observation, capture_units, saved = [], None, None, None
    for update in range(cfg["updates"]):
        chosen = sampler.integers(len(frames), size=cfg["batchSize"]).tolist()
        batches, rows, trajectories, episode_units = [], [], [], []
        for slot, index in enumerate(chosen):
            active = Capture(model, wrapper, cfg["latentStd"]) if update == 0 else model
            batch, row, trajectory = h.episode(active, source, wrapper, frames[index], cfg, arm,
                sampling_seed=seed * 100000 + update * cfg["batchSize"] + slot)
            batches.append(batch); rows.append(row); trajectories.append(trajectory)
            if update == 0:
                episode_units.extend(active.rows)
        rollout = h.r.combine(batches)
        if update == 0:
            if len(episode_units) != len(rollout["advantage"]):
                raise ValueError("capture row count mismatch")
            capture_observation = {k: v.clone() for k, v in rollout["observation"].items()}
            capture_units = episode_units
        checks = h.diagnostics(model, rollout)
        indices = h.select_rows(len(rollout["advantage"]), cfg["rowsPerUpdate"], seed * 100000 + update)
        selected = h.subset(rollout, indices)
        if null:
            selected = permute_advantage(selected, seed, update)
        with torch.no_grad():
            old_mean = model(rollout["observation"])["mean"].detach().clone()
        step = h.r.ppo_update(model, optimizer, selected, cfg)
        with torch.no_grad():
            _, prediction = model.evaluate_latents(rollout["observation"], rollout["action_type"], rollout["latent"])
            scale = torch.tensor([50., 40.])
            shifts = (torch.tanh(prediction["mean"]) - torch.tanh(old_mean)) * scale
            selected_shift = shifts.norm(dim=-1)[prediction["move_mask"]]
        if semantic_state_digest(source.state_dict()) != semantic_state_digest(model.geometry.source.state_dict()):
            raise ValueError("null-horizon inherited source changed")
        step.update({"update": update + 1, "seeds": [frames[i]["seed"] for i in chosen], **checks, "null": null,
            "selectedRows": indices.tolist(), "presentedRows": len(indices),
            "uniqueSelectedRows": len(set(indices.tolist())), "collectedRows": len(rollout["advantage"]),
            "successes": sum(x["success"] for x in rows), "simulatorDecisions": sum(x["simulatorDecisions"] for x in rows),
            "learnedDecisions": sum(x["learnedDecisions"] for x in rows), "tailDecisions": sum(x["tailDecisions"] for x in rows),
            "meanWorldTargetUpdate": float(selected_shift.mean()) if selected_shift.numel() else None})
        history.append(step)
        write_jsonl(root / f"events-{update + 1:03d}.jsonl.gz", trajectories)
        print(json.dumps({"seed": seed, "null": null, "update": update + 1, "successes": step["successes"],
            "optimizerSteps": step["optimizerSteps"]}), flush=True)
        if update + 1 == cfg["updates"]:
            saved = h.r.recovery_checkpoint.save(root / "final", model, optimizer, source=metadata, config=full_cfg,
                seed=seed, dataset_digest=digest, update=update + 1, sampler=sampler.bit_generator.state, history=history)
    if semantic_state_digest(source.state_dict()) != semantic_state_digest(model.geometry.source.state_dict()):
        raise ValueError("null-horizon inherited source changed")
    change = sum(float((p.detach() - initial[n]).square().sum()) for n, p in model.named_parameters() if n in initial) ** .5
    report = {"arm": arm, "null": null, "history": history, "actorParameterL2Change": change,
        "initialStateDigest": initial_digest, "checkpointDigest": saved["checkpointDigest"],
        "datasetDigest": digest, "sourceUnchanged": True, **h.r.ASSIST_FIELDS}
    write_json(root / "training.json", report)
    return model, report, {"observation": capture_observation, "units": capture_units}, initial


def evaluate_point(model, source, client, data, cfg, account, root, *, parity=False):
    """Every predeclared per-checkpoint measurement except (2) and (3), which need training-time state."""
    result = {}
    for split in ("historical", "replication"):
        rows = h.evaluate(model, source, client, data["datasets"][split], cfg, "full",
            root / "evaluation" / split, parity=parity)
        account(sum(x["simulatorDecisions"] for x in rows))
        result[split] = rows
    training_rows = h.evaluate(model, source, client, data["datasets"]["training"], cfg, "full",
        root / "training-evaluation", parity=parity)
    account(sum(x["simulatorDecisions"] for x in training_rows))
    result["training"] = training_rows
    result["stochasticHistorical"] = stochastic_evaluate(model, source, client, data["datasets"]["historical"], cfg, account)
    return result


def analyze_point(name, evaluation, baseline, history, deterministic_by_seed, opportunities, initial, final_state,
                   steps, cfg):
    point = {"name": name,
        "developmentGates": {split: h.paired(evaluation[split], baseline[split]) for split in ("historical", "replication")},
        "trainingDeterministicSuccesses": sum(x["success"] for x in evaluation["training"]),
        "stochasticHistoricalSuccesses": sum(x["success"] for x in evaluation["stochasticHistorical"]),
        "stochasticHistoricalDraws": len(evaluation["stochasticHistorical"])}
    if history is not None:
        point["trainingSuccessGain"] = training_success_gain(history, deterministic_by_seed)
    if opportunities is not None:
        point["update1ConstantVectorFit"] = constant_vector_fit(opportunities, gap_threshold=cfg["gapThreshold"])
    if initial is not None:
        point["parameterBreakdown"] = parameter_breakdown(initial, final_state, steps, cfg["learningRate"])
    return point


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    cfg = configuration()
    source, metadata, data, lineage = inputs()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = semantic_state_digest(source.state_dict())
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "source": metadata, "datasetDigest": data["digest"],
        "sourceManifest": h.b.file_digest(h.b.ARCHIVE / "manifest.json"), "s8Manifest": h.b.file_digest(h.ARCHIVE / "manifest.json"),
        "s9Manifest": lineage["s9"]["manifestDigest"], "s10Manifest": lineage["s10"]["manifestDigest"],
        "implementationDigest": h.b.file_digest(Path(__file__)),
        "declarationDigest": h.b.file_digest(h.b.TRAINING / "reviews/m7b_r1m_s11_declaration.md"),
        "gitCommit": resolve_git_commit(), **h.r.ASSIST_FIELDS})
    steps = 0
    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["simulatorBudget"]:
            raise ValueError("null-horizon simulator budget exceeded")
    deterministic_by_seed = {f["seed"]: f["baseline"]["success"]
        for f in json.loads((h.b.ARCHIVE / "snapshots.json").read_text())["datasets"]["training"]}
    points = {}
    with SnowGymBatchClient() as client:
        h.b.require_capabilities(client)
        torch.manual_seed(cfg["trainingRngs"][0])
        initializer = RecoveryPolicy(copy.deepcopy(source), standard_deviation=cfg["latentStd"])
        baseline = {}
        for split in ("historical", "replication"):
            rows = h.evaluate(initializer, source, client, data["datasets"][split], cfg, "full",
                root / "initialization" / split, parity=True)
            account(sum(x["simulatorDecisions"] for x in rows)); baseline[split] = rows
        baseline_training = h.evaluate(initializer, source, client, data["datasets"]["training"], cfg, "full",
            root / "initialization" / "training", parity=True)
        account(sum(x["simulatorDecisions"] for x in baseline_training))
        baseline_stochastic = stochastic_evaluate(initializer, source, client, data["datasets"]["historical"], cfg, account)
        write_json(root / "initialization.json", {**baseline, "training": baseline_training,
            "stochasticHistorical": baseline_stochastic})
        points["initializer"] = {"name": "initializer",
            "trainingDeterministicSuccesses": sum(x["success"] for x in baseline_training),
            "stochasticHistoricalSuccesses": sum(x["success"] for x in baseline_stochastic),
            "stochasticHistoricalDraws": len(baseline_stochastic)}
        for seed in (99302, 99303):
            directory = root / f"{seed}-real"
            model, report, capture, initial = train(source, metadata, client, data["datasets"]["training"], cfg,
                directory / "train", seed, null=False)
            account(sum(x["simulatorDecisions"] for x in report["history"]))
            evaluation = evaluate_point(model, source, client, data, cfg, account, root=directory)
            write_json(directory / "evaluation.json", {k: v for k, v in evaluation.items()})
            opportunities = finish_capture(capture, model)
            write_jsonl(directory / "opportunities-update-001.jsonl.gz", opportunities)
            points[f"{seed}-real"] = analyze_point(f"{seed}-real", evaluation, baseline, report["history"],
                deterministic_by_seed, opportunities, initial, model.state_dict(),
                sum(x["optimizerSteps"] for x in report["history"]), cfg)
            print(json.dumps({"point": f"{seed}-real", "simulatorDecisions": steps}), flush=True)
        for seed in (99301, 99302, 99303):
            directory = root / f"{seed}-null"
            model, report, capture, initial = train(source, metadata, client, data["datasets"]["training"], cfg,
                directory / "train", seed, null=True)
            account(sum(x["simulatorDecisions"] for x in report["history"]))
            evaluation = evaluate_point(model, source, client, data, cfg, account, root=directory)
            write_json(directory / "evaluation.json", {k: v for k, v in evaluation.items()})
            opportunities = finish_capture(capture, model)
            write_jsonl(directory / "opportunities-update-001.jsonl.gz", opportunities)
            points[f"{seed}-null"] = analyze_point(f"{seed}-null", evaluation, baseline, report["history"],
                deterministic_by_seed, opportunities, initial, model.state_dict(),
                sum(x["optimizerSteps"] for x in report["history"]), cfg)
            print(json.dumps({"point": f"{seed}-null", "simulatorDecisions": steps}), flush=True)
        # Archived S9-full/99301 real point: reused checkpoint, freshly evaluated on measurements 5-6 only.
        archived_meta = json.loads((S9_ARCHIVE / "99301/full/training.json").read_text())
        torch.manual_seed(99301)
        rebuilt_initial = RecoveryPolicy(copy.deepcopy(source), standard_deviation=cfg["latentStd"])
        if semantic_state_digest(rebuilt_initial.state_dict()) != archived_meta["initialStateDigest"]:
            raise ValueError("archived S9-full/99301 initializer digest mismatch")
        archived_model, _, archived_checkpoint = h.r.recovery_checkpoint.load(S9_ARCHIVE / "99301/full/final")
        if archived_checkpoint["checkpointDigest"] != archived_meta["checkpointDigest"]:
            raise ValueError("archived S9-full/99301 checkpoint digest mismatch")
        archived_model.eval().requires_grad_(False)
        archived_evaluation_dev = json.loads((S9_ARCHIVE / "99301/full/evaluation.json").read_text())
        archived_evaluation = dict(archived_evaluation_dev)
        archived_training = h.evaluate(archived_model, source, client, data["datasets"]["training"], cfg, "full",
            root / "99301-real-archived" / "training-evaluation")
        account(sum(x["simulatorDecisions"] for x in archived_training))
        archived_evaluation["training"] = archived_training
        archived_stochastic = stochastic_evaluate(archived_model, source, client, data["datasets"]["historical"], cfg, account)
        archived_evaluation["stochasticHistorical"] = archived_stochastic
        write_json(root / "99301-real-archived" / "evaluation.json", archived_evaluation)
        archived_opportunities = [json.loads(line) for line in gzip.open(S10_ARCHIVE / "opportunities-full-001.jsonl.gz", "rt")]
        # S10's opportunity rows are per-unit-per-decision with worldDirection/finalGapReduction/finalWorldShift already computed.
        points["99301-real-archived"] = analyze_point("99301-real-archived", archived_evaluation, baseline,
            archived_meta["history"], deterministic_by_seed, [r for r in archived_opportunities if r["recommendationAvailable"]],
            None, None, sum(x["optimizerSteps"] for x in archived_meta["history"]), cfg)
        points["99301-real-archived"]["parameterBreakdown"] = {
            "l2": archived_meta["actorParameterL2Change"], "optimizerSteps": sum(x["optimizerSteps"] for x in archived_meta["history"]),
            "lrSqrtSteps": cfg["learningRate"] * sum(x["optimizerSteps"] for x in archived_meta["history"]) ** .5}
    if inputs()[-1] != lineage or semantic_state_digest(source.state_dict()) != before:
        raise ValueError("null-horizon source or ancestry changed")
    report = {"format": "snowgym.horizon-null-report.v0", **h.r.ASSIST_FIELDS, "points": points,
        "simulatorDecisions": steps, "realNewRuns": ["99302-real", "99303-real"],
        "nullRuns": ["99301-null", "99302-null", "99303-null"], "reusedArchivedPoint": "99301-real-archived"}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.horizon-null-manifest.v0", **h.r.ASSIST_FIELDS,
        "artifacts": {str(p.relative_to(root)): h.b.file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    h.b.verified_manifest(root)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
