"""R1l fixed-architecture state-support x conditional-label factorial."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient

from ..checkpoint import load_checkpoint, semantic_state_digest
from ..executor.geometry_probe import GeometryProbe
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .evaluate import evaluate_option_episode
from .geometry_probe import load_probe
from .identity import checkpoint_model, recover_initializer
from .interventions import require_capabilities
from .opportunity_audit import REFERENCE_DIGEST, collect, write_jsonl
from .opportunity_metrics import conditional_loss
from .recovery_lineage import TRAINING_ROOT, audit_ancestry
from .recovery_report import audit_artifact_manifest
from .reservoir import file_digest
from .supervised_probe import summarize_rows, write_json
from .train import DEFAULT_INITIALIZER

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1l_corrective_v0.json"


def load_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text())
    expected = {"format": "snowgym.corrective-data-config.v0", "checkpointDigest": REFERENCE_DIGEST,
        "arms": ["A", "B", "C", "D"], "trainingSeeds": [93001, 93002, 93003], "replicationCandidate": "D",
        "optimizerSteps": 420, "minibatchSize": 256, "learningRate": .0003, "maxGradNorm": .5,
        "learnerFraction": .5, "developmentSeeds": [200000, 200039],
        "replicationDevelopmentSeeds": [210000, 210039], "teacherRegressionSeeds": [108000, 108039],
        "learnerValidationSeeds": [108100, 108139], "selection": "final-step-only",
        "bootstrapSamples": 10000, "bootstrapSeed": 790001, "ppoUpdates": 0, "runtimeAssistance": False}
    if value != expected:
        raise ValueError("configuration differs from frozen R1l factorial")
    return value


class OpportunityDataset:
    def __init__(self, states):
        if not states:
            raise ValueError("empty corrective dataset")
        self.observation = {k: torch.tensor(np.stack([s["observation"][k][0] for s in states]), dtype=torch.float32)
                            for k in states[0]["observation"]}
        self.conditional = {k: torch.tensor(np.stack([s["labels"][k][0] for s in states]),
                                dtype=torch.bool if k.endswith("mask") else torch.float32)
                            for k in states[0]["labels"]}
        teacher_type = torch.tensor(np.stack([s["teacher"]["action_type"][0] for s in states]), dtype=torch.long)
        target = torch.tensor(np.stack([s["teacher"]["target"][0] for s in states]), dtype=torch.float32)
        self.old = {"move_target": target, "shot_target": target,
                    "power": torch.tensor(np.stack([s["teacher"]["power"][0] for s in states]), dtype=torch.float32),
                    "move_mask": teacher_type == 1, "shot_mask": teacher_type == 2}
        self.seeds = np.asarray([s["seed"] for s in states], dtype=np.int64)
        self.episodes = np.unique(self.seeds)
        self.by_episode = {int(seed): np.flatnonzero(self.seeds == seed) for seed in self.episodes}

    @classmethod
    def read(cls, path):
        with gzip.open(path, "rt") as stream:
            return cls([json.loads(line) for line in stream])

    def sample(self, count, rng):
        episodes = rng.choice(self.episodes, count)
        return np.asarray([rng.choice(self.by_episode[int(seed)]) for seed in episodes])

    def batch(self, indices, conditional):
        labels = self.conditional if conditional else self.old
        return ({k: v[indices] for k, v in self.observation.items()}, {k: v[indices] for k, v in labels.items()})


def sample_batch(teacher, learner, *, arm, count, rng):
    if arm not in {"A", "B", "C", "D"} or count % 2:
        raise ValueError("invalid factorial arm or odd minibatch")
    count_learner = count//2 if arm in {"B", "D"} else 0
    conditional = arm in {"C", "D"}
    sources = [("teacher", teacher, count-count_learner), ("learner", learner, count_learner)]
    observations, labels, selected = [], [], {}
    for name, dataset, size in sources:
        if not size:
            continue
        indices = dataset.sample(size, rng)
        observation, label = dataset.batch(indices, conditional)
        observations.append(observation)
        labels.append(label)
        selected[name] = indices
    return ({k: torch.cat([o[k] for o in observations]) for k in observations[0]},
            {k: torch.cat([o[k] for o in labels]) for k in labels[0]}, selected)


def evaluate_fit(model, dataset, *, conditional):
    # Normalize each head within each episode, then average episodes. This is
    # a fixed-distribution diagnostic, not an additional fitting objective.
    totals = []
    with torch.no_grad():
        for indices in dataset.by_episode.values():
            observation, labels = dataset.batch(indices, conditional)
            losses = conditional_loss(model(observation), observation, labels)
            totals.append({k: float(v) for k, v in losses.items()})
    return {k: float(np.mean([r[k] for r in totals])) for k in totals[0]}


def save_corrective(path, model, parent, *, arm, seed, steps, optimizer, config):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite corrective checkpoint {path}")
    path.mkdir()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path / "state.pt")
    value = {"format": "snowgym.corrective-checkpoint.v0", "source": parent["source"],
             "parentCheckpointDigest": parent["checkpointDigest"], "arm": arm, "trainingSeed": seed,
             "optimizerSteps": steps, "execution": "deterministic-only", "ppoCompatible": False,
             "config": config, "stateFileDigest": file_digest(path / "state.pt"),
             "modelStateDigest": semantic_state_digest(model.state_dict())}
    value["checkpointDigest"] = json_digest(value)
    write_json(path / "checkpoint.json", value)
    return value


def load_corrective(path):
    path = Path(path)
    value = json.loads((path / "checkpoint.json").read_text())
    if (value.get("format") != "snowgym.corrective-checkpoint.v0"
            or value.get("checkpointDigest") != json_digest({k: v for k, v in value.items() if k != "checkpointDigest"})
            or value["stateFileDigest"] != file_digest(path / "state.pt")):
        raise ValueError("corrective checkpoint provenance mismatch")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    model = GeometryProbe(checkpoint_model(value["source"]), relative=False)
    model.load_state_dict(state["model"])
    if semantic_state_digest(model.state_dict()) != value["modelStateDigest"]:
        raise ValueError("corrective checkpoint model digest mismatch")
    return model.eval(), value


def fit_arm(reference, teacher, learner, *, arm, seed, config):
    model = copy.deepcopy(reference)
    inherited_digest = semantic_state_digest(model.source.state_dict())
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learningRate"])
    rng = np.random.default_rng(seed)
    exposure = {"teacher": set(), "learner": set()}
    counts = {"teacherStates": 0, "learnerStates": 0, "moveLabels": 0, "shotLabels": 0}
    traces = []
    for step in range(config["optimizerSteps"]):
        observation, labels, selected = sample_batch(teacher, learner, arm=arm, count=config["minibatchSize"], rng=rng)
        optimizer.zero_grad(set_to_none=True)
        losses = conditional_loss(model(observation), observation, labels)
        if not all(torch.isfinite(v) for v in losses.values()):
            raise ValueError("non-finite corrective loss")
        losses["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["maxGradNorm"], error_if_nonfinite=True)
        optimizer.step()
        for name, indices in selected.items():
            exposure[name].update(int(i) for i in indices)
            counts[name+"States"] += len(indices)
        counts["moveLabels"] += int(labels["move_mask"].sum())
        counts["shotLabels"] += int(labels["shot_mask"].sum())
        traces.append({"step": step+1, **{k: float(v.detach()) for k, v in losses.items()},
                       "gradientNorm": float(norm)})
    if semantic_state_digest(model.source.state_dict()) != inherited_digest:
        raise RuntimeError("corrective fitting changed the frozen inherited actor/critic")
    return model, optimizer, {"trace": traces, "exposure": counts,
        "distinctStates": {k: len(v) for k, v in exposure.items()},
        "newParameterL2Change": sum(float((p.detach()-initial[n]).square().sum()) for n, p in model.named_parameters() if n in initial)**.5,
        "inheritedUnchanged": True, "optimizerSteps": config["optimizerSteps"]}


def paired_effect(left, right, config):
    if [r["seed"] for r in left] != [r["seed"] for r in right]:
        raise ValueError("unpaired evaluation episodes")
    rng = np.random.default_rng(config["bootstrapSeed"])
    draws = rng.integers(len(left), size=(config["bootstrapSamples"], len(left)))
    result = {}
    for key in ("success", "progress", "physicalWin"):
        difference = np.asarray([float(a[key])-float(b[key]) for a, b in zip(left, right, strict=True)])
        bootstrap = difference[draws].mean(1)
        result[key] = {"mean": float(difference.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    result["progressWinFraction"] = float(np.mean([a["progress"] > b["progress"] for a, b in zip(left, right, strict=True)]))
    return result


def replication_allowed(a, d):
    success_gain = sum(r["success"] for r in d) > sum(r["success"] for r in a)
    def rejected(rows):
        return sum(r["rejectedActions"] for r in rows)/max(sum(r["totalActions"] for r in rows), 1)
    return success_gain and rejected(d) <= rejected(a)


def bootstrap_gate(correct, wrong, initializer):
    success = float(np.mean([r["success"] for r in correct]))
    values = {"successMinimum": success >= .5,
              "wrongPlanImprovement": success-float(np.mean([r["success"] for r in wrong])) >= .2-1e-12,
              "initializerImprovement": success-float(np.mean([r["success"] for r in initializer])) >= .2-1e-12,
              "contactMinimum": float(np.mean([r["firstContactDecision"] is not None for r in correct])) >= .8,
              "rejections": sum(r["rejectedActions"] for r in correct)/max(sum(r["totalActions"] for r in correct), 1) < .001}
    return {"passed": all(values.values()), "criteria": values}


def evaluate_model(model, initializer, client, config):
    result = {}
    for split, key in (("historical", "developmentSeeds"), ("replication", "replicationDevelopmentSeeds")):
        start, end = config[key]
        result[split] = {condition: [evaluate_option_episode(model, initializer, option="engage", seed=seed,
                                     condition=condition, client=client) for seed in range(start, end+1)]
                         for condition in ("correct", "shuffled")}
        print(json.dumps({"phase": "evaluate", "split": split,
                          "successes": sum(r["success"] for r in result[split]["correct"])}), flush=True)
    return result


def run_factorial(*, checkpoint, reservoir_path, audit_path, output, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite corrective experiment {destination}")
    audit_root = Path(audit_path)
    audit_manifest = audit_artifact_manifest(audit_root, "manifest.json")
    if audit_manifest["manifestDigest"] != json_digest({k: v for k, v in audit_manifest.items() if k != "manifestDigest"}):
        raise ValueError("opportunity audit manifest digest mismatch")
    audit_report = json.loads((audit_root / "report.json").read_text())
    if not audit_report["r1lAllowed"] or audit_manifest["checkpoint"]["checkpointDigest"] != config["checkpointDigest"]:
        raise ValueError("R1l requires the passed, matching R1k audit")
    lineage = audit_ancestry(checkpoint=checkpoint, reservoir_path=reservoir_path,
                             checkpoint_root=TRAINING_ROOT / "runs", dataset_roots=[TRAINING_ROOT / "artifacts"])
    if not lineage["passed"]:
        raise ValueError("R1l ancestry unresolved; run recovery_lineage before any collection")
    reference, parent = load_probe(checkpoint)
    teacher = OpportunityDataset.read(audit_root / "teacher-states.jsonl.gz")
    learner = OpportunityDataset.read(audit_root / "learner-states.jsonl.gz")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference_digest = semantic_state_digest(reference.state_dict())
    old_metadata, old_state = load_ppo_checkpoint(TRAINING_ROOT / "runs/m7b_engage_r1f_supervised_probe_v0/epoch-020")
    source_metadata, source_state = load_checkpoint(DEFAULT_INITIALIZER)
    initializer, initializer_identity = recover_initializer(old_metadata, old_state, source_metadata, source_state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        write_json(root / "lineage.json", lineage)
        records, training = {}, {}
        with SnowGymBatchClient() as client:
            capabilities = require_capabilities(client)
            for key in ("simulationVersion", "stateHashVersion"):
                if capabilities[key] != audit_manifest["capabilities"][key]:
                    raise ValueError("R1l simulator differs from R1k")
            validation = {}
            for name, kind, seed_key in (("teacher", "teacher", "teacherRegressionSeeds"), ("learner", "learner", "learnerValidationSeeds")):
                states, opportunities, episodes = collect(reference, client, {"trainingSeeds": config[seed_key]}, kind=kind)
                write_jsonl(root / f"fresh-{name}-states.jsonl.gz", states)
                write_jsonl(root / f"fresh-{name}-opportunities.jsonl.gz", opportunities)
                write_jsonl(root / f"fresh-{name}-episodes.jsonl.gz", episodes)
                validation[name] = OpportunityDataset(states)
                del states, opportunities
            for name, baseline in (("reference", reference), ("initializer", initializer)):
                records[name] = evaluate_model(baseline, initializer, client, config)
                write_json(root / f"{name}-evaluation.json", records[name])
            for rng_index, seed in enumerate(config["trainingSeeds"]):
                if rng_index and not replication_allowed(records["A-93001"]["historical"]["correct"], records["D-93001"]["historical"]["correct"]):
                    break
                for arm in config["arms"] if rng_index == 0 else ("A", "D"):
                    name = f"{arm}-{seed}"
                    print(json.dumps({"phase": "fit", "arm": arm, "seed": seed}), flush=True)
                    model, optimizer, fit = fit_arm(reference, teacher, learner, arm=arm, seed=seed, config=config)
                    fit["freshRegression"] = {kind: {mode: evaluate_fit(model, data, conditional=mode == "conditional")
                                                  for mode in ("old", "conditional")} for kind, data in validation.items()}
                    saved = save_corrective(root / name, model, parent, arm=arm, seed=seed,
                                            steps=config["optimizerSteps"], optimizer=optimizer, config=config)
                    restored, _ = load_corrective(root / name)
                    if semantic_state_digest(restored.state_dict()) != semantic_state_digest(model.state_dict()):
                        raise RuntimeError("corrective checkpoint reload failed")
                    training[name] = {**fit, "checkpointDigest": saved["checkpointDigest"]}
                    records[name] = evaluate_model(restored, initializer, client, config)
                    write_json(root / f"{name}-evaluation.json", records[name])
                    write_json(root / f"{name}-fit.json", training[name])
        report = {"format": "snowgym.corrective-data-report.v0", "ppoUpdates": 0, "runtimeAssistance": False,
            "qualificationEligible": False, "replicated": "D-93003" in records,
            "sourceUnchanged": semantic_state_digest(reference.state_dict()) == reference_digest,
            "summary": {name: {split: {condition: summarize_rows(rows) for condition, rows in conditions.items()}
                               for split, conditions in record.items()} for name, record in records.items()},
            "contrasts": {split: {f"{left}-{right}": paired_effect(records[left+"-93001"][split]["correct"], records[right+"-93001"][split]["correct"], config)
                                   for left, right in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C"), ("D", "A"))}
                          for split in ("historical", "replication")},
            "baselineFit": {kind: {mode: evaluate_fit(reference, data, conditional=mode == "conditional")
                                    for mode in ("old", "conditional")} for kind, data in validation.items()},
            "predeclaredCandidate": "D", "trainingRngs": config["trainingSeeds"] if "D-93003" in records else [93001],
            "selectionPolicy": "final-step-only; D replication decided on historical success versus A",
            "initializerIdentity": initializer_identity}
        report["bootstrapGates"] = {
            name: {split: bootstrap_gate(record[split]["correct"], record[split]["shuffled"], records["initializer"][split]["correct"])
                   for split in ("historical", "replication")}
            for name, record in records.items() if name not in {"reference", "initializer"}}
        report["referenceComparisons"] = {
            name: {split: paired_effect(record[split]["correct"], records["reference"][split]["correct"], config)
                   for split in ("historical", "replication")}
            for name, record in records.items() if name not in {"reference", "initializer"}}
        for split in ("historical", "replication"):
            def differences(a, b):
                return [{"seed": x["seed"], **{k: float(x[k])-float(y[k]) for k in ("success", "progress", "physicalWin")}}
                        for x, y in zip(records[a+"-93001"][split]["correct"], records[b+"-93001"][split]["correct"], strict=True)]
            report["contrasts"][split]["interaction-D-C-B+A"] = paired_effect(differences("D", "C"), differences("B", "A"), config)
        if not report["sourceUnchanged"]:
            raise RuntimeError("R1l changed its reference")
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        manifest = {"format": "snowgym.corrective-data-run.v0", "gitCommit": resolve_git_commit(),
            "config": config, "configDigest": json_digest(config), "sourceCheckpoint": parent,
            "opportunityAuditDigest": audit_manifest["manifestDigest"],
            "sourceFiles": {str(p.relative_to(repository)): file_digest(p) for p in sorted(module_root.rglob("*.py"))},
            "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
        manifest["manifestDigest"] = json_digest(manifest)
        write_json(root / "manifest.json", manifest)
        root.replace(destination)
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = run_factorial(checkpoint=args.checkpoint, reservoir_path=args.reservoir, audit_path=args.audit,
                           output=args.output, config_path=args.config)
    print(json.dumps({"replicated": result["replicated"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
