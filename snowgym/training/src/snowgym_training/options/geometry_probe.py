"""Frozen R1i matched feature fitting and closed-loop diagnostic runner."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from snowgym_client.batch import SnowGymBatchClient

from ..checkpoint import semantic_state_digest
from ..executor.geometry_probe import GeometryProbe, geometry_loss
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .evaluate import evaluate_option_episode
from .identity import checkpoint_model
from .interventions import require_capabilities
from .probe_metrics import teacher_agreement
from .reservoir import TeacherBcReservoir, file_digest, load_teacher_bc_reservoir
from .supervised_probe import summarize_rows, validate_inputs, write_json

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1i_geometry_probe_v0.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    expected = {
        "format": "snowgym.geometry-probe-config.v0",
        "sourceCheckpointDigest": "sha256:10d924ecdfbc554a8e0324387d8d049b9ffe719e8d8f2768123e4886c265a697",
        "reservoirDigest": "sha256:a1410c32a718c53664b91878852a2203247454ae0bf2dcb4caeb904b0ac334a6",
        "arms": ["absolute", "relative"], "trainingEvaluationSeeds": [100000, 100039],
        "developmentEvaluationSeeds": [200000, 200039], "smallBatchSteps": 200,
        "smallBatchLearningRate": .001, "smallBatchRequiredReduction": .5, "epochs": 20,
        "minibatchSize": 256, "learningRate": .0003, "maxGradNorm": .5,
        "trainingSeed": 92001, "selectionPolicy": "final-epoch-only",
    }
    if value != expected:
        raise ValueError("geometry configuration differs from frozen R1i design")
    return value


def build_probe(metadata, state, *, relative: bool, seed: int) -> GeometryProbe:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source = checkpoint_model(metadata)
        source.load_state_dict(state["model"])
        return GeometryProbe(source, relative=relative).eval()


def optimizer_for(model: GeometryProbe, rate: float):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=rate)


def train_step(model, optimizer, observation, teacher, clip):
    losses = geometry_loss(model(observation), teacher, observation)
    if not all(torch.isfinite(v) for v in losses.values()):
        raise ValueError("non-finite geometry loss")
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip, error_if_nonfinite=True)
    optimizer.step()
    return {k: float(v.detach()) for k, v in losses.items()}


def gate_indices(reservoir: TeacherBcReservoir) -> torch.Tensor:
    labels = reservoir.actions["action_type"]
    throws = (labels == 2).any(-1)
    moves = (labels == 1).any(-1) & ~throws
    selected = torch.cat([throws.nonzero().flatten()[:16], moves.nonzero().flatten()[:16]])
    if selected.numel() != 32:
        raise ValueError("small-batch gate requires 16 throw and 16 move-only states")
    return selected


def small_batch_gate(model, reservoir, config):
    probe = copy.deepcopy(model)
    source_digest = semantic_state_digest(probe.source.state_dict())
    observation, teacher = reservoir.batch(gate_indices(reservoir))
    with torch.no_grad():
        initial = float(geometry_loss(probe(observation), teacher, observation)["total"])
    optimizer = optimizer_for(probe, config["smallBatchLearningRate"])
    for _ in range(config["smallBatchSteps"]):
        train_step(probe, optimizer, observation, teacher, config["maxGradNorm"])
    with torch.no_grad():
        final = float(geometry_loss(probe(observation), teacher, observation)["total"])
    gradients = {}
    for component in ("move", "direction", "power"):
        probe.zero_grad(set_to_none=True)
        geometry_loss(probe(observation), teacher, observation)[component].backward()
        gradients[component] = {
            group: sum(float(p.grad.square().sum()) for name, p in probe.named_parameters()
                       if name.startswith(group + ".") and p.grad is not None) ** .5
            for group in ("encoders", "move", "shot", "source")
        }
    if source_digest != semantic_state_digest(probe.source.state_dict()):
        raise RuntimeError("small-batch fitting changed frozen source")
    reachable = all(gradients[k]["encoders"] > 0 for k in gradients)
    return {"initialLoss": initial, "finalLoss": final, "reduction": 1-final/max(initial, 1e-12),
            "gradientNorms": gradients, "sourceUnchanged": True,
            "passed": final <= initial * (1-config["smallBatchRequiredReduction"]) and reachable,
            "indices": gate_indices(reservoir).tolist()}


def save_probe(path: Path, model: GeometryProbe, source_metadata, *, epoch: int, config, optimizer=None):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite geometry checkpoint {path}")
    path.mkdir()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None}, path / "state.pt")
    metadata = {"format": "snowgym.geometry-probe-checkpoint.v0", "relative": model.relative,
                "source": source_metadata, "epoch": epoch, "config": config,
                "stateFileDigest": file_digest(path / "state.pt"),
                "modelStateDigest": semantic_state_digest(model.state_dict()),
                "execution": "deterministic-only", "ppoCompatible": False}
    metadata["checkpointDigest"] = json_digest(metadata)
    write_json(path / "checkpoint.json", metadata)
    return metadata


def load_probe(path: str | Path) -> tuple[GeometryProbe, dict[str, Any]]:
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (metadata.get("format") != "snowgym.geometry-probe-checkpoint.v0"
        or metadata.get("checkpointDigest") != json_digest({k: v for k, v in metadata.items() if k != "checkpointDigest"})
        or metadata["stateFileDigest"] != file_digest(path / "state.pt")):
        raise ValueError("geometry checkpoint provenance mismatch")
    state = torch.load(path / "state.pt", weights_only=True, map_location="cpu")
    model = GeometryProbe(checkpoint_model(metadata["source"]), relative=metadata["relative"])
    model.load_state_dict(state["model"])
    if semantic_state_digest(model.state_dict()) != metadata["modelStateDigest"]:
        raise ValueError("geometry checkpoint semantic digest mismatch")
    return model.eval(), metadata


def run_probe(*, checkpoint: str | Path, reservoir_path: str | Path, output: str | Path, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite geometry experiment {destination}")
    source, state = load_ppo_checkpoint(checkpoint)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    validate_inputs(config, source, reservoir)
    if reservoir.metadata["simulationVersion"] != "snowgym.sim.v2" or reservoir.metadata["stateHashVersion"] != "snowgym.state.v2":
        raise ValueError("geometry probe requires v2 reservoir provenance")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {arm: build_probe(source, state, relative=arm == "relative", seed=config["trainingSeed"]) for arm in config["arms"]}
    source_digest = semantic_state_digest(state["model"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        gates = {arm: small_batch_gate(model, reservoir, config) for arm, model in models.items()}
        write_json(root / "small-batch-gates.json", gates)
        print(json.dumps({"phase": "small-batch-gates", "results": gates}), flush=True)
        passed = all(gate["passed"] for gate in gates.values())
        report = {"format": "snowgym.geometry-probe-report.v0", "gatesPassed": passed,
                  "ppoUpdates": 0, "criticUpdates": 0, "qualificationEligible": False,
                  "trainingMethod": "supervised-only", "arms": {}}
        if passed:
            for arm, model in models.items():
                initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                before = teacher_agreement(model, reservoir)
                optimizer = optimizer_for(model, config["learningRate"])
                save_probe(root / f"{arm}-epoch-000", model, source, epoch=0, config=config)
                epochs = []
                for epoch in range(1, config["epochs"]+1):
                    order = torch.randperm(reservoir.size, generator=torch.Generator().manual_seed(config["trainingSeed"]+epoch*1_000_003))
                    total = 0.
                    for start in range(0, reservoir.size, config["minibatchSize"]):
                        indices = order[start:start+config["minibatchSize"]]
                        observation, teacher = reservoir.batch(indices)
                        losses = train_step(model, optimizer, observation, teacher, config["maxGradNorm"])
                        total += losses["total"] * len(indices)
                    epochs.append({"epoch": epoch, "loss": total/reservoir.size})
                    print(json.dumps({"phase": "training", "arm": arm, **epochs[-1]}), flush=True)
                if semantic_state_digest(model.source.state_dict()) != source_digest:
                    raise RuntimeError("geometry fitting changed frozen actor/critic")
                final_path = root / f"{arm}-epoch-020"
                save_probe(final_path, model, source, epoch=config["epochs"], config=config, optimizer=optimizer)
                loaded, _ = load_probe(final_path)
                observation, _ = reservoir.batch(gate_indices(reservoir))
                with torch.no_grad():
                    original, _, _ = model.act(observation, deterministic=True)
                    restored, _, _ = loaded.act(observation, deterministic=True)
                    if any(not torch.equal(original[k], restored[k]) for k in original):
                        raise RuntimeError("geometry checkpoint reload changed actions")
                report["arms"][arm] = {"epochs": epochs, "teacherAgreementBefore": before,
                    "teacherAgreementAfter": teacher_agreement(model, reservoir),
                    "newParameterCount": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "newParameterL2Change": sum(float((p.detach()-initial[n]).square().sum()) for n, p in model.named_parameters() if n in initial) ** .5,
                    "sourceUnchanged": True, "checkpointReloadExact": True}
            baseline = models["absolute"].source
            with SnowGymBatchClient() as client:
                capabilities = require_capabilities(client)
                if any(capabilities[k] != reservoir.metadata[k] for k in ("simulationVersion", "stateHashVersion")):
                    raise ValueError("geometry evaluation simulator version mismatch")
                report["capabilities"] = capabilities
                for name, model in {"source": baseline, **models}.items():
                    rows = {}
                    for condition in ("correct", "shuffled"):
                        rows[condition] = [evaluate_option_episode(model, baseline, option="engage", seed=seed,
                            condition=condition, client=client) for seed in range(200000, 200040)]
                        print(json.dumps({"phase": "evaluation", "arm": name, "condition": condition,
                            "summary": summarize_rows(rows[condition])}), flush=True)
                    write_json(root / f"{name}-development.json", rows)
                    report.setdefault("development", {})[name] = {c: summarize_rows(r) for c, r in rows.items()}
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        manifest = {"format": "snowgym.geometry-probe-run.v0", "gitCommit": resolve_git_commit(), "config": config,
                    "configDigest": json_digest(config), "sourceCheckpoint": source, "reservoir": reservoir.metadata,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = run_probe(checkpoint=args.checkpoint, reservoir_path=args.reservoir, output=args.output, config_path=args.config)
    print(json.dumps({"gatesPassed": report["gatesPassed"], "development": report.get("development")}), flush=True)


if __name__ == "__main__":
    main()
