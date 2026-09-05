"""Frozen R1j matched 2x2 decoder fitting and development evaluation."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from snowgym_client.batch import SnowGymBatchClient

from ..checkpoint import semantic_state_digest
from ..executor.decoder_probe import ARMS, DecoderProbe
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import geometry_probe as geometry
from .evaluate import evaluate_option_episode
from .identity import checkpoint_model
from .interventions import require_capabilities
from .probe_metrics import teacher_agreement
from .reservoir import file_digest, load_teacher_bc_reservoir
from .supervised_probe import summarize_rows, validate_inputs, write_json

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/m7b_engage_r1j_decoder_probe_v0.json"


def load_config(path=DEFAULT_CONFIG):
    expected = {**geometry.load_config(), "format": "snowgym.decoder-probe-config.v0", "arms": list(ARMS),
                "movementCorrectionWorld": 10, "bootstrapSamples": 10000, "bootstrapSeed": 770001}
    value = json.loads(Path(path).read_text())
    if value != expected:
        raise ValueError("decoder configuration differs from frozen R1j design")
    return value


def build_probe(metadata, state, *, arm, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source = checkpoint_model(metadata)
        source.load_state_dict(state["model"])
        return DecoderProbe(source, arm=arm).eval()


def save_probe(path, model, source, config, epoch, optimizer=None):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite decoder checkpoint {path}")
    path.mkdir()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None}, path / "state.pt")
    metadata = {"format": "snowgym.decoder-probe-checkpoint.v0", "arm": model.arm, "source": source,
                "config": config, "epoch": epoch, "execution": "deterministic-only", "ppoCompatible": False,
                "stateFileDigest": file_digest(path / "state.pt"),
                "modelStateDigest": semantic_state_digest(model.state_dict()),
                "sourceModelStateDigest": semantic_state_digest(model.source.state_dict())}
    metadata["checkpointDigest"] = json_digest(metadata)
    write_json(path / "checkpoint.json", metadata)
    return metadata


def load_probe(path):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (metadata.get("format") != "snowgym.decoder-probe-checkpoint.v0"
        or metadata.get("execution") != "deterministic-only" or metadata.get("ppoCompatible") is not False
        or metadata.get("checkpointDigest") != json_digest({k: v for k, v in metadata.items() if k != "checkpointDigest"})
        or metadata["stateFileDigest"] != file_digest(path / "state.pt")):
        raise ValueError("decoder checkpoint provenance mismatch")
    state = torch.load(path / "state.pt", weights_only=True, map_location="cpu")
    model = DecoderProbe(checkpoint_model(metadata["source"]), arm=metadata["arm"])
    model.load_state_dict(state["model"])
    if (semantic_state_digest(model.state_dict()) != metadata["modelStateDigest"]
        or semantic_state_digest(model.source.state_dict()) != metadata["sourceModelStateDigest"]):
        raise ValueError("decoder checkpoint semantic digest mismatch")
    return model.eval(), metadata


def contrasts(rows, config):
    seeds = [r["seed"] for r in rows["absolute"]["correct"]]
    if not seeds or len(seeds) != len(set(seeds)) or any([r["seed"] for r in rows[a]["correct"]] != seeds for a in ARMS):
        raise ValueError("decoder evaluation seeds must be unique and paired")
    indices = np.random.default_rng(config["bootstrapSeed"]).integers(0, len(seeds), (config["bootstrapSamples"], len(seeds)))
    weights = {"displacementWithAbsoluteShot": {"displacement": 1, "absolute": -1},
               "directionWithAbsoluteMove": {"direction": 1, "absolute": -1},
               "displacementWithDirection": {"both": 1, "direction": -1},
               "directionWithDisplacement": {"both": 1, "displacement": -1},
               "interaction": {"both": 1, "direction": -1, "displacement": -1, "absolute": 1}}
    output = {}
    for name, coefficients in weights.items():
        output[name] = {}
        for metric in ("success", "progress"):
            delta = sum(w*np.asarray([float(r[metric]) for r in rows[a]["correct"]]) for a, w in coefficients.items())
            output[name][metric] = {"mean": float(delta.mean()), "bootstrap95": np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    return output


def run_probe(*, checkpoint, reservoir_path, output, config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite decoder experiment {destination}")
    source, state = load_ppo_checkpoint(checkpoint)
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    validate_inputs(config, source, reservoir)
    if reservoir.metadata["simulationVersion"] != "snowgym.sim.v2" or reservoir.metadata["stateHashVersion"] != "snowgym.state.v2":
        raise ValueError("decoder probe requires v2 reservoir")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {a: build_probe(source, state, arm=a, seed=config["trainingSeed"]) for a in ARMS}
    source_digest = semantic_state_digest(state["model"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    root = temporary / destination.name
    root.mkdir()
    try:
        gates = {a: geometry.small_batch_gate(m, reservoir, config) for a, m in models.items()}
        write_json(root / "small-batch-gates.json", gates)
        print(json.dumps({"phase": "gates", "results": gates}), flush=True)
        report = {"format": "snowgym.decoder-probe-report.v0", "gatesPassed": all(g["passed"] for g in gates.values()),
                  "ppoUpdates": 0, "criticUpdates": 0, "qualificationEligible": False, "arms": {}}
        if report["gatesPassed"]:
            for arm, model in models.items():
                initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                before = teacher_agreement(model, reservoir)
                save_probe(root / f"{arm}-epoch-000", model, source, config, 0)
                optimizer = geometry.optimizer_for(model, config["learningRate"])
                epochs = []
                for epoch in range(1, config["epochs"]+1):
                    order = torch.randperm(reservoir.size, generator=torch.Generator().manual_seed(config["trainingSeed"]+epoch*1_000_003))
                    total = 0.
                    for start in range(0, reservoir.size, config["minibatchSize"]):
                        indices = order[start:start+config["minibatchSize"]]
                        obs, teacher = reservoir.batch(indices)
                        losses = geometry.train_step(model, optimizer, obs, teacher, config["maxGradNorm"])
                        total += losses["total"] * len(indices)
                    epochs.append({"epoch": epoch, "loss": total/reservoir.size})
                    print(json.dumps({"phase": "training", "arm": arm, **epochs[-1]}), flush=True)
                if semantic_state_digest(model.source.state_dict()) != source_digest:
                    raise RuntimeError("decoder fitting changed frozen source")
                path = root / f"{arm}-epoch-020"
                save_probe(path, model, source, config, config["epochs"], optimizer)
                loaded, _ = load_probe(path)
                obs, _ = reservoir.batch(geometry.gate_indices(reservoir))
                with torch.no_grad():
                    original, _, _ = model.act(obs, deterministic=True)
                    restored, _, _ = loaded.act(obs, deterministic=True)
                if any(not torch.equal(original[k], restored[k]) for k in original):
                    raise RuntimeError("decoder reload changed actions")
                report["arms"][arm] = {"epochs": epochs, "teacherAgreementBefore": before,
                    "teacherAgreementAfter": teacher_agreement(model, reservoir), "sourceUnchanged": True,
                    "checkpointReloadExact": True, "newParameterCount": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "newParameterL2Change": sum(float((p.detach()-initial[n]).square().sum()) for n, p in model.named_parameters() if n in initial) ** .5}
            baseline = models["absolute"].source
            rows = {}
            with SnowGymBatchClient() as client:
                capabilities = require_capabilities(client)
                if any(capabilities[k] != reservoir.metadata[k] for k in ("simulationVersion", "stateHashVersion")):
                    raise ValueError("decoder evaluation simulator mismatch")
                report["capabilities"] = capabilities
                for arm, model in {"source": baseline, **models}.items():
                    rows[arm] = {}
                    for condition in ("correct", "shuffled"):
                        rows[arm][condition] = [evaluate_option_episode(model, baseline, option="engage", seed=seed,
                            condition=condition, client=client) for seed in range(200000, 200040)]
                        print(json.dumps({"phase": "evaluation", "arm": arm, "condition": condition,
                                         "summary": summarize_rows(rows[arm][condition])}), flush=True)
                    write_json(root / f"{arm}-development.json", rows[arm])
            report["development"] = {a: {c: summarize_rows(r) for c, r in v.items()} for a, v in rows.items()}
            report["pairedContrasts"] = contrasts(rows, config)
        write_json(root / "report.json", report)
        module_root = Path(__file__).resolve().parents[1]
        repository = module_root.parents[3]
        manifest = {"format": "snowgym.decoder-probe-run.v0", "gitCommit": resolve_git_commit(), "config": config,
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
    result = run_probe(checkpoint=args.checkpoint, reservoir_path=args.reservoir, output=args.output, config_path=args.config)
    print(json.dumps({"gatesPassed": result["gatesPassed"], "development": result.get("development")}), flush=True)


if __name__ == "__main__":
    main()
