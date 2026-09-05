"""Recovery-specific checkpoints; legacy movement loaders remain unchanged."""

import json
from pathlib import Path

import torch

from ..checkpoint import semantic_state_digest
from ..executor.recovery_ppo import RecoveryPolicy, RECOVERY_INPUT
from ..trajectory import json_digest
from .identity import checkpoint_model
from .movement_collect import ASSIST_FIELDS
from .reservoir import file_digest
from .supervised_probe import write_json


def save(path, model, optimizer, *, source, config, seed, dataset_digest, update, sampler, history):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torchRng": torch.get_rng_state()}
    metadata = {"format": "snowgym.recovery-checkpoint.v0", "recoveryInput": RECOVERY_INPUT,
        **ASSIST_FIELDS, "source": source, "config": config, "trainingSeed": seed,
        "datasetDigest": dataset_digest, "update": update, "sampler": sampler, "history": history,
        "stateDigest": semantic_state_digest(state)}
    path.mkdir(parents=True)
    torch.save(state, path / "state.pt")
    metadata["stateFileDigest"] = file_digest(path / "state.pt")
    metadata["checkpointDigest"] = json_digest(metadata)
    write_json(path / "checkpoint.json", metadata)
    return metadata


def load(path):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (metadata.get("format") != "snowgym.recovery-checkpoint.v0" or metadata.get("recoveryInput") != RECOVERY_INPUT
        or any(metadata.get(k) != v for k, v in ASSIST_FIELDS.items())
        or metadata["checkpointDigest"] != json_digest({k: v for k, v in metadata.items() if k != "checkpointDigest"})
        or metadata["stateFileDigest"] != file_digest(path / "state.pt")):
        raise ValueError("recovery checkpoint identity mismatch")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if semantic_state_digest(state) != metadata["stateDigest"]:
        raise ValueError("recovery checkpoint state mismatch")
    model = RecoveryPolicy(checkpoint_model(metadata["source"]), standard_deviation=metadata["config"]["latentStd"])
    model.load_state_dict(state["model"])
    if float(model.standard_deviation) != float(torch.tensor(metadata["config"]["latentStd"])):
        raise ValueError("recovery noise mismatch")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=metadata["config"]["learningRate"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torchRng"])
    return model, optimizer, metadata
