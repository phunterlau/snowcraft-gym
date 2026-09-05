"""Digest-bound checkpoints for the scoped assisted movement contract."""

import json
from pathlib import Path

import torch

from ..checkpoint import semantic_state_digest
from ..executor.movement_ppo import AssistedMovementPolicy
from ..trajectory import json_digest
from .engage_v1 import OPTION_STATE_VERSION
from .identity import checkpoint_model
from .movement_collect import ASSIST_FIELDS
from .reservoir import file_digest
from .supervised_probe import write_json


def save_movement(path, model, optimizer, *, source, config, seed, update, schedule, collection=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite movement checkpoint {path}")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "torchRng": torch.get_rng_state(), "collection": collection}
    metadata = {"format": "snowgym.assisted-movement-checkpoint.v0", **ASSIST_FIELDS,
        "optionStateVersion": OPTION_STATE_VERSION, "source": source,
        "config": config, "configDigest": json_digest(config), "trainingSeed": seed,
        "updateIndex": update, "seedSchedule": schedule,
        "stateDigest": semantic_state_digest(state)}
    path.mkdir(parents=True)
    torch.save(state, path / "state.pt")
    metadata["stateFileDigest"] = file_digest(path / "state.pt")
    metadata["checkpointDigest"] = json_digest(metadata)
    write_json(path / "checkpoint.json", metadata)
    return metadata


def load_movement(path):
    path = Path(path)
    metadata = json.loads((path / "checkpoint.json").read_text())
    if (metadata.get("format") != "snowgym.assisted-movement-checkpoint.v0"
        or metadata.get("optionStateVersion") != OPTION_STATE_VERSION
        or any(metadata.get(k) != v for k, v in ASSIST_FIELDS.items())
        or metadata["checkpointDigest"] != json_digest({k: v for k, v in metadata.items() if k != "checkpointDigest"})
        or metadata["configDigest"] != json_digest(metadata["config"])
        or metadata["stateFileDigest"] != file_digest(path / "state.pt")):
        raise ValueError("movement checkpoint identity mismatch")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if semantic_state_digest(state) != metadata["stateDigest"]:
        raise ValueError("movement checkpoint state digest mismatch")
    model = AssistedMovementPolicy(checkpoint_model(metadata["source"]),
        standard_deviation=metadata["config"]["latentStd"])
    model.load_state_dict(state["model"])
    if float(model.standard_deviation) != float(torch.tensor(metadata["config"]["latentStd"])):
        raise ValueError("movement noise buffer differs from configuration")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=metadata["config"]["learningRate"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torchRng"])
    return model, optimizer, metadata, state["collection"]
