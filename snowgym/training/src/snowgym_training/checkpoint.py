"""Versioned, semantically hashed SnowGym Torch checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .trajectory import canonical_json, json_digest

CHECKPOINT_FORMAT = "snowgym.checkpoint.v0"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {destination}")
    destination.mkdir(parents=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    state_digest = semantic_state_digest(state)
    value = {"format": CHECKPOINT_FORMAT, **metadata, "stateDigest": state_digest}
    validate_checkpoint_metadata(value, require_digest=False)
    value["checkpointDigest"] = json_digest(value)
    torch.save(state, destination / "state.pt")
    (destination / "metadata.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load checkpoint metadata: {error}") from error
    validate_checkpoint_metadata(metadata, require_digest=True)
    try:
        state = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot load checkpoint state: {error}") from error
    if not isinstance(state, dict) or set(state) != {"model", "optimizer"}:
        raise ValueError("checkpoint state must contain model and optimizer")
    if semantic_state_digest(state) != metadata["stateDigest"]:
        raise ValueError("checkpoint state digest mismatch")
    return metadata, state


def validate_checkpoint_metadata(value: Any, *, require_digest: bool) -> None:
    if not isinstance(value, dict) or value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format must be {CHECKPOINT_FORMAT}")
    required = {
        "gitCommit",
        "datasetManifestHash",
        "versions",
        "architecture",
        "optimizer",
        "loss",
        "trainingSeed",
        "step",
        "evaluationSuite",
        "stateDigest",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"checkpoint metadata missing fields: {sorted(missing)}")
    for key in ("gitCommit", "datasetManifestHash", "stateDigest", "evaluationSuite"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"checkpoint {key} must be a non-empty string")
    if not isinstance(value["trainingSeed"], int) or not isinstance(value["step"], int):
        raise ValueError("checkpoint trainingSeed and step must be integers")
    for key in ("versions", "architecture", "optimizer", "loss"):
        if not isinstance(value[key], dict):
            raise ValueError(f"checkpoint {key} must be an object")
    if require_digest:
        claimed = value.get("checkpointDigest")
        source = {key: item for key, item in value.items() if key != "checkpointDigest"}
        if claimed != json_digest(source):
            raise ValueError("checkpoint metadata digest mismatch")


def semantic_state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return "sha256:" + digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        header = canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
        digest.update(b"tensor")
        digest.update(header.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: canonical_json(item)):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(canonical_json(value).encode("utf-8"))
    else:
        raise ValueError(f"unsupported checkpoint state value: {type(value).__name__}")
