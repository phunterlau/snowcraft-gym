"""Audited loader and transition join for exported command-plan tensors."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import json_digest

PLAN_TENSOR_DATASET_FORMAT = "snowgym.plan-tensor-dataset.v0"
SYNTHETIC_PLAN_CURRICULUM_FORMAT = "snowgym.synthetic-plan-curriculum.v0"
PLAN_GROUP_SLOTS = 3
PLAN_FEATURE_VECTOR_SIZE = 38


def javascript_json_digest(value: Any) -> str:
    """Match the sorted-key JSON.stringify digest used by TypeScript artifacts."""
    payload = _javascript_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _javascript_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _javascript_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_javascript_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            _javascript_json(str(key)) + ":" + _javascript_json(value[key])
            for key in sorted(value)
        ) + "}"
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def _javascript_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON numbers must be finite")
    if value == 0:
        return "0"
    negative = value < 0
    text = repr(abs(value)).lower()
    if "e" not in text:
        if text.endswith(".0"):
            text = text[:-2]
        return ("-" if negative else "") + text
    mantissa, exponent_text = text.split("e")
    exponent = int(exponent_text)
    digits = mantissa.replace(".", "").rstrip("0")
    decimal_position = 1 + exponent
    absolute = abs(value)
    if 1e-6 <= absolute < 1e21:
        if decimal_position <= 0:
            rendered = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
    else:
        rendered = digits[0]
        if len(digits) > 1:
            rendered += "." + digits[1:]
        rendered += "e" + ("+" if exponent >= 0 else "-") + str(abs(exponent))
    return ("-" if negative else "") + rendered


class PlanTensorDataset:
    """Strict immutable NumPy view of one TypeScript-exported plan dataset."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load plan tensor dataset {self.path}: {error}") from error
        audit_plan_tensor_dataset(value)
        self.manifest = value
        self.plan_groups = np.asarray(
            [item["groups"] for item in value["tensors"]], dtype=np.float32
        ).reshape(-1, PLAN_GROUP_SLOTS, PLAN_FEATURE_VECTOR_SIZE)
        self.plan_group_mask = np.asarray(
            [item["groupMask"] for item in value["tensors"]], dtype=np.int8
        )
        self.source_seeds = np.asarray(
            [item["sourceSeed"] for item in value["tensors"]], dtype=np.int64
        )
        for array in (self.plan_groups, self.plan_group_mask, self.source_seeds):
            array.setflags(write=False)

    def __len__(self) -> int:
        return int(self.plan_groups.shape[0])

    def batch_for_transitions(
        self, plan_indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        indices = np.asarray(plan_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("plan indices must be a one-dimensional integer array")
        if len(indices) > 0 and (int(indices.min()) < 0 or int(indices.max()) >= len(self)):
            raise IndexError("plan index is outside the plan tensor dataset")
        return {
            "plan_groups": np.array(self.plan_groups[indices], copy=True),
            "plan_group_mask": np.array(self.plan_group_mask[indices], copy=True),
            "plan_source_seed": np.array(self.source_seeds[indices], copy=True),
        }


def audit_plan_tensor_dataset(value: Any) -> dict[str, Any]:
    required = {
        "format",
        "apiVersion",
        "simulationVersion",
        "stateHashVersion",
        "upstreamBaseCommit",
        "scenario",
        "environmentSeed",
        "sourceStateHash",
        "curriculum",
        "tensors",
        "datasetDigest",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("plan tensor dataset fields are invalid")
    if value["format"] != PLAN_TENSOR_DATASET_FORMAT:
        raise ValueError(f"plan tensor dataset format must be {PLAN_TENSOR_DATASET_FORMAT}")
    curriculum = value["curriculum"]
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("format") != SYNTHETIC_PLAN_CURRICULUM_FORMAT
        or not isinstance(curriculum.get("samples"), list)
        or curriculum.get("sampleCount") != len(curriculum["samples"])
    ):
        raise ValueError("plan tensor curriculum is invalid")
    tensors = value["tensors"]
    if not isinstance(tensors, list) or len(tensors) != len(curriculum["samples"]):
        raise ValueError("plan tensor sample counts do not match")
    for index, (sample, tensor) in enumerate(zip(curriculum["samples"], tensors)):
        if not isinstance(sample, dict) or not isinstance(tensor, dict):
            raise ValueError(f"plan tensor sample {index} is invalid")
        if set(tensor) != {"sourceSeed", "groups", "groupMask"}:
            raise ValueError(f"plan tensor sample {index} fields are invalid")
        if tensor["sourceSeed"] != sample.get("sourceSeed"):
            raise ValueError(f"plan tensor sample {index} seed is misaligned")
        groups = np.asarray(tensor["groups"])
        mask = np.asarray(tensor["groupMask"])
        if groups.shape != (PLAN_GROUP_SLOTS * PLAN_FEATURE_VECTOR_SIZE,):
            raise ValueError(f"plan tensor sample {index} group shape is invalid")
        if mask.shape != (PLAN_GROUP_SLOTS,):
            raise ValueError(f"plan tensor sample {index} mask shape is invalid")
        if groups.dtype.kind not in "iuf" or not np.isfinite(groups).all():
            raise ValueError(f"plan tensor sample {index} values must be finite numbers")
        if np.any(groups < -1) or np.any(groups > 1):
            raise ValueError(f"plan tensor sample {index} values are outside [-1, 1]")
        if mask.dtype.kind not in "iu" or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"plan tensor sample {index} mask must be binary integers")
    body = {name: item for name, item in value.items() if name != "datasetDigest"}
    if value["datasetDigest"] != json_digest(body):
        raise ValueError("plan tensor dataset digest mismatch")
    return value
