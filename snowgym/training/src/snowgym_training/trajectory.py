"""Versioned, deterministic SnowGym trajectory shards and audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

DATASET_FORMAT = "snowgym.trajectory.v0"
EXPORT_SPEC_FORMAT = "snowgym.trajectory-export.v0"
STATE_HASH_PATTERN = re.compile(r"^fnv1a64:[0-9a-f]{16}$")
REQUIRED_ARRAYS = {
    "action__action_type",
    "action__power",
    "action__target",
    "episode_index",
    "next_tick",
    "observation__ally_mask",
    "observation__unit_action_mask",
    "post_state_hash",
    "pre_state_hash",
    "reward",
    "seed",
    "teacher_accepted",
    "teacher_reason",
    "terminated",
    "tick",
    "truncated",
}
COUNTERFACTUAL_ARRAYS = {
    "action__counterfactual_action_type",
    "action__counterfactual_power",
    "action__counterfactual_target",
    "observation__counterfactual_plan_group_mask",
    "observation__counterfactual_plan_groups",
}
PLAN_ROLE_ARRAYS = {
    "observation__plan_unit_roles",
    "observation__counterfactual_plan_unit_roles",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def tensor_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash tensor meaning, independent of NPZ compression or timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError(f"object array is forbidden: {name}")
        metadata = canonical_json(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def validate_export_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("format") != EXPORT_SPEC_FORMAT:
        raise ValueError(f"export spec format must be {EXPORT_SPEC_FORMAT}")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("export spec name must be a non-empty string")
    if not isinstance(spec.get("teacher"), str) or not spec["teacher"]:
        raise ValueError("export spec teacher must be a non-empty string")
    unknown = set(spec) - {
        "format",
        "name",
        "teacher",
        "maxTeamUnits",
        "shardSize",
        "splits",
    }
    if unknown:
        raise ValueError(f"unknown export spec fields: {sorted(unknown)}")
    capacity = spec.get("maxTeamUnits")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or not 1 <= capacity <= 10:
        raise ValueError("maxTeamUnits must be an integer in [1, 10]")
    shard_size = spec.get("shardSize")
    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size <= 0:
        raise ValueError("shardSize must be a positive integer")
    splits = spec.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "evaluation"}:
        raise ValueError("splits must contain exactly train, validation, and evaluation")
    seen: dict[int, str] = {}
    for split, episodes in splits.items():
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"split {split} must contain episodes")
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"{split}[{index}] must be an object")
            seed = episode.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError(f"{split}[{index}].seed must be non-negative")
            if seed in seen:
                raise ValueError(
                    f"seed {seed} overlaps splits {seen[seed]} and {split}"
                )
            seen[seed] = split
            if not isinstance(episode.get("scenario"), dict):
                raise ValueError(f"{split}[{index}].scenario must be an object")
    return spec


def load_export_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load export spec {source}: {error}") from error
    return validate_export_spec(value)


class TrajectoryWriter:
    """Write stable tensor shards and a manifest without holding a full corpus."""

    def __init__(self, output: str | Path, *, shard_size: int) -> None:
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite existing dataset {self.output}")
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output.mkdir(parents=True)
        self.shard_size = shard_size
        self._buffer: list[dict[str, np.ndarray]] = []
        self._schema: dict[str, tuple[str, tuple[int, ...]]] | None = None
        self._shards: list[dict[str, Any]] = []
        self.transition_count = 0

    def add(self, transition: dict[str, np.ndarray]) -> None:
        normalized = {
            name: np.asarray(value) for name, value in sorted(transition.items())
        }
        schema = {
            name: (value.dtype.str, value.shape) for name, value in normalized.items()
        }
        if self._schema is None:
            missing = REQUIRED_ARRAYS - set(normalized)
            if missing:
                raise ValueError(f"transition is missing arrays: {sorted(missing)}")
            self._schema = schema
        elif schema != self._schema:
            raise ValueError("transition tensor schema changed within dataset")
        if any(value.dtype.hasobject for value in normalized.values()):
            raise ValueError("object arrays are forbidden")
        self._buffer.append(normalized)
        self.transition_count += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def finish(self, metadata: dict[str, Any]) -> dict[str, Any]:
        self._flush()
        if not self._shards:
            raise ValueError("cannot finish an empty trajectory dataset")
        manifest = {
            "format": DATASET_FORMAT,
            **metadata,
            "transitions": self.transition_count,
            "shards": self._shards,
        }
        manifest["datasetDigest"] = json_digest(manifest)
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    def _flush(self) -> None:
        if not self._buffer:
            return
        arrays = {
            name: np.stack([transition[name] for transition in self._buffer])
            for name in self._buffer[0]
        }
        digest = tensor_digest(arrays)
        name = f"shard-{len(self._shards):05d}.npz"
        np.savez_compressed(self.output / name, **arrays)
        self._shards.append(
            {"path": name, "transitions": len(self._buffer), "tensorDigest": digest}
        )
        self._buffer = []


def audit_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load dataset manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != DATASET_FORMAT:
        raise ValueError(f"dataset format must be {DATASET_FORMAT}")
    claimed_dataset_digest = manifest.get("datasetDigest")
    digest_source = {key: value for key, value in manifest.items() if key != "datasetDigest"}
    if claimed_dataset_digest != json_digest(digest_source):
        raise ValueError("dataset manifest digest mismatch")
    if manifest.get("split") not in {"train", "validation", "evaluation"}:
        raise ValueError("dataset manifest has invalid split")
    versions = manifest.get("versions")
    if not isinstance(versions, dict) or not all(
        isinstance(versions.get(key), str) and versions[key]
        for key in (
            "apiVersion",
            "simulationVersion",
            "stateHashVersion",
            "upstreamBaseCommit",
        )
    ):
        raise ValueError("dataset manifest has invalid version provenance")
    split_seeds = manifest.get("splitSeeds")
    if not isinstance(split_seeds, dict):
        raise ValueError("dataset manifest is missing splitSeeds")
    flattened: list[int] = []
    for split in ("train", "validation", "evaluation"):
        seeds = split_seeds.get(split)
        if not isinstance(seeds, list) or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
        ):
            raise ValueError(f"invalid split seed list: {split}")
        flattened.extend(seeds)
    if len(flattened) != len(set(flattened)):
        raise ValueError("dataset split seeds overlap")

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest has no episodes")
    episode_by_index: dict[int, dict[str, Any]] = {}
    for expected_index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or episode.get("index") != expected_index:
            raise ValueError("dataset episode indices must be contiguous")
        if episode.get("seed") not in split_seeds[manifest["split"]]:
            raise ValueError("dataset episode seed is outside its declared split")
        episode_by_index[expected_index] = episode

    transition_count = 0
    episode_last: dict[int, tuple[str, int]] = {}
    episode_counts = {index: 0 for index in episode_by_index}
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("dataset manifest has no shards")
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValueError("invalid shard entry")
        with np.load(root / shard["path"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        missing = REQUIRED_ARRAYS - set(arrays)
        if missing:
            raise ValueError(f"shard is missing arrays: {sorted(missing)}")
        counterfactual = manifest.get("counterfactualPlanLabels")
        if counterfactual is not None:
            if counterfactual != {
                "teacher": "plan-teacher-action.v0",
                "pairing": "same-physical-state",
            }:
                raise ValueError("dataset counterfactual plan-label provenance is invalid")
            missing_counterfactual = COUNTERFACTUAL_ARRAYS - set(arrays)
            if missing_counterfactual:
                raise ValueError(
                    "shard is missing counterfactual arrays: "
                    f"{sorted(missing_counterfactual)}"
                )
            if manifest.get("planUnitRoles") is True:
                missing_roles = PLAN_ROLE_ARRAYS - set(arrays)
                if missing_roles:
                    raise ValueError(f"shard is missing plan role arrays: {sorted(missing_roles)}")
        if tensor_digest(arrays) != shard.get("tensorDigest"):
            raise ValueError(f"tensor digest mismatch: {shard['path']}")
        lengths = {array.shape[0] for array in arrays.values()}
        if len(lengths) != 1 or lengths != {shard.get("transitions")}:
            raise ValueError(f"inconsistent transition dimension: {shard['path']}")
        _audit_actions(arrays, shard["path"])
        if counterfactual is not None:
            _audit_counterfactual_actions(arrays, shard["path"])
            if manifest.get("planUnitRoles") is True:
                _audit_plan_roles(arrays, shard["path"])
        _audit_continuity(arrays, shard["path"], episode_last)
        for index, seed in zip(arrays["episode_index"], arrays["seed"], strict=True):
            episode_index = int(index)
            episode = episode_by_index.get(episode_index)
            if episode is None or int(seed) != episode.get("seed"):
                raise ValueError(f"transition episode metadata mismatch: {shard['path']}")
            episode_counts[episode_index] += 1
        transition_count += int(shard["transitions"])
    if transition_count != manifest.get("transitions"):
        raise ValueError("manifest transition count mismatch")
    for index, episode in episode_by_index.items():
        if episode_counts[index] != episode.get("transitions"):
            raise ValueError(f"episode transition count mismatch: {index}")
    return {
        "ok": True,
        "format": DATASET_FORMAT,
        "split": manifest.get("split"),
        "transitions": transition_count,
        "shards": len(manifest.get("shards", [])),
        "datasetDigest": claimed_dataset_digest,
    }


def _audit_actions(arrays: dict[str, np.ndarray], shard: str) -> None:
    action_types = arrays["action__action_type"]
    masks = arrays["observation__unit_action_mask"]
    ally_mask = arrays["observation__ally_mask"].astype(bool)
    if action_types.shape != ally_mask.shape or masks.shape[:2] != action_types.shape:
        raise ValueError(f"action/mask shape mismatch: {shard}")
    if (
        arrays["teacher_accepted"].shape != action_types.shape
        or arrays["teacher_reason"].shape != action_types.shape
    ):
        raise ValueError(f"teacher result shape mismatch: {shard}")
    if np.any((action_types < 0) | (action_types >= masks.shape[-1])):
        raise ValueError(f"teacher action type is out of range: {shard}")
    selected = np.take_along_axis(masks, action_types[..., None], axis=-1)[..., 0]
    if np.any(selected[ally_mask] != 1):
        raise ValueError(f"teacher action violates action mask: {shard}")
    if np.any(action_types[~ally_mask] != 0):
        raise ValueError(f"absent unit has non-noop action: {shard}")
    if not np.all(arrays["teacher_accepted"]):
        raise ValueError(f"dataset contains rejected teacher action: {shard}")
    targets = arrays["action__target"]
    powers = arrays["action__power"]
    if not np.all(np.isfinite(targets)) or np.any((targets < -1) | (targets > 1)):
        raise ValueError(f"teacher target is outside [-1, 1]: {shard}")
    if not np.all(np.isfinite(powers)) or np.any((powers < 0) | (powers > 1)):
        raise ValueError(f"teacher power is outside [0, 1]: {shard}")
    if not np.all(np.isfinite(arrays["reward"])):
        raise ValueError(f"reward is not finite: {shard}")
    for key in ("pre_state_hash", "post_state_hash"):
        if not all(STATE_HASH_PATTERN.fullmatch(str(value)) for value in arrays[key]):
            raise ValueError(f"invalid {key}: {shard}")


def _audit_counterfactual_actions(arrays: dict[str, np.ndarray], shard: str) -> None:
    remapped = dict(arrays)
    for field in ("action_type", "target", "power"):
        remapped[f"action__{field}"] = arrays[f"action__counterfactual_{field}"]
    remapped["teacher_accepted"] = np.ones_like(arrays["teacher_accepted"])
    remapped["teacher_reason"] = np.zeros_like(arrays["teacher_reason"])
    _audit_actions(remapped, f"{shard}:counterfactual")
    groups = arrays["observation__counterfactual_plan_groups"]
    mask = arrays["observation__counterfactual_plan_group_mask"]
    if groups.shape[1:] != (3, 38) or mask.shape[1:] != (3,):
        raise ValueError(f"counterfactual plan tensor shape mismatch: {shard}")
    if not np.all(np.isfinite(groups)) or np.any((groups < -1) | (groups > 1)):
        raise ValueError(f"counterfactual plan tensor is outside [-1, 1]: {shard}")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError(f"counterfactual plan mask is invalid: {shard}")


def _audit_plan_roles(arrays: dict[str, np.ndarray], shard: str) -> None:
    ally_mask = arrays["observation__ally_mask"].astype(bool)
    for name in PLAN_ROLE_ARRAYS:
        roles = arrays[name]
        if roles.shape != (*ally_mask.shape, 3):
            raise ValueError(f"plan unit-role tensor shape mismatch: {shard}")
        if np.any((roles != 0) & (roles != 1)) or np.any(roles.sum(axis=-1) > 1):
            raise ValueError(f"plan unit-role tensor is not one-hot: {shard}")
        if np.any(roles[~ally_mask] != 0) or np.any(roles[ally_mask].sum(axis=-1) != 1):
            raise ValueError(f"plan unit-role tensor does not cover living allies: {shard}")


def _audit_continuity(
    arrays: dict[str, np.ndarray],
    shard: str,
    episode_last: dict[int, tuple[str, int]],
) -> None:
    ticks = arrays["tick"]
    next_ticks = arrays["next_tick"]
    observation_ticks = arrays.get("observation__tick")
    if observation_ticks is None or observation_ticks.shape != (len(ticks), 1):
        raise ValueError(f"observation tick shape mismatch: {shard}")
    for index in range(len(ticks)):
        episode = int(arrays["episode_index"][index])
        tick = int(ticks[index])
        next_tick = int(next_ticks[index])
        if int(observation_ticks[index, 0]) != tick or next_tick <= tick:
            raise ValueError(f"invalid transition tick: {shard}[{index}]")
        previous = episode_last.get(episode)
        if previous is not None and (
            str(arrays["pre_state_hash"][index]) != previous[0]
            or tick != previous[1]
        ):
            raise ValueError(f"state trajectory discontinuity: {shard}[{index}]")
        episode_last[episode] = (
            str(arrays["post_state_hash"][index]),
            next_tick,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a SnowGym trajectory dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_dataset(args.dataset)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"SnowGym dataset passed: {result['transitions']} transitions, "
            f"{result['shards']} shards"
        )


if __name__ == "__main__":
    main()
