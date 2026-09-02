"""Deterministically aggregate audited SnowGym trajectory datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import TrajectoryWriter, audit_dataset, json_digest


def merge_datasets(
    *, output: str | Path, inputs: list[str | Path], shard_size: int = 256,
    independent_sources: bool = False,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input dataset is required")
    manifests: list[dict[str, Any]] = []
    roots: list[Path] = []
    for value in inputs:
        root = Path(value)
        audit_dataset(root)
        manifests.append(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
        roots.append(root)
    reference = manifests[0]
    compatibility = ("split", "maxTeamUnits", "versions") if independent_sources else (
        "split", "splitSeeds", "sourceSpecDigest", "maxTeamUnits", "versions"
    )
    for manifest in manifests[1:]:
        for key in compatibility:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"trajectory inputs disagree on {key}")
    plan_conditioned = reference.get("planConditioned") is True
    if any((manifest.get("planConditioned") is True) != plan_conditioned for manifest in manifests):
        raise ValueError("trajectory inputs disagree on planConditioned")
    field_sets = [_dataset_fields(root, manifest) for root, manifest in zip(roots, manifests, strict=True)]
    retained_fields = set.intersection(*field_sets) if independent_sources else field_sets[0]
    if not independent_sources and any(fields != retained_fields for fields in field_sets[1:]):
        raise ValueError("trajectory inputs disagree on tensor fields")
    if plan_conditioned and not {
        "observation__plan_groups", "observation__plan_group_mask"
    } <= retained_fields:
        raise ValueError("plan-conditioned aggregate is missing plan tensor fields")
    combined_split_seeds = _combined_split_seeds(manifests, independent_sources)

    writer = TrajectoryWriter(output, shard_size=shard_size)
    episodes: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    episode_offset = 0
    for source_index, (root, manifest) in enumerate(zip(roots, manifests, strict=True)):
        transition_offset = writer.transition_count
        for shard in manifest["shards"]:
            with np.load(root / shard["path"], allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files if name in retained_fields
                }
            for row in range(int(shard["transitions"])):
                transition = {name: value[row] for name, value in arrays.items()}
                transition["episode_index"] = np.asarray(
                    int(transition["episode_index"]) + episode_offset, dtype=np.int32
                )
                writer.add(transition)
        for episode in manifest["episodes"]:
            episodes.append(
                {
                    **episode,
                    "index": int(episode["index"]) + episode_offset,
                    "startTransition": int(episode["startTransition"]) + transition_offset,
                    "sourceIndex": source_index,
                }
            )
        sources.append(
            {
                "index": source_index,
                "datasetDigest": manifest["datasetDigest"],
                "transitions": manifest["transitions"],
                "teacher": manifest.get("teacher"),
                "rolloutCheckpointDigest": manifest.get("rolloutCheckpointDigest"),
                "sourceSpecDigest": manifest.get("sourceSpecDigest"),
                "sourcePlanRolloutDigest": manifest.get("sourcePlanRolloutDigest"),
                "sourceIdentity": _source_identity(manifest),
                "splitSeeds": manifest.get("splitSeeds"),
                "droppedFields": sorted(field_sets[source_index] - retained_fields),
            }
        )
        episode_offset += len(manifest["episodes"])

    result = writer.finish(
        {
            "name": "snowgym-trajectory-aggregate",
            "teacher": "ordered-aggregate.v0",
            "split": reference["split"],
            "splitSeeds": combined_split_seeds,
            "sourceSpecDigest": _source_identity(reference) if not independent_sources else json_digest(
                [_source_identity(manifest) for manifest in manifests]
            ),
            "maxTeamUnits": reference["maxTeamUnits"],
            "versions": reference["versions"],
            **({"planConditioned": True} if plan_conditioned else {}),
            "independentSources": independent_sources,
            "retainedFields": sorted(retained_fields),
            "sources": sources,
            "episodes": episodes,
        }
    )
    audit_dataset(output)
    return result


def _dataset_fields(root: Path, manifest: dict[str, Any]) -> set[str]:
    fields: set[str] | None = None
    for shard in manifest["shards"]:
        with np.load(root / shard["path"], allow_pickle=False) as archive:
            current = set(archive.files)
        if fields is None:
            fields = current
        elif fields != current:
            raise ValueError("tensor fields change between shards")
    if not fields:
        raise ValueError("trajectory dataset has no tensor fields")
    return fields


def _source_identity(manifest: dict[str, Any]) -> str:
    for key in ("sourceSpecDigest", "sourcePlanRolloutDigest", "datasetDigest"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("trajectory source has no audited identity")


def _combined_split_seeds(
    manifests: list[dict[str, Any]], independent_sources: bool
) -> dict[str, list[int]]:
    if not independent_sources:
        return manifests[0]["splitSeeds"]
    combined = {name: [] for name in ("train", "validation", "evaluation")}
    seen: set[int] = set()
    for manifest in manifests:
        for split, seeds in manifest["splitSeeds"].items():
            for seed in seeds:
                if seed in seen:
                    raise ValueError(f"independent trajectory sources overlap seed {seed}")
                seen.add(seed)
                combined[split].append(seed)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge audited SnowGym trajectories")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--independent-sources", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = merge_datasets(
            output=args.output, inputs=args.input, shard_size=args.shard_size,
            independent_sources=args.independent_sources,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "ok": True,
        "format": result["format"],
        "sources": len(result["sources"]),
        "transitions": result["transitions"],
        "datasetDigest": result["datasetDigest"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True) if args.json else summary)


if __name__ == "__main__":
    main()
