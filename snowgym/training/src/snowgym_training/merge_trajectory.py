"""Deterministically aggregate audited SnowGym trajectory datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import TrajectoryWriter, audit_dataset


def merge_datasets(
    *, output: str | Path, inputs: list[str | Path], shard_size: int = 256
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
    for manifest in manifests[1:]:
        for key in ("split", "splitSeeds", "sourceSpecDigest", "maxTeamUnits", "versions"):
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"trajectory inputs disagree on {key}")

    writer = TrajectoryWriter(output, shard_size=shard_size)
    episodes: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    episode_offset = 0
    for source_index, (root, manifest) in enumerate(zip(roots, manifests, strict=True)):
        transition_offset = writer.transition_count
        for shard in manifest["shards"]:
            with np.load(root / shard["path"], allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
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
            }
        )
        episode_offset += len(manifest["episodes"])

    result = writer.finish(
        {
            "name": "snowgym-trajectory-aggregate",
            "teacher": "ordered-aggregate.v0",
            "split": reference["split"],
            "splitSeeds": reference["splitSeeds"],
            "sourceSpecDigest": reference["sourceSpecDigest"],
            "maxTeamUnits": reference["maxTeamUnits"],
            "versions": reference["versions"],
            "sources": sources,
            "episodes": episodes,
        }
    )
    audit_dataset(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge audited SnowGym trajectories")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = merge_datasets(
            output=args.output, inputs=args.input, shard_size=args.shard_size
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
