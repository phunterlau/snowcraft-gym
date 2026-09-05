"""Fail-closed ancestry and fresh-seed preflight before R1l collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..checkpoint import load_checkpoint
from ..ppo_checkpoint import load_ppo_checkpoint
from ..trajectory import audit_dataset, json_digest
from .geometry_probe import load_probe
from .opportunity_audit import REFERENCE_DIGEST, RESERVOIR_DIGEST
from .reservoir import file_digest, load_teacher_bc_reservoir
from .supervised_probe import write_json

TRAINING_ROOT = Path(__file__).resolve().parents[3]
RESERVED = {"teacherRegression": [108000, 108039], "learnerValidation": [108100, 108139],
            "replicationDevelopment": [210000, 210039]}


def reserve_seeds(seen, reserved):
    claimed = set()
    conflicts = {}
    for name, interval in reserved.items():
        if (len(interval) != 2 or any(type(v) is not int for v in interval)
                or interval[0] > interval[1]):
            raise ValueError("invalid reserved seed interval")
        values = set(range(interval[0], interval[1]+1))
        if claimed & values:
            raise ValueError("reserved seed blocks overlap")
        claimed |= values
        collisions = sorted(set(seen) & values)
        if collisions:
            conflicts[name] = collisions
    return conflicts


def load_ancestor(path):
    metadata = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else json.loads((path / "checkpoint.json").read_text())
    kind = metadata.get("format")
    if kind == "snowgym.checkpoint.v0":
        return load_checkpoint(path)[0]
    if kind == "snowgym.ppo-checkpoint.v0":
        return load_ppo_checkpoint(path)[0]
    if kind == "snowgym.geometry-probe-checkpoint.v0":
        return load_probe(path)[1]
    raise ValueError(f"unsupported ancestor checkpoint format: {kind}")


def audit_ancestry(*, checkpoint, reservoir_path, checkpoint_root, dataset_roots):
    """Require original digest-matching datasets, never infer from a descendant.

    A descendant's source record is useful recovery evidence but cannot establish
    that the original manifest and shards pass the repository's dataset audit.
    """
    reference = load_ancestor(Path(checkpoint))
    reservoir = load_teacher_bc_reservoir(reservoir_path)
    if reference["checkpointDigest"] != REFERENCE_DIGEST or reservoir.metadata["digest"] != RESERVOIR_DIGEST:
        raise ValueError("R1l preflight requires the frozen R1i reference and teacher reservoir")
    registry = {}
    for name in ("metadata.json", "checkpoint.json"):
        for path in sorted(Path(checkpoint_root).rglob(name)):
            value = json.loads(path.read_text())
            digest = value.get("checkpointDigest")
            if digest:
                registry.setdefault(digest, []).append(path.parent)
    registry.setdefault(reference["checkpointDigest"], []).insert(0, Path(checkpoint))
    datasets, indirect = {}, {}
    for directory in dataset_roots:
        for path in sorted(Path(directory).rglob("manifest.json")):
            value = json.loads(path.read_text())
            if value.get("datasetDigest"):
                datasets.setdefault(value["datasetDigest"], []).append(path)
            for source in value.get("sources", []):
                if source.get("datasetDigest"):
                    indirect.setdefault(source["datasetDigest"], []).append({
                        "path": str(path), "manifestFileDigest": file_digest(path),
                        "descendantDatasetDigest": value.get("datasetDigest"),
                        "sourceRecord": source, "acceptedAsOriginal": False})
    with np.load(reservoir_path, allow_pickle=False) as arrays:
        seen = set(int(v) for v in arrays["episode_seed"])
    pending = [reference["checkpointDigest"]]
    checked, data_checked, missing = {}, {}, []
    while pending:
        digest = pending.pop()
        if digest in checked:
            continue
        paths = registry.get(digest, [])
        if not paths:
            missing.append({"kind": "checkpoint", "digest": digest})
            continue
        path = paths[0]
        value = load_ancestor(path)
        if value["checkpointDigest"] != digest:
            raise ValueError("ancestor registry digest mismatch")
        checked[digest] = {"path": str(path), "format": value["format"],
                           "metadataFileDigest": file_digest(path / ("metadata.json" if (path / "metadata.json").exists() else "checkpoint.json"))}
        schedule = value.get("seedSchedule")
        if schedule is not None:
            minimum, maximum, following = (schedule[k] for k in ("minimum", "maximum", "nextSeed"))
            if not all(type(v) is int for v in (minimum, maximum, following)) or not minimum <= following <= maximum+1:
                raise ValueError("invalid ancestor seed schedule")
            seen.update(range(minimum, following))
            checked[digest]["consumedSeedInterval"] = [minimum, following-1]
        for parent in (value.get("source", {}).get("checkpointDigest"),
                       value.get("initialization", {}).get("checkpointDigest"),
                       value.get("initializerSourceDigest")):
            if parent and parent not in checked:
                pending.append(parent)
        dataset_digest = value.get("datasetManifestHash")
        if dataset_digest and dataset_digest not in data_checked:
            candidates = datasets.get(dataset_digest, [])
            if not candidates:
                missing.append({"kind": "dataset", "digest": dataset_digest,
                                "requiredBy": digest, "indirectRecoveryEvidence": indirect.get(dataset_digest, [])})
                continue
            manifest_path = candidates[0]
            audit_dataset(manifest_path.parent)
            manifest = json.loads(manifest_path.read_text())
            episode_seeds = [e["seed"] for e in manifest["episodes"]]
            if not episode_seeds or any(type(v) is not int for v in episode_seeds):
                raise ValueError("ancestor dataset lacks explicit episode seed evidence")
            seen.update(episode_seeds)
            data_checked[dataset_digest] = {"path": str(manifest_path),
                "manifestFileDigest": file_digest(manifest_path), "episodeSeeds": sorted(set(episode_seeds))}
    collisions = reserve_seeds(seen, RESERVED)
    return {"format": "snowgym.recovery-lineage-preflight.v0", "passed": not missing and not collisions,
            "checkpointDigest": reference["checkpointDigest"], "reservoir": reservoir.metadata,
            "checkedCheckpoints": checked, "checkedDatasets": data_checked,
            "missing": missing, "collisions": collisions, "reserved": RESERVED,
            "knownTrainingSeeds": sorted(seen), "freshHoldoutsCollected": False,
            "r1lTrainingUpdates": 0, "qualificationEligible": False,
            "remediation": "Restore each original missing dataset manifest and all its shards, then rerun this preflight. Do not substitute a descendant source record or change the reserved seeds to bypass missing ancestry."}


def run_preflight(*, checkpoint, reservoir_path, output, checkpoint_root=TRAINING_ROOT / "runs", dataset_roots=None):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite lineage preflight {destination}")
    report = audit_ancestry(checkpoint=checkpoint, reservoir_path=reservoir_path, checkpoint_root=checkpoint_root,
                            dataset_roots=dataset_roots or [TRAINING_ROOT / "artifacts"])
    destination.mkdir(parents=True)
    write_json(destination / "report.json", report)
    manifest = {"format": "snowgym.recovery-lineage-run.v0", "sourceFileDigest": file_digest(Path(__file__)),
                "artifacts": {"report.json": file_digest(destination / "report.json")}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(destination / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reservoir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", default=str(TRAINING_ROOT / "runs"))
    parser.add_argument("--dataset-root", action="append")
    args = parser.parse_args()
    report = run_preflight(checkpoint=args.checkpoint, reservoir_path=args.reservoir, output=args.output,
                           checkpoint_root=args.checkpoint_root, dataset_roots=args.dataset_root)
    print(json.dumps({"passed": report["passed"], "missing": report["missing"], "collisions": report["collisions"]}), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
