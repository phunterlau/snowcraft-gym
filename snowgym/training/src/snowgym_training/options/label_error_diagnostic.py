"""R1n-g: label-error diagnostic for why blue's offense fails against scripted red.

No training, no PPO, no new checkpoint. Reuses `full_authority_imitation.py`'s
`Labeler`/`collect`/`label_error` and `opponent_transfer.py`'s
`scenario_override`/`load_initializer`/`load_final`, all unchanged, to measure
the six R1n-c/R1n-e frozen policies' label error against three opponent arms
(random, scripted easy, scripted normal). See `reviews/m7b_r1n_g_declaration.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_imitation as fi
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, write_episodes
from .opponent_transfer import load_final, load_initializer, scenario_override
from .pre_ppo_diagnostics import verify_source_run
from .reservoir import file_digest
from .supervised_probe import write_json

COMPARATOR_NAMES = ("init-97101", "init-97102", "init-97103",
                     "final-97101", "final-97102", "final-97103")
ASSUMED_DECISIONS_PER_EPISODE = 110  # conservative overestimate; declaration §6


def configuration():
    return {**fi.configuration(), "format": "snowgym.r1n-g-label-error-config.v0",
        "initializerSeeds": [97101, 97102, 97103],
        "sourceRun": "runs/m7b_engage_r1n_c_v0", "policyCheckpoint": "fit-4.pt",
        "finalRun": "runs/m7b_engage_r1n_e_v0", "finalCheckpoint": "update-200.pt",
        "sigmaScale": .5,
        "armOrder": ["R", "E", "N"],
        "arms": {"R": {"redController": "random"},
                 "E": {"redController": "scripted", "redDifficulty": "easy"},
                 "N": {"redController": "scripted", "redDifficulty": "normal"}},
        "armSeeds": {"R": [994000, 994127], "E": [995000, 995127], "N": [996000, 996127]},
        "episodesPerArm": 128, "collectionBlockWorlds": 64,
        "simulatorBudget": 260000, "budgetCap": 400000,
        "assistType": "none at runtime; frozen checkpoints, no training",
        "assistVersion": "snowgym.r1n-g-label-error.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    cells = len(COMPARATOR_NAMES) * len(cfg["armOrder"])
    total = cells * cfg["episodesPerArm"] * ASSUMED_DECISIONS_PER_EPISODE
    return {"cells": cells, "episodesPerArm": cfg["episodesPerArm"],
            "assumedDecisionsPerEpisode": ASSUMED_DECISIONS_PER_EPISODE, "total": total}


# -- Per-cell collection and measurement (declaration §1, §2) -----------------------------


def collect_cell(client, cfg, model, seeds, source, account):
    """One (comparator, arm) cell: the frozen policy acts deterministically while the
    teacher labels every visited state (`fi.collect`, unedited); `label_error` (unedited)
    scores the policy's heads against those labels. `fi.collect` calls `account` itself,
    once per collected block."""
    episodes, part = fi.collect(client, seeds, cfg, model=model, source=source,
                                block_worlds=cfg["collectionBlockWorlds"], account=account,
                                deterministic=True, record=True)
    errors = fi.label_error(model, part, cfg)
    return episodes, errors


def run_arm(root, cfg, arm):
    root = Path(root)
    directory = root / f"arm-{arm}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    low, high = cfg["armSeeds"][arm]
    seeds = list(range(low, high + 1))
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("R1n-g budget exceeded")

    loaders = {"init": load_initializer, "final": load_final}
    cells = {}
    with SnowGymBatchClient() as client, scenario_override(cfg["arms"][arm]):
        for index, seed in enumerate(cfg["initializerSeeds"]):
            for prefix, loader in loaders.items():
                name = f"{prefix}-{seed}"
                model = loader(cfg, index)
                episodes, errors = collect_cell(client, cfg, model, seeds, f"{arm}-{name}", account)
                write_episodes(directory / name, episodes)
                write_json(directory / name / "label-error.json", errors)
                cells[name] = errors
    write_json(directory / "arm-report.json", {"arm": arm, "cells": cells, "simulatorDecisions": steps})
    dr.seal(directory, "snowgym.r1n-g-arm-manifest.v0")
    return cells


# -- Cross-checks and localization (declaration §2, §4) ------------------------------------


def _delta(current, archived):
    """None-propagating difference: at full scale (128 episodes) every measure below has
    enough support to be a real number, but the tiny test configuration can starve a rare
    label (e.g. zero throws in a handful of episodes) to `None`, which must not be silently
    read as "zero divergence"."""
    return None if current is None or archived is None else current - archived


def archive_cross_check(cfg, arm_r_cells):
    """Arm R's initializer cells against R1n-c's own archived split-B label error for the
    same checkpoints — different worlds, same opponent, same checkpoint. A large divergence
    flags a measurement problem in this diagnostic, not a new finding (declaration §2, §8).
    Finals have no R1n-c archive (they postdate R1n-c), so only initializers are checked."""
    rows = {}
    for seed in cfg["initializerSeeds"]:
        archived = json.loads((TRAINING / cfg["sourceRun"] / f"seed-{seed}" / "label-error.json")
                              .read_text(encoding="utf-8"))
        current = arm_r_cells[f"init-{seed}"]
        rows[str(seed)] = {"archived": archived, "current": current,
            "typeAccuracyDelta": _delta(current["typeAccuracy"], archived["typeAccuracy"]),
            "throwRecallDelta": _delta(current["perType"]["throw"]["recall"], archived["perType"]["throw"]["recall"]),
            "throwAimHeadingErrorDegreesDelta": _delta(current["throwAimHeadingErrorDegrees"],
                                                        archived["throwAimHeadingErrorDegrees"])}
    return rows


def throw_recall_flags(cells_by_arm, cfg):
    """Type/detection candidate (declaration §2, §3): mean throw recall across the six
    comparators, per scripted arm, against R1n-c's own `throwRecallFlag` threshold — the
    only numeric flag this diagnostic pre-declares. Aim and power/range are reported as raw
    deltas (see `metric_deltas`), not thresholded, since no other numeric flag was declared."""
    flags = {}
    for arm in ("E", "N"):
        recalls = [cells_by_arm[arm][name]["perType"]["throw"]["recall"] for name in COMPARATOR_NAMES]
        recalls = [r for r in recalls if r is not None]
        mean_recall = float(np.mean(recalls)) if recalls else None
        flags[arm] = {"meanThrowRecall": mean_recall,
                      "belowThrowRecallFlag": mean_recall is not None and mean_recall < cfg["throwRecallFlag"]}
    return flags


def metric_deltas(cells_by_arm):
    """Arm E/N minus arm R, per metric, averaged over the six comparators (declaration §2's
    primary comparison). `support` is carried alongside every error mean so a delta computed
    on a handful of throw-labelled rows is never read like one computed on thousands."""
    metrics = ("typeAccuracy", "moveHeadingErrorDegrees", "moveEndpointErrorWorld",
               "throwAimHeadingErrorDegrees", "powerMeanAbsoluteError")
    out = {}
    for arm in ("E", "N"):
        row = {}
        for metric in metrics:
            reference = [cells_by_arm["R"][name][metric] for name in COMPARATOR_NAMES]
            scripted = [cells_by_arm[arm][name][metric] for name in COMPARATOR_NAMES]
            paired = [(r, s) for r, s in zip(reference, scripted) if r is not None and s is not None]
            row[metric] = {"armRMean": float(np.mean([p[0] for p in paired])) if paired else None,
                          "armMean": float(np.mean([p[1] for p in paired])) if paired else None,
                          "delta": float(np.mean([p[1] - p[0] for p in paired])) if paired else None,
                          "cells": len(paired)}
        row["throwSupport"] = {name: cells_by_arm[arm][name]["perType"]["throw"]["support"]
                               for name in COMPARATOR_NAMES}
        out[arm] = row
    return out


# -- Declaration and run (declaration §1, §5, §8) -------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    source_run = verify_source_run({**cfg, "policySeeds": cfg["initializerSeeds"]})
    final_run = dr.verify_sealed(TRAINING / cfg["finalRun"])
    root.mkdir(parents=True)
    here = Path(__file__).resolve()
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_g_declaration.md"),
        "implementationDigest": file_digest(here),
        "imitationImplementationDigest": file_digest(here.parent / "full_authority_imitation.py"),
        "opponentTransferImplementationDigest": file_digest(here.parent / "opponent_transfer.py"),
        "sourceRun": source_run, "finalRunManifest": final_run, "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    arm_manifests = {arm: dr.verify_sealed(root / f"arm-{arm}") for arm in cfg["armOrder"]}
    cells_by_arm = {arm: json.loads((root / f"arm-{arm}" / "arm-report.json").read_text(encoding="utf-8"))["cells"]
                    for arm in cfg["armOrder"]}
    report = {"format": "snowgym.r1n-g-label-error-report.v0",
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "armManifests": arm_manifests, "cells": cells_by_arm,
        "archiveCrossCheck": archive_cross_check(cfg, cells_by_arm["R"]),
        "throwRecallFlags": throw_recall_flags(cells_by_arm, cfg),
        "metricDeltas": metric_deltas(cells_by_arm),
        "simulatorDecisions": sum(json.loads((root / f"arm-{arm}" / "arm-report.json").read_text(encoding="utf-8"))
                                  ["simulatorDecisions"] for arm in cfg["armOrder"])}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.r1n-g-manifest.v0")
    return report


def run(output):
    cfg = configuration()
    declare(output, cfg)
    for arm in cfg["armOrder"]:
        run_arm(output, cfg, arm)
    return aggregate(output, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "declare", "arm", "aggregate"], default="all")
    parser.add_argument("--arm", choices=["R", "E", "N"])
    arguments = parser.parse_args()
    if arguments.stage == "all":
        run(arguments.output)
    elif arguments.stage == "declare":
        declare(arguments.output, configuration())
    elif arguments.stage == "arm":
        if arguments.arm is None:
            raise SystemExit("--arm is required for --stage arm")
        run_arm(arguments.output, configuration(), arguments.arm)
    else:
        aggregate(arguments.output, configuration())
