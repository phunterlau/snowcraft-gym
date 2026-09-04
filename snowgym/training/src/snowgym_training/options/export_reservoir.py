"""Export a successful production-teacher BC reservoir on training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from snowgym_client.batch import SnowGymBatchClient, SnowGymBatchEnv

from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .environment import FixedPlanOptionBatchEnv
from .interventions import require_capabilities
from .plans import teacher_option_plan, teacher_option_scenario
from .protocol import load_option_protocol
from .reservoir import file_digest


def export_teacher_bc_reservoir(
    *, output: str | Path, seed_count: int = 40
) -> dict[str, Any]:
    if not isinstance(seed_count, int) or isinstance(seed_count, bool) or not 1 <= seed_count <= 100:
        raise ValueError("teacher reservoir seed_count must be in [1,100]")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite teacher reservoir {destination}")
    protocol = load_option_protocol()
    start = int(protocol["seeds"]["training"][0])
    seeds = list(range(start, start + seed_count))
    plan, spec = teacher_option_plan("engage")
    scenario = teacher_option_scenario("engage")
    observation_rows: dict[str, list[np.ndarray]] = {}
    action_rows: dict[str, list[np.ndarray]] = {
        "teacher_action_type": [], "teacher_target": [], "teacher_power": []
    }
    decisions: list[int] = []
    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        for seed in seeds:
            base = SnowGymBatchEnv(1, client=client, observation_version=3)
            wrapped = FixedPlanOptionBatchEnv(base, gamma=0.9976921765)
            observation, _ = wrapped.reset(
                [seed], [scenario], [f"engage-r1-teacher-{seed}"], [plan], [spec]
            )
            completed = False
            for decision in range(spec.horizon):
                teacher = base.plan_teacher_tensor_actions()
                for name, value in observation.items():
                    observation_rows.setdefault(f"observation__{name}", []).append(
                        np.array(value[0], copy=True)
                    )
                action_rows["teacher_action_type"].append(
                    np.array(teacher["action_type"][0], copy=True)
                )
                action_rows["teacher_target"].append(
                    np.array(teacher["target"][0], copy=True)
                )
                action_rows["teacher_power"].append(
                    np.array(teacher["power"][0], copy=True)
                )
                observation, _, terminated, truncated, infos = wrapped.step(
                    teacher=True
                )
                if bool(terminated[0] or truncated[0]):
                    if not bool(infos[0]["option"]["success"]):
                        raise RuntimeError(
                            f"production teacher failed Engage training seed {seed}"
                        )
                    decisions.append(decision + 1)
                    completed = True
                    break
            if not completed:
                raise RuntimeError(f"teacher reservoir seed {seed} did not complete")
    destination.mkdir(parents=True)
    artifact = destination / "teacher_states.npz"
    np.savez_compressed(
        artifact,
        **{
            name: np.stack(rows)
            for name, rows in {**observation_rows, **action_rows}.items()
        },
        episode_success=np.ones(seed_count, dtype=np.bool_),
        episode_seed=np.asarray(seeds, dtype=np.int64),
        episode_decisions=np.asarray(decisions, dtype=np.int64),
    )
    manifest = {
        "format": "snowgym.teacher-bc-reservoir-export.v0",
        "implementationGitCommit": resolve_git_commit(),
        "simulationVersion": capabilities["simulationVersion"],
        "stateHashVersion": capabilities["stateHashVersion"],
        "upstreamBaseCommit": capabilities["upstreamBaseCommit"],
        "protocolDigest": json_digest(protocol),
        "seedPartition": "training",
        "seeds": seeds,
        "episodes": seed_count,
        "samples": sum(decisions),
        "allSuccessful": True,
        "artifacts": {"teacher_states.npz": file_digest(artifact)},
    }
    manifest["manifestDigest"] = json_digest(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    result = export_teacher_bc_reservoir(
        output=args.output, seed_count=args.seed_count
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
