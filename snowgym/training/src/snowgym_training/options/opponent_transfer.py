"""R1n-f: opponent-transfer evaluation of R1n-e's frozen policies. No training.

Evaluates the R1n-c initializers, the R1n-e finals, and the scripted blue teacher
against three opponents (random, scripted easy, scripted normal) on a fresh
world split, deterministic execution only. See
`reviews/m7b_r1n_f_declaration.md`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import death_rate_ppo as dr
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .full_authority_imitation import paired_difference
from .interventions import require_capabilities
from .plans import teacher_option_plan
from .pre_ppo_diagnostics import MODES, mode_chooser, verify_source_run
from .reservoir import file_digest
from .supervised_probe import write_json

COMPARATOR_NAMES = ("init-97101", "init-97102", "init-97103",
                     "final-97101", "final-97102", "final-97103", "teacher")


def configuration():
    return {**v1.configuration(), "format": "snowgym.opponent-transfer-config.v0", "destination": "global",
        "initializerSeeds": [97101, 97102, 97103],
        "sourceRun": "runs/m7b_engage_r1n_c_v0", "policyCheckpoint": "fit-4.pt",
        "finalRun": "runs/m7b_engage_r1n_e_v0", "finalCheckpoint": "update-200.pt",
        "sigmaScale": .5,
        "armOrder": ["R", "E", "N"],
        "arms": {"R": {"redController": "random"},
                 "E": {"redController": "scripted", "redDifficulty": "easy"},
                 "N": {"redController": "scripted", "redDifficulty": "normal"}},
        "evaluationSeeds": [871000, 871399], "evaluationBlockWorlds": 50,
        "reproductionSeeds": [870000, 870049], "reproductionInitializer": 97101,
        "bootstrapSeed": 983001, "bootstrapSamples": 10000,
        "teacherMinSuccess": .80, "floorMaxSuccess": .05,
        "failThreshold": -.50, "partialThreshold": -.10,
        "referenceSuccessRange": [.88, .96], "referenceDeathRange": [.03, .11],
        "simulatorBudget": 1700000, "budgetCap": 2000000,
        "assistType": "none at runtime; frozen checkpoints, no training",
        "assistVersion": "snowgym.opponent-transfer.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    worlds = cfg["evaluationSeeds"][1] - cfg["evaluationSeeds"][0] + 1
    per_arm = len(COMPARATOR_NAMES) * worlds * horizon
    reproduction = (cfg["reproductionSeeds"][1] - cfg["reproductionSeeds"][0] + 1) * horizon
    total = per_arm * len(cfg["arms"]) + reproduction
    return {"perArm": per_arm, "reproduction": reproduction, "total": total}


# -- Opponent override, without editing full_authority_train_v1.py ----------------------


@contextlib.contextmanager
def scenario_override(arm_spec):
    """Override `full_authority_train_v1.scenario` for one arm's `redController` /
    `redDifficulty` while this context is active (declaration §2). `v1.py` is not
    edited; only its module-level name binding is swapped, and restored on exit even if
    collection raises."""
    original = v1.scenario

    def overridden():
        return {**original(), **arm_spec}

    v1.scenario = overridden
    try:
        yield
    finally:
        v1.scenario = original


# -- Policy loading -----------------------------------------------------------------------


def load_initializer(cfg, index):
    """R1n-c's fit-4 checkpoint, loaded exactly as R1n-e loaded it (§1): the frozen
    reference copy `death_rate_ppo.prepare_policy` returns, including the sigma-scale
    log-std shift. Deterministic execution never samples, so sigma has no effect here;
    the load path is kept identical only so the initializer arm is the same object R1n-e
    started PPO from."""
    _, reference = dr.prepare_policy(cfg, index)
    return reference


def load_final(cfg, index):
    """R1n-e's `update-200.pt`. Its log-stds are already sigma-scaled and were frozen
    throughout training, so no further adjustment is made."""
    model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
                                  target_world_sigma=cfg["targetWorldSigma"],
                                  initial_power_log_std=cfg["initialPowerLogStd"])
    seed = cfg["initializerSeeds"][index]
    path = TRAINING / cfg["finalRun"] / f"policy-{seed}" / cfg["finalCheckpoint"]
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.eval().requires_grad_(False)


# -- Logged collection (declaration §4, §7) ------------------------------------------------


def collect_logged(wrapper, seeds, cfg, *, choose, source):
    """`full_authority_train_v1.run_block`'s loop, reimplemented (not called) so every
    decision's raw blue/red/projectile state can be logged for the projectile-level
    mechanism analysis. `choose=None` steps the scripted teacher, exactly as `run_block`
    does. Calls `v1.scenario()` for the reset, so `scenario_override` governs this path
    the same way it governs any future direct use of `run_block`."""
    count = len(seeds)
    if wrapper.batch_size != count:
        raise ValueError("block seeds must match the wrapper batch size")
    plan, _ = teacher_option_plan("engage")
    spec = v1.engage_spec(cfg)
    observation, _ = wrapper.reset(list(seeds), [v1.scenario()] * count,
                                   [f"r1n-f-{source}-{s}" for s in seeds], [plan] * count, [spec] * count)
    current = v1.tensor_dict(observation)
    episodes = [{"seed": int(seed), "source": source, "rewards": [], "remainingFraction": [],
                 "distances": [], "targetDamage": [], "rejectedActions": 0, "totalActions": 0}
                for seed in seeds]
    logs = {int(seed): [] for seed in seeds}
    decisions = 0
    while True:
        active = [i for i in range(count) if not wrapper.trackers[i].finished]
        if not active:
            break
        rows = {key: value[active] for key, value in current.items()}
        raws = [wrapper.environment.raw_observations[i] for i in active]
        distances = [v1.target_distance(raw, wrapper.trackers[i]) for raw, i in zip(raws, active)]
        for i, raw in zip(active, raws):
            b, r = raw["allies"][0], raw["enemies"][0]
            logs[seeds[i]].append((b["health"], r["health"], b["x"], b["y"], b["vx"], b["vy"],
                                   r["x"], r["y"], r["vx"], r["vy"],
                                   tuple((p["id"], p["team"], p["x"], p["y"]) for p in raw["projectiles"])))
        if choose is None:
            raw_next, rewards, _, _, infos = wrapper.step_scripted_indices(active)
        else:
            raw_next, rewards, _, _, infos = wrapper.step_indices(active, choose(wrapper, active, rows, raws))
        decisions += len(active)
        following = v1.tensor_dict(raw_next)
        for key in current:
            current[key][active] = following[key]
        for row, index in enumerate(active):
            episode, option = episodes[index], infos[row]["option"]
            episode["remainingFraction"].append(float(rows["option_state"][row, 0]))
            episode["rewards"].append(float(rewards[row]))
            episode["distances"].append(distances[row])
            episode["targetDamage"].append(float(option["metrics"]["targetDamage"]))
            results = infos[row].get("actionResults", [])
            episode["rejectedActions"] += sum(r.get("accepted") is False for r in results)
            episode["totalActions"] += len(results)
            if wrapper.trackers[index].finished:
                allies = wrapper.environment.raw_observations[index]["allies"]
                episode.update(success=bool(option["success"]), failed=bool(option["failed"]),
                    timedOut=bool(option["timedOut"]), finalDecision=int(option["decision"]),
                    blueAliveAtEnd=any(u["alive"] and u["id"] in wrapper.trackers[index].assigned_ids
                                       for u in allies))
    return episodes, logs, decisions


def collect_comparator(client, seeds, cfg, *, scripted, choose, source, block_worlds, account):
    episodes, logs, total = [], {}, 0
    for start in range(0, len(seeds), block_worlds):
        block = seeds[start:start + block_worlds]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"], scripted=scripted)
        found, block_logs, used = collect_logged(wrapper, block, cfg, choose=choose, source=source)
        account(used)
        total += used
        episodes.extend(found)
        logs.update(block_logs)
    return episodes, logs, total


# -- Projectile attribution and mechanism counters (declaration §4, §7) -----------------


def shots_from_log(row, log, team):
    """Every projectile of `team` seen in the log, from its first to last appearance:
    spawn decision, spawn distance, the *target*'s speed and displacement over the
    projectile's lifetime, and whether it is the hit attributed to a health drop at its
    last-seen decision (the nearest still-present same-team projectile to the target,
    matching `m7b_r1n_e_results.md` §7's attribution rule)."""
    first, last = {}, {}
    for t, step in enumerate(log):
        for pid, pteam, _px, _py in step[10]:
            if pteam != team:
                continue
            first.setdefault(pid, t)
            last[pid] = t
    victim_health_index = 0 if team == "red" else 1  # red throws at blue (index 0); blue throws at red (index 1)
    by_end = {}
    for pid, t1 in last.items():
        by_end.setdefault(t1, []).append(pid)
    used_hits = set()
    out = []
    for pid, t0 in first.items():
        t1 = last[pid]
        spawn = log[t0]
        bx, by, bvx, bvy = spawn[2], spawn[3], spawn[4], spawn[5]
        rx, ry, rvx, rvy = spawn[6], spawn[7], spawn[8], spawn[9]
        spawn_distance = math.hypot(bx - rx, by - ry)
        if team == "red":
            target_speed, target_start, target_end = math.hypot(bvx, bvy), (bx, by), (log[t1][2], log[t1][3])
        else:
            target_speed, target_start, target_end = math.hypot(rvx, rvy), (rx, ry), (log[t1][6], log[t1][7])
        displacement = math.hypot(target_end[0] - target_start[0], target_end[1] - target_start[1])
        if t1 + 1 < len(log):
            dropped = log[t1 + 1][victim_health_index] < log[t1][victim_health_index]
        else:
            dropped = (not row["blueAliveAtEnd"]) if team == "red" else bool(row["success"])
        hit = False
        if dropped and t1 not in used_hits:
            candidates = [(math.hypot(px - target_end[0], py - target_end[1]), p)
                          for p, pteam, px, py in log[t1][10] if p in by_end.get(t1, []) and pteam == team]
            if candidates and min(candidates)[1] == pid:
                hit = True
                used_hits.add(t1)
        out.append({"spawnDecision": t0, "spawnDistance": spawn_distance, "targetSpeedAtSpawn": target_speed,
                    "displacement": displacement, "flightDecisions": t1 - t0 + 1, "hit": hit})
    return out


def health_drop_events(row, log, team):
    """Number of decisions where `team`'s health fell, including a hit that coincides with
    the episode's final logged decision (declaration §4's unattributed-drop check; mirrors
    `shots_from_log`'s terminal-hit fallback, since the log has no state after the episode
    ends to compare against)."""
    index = 0 if team == "blue" else 1
    within = sum(1 for a, b in zip(log, log[1:]) if b[index] < a[index])
    terminal = (not row["blueAliveAtEnd"]) if team == "blue" else bool(row["success"])
    return within + (1 if terminal else 0)


def mechanism_summary(episodes, logs):
    rows = {e["seed"]: v1.episode_row(e) for e in episodes}
    red_shots, blue_shots = [], []
    red_drops, red_hits, blue_drops, blue_hits = 0, 0, 0, 0
    for seed, log in logs.items():
        row = rows[seed]
        red = shots_from_log(row, log, "red")
        blue = shots_from_log(row, log, "blue")
        red_shots += red
        blue_shots += blue
        red_drops += health_drop_events(row, log, "blue")   # blue's health drops = red's hits
        blue_drops += health_drop_events(row, log, "red")   # red's health drops = blue's hits
        red_hits += sum(s["hit"] for s in red)
        blue_hits += sum(s["hit"] for s in blue)
    n = len(episodes)

    def rate(shots):
        return sum(s["hit"] for s in shots) / len(shots) if shots else None

    return {"episodes": n,
        "redProjectilesPerEpisode": len(red_shots) / n if n else None,
        "redSpawnDistanceMedian": float(np.median([s["spawnDistance"] for s in red_shots])) if red_shots else None,
        "redHitRate": rate(red_shots),
        "blueDisplacementDuringRedFlightMedian": float(np.median([s["displacement"] for s in red_shots]))
            if red_shots else None,
        "blueProjectilesPerEpisode": len(blue_shots) / n if n else None,
        "blueSpawnDistanceMedian": float(np.median([s["spawnDistance"] for s in blue_shots])) if blue_shots else None,
        "blueHitRate": rate(blue_shots),
        "redDeathAttributedFraction": red_hits / red_drops if red_drops else None,
        "blueDeathAttributedFraction": blue_hits / blue_drops if blue_drops else None}


def failure_label(row):
    """The six declared labels (§4). `firstHitDecision` comes from `episode_row`."""
    if row["success"] and row["blueAliveAtEnd"]:
        return "win"
    if row["success"]:
        return "win-but-dead"
    if not row["blueAliveAtEnd"]:
        return "death"
    if row["timedOut"]:
        return "timeout-with-hits" if row["firstHitDecision"] is not None else "timeout-no-hits"
    return "unresolved"


def outcome_table(episodes):
    rows = [v1.episode_row(e) for e in episodes]
    n = len(rows)
    labels = [failure_label(r) for r in rows]
    counts = {name: labels.count(name) for name in
              ("win", "win-but-dead", "death", "timeout-with-hits", "timeout-no-hits", "unresolved")}
    return {"episodes": n, "successFraction": sum(r["success"] for r in rows) / n if n else None,
        "deathFraction": sum(not r["blueAliveAtEnd"] for r in rows) / n if n else None,
        "timeoutFraction": sum(r["timedOut"] for r in rows) / n if n else None,
        "labels": counts}


def save_comparator(directory, episodes, logs, table, mechanics):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episodes.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text("".join(json.dumps(v1.episode_row(e), sort_keys=True, allow_nan=False) + "\n" for e in episodes),
                    encoding="utf-8")
    write_json(directory / "summary.json", {**table, "mechanism": mechanics})
    rows_by_seed = {e["seed"]: v1.episode_row(e) for e in episodes}
    ids, offsets, spawn, dist, speed, disp, flight, hit, team = [], [0], [], [], [], [], [], [], []
    for seed, log in sorted(logs.items()):
        rows = {"red": shots_from_log(rows_by_seed[seed], log, "red"),
                "blue": shots_from_log(rows_by_seed[seed], log, "blue")}
        count = 0
        for t, shots in rows.items():
            for s in shots:
                spawn.append(s["spawnDecision"]); dist.append(s["spawnDistance"]); speed.append(s["targetSpeedAtSpawn"])
                disp.append(s["displacement"]); flight.append(s["flightDecisions"]); hit.append(s["hit"])
                team.append(t)
                count += 1
        ids.append(seed)
        offsets.append(offsets[-1] + count)
    np.savez_compressed(directory / "shots.npz", seeds=np.asarray(ids), offsets=np.asarray(offsets),
        spawnDecision=np.asarray(spawn), spawnDistance=np.asarray(dist), targetSpeedAtSpawn=np.asarray(speed),
        displacement=np.asarray(disp), flightDecisions=np.asarray(flight), hit=np.asarray(hit),
        team=np.asarray(team))


# -- Reproduction check (declaration §9) -------------------------------------------------


def reproduction_check(client, cfg):
    """Rerun 50 worlds of split E for initializer 97101 under arm R using this module's
    own collection loop, and require an exact match against the archived R1n-e episodes
    on success, `blueAliveAtEnd`, and `finalDecision`. Must pass before any arm collects."""
    seeds = list(range(cfg["reproductionSeeds"][0], cfg["reproductionSeeds"][1] + 1))
    archive_path = (TRAINING / cfg["finalRun"] / f"policy-{cfg['reproductionInitializer']}"
                     / "evaluation" / "initializer-deterministic" / "episodes.jsonl")
    archived = {row["seed"]: row for row in
                (json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines())
                if row["seed"] in seeds}
    index = cfg["initializerSeeds"].index(cfg["reproductionInitializer"])
    reference = load_initializer(cfg, index)
    with scenario_override(cfg["arms"]["R"]):
        episodes, _, used = collect_comparator(client, seeds, cfg, scripted=False,
            choose=mode_chooser(reference, MODES["det"]), source="reproduction",
            block_worlds=cfg["evaluationBlockWorlds"], account=lambda _n: None)
    rows = [v1.episode_row(e) for e in episodes]
    mismatches = [r["seed"] for r in rows if (r["success"], r["blueAliveAtEnd"], r["finalDecision"]) !=
                  (archived[r["seed"]]["success"], archived[r["seed"]]["blueAliveAtEnd"], archived[r["seed"]]["finalDecision"])]
    return {"worlds": len(seeds), "decisions": used, "mismatches": mismatches, "passed": not mismatches}


# -- Per-arm collection, analysis, and decision rules (declaration §3-§5) ---------------


def run_arm(root, cfg, arm):
    root = Path(root)
    directory = root / f"arm-{arm}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    seeds = list(range(cfg["evaluationSeeds"][0], cfg["evaluationSeeds"][1] + 1))
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("R1n-f budget exceeded")

    tables = {}
    with SnowGymBatchClient() as client, scenario_override(cfg["arms"][arm]):
        comparators = {}
        for index, seed in enumerate(cfg["initializerSeeds"]):
            comparators[f"init-{seed}"] = (False, mode_chooser(load_initializer(cfg, index), MODES["det"]))
            comparators[f"final-{seed}"] = (False, mode_chooser(load_final(cfg, index), MODES["det"]))
        comparators["teacher"] = (True, None)
        for name, (scripted, choose) in comparators.items():
            episodes, logs, _ = collect_comparator(client, seeds, cfg, scripted=scripted, choose=choose,
                source=f"{arm}-{name}", block_worlds=cfg["evaluationBlockWorlds"], account=account)
            table = outcome_table(episodes)
            mechanics = mechanism_summary(episodes, logs)
            save_comparator(directory / name, episodes, logs, table, mechanics)
            tables[name] = table
    write_json(directory / "arm-report.json", {"arm": arm, "comparators": tables, "simulatorDecisions": steps})
    dr.seal(directory, "snowgym.opponent-transfer-arm-manifest.v0")
    return tables


# -- Aggregation and decision rules ------------------------------------------------------


def world_paired_difference(first_by_seed, second_by_world, cfg, metric):
    """Seed-averaged `first` minus `second`, resampled over the shared 400 worlds
    (declaration §4: "a 400-world bootstrap 95% interval"), not over the 3 policies.
    `first_by_seed`: {seed: {world: outcomes}}. `second_by_world`: {world: outcomes}
    (the teacher, or another seed-averaged comparator)."""
    worlds = sorted(second_by_world)
    for per_world in first_by_seed.values():
        if sorted(per_world) != worlds:
            raise RuntimeError("evaluation worlds differ across comparators")
    first = [float(np.mean([first_by_seed[s][w][metric] for s in first_by_seed])) for w in worlds]
    second = [second_by_world[w][metric] for w in worlds]
    return paired_difference(first, second, samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])


def arm_analysis(root, cfg, arm):
    directory = Path(root) / f"arm-{arm}"
    report = json.loads((directory / "arm-report.json").read_text(encoding="utf-8"))
    tables = report["comparators"]

    def episode_outcomes(name):
        return dr.outcomes(directory / name / "episodes.jsonl")

    init_outcomes = {s: episode_outcomes(f"init-{s}") for s in cfg["initializerSeeds"]}
    final_outcomes = {s: episode_outcomes(f"final-{s}") for s in cfg["initializerSeeds"]}
    teacher_outcomes = episode_outcomes("teacher")

    def averaged(prefix, metric):
        return float(np.mean([tables[f"{prefix}-{s}"][metric] for s in cfg["initializerSeeds"]]))

    teacher = tables["teacher"]
    analysis = {"teacherSuccess": teacher["successFraction"],
        "initializerSuccessMean": averaged("init", "successFraction"),
        "finalSuccessMean": averaged("final", "successFraction"),
        "finalDeathMean": averaged("final", "deathFraction"),
        "finalLabelCounts": {label: sum(tables[f"final-{s}"]["labels"][label] for s in cfg["initializerSeeds"])
                             for label in tables["teacher"]["labels"]}}
    for outcomes, prefix, label in ((init_outcomes, "init", "initializer"), (final_outcomes, "final", "final")):
        for metric, key in (("success", "Success"), ("death", "Death")):
            analysis[f"{label}{key}GapToTeacher"] = world_paired_difference(
                outcomes, teacher_outcomes, cfg, metric)
    for metric, key in (("success", "Success"), ("death", "Death")):
        analysis[f"finalMinusInitializer{key}"] = world_paired_difference(
            final_outcomes, {w: {metric: float(np.mean([init_outcomes[s][w][metric] for s in init_outcomes]))}
                             for w in next(iter(init_outcomes.values()))}, cfg, metric)
    return analysis


def floor_failure_mode(counts):
    """Amendment A4: which failure dominates the finals' floor, since arm E (timeouts) and
    arm N (deaths) both otherwise collapse to the same "no-transfer-floor" label and the
    whole point of the failure-mode decomposition (declaration §4) was to keep them apart."""
    death = counts["death"] + counts["win-but-dead"]
    timeout = counts["timeout-with-hits"] + counts["timeout-no-hits"]
    if death > timeout:
        return "death"
    if timeout > death:
        return "timeout"
    return "mixed"


def arm_outcome(analysis, cfg):
    if analysis["teacherSuccess"] < cfg["teacherMinSuccess"]:
        return "invalid"
    if analysis["finalSuccessMean"] < cfg["floorMaxSuccess"] and analysis["initializerSuccessMean"] < cfg["floorMaxSuccess"]:
        return f"no-transfer-floor-{floor_failure_mode(analysis['finalLabelCounts'])}"
    gap = analysis["finalSuccessGapToTeacher"]["mean"]
    if gap <= cfg["failThreshold"]:
        return "fails"
    if gap <= cfg["partialThreshold"]:
        return "partial"
    return "transfers"


RECOMMENDATION = {
    "transfers": "R1n-g replication as R1n-e recommended; opponent variety is a later concern.",
    "partial": "R1n-g replication as R1n-e recommended; opponent variety is a later concern.",
    "no-transfer-floor-death": "The generalization loss predates PPO, and the finals mostly die rather than "
        "time out. The next declaration should address the imitation stage: train against a mixture of "
        "opponents, holding one out for evaluation. A death-dominant floor has no learnable gradient at this "
        "difficulty; consider starting the mixture curriculum from an easier opponent.",
    "no-transfer-floor-timeout": "The generalization loss predates PPO, and the finals mostly time out rather "
        "than die: they survive but cannot finish a purposeful opponent. The next declaration should address "
        "the imitation stage with an opponent mixture; this arm's difficulty is a plausible starting point "
        "since a timeout-dominant floor still carries a learnable gradient.",
    "no-transfer-floor-mixed": "The generalization loss predates PPO, with no single dominant failure mode. "
        "The next declaration should address the imitation stage: train against a mixture of opponents, "
        "holding one out for evaluation.",
    "fails": "Same as no-transfer-floor, plus: the R1n-e policy is worse than its initializer against this "
        "opponent, which would make the anchor and sigma choices suspect.",
    "invalid": "Fix the arm before drawing conclusions.",
}


def aggregate(root, cfg):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    arm_manifests = {arm: dr.verify_sealed(root / f"arm-{arm}") for arm in cfg["armOrder"]}
    analyses = {arm: arm_analysis(root, cfg, arm) for arm in cfg["armOrder"]}
    outcomes = {arm: arm_outcome(analyses[arm], cfg) for arm in ("E", "N")}
    reference = analyses["R"]
    reference_in_range = (cfg["referenceSuccessRange"][0] <= reference["finalSuccessMean"] <= cfg["referenceSuccessRange"][1])
    report = {"format": "snowgym.opponent-transfer-report.v0",
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "armManifests": arm_manifests, "analyses": analyses, "outcomes": outcomes,
        "recommendations": {arm: RECOMMENDATION[outcomes[arm]] for arm in outcomes},
        "referenceArmReproducesR1nERange": reference_in_range,
        "simulatorDecisions": sum(json.loads((root / f"arm-{arm}" / "arm-report.json").read_text())["simulatorDecisions"]
                                  for arm in cfg["armOrder"])}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.opponent-transfer-manifest.v0")
    return report


# -- Declaration and run ------------------------------------------------------------------


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
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_f_declaration.md"),
        "implementationDigest": file_digest(here),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "deathRatePpoImplementationDigest": file_digest(here.parent / "death_rate_ppo.py"),
        "sourceRun": source_run, "finalRunManifest": final_run, "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def run(output):
    cfg = configuration()
    declare(output, cfg)
    with SnowGymBatchClient() as client:
        check = reproduction_check(client, cfg)
    write_json(Path(output) / "reproduction-check.json", check)
    if not check["passed"]:
        raise RuntimeError(f"R1n-f reproduction check failed on seeds {check['mismatches']}")
    for arm in cfg["armOrder"]:
        run_arm(output, cfg, arm)
    return aggregate(output, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "declare", "reproduction", "arm", "aggregate"], default="all")
    parser.add_argument("--arm", choices=["R", "E", "N"])
    arguments = parser.parse_args()
    if arguments.stage == "all":
        run(arguments.output)
    elif arguments.stage == "declare":
        declare(arguments.output, configuration())
    elif arguments.stage == "reproduction":
        cfg = configuration()
        with SnowGymBatchClient() as client:
            check = reproduction_check(client, cfg)
        write_json(Path(arguments.output) / "reproduction-check.json", check)
        if not check["passed"]:
            raise SystemExit(f"R1n-f reproduction check failed on seeds {check['mismatches']}")
    elif arguments.stage == "arm":
        if arguments.arm is None:
            raise SystemExit("--arm is required for --stage arm")
        run_arm(arguments.output, configuration(), arguments.arm)
    else:
        aggregate(arguments.output, configuration())
