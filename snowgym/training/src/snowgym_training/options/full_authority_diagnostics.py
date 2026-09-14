"""R1n-b pre-actor diagnostics (declaration section 3): teacher precondition and
ceiling with a label audit, a plan-blind scripted reference, the uniform floor,
and the corrected Monte Carlo critic warm start, plus reported D research.

No actor is trained. See `reviews/m7b_r1n_b_declaration.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW, decode_action
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import full_authority_train_v1 as v1
from .full_authority_train import scenario
from .interventions import require_capabilities
from .movement_train import TRAINING
from .plans import teacher_option_plan
from .reservoir import file_digest
from .supervised_probe import write_json

E3_ARCHIVE = TRAINING / "runs/m7b_engage_r1n_v0/declaration.json"
E3_SOURCES = {"implementationDigest": "src/snowgym_training/options/full_authority_train.py",
              "policyImplementationDigest": "src/snowgym_training/executor/full_authority_ppo.py"}
ACTION_NAMES = ("noop", "move", "throw", "hold")


def configuration():
    return {**v1.configuration(), "format": "snowgym.full-authority-diagnostics.v0",
        "calibrationSeeds": [620000, 620099], "diagnosticBlockWorlds": 50, "preconditionDecisions": 5,
        "floorRng": 981002, "simulatorBudget": 600000, "teacherMinSuccessFraction": .8,
        "teacherMaxRejectedRate": .001, "contactAdequateFraction": .25, "localReachWorld": 8.0,
        "labelBeyondRadiusFlagFraction": .2, "potentialFirstDecisions": 50}


def e3_digests_unchanged():
    archived = json.loads(E3_ARCHIVE.read_text(encoding="utf-8"))
    return {key: {"archived": archived[key], "current": file_digest(TRAINING / path),
                  "match": archived[key] == file_digest(TRAINING / path)} for key, path in E3_SOURCES.items()}


# -- C1a / C1b / C1c / C2 controllers -------------------------------------------------


def teacher_precondition(client, cfg):
    """C1a: a structured pass/fail, never an exception."""
    decisions, rejected = 0, 0
    try:
        wrapper = v1.make_wrapper(client, 1, cfg["gamma"])
        plan, _ = teacher_option_plan("engage")
        seed = cfg["calibrationSeeds"][0]
        wrapper.reset([seed], [scenario()], [f"r1n-b-c1a-{seed}"], [plan], [v1.engage_spec(cfg)])
        for _ in range(cfg["preconditionDecisions"]):
            labels = wrapper.environment.plan_teacher_tensor_actions_indices([0])
            _, _, done, _, infos = wrapper.step_indices([0], labels)
            decisions += 1
            rejected += sum(r.get("accepted") is False for r in infos[0].get("actionResults", []))
            if done[0]:
                break
        return {"passed": decisions > 0 and rejected == 0, "decisions": decisions,
                "rejectedActions": rejected, "error": None}
    except Exception as error:  # the declaration records any failure as a finding
        return {"passed": False, "decisions": decisions, "rejectedActions": rejected,
                "error": f"{type(error).__name__}: {error}"}


class TeacherLabels:
    """C1b: plan-teacher labels, stepped through the learner's tensor round trip, audited."""

    def __init__(self):
        self.counts = {name: 0 for name in ACTION_NAMES}
        self.move_distance, self.move_saturated = [], 0
        self.throw_aim_distance, self.throw_saturated, self.power, self.throw_target_distance = [], 0, [], []
        self.seconds, self.labels = 0., 0

    def __call__(self, wrapper, active, _rows, raws):
        started = time.perf_counter()
        semantic = wrapper.environment.plan_teacher_actions_indices(active)
        self.seconds += time.perf_counter() - started
        self.labels += len(active)
        decoded = [decode_action(action, raw, wrapper.environment.max_team_units) for action, raw in zip(semantic, raws)]
        for action, raw, tensor, index in zip(semantic, raws, decoded, active):
            own = {u["id"]: u for u in raw["allies"]}
            distance = v1.target_distance(raw, wrapper.trackers[index])
            for unit in action["actions"]:
                self.counts[unit["type"]] += 1
                slot = next(i for i, u in enumerate(raw["allies"]) if u["id"] == unit["unitId"])
                if unit["type"] == "move":
                    self.move_distance.append(math.hypot(unit["x"] - own[unit["unitId"]]["x"], unit["y"] - own[unit["unitId"]]["y"]))
                    self.move_saturated += bool((np.abs(tensor["target"][slot]) >= .999).any())
                elif unit["type"] == "throw":
                    self.throw_aim_distance.append(math.hypot(unit["x"] - own[unit["unitId"]]["x"], unit["y"] - own[unit["unitId"]]["y"]))
                    self.throw_saturated += bool((np.abs(tensor["target"][slot]) >= .999).any())
                    self.power.append(float(unit["power"]))
                    if distance is not None:
                        self.throw_target_distance.append(distance)
        return {name: np.stack([d[name] for d in decoded]) for name in ("action_type", "target", "power")}

    def audit(self, cfg):
        def quantiles(values, points):
            return {name: float(np.quantile(values, q)) for name, q in points} if values else None
        moves, throws = len(self.move_distance), len(self.throw_aim_distance)
        return {"actionTypeCounts": self.counts,
            "moveLabels": moves, "moveSaturatedFraction": self.move_saturated / moves if moves else None,
            "moveDistanceWorld": quantiles(self.move_distance, (("median", .5), ("p90", .9))),
            "moveBeyondLocalReachFraction": (sum(d > cfg["localReachWorld"] for d in self.move_distance) / moves
                                             if moves else None),
            "throwLabels": throws, "throwSaturatedFraction": self.throw_saturated / throws if throws else None,
            "throwAimDistanceWorld": quantiles(self.throw_aim_distance, (("median", .5), ("p90", .9))),
            "throwPower": quantiles(self.power, (("p10", .1), ("median", .5), ("p90", .9))),
            "throwTargetDistanceMedianWorld": float(np.median(self.throw_target_distance)) if self.throw_target_distance else None,
            "labelsPerSecondHardwareSpecific": self.labels / self.seconds if self.seconds > 0 else None}


def uniform_floor(seed):
    generator = np.random.default_rng(seed)

    def choose(_wrapper, _active, rows, _raws):
        mask = rows["unit_action_mask"].numpy().astype(bool)
        action_type = np.array([[generator.choice(np.flatnonzero(unit)) if unit.any() else 0 for unit in world]
                                for world in mask])
        target = generator.uniform(-1, 1, size=(*mask.shape[:2], 2)).astype(np.float32)
        power = generator.uniform(0, 1, size=mask.shape[:2]).astype(np.float32)
        return {"action_type": action_type, "target": target, "power": power}

    return choose


def run_source(client, cfg, *, source, choose, scripted=False, account):
    seeds = list(range(cfg["calibrationSeeds"][0], cfg["calibrationSeeds"][1] + 1))
    episodes = []
    for start in range(0, len(seeds), cfg["diagnosticBlockWorlds"]):
        block = seeds[start:start + cfg["diagnosticBlockWorlds"]]
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"], scripted=scripted)
        found, _, used = v1.run_block(wrapper, block, cfg, choose=choose, source=source)
        account(used)
        episodes.extend(found)
    return episodes


# -- Summaries, D research and decision rules -----------------------------------------


def outcome_summary(episodes, cfg):
    rows = [v1.episode_row(e) for e in episodes]
    n = len(rows)
    successes = [r for r in rows if r["success"]]
    hits = [r["firstHitDecision"] for r in rows if r["firstHitDecision"] is not None]
    total = sum(r["totalActions"] for r in rows)
    completion = [r["finalDecision"] for r in successes]

    def quantile(values, q):
        return float(np.quantile(values, q)) if values else None

    p95 = quantile(completion, .95)
    return {"episodes": n, "successes": len(successes), "successFraction": len(successes) / n if n else None,
        "completionDecisionP50": quantile(completion, .5), "completionDecisionP95": p95,
        "suggestedHorizon": math.ceil(1.5 * p95) if p95 is not None else None,
        "contactFraction": len(hits) / n if n else None,
        "firstHitDecisionP50": quantile(hits, .5), "firstHitDecisionP95": quantile(hits, .95),
        "blueDeathFraction": sum(not r["blueAliveAtEnd"] for r in rows) / n if n else None,
        "rejectedActions": sum(r["rejectedActions"] for r in rows), "totalActions": total,
        "rejectedActionRate": sum(r["rejectedActions"] for r in rows) / total if total else None}


def potential_statistics(episodes, d_star, cfg):
    """D2, offline: Phi_d(s) = -max(0, d - d*) / D on recorded trajectories. No reward change.

    `nonzeroShapingFraction` is the declared statistic; because gamma < 1 it is nonzero
    whenever Phi_d != 0, so the distance-dependent `potentialChangeFraction` and the
    mean |Delta Phi_d| are reported alongside it."""
    if d_star is None:
        return {"computed": False, "reason": "d* undefined (C1b did not run or recorded no THROW)"}
    diagonal = math.hypot(100, 80)
    gamma = cfg["gamma"]

    def phi(distance):
        return None if distance is None else -max(0., distance - d_star) / diagonal

    transitions = nonzero = changed = 0
    magnitude, means, contact = 0., [], []
    for episode in episodes:
        first = next((t for t, damage in enumerate(episode["targetDamage"]) if damage > 0), None)
        values = [phi(d) for d in episode["distances"]]
        limit = len(values) - 1 if first is None else min(first, len(values) - 1)
        for t in range(limit):
            if values[t] is None or values[t + 1] is None:
                continue
            transitions += 1
            nonzero += abs(gamma * values[t + 1] - values[t]) > 1e-12
            delta = abs(values[t + 1] - values[t])
            changed += delta > 1e-12
            magnitude += delta
        head = [v for v in values[:cfg["potentialFirstDecisions"]] if v is not None]
        if head:
            means.append(float(np.mean(head)))
            contact.append(float(first is not None))
    correlation = None
    if len(means) > 1 and np.std(means) > 0 and np.std(contact) > 0:
        correlation = float(np.corrcoef(means, contact)[0, 1])
    return {"computed": True, "dStarWorld": d_star, "preContactTransitions": transitions,
        "nonzeroShapingFraction": nonzero / transitions if transitions else None,
        "potentialChangeFraction": changed / transitions if transitions else None,
        "meanAbsolutePotentialChange": magnitude / transitions if transitions else None,
        "earlyPotentialContactPointBiserial": correlation, "episodes": len(means)}


def decision_rules(precondition, teacher, audit, warm, pooled, cfg):
    teacher_usable = bool(precondition["passed"] and teacher is not None
        and teacher["successFraction"] >= cfg["teacherMinSuccessFraction"]
        and (teacher["rejectedActionRate"] or 0.) < cfg["teacherMaxRejectedRate"])
    arms = {}
    for arm in cfg["arms"]:
        healthy = all(report["gatePassed"] for report in warm[arm].values())
        adequate = pooled[arm]["contactFraction"] >= cfg["contactAdequateFraction"]
        if not healthy:
            recommendation = "no PPO branch until a declared critic diagnosis"
        elif adequate:
            recommendation = "D1 informative" + ("; D3 also open" if teacher_usable else "")
        elif teacher_usable:
            recommendation = "D3"
        else:
            recommendation = "D2 or a separately declared teacher"
        arms[arm] = {"criticHealthy": healthy, "contactAdequate": adequate, "recommendation": recommendation,
            "perRng": {rng: {key: report.get(key) for key in ("gatePassed", "gateConditions", "predictiveR2",
                                                              "timeOnlyR2", "clockSkillScore")}
                       for rng, report in warm[arm].items()}}
    beyond = None if audit is None else audit["moveBeyondLocalReachFraction"]
    return {"teacherUsable": teacher_usable, "arms": arms,
        "labelAuditRequiresDecoderStatement": None if beyond is None else beyond > cfg["labelBeyondRadiusFlagFraction"],
        "authorizes": "nothing; a Phase D branch needs its own declaration"}


# -- Artifact writing -----------------------------------------------------------------


def write_episodes(directory, episodes):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episodes.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text("".join(json.dumps(v1.episode_row(e), sort_keys=True, allow_nan=False) + "\n" for e in episodes),
                    encoding="utf-8")
    distances = [np.asarray([np.nan if d is None else d for d in e["distances"]], dtype=np.float64) for e in episodes]
    np.savez_compressed(directory / "trajectory-distances.npz", seeds=np.asarray([e["seed"] for e in episodes]),
        offsets=np.cumsum([0] + [len(d) for d in distances]), distances=np.concatenate(distances),
        targetDamage=np.concatenate([np.asarray(e["targetDamage"]) for e in episodes]))


def execute(output, cfg):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    torch.set_num_threads(1)
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    root.mkdir(parents=True)
    here = Path(__file__).resolve()
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_b_declaration.md"),
        "diagnosticsImplementationDigest": file_digest(here),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "policyImplementationDigest": file_digest(here.parents[1] / "executor/full_authority_ppo_v1.py"),
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"], "assistVersion": cfg["assistVersion"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["simulatorBudget"]:
            raise ValueError("R1n-b simulator budget exceeded")

    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        precondition = teacher_precondition(client, cfg)
        account(precondition["decisions"])
        write_json(root / "c1a-precondition.json", precondition)
        print(json.dumps({"c1aPassed": precondition["passed"], "error": precondition["error"]}), flush=True)

        teacher = audit = None
        teacher_episodes = []
        if precondition["passed"]:
            labels = TeacherLabels()
            teacher_episodes = run_source(client, cfg, source="c1b-plan-teacher", choose=labels, account=account)
            write_episodes(root / "c1b-plan-teacher", teacher_episodes)
            teacher, audit = outcome_summary(teacher_episodes, cfg), labels.audit(cfg)
            write_json(root / "c1b-plan-teacher/summary.json", {**teacher, "assistance": "none (coded production teacher)",
                "observationVersion": 3, "task": "Engage option success (target <= 20% health)"})
            write_json(root / "c1b-plan-teacher/label-audit.json", audit)
            print(json.dumps({"c1bSuccesses": teacher["successes"], "contact": teacher["contactFraction"]}), flush=True)

        scripted_episodes = run_source(client, cfg, source="c1c-scripted", choose=None, scripted=True, account=account)
        write_episodes(root / "c1c-scripted", scripted_episodes)
        scripted = outcome_summary(scripted_episodes, cfg)
        write_json(root / "c1c-scripted/summary.json", {**scripted, "role": "plan-blind reference, not a ceiling"})

        floor_episodes = run_source(client, cfg, source="c2-floor", choose=uniform_floor(cfg["floorRng"]), account=account)
        write_episodes(root / "c2-floor", floor_episodes)
        floor = outcome_summary(floor_episodes, cfg)
        write_json(root / "c2-floor/summary.json", floor)
        print(json.dumps({"c1cSuccesses": scripted["successes"], "floorSuccesses": floor["successes"]}), flush=True)

        warm, pooled_episodes, calibration, contact = {}, {}, {}, {}
        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        for arm in cfg["arms"]:
            warm[arm], pooled_episodes[arm] = {}, []
            for rng_index, rng in enumerate(cfg["trainingRngs"]):
                directory = root / "c3" / arm / str(rng)
                directory.mkdir(parents=True)
                torch.manual_seed(rng)
                model = FullAuthorityPolicyV1(destination=arm, local_radius=cfg["localRadius"],
                    target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
                report, arrays, episodes, held, _ = v1.warm_start_critic_mc(model, wrapper, cfg, rng_index,
                                                                            source=f"c3-{arm}-{rng}")
                account(report["simulatorDecisions"])
                summary = outcome_summary(episodes, cfg)
                report["randomInitOutcomes"] = summary
                write_json(directory / "critic-warm-start.json", report)
                np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
                torch.save({"model": model.state_dict(), "note": "untrained actor; warm-started critic; not promotable"},
                           directory / "initial-policy.pt")
                write_episodes(directory, episodes)
                if rng_index == 0:
                    calibration[arm] = v1.exploration_calibration(model, held["observation"], cfg)
                    write_json(directory / "calibration.json", calibration[arm])
                warm[arm][str(rng)] = report
                pooled_episodes[arm].extend(episodes)
                print(json.dumps({"arm": arm, "rng": rng, "predictiveR2": report["predictiveR2"],
                    "timeOnlyR2": report["timeOnlyR2"], "gatePassed": report["gatePassed"],
                    "contact": summary["contactFraction"], "simulatorDecisions": steps}), flush=True)
                del arrays, episodes, held, model
            contact[arm] = outcome_summary(pooled_episodes[arm], cfg)

    d_star = None if audit is None else audit["throwTargetDistanceMedianWorld"]
    research = {"D1": {arm: {key: contact[arm][key] for key in ("episodes", "successFraction", "contactFraction",
                                                              "firstHitDecisionP50", "firstHitDecisionP95")}
                       for arm in cfg["arms"]},
        "D2": {"teacher": potential_statistics(teacher_episodes, d_star, cfg),
               "floor": potential_statistics(floor_episodes, d_star, cfg),
               **{f"randomInit-{arm}": potential_statistics(pooled_episodes[arm], d_star, cfg) for arm in cfg["arms"]}},
        "D3": {"preconditionPassed": precondition["passed"], "teacher": teacher, "labelAudit": audit},
        "calibration": calibration}
    write_json(root / "d-research.json", research)
    rules = decision_rules(precondition, teacher, audit, warm, contact, cfg)
    report = {"format": "snowgym.full-authority-diagnostics-report.v0", "capabilities": capabilities,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "c1aPrecondition": precondition, "c1bPlanTeacher": teacher, "c1cScripted": scripted, "c2Floor": floor,
        "c3WarmStart": warm, "randomInitPooled": contact, "decisionRules": rules, "simulatorDecisions": steps}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.full-authority-diagnostics-manifest.v0",
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


def run(output):
    return execute(output, configuration())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
