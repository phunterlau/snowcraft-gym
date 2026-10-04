"""M8-S16: command-controllability audit of the frozen S14 fighters (`reviews/m8_s16_declaration.md`).

From one reset state with three singleton Red clusters, the execution plan commands `leftmost`, `rightmost`,
`nearest` or a shuffled flank, and every episode is scored against the *requested* flank with a stopping rule that
ignores execution-option completion. The plan-aware teacher is the positive control. A regression gate first replays
S14's archived first evaluation block through this runner on the default layout. No training; no existing module is
edited."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_THROW
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT
from ..ppo_collect import numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import enemy_relative_throw_ppo_retention as s15
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from .engage_v1 import EngageOptionBatchV1, FrozenEngageTracker
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .interventions import require_capabilities
from .plans import teacher_option_plan
from .reservoir import file_digest
from .supervised_probe import write_json
from .tracker import FixedOptionTracker

POLICIES = ("initializer", "final")
MODES = ("deterministic", "stochastic")
TASKS = ("leftmost", "rightmost")
MIRROR = {"leftmost": "rightmost", "rightmost": "leftmost"}
CONDITIONS = ("correct", "other", "canonical", "shuffled", "zero-plan")
TEACHER_CONDITIONS = CONDITIONS[:4]
SPREADS = (10, 20, 30)
BLUE_SPAWNS = ({"x": -30, "y": -5}, {"x": -30, "y": 0}, {"x": -30, "y": 5})
DEFAULT_RED_SPAWNS = ({"x": 30, "y": -5}, {"x": 30, "y": 0}, {"x": 30, "y": 5})
ZEROED_PLAN_KEYS = ("plan_groups", "plan_role_state")
CLASSES = ("controllable", "sensitive", "insensitive", "counter-sensitive", "uninformative")


def configuration():
    return {**s15.configuration(), "format": "snowgym.m8-s16-command-control-audit-config.v0",
        "s15Run": "m8_s15_ppo_retention_eval_v0",
        "worldSeedBase": 2720000, "worlds": 192, "spreads": SPREADS,
        "reservedQualificationSeeds": [2730000, 2730399],
        "commandHorizon": 200, "thresholdFraction": .2, "earlyWindow": 100, "lateralDecision": 40,
        "shuffleSeed": 986002, "bootstrapSeed": 986001, "torchSeedBase": 986000,
        "teacherOrderMin": .90, "teacherDeltaLowerMin": .70, "teacherSpreadOrderMin": .85,
        "informativeMin": .50, "controllableDelta": .40,
        "teacherIdentityTolerance": .02, "learnedIdentityTolerance": .10,
        "budgetCap": 5_100_000,
        "assistType": "none at learned-policy runtime; plan-aware teacher only as a separate positive control",
        "assistVersion": "snowgym.m8-s16.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["commandHorizon"]
    gate = len(cfg["cohorts"]) * len(POLICIES) * len(MODES) * cfg["fidelityWorlds"] * horizon
    teacher = cfg["worlds"] * len(TASKS) * len(TEACHER_CONDITIONS) * horizon
    learned = (cfg["worlds"] * len(TASKS) * len(CONDITIONS) * len(cfg["cohorts"]) * len(POLICIES) * len(MODES)
               * horizon)
    return {"gate": gate, "teacher": teacher, "learned": learned, "total": gate + teacher + learned}


# -- Worlds, layouts, plans (declaration §2, §3) ---------------------------------------------------


def panel(cfg):
    """(world index, seed, spread) for the development panel."""
    return [(i, cfg["worldSeedBase"] + i, cfg["spreads"][i % len(cfg["spreads"])]) for i in range(cfg["worlds"])]


def scenario_for(spread):
    red = DEFAULT_RED_SPAWNS if spread is None else (
        {"x": 30, "y": -spread}, {"x": 30, "y": 0}, {"x": 30, "y": spread})
    return {**v1.scenario(), **rb.arm_scenario("normal"), "blueSpawns": [dict(p) for p in BLUE_SPAWNS],
            "redSpawns": [dict(p) for p in red]}


def engage_plan(select):
    plan, _ = teacher_option_plan("engage")
    plan = copy.deepcopy(plan)
    plan["groups"][0]["order"]["objective"]["select"] = select
    return plan


def shuffle_table(cfg):
    draws = np.random.default_rng(cfg["shuffleSeed"]).integers(0, len(TASKS), size=(cfg["worlds"], len(TASKS)))
    return {(i, task): TASKS[int(draws[i, t])] for i in range(cfg["worlds"]) for t, task in enumerate(TASKS)}


def execution_select(condition, task, world_index, shuffled):
    if condition in ("correct", "zero-plan"):
        return task
    if condition == "other":
        return MIRROR[task]
    if condition == "canonical":
        return "nearest"
    if condition == "shuffled":
        return shuffled[(world_index, task)]
    raise ValueError(f"unknown condition {condition!r}")


def stochastic_torch_seed(cfg, cohort, policy, condition, task):
    return (cfg["torchSeedBase"] + 100 * cohort + 20 * CONDITIONS.index(condition) + 2 * TASKS.index(task)
            + POLICIES.index(policy))


# -- Scoring (declaration §4) -------------------------------------------------------------------------


def flank_order(requested_cross, mirror_cross):
    """1 if the requested flank crosses strictly first, 0 if the mirror does, 0.5 for a tie or neither."""
    if requested_cross is None and mirror_cross is None:
        return .5
    if mirror_cross is None:
        return 1.
    if requested_cross is None:
        return 0.
    return 1. if requested_cross < mirror_cross else 0. if mirror_cross < requested_cross else .5


def flank_contrast(requested_damage, mirror_damage):
    total = requested_damage + mirror_damage
    return 0. if total <= 0 else (requested_damage - mirror_damage) / total


def bearing_closest(origin, target, enemies):
    """ID of the living enemy whose bearing from `origin` is closest to the direction toward `target`."""
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    if math.hypot(dx, dy) <= 1e-9:
        return None
    heading = math.atan2(dy, dx)
    best, best_gap = None, math.inf
    for enemy in enemies:
        if not enemy["alive"]:
            continue
        gap = abs((math.atan2(enemy["y"] - origin[1], enemy["x"] - origin[0]) - heading + math.pi)
                  % (2 * math.pi) - math.pi)
        if gap < best_gap:
            best, best_gap = enemy["id"], gap
    return best


class CommandAuditTracker(FrozenEngageTracker):
    """The execution plan's Engage tracker, continued past its own completion until the declared stopping rule
    (requested target at the threshold, blue wiped, environment done, or the horizon)."""

    def __init__(self, spec, plan, observation, plan_observation):
        super().__init__(spec, plan, observation, plan_observation)
        self.initial_enemy_health = {u["id"]: max(0., float(u["health"])) for u in observation["enemies"]}
        self.requested_ids = self.mirror_ids = None
        self.crossed, self.early_damage, self.first_damage = {}, {}, {}
        self.execution_success_decision = self.stop_reason = self.last_step = None

    def configure(self, requested_ids, mirror_ids, *, horizon, threshold, early_window):
        self.requested_ids, self.mirror_ids = tuple(requested_ids), tuple(mirror_ids)
        self.horizon, self.threshold, self.early_window = horizon, threshold, early_window
        self.early_damage = {i: 0. for i in self.initial_enemy_health}
        self._health = dict(self.initial_enemy_health)

    def requested_fraction(self, health):
        initial = sum(self.initial_enemy_health[i] for i in self.requested_ids)
        return sum(health.get(i, 0.) for i in self.requested_ids) / max(initial, 1e-9)

    def update(self, observation, plan_observation, *, canonical_reward, gamma, environment_done=False):
        if self.finished:
            raise RuntimeError("command-audit episode is already complete")
        if self.requested_ids is None:
            raise RuntimeError("configure the requested target before stepping")
        step = FixedOptionTracker.update(self, observation, plan_observation, canonical_reward=canonical_reward,
                                         gamma=gamma, environment_done=environment_done)
        if step.success and self.execution_success_decision is None:
            self.execution_success_decision = self.decision
        health = {u["id"]: max(0., float(u["health"])) if u["alive"] else 0. for u in observation["enemies"]}
        for unit_id, before in self._health.items():
            damage = max(0., before - health.get(unit_id, 0.))
            if damage > 0 and unit_id not in self.first_damage:
                self.first_damage[unit_id] = self.decision
            if self.decision <= self.early_window:
                self.early_damage[unit_id] += damage
        self._health = health
        for unit_id, initial in self.initial_enemy_health.items():
            if unit_id not in self.crossed and health.get(unit_id, 0.) <= self.threshold * initial:
                self.crossed[unit_id] = self.decision
        blue_alive = any(u["alive"] and u["id"] in self.assigned_ids for u in observation["allies"])
        reasons = [("requested-complete", self.requested_fraction(health) <= self.threshold),
                   ("blue-wiped", not blue_alive), ("environment-done", environment_done),
                   ("horizon", self.decision >= self.horizon)]
        self.stop_reason = next((name for name, hit in reasons if hit), None)
        self.finished = self.stop_reason is not None
        self.last_step = step
        return step


class CommandAuditBatch(EngageOptionBatchV1):
    def _install_trackers(self, indices, plans, specs, bodies):
        for index, plan, spec, body in zip(indices, plans, specs, bodies, strict=True):
            raw = self.environment.raw_observations[index]
            if raw is None:
                raise RuntimeError("missing activation observation")
            self.trackers[index] = CommandAuditTracker(spec, plan, raw, body)


def preview_ids(wrapper, seeds, select, source):
    plans = [engage_plan(select)] * len(seeds)
    _, _, bodies = wrapper.environment.preview_plans([f"s16-preview-{source}-{seed}-{select}" for seed in seeds],
                                                     plans)
    ids = []
    for body in bodies:
        main = next(item for item in body["activationObjectives"] if item["role"] == "main")
        ids.append(tuple(main["enemyIds"]))
    return ids


# -- Collection ------------------------------------------------------------------------------------------


class ZeroPlan:
    """Diagnostic actor input with the command channels removed (declaration §3; outside the production contract)."""

    def __init__(self, model):
        self.model = model

    def act(self, observation, *, deterministic=False):
        stripped = {key: (torch.zeros_like(value) if key in ZEROED_PLAN_KEYS else value)
                    for key, value in observation.items()}
        return self.model.act(stripped, deterministic=deterministic)


def model_chooser(model, deterministic):
    def choose(_wrapper, _active, rows, _raws):
        with torch.no_grad():
            action, *_ = model.act(rows, deterministic=deterministic)
        return numpy_actions(action)
    return choose


def teacher_chooser(wrapper, active, _rows, _raws):
    return wrapper.environment.plan_teacher_tensor_actions_indices(active)


def run_audit_block(wrapper, block, cfg, *, requested_select, mirror_select, execution_selects, choose, source):
    """One block of worlds (`block` = [(index, seed, spread)]): reset with each world's execution plan, ground the
    requested/mirror/centre targets by preview, then step until the declared stopping rule. Returns per-episode rows
    carrying S14's row fields (for the regression gate) and the audit fields."""
    count = len(block)
    seeds = [seed for _, seed, _ in block]
    if wrapper.batch_size != count:
        raise ValueError("block size must match the wrapper batch size")
    plans = [engage_plan(select) for select in execution_selects]
    observation, _ = wrapper.reset(seeds, [scenario_for(spread) for _, _, spread in block],
                                   [f"s16-{source}-{seed}" for seed in seeds], plans, [v1.engage_spec(cfg)] * count)
    requested = preview_ids(wrapper, seeds, requested_select, source)
    mirror = preview_ids(wrapper, seeds, mirror_select, source) if mirror_select else [()] * count
    centre = preview_ids(wrapper, seeds, "nearest", source) if mirror_select else [()] * count
    for index in range(count):
        if mirror_select and not (len(requested[index]) == len(mirror[index]) == len(centre[index]) == 1
                                  and len({*requested[index], *mirror[index], *centre[index]}) == 3):
            raise RuntimeError(f"world {seeds[index]}: requested/mirror/centre do not ground to three singletons")
        wrapper.trackers[index].configure(requested[index], mirror[index], horizon=cfg["commandHorizon"],
            threshold=cfg["thresholdFraction"], early_window=cfg["earlyWindow"])
    side = []
    for index in range(count):  # +1 when the requested flank lies at positive lateral (y) offset
        raw = wrapper.environment.raw_observations[index]
        ys = [u["y"] for u in raw["enemies"] if u["id"] in requested[index]]
        side.append(float(np.sign(np.mean(ys))) if mirror_select else 0.)
    start_y = [float(np.mean([u["y"] for u in wrapper.environment.raw_observations[i]["allies"] if u["alive"]]))
               for i in range(count)]
    current = tensor_dict(observation)
    episodes = [{"seed": seed, "source": source, "rewards": [], "distances": [], "targetDamage": [],
                 "rejectedActions": 0, "totalActions": 0, "aim": {"requested": 0, "mirror": 0, "centre": 0,
                 "other": 0}, "lateral": None} for seed in seeds]
    decisions = 0
    while True:
        active = [i for i in range(count) if not wrapper.trackers[i].finished]
        if not active:
            break
        rows = {key: value[active] for key, value in current.items()}
        raws = [wrapper.environment.raw_observations[i] for i in active]
        distances = [v1.target_distance(raw, wrapper.trackers[i]) for raw, i in zip(raws, active)]
        actions = choose(wrapper, active, rows, raws)
        for row, index in enumerate(active):
            tracker, raw = wrapper.trackers[index], raws[row]
            if tracker.decision < cfg["earlyWindow"]:
                record_aim(episodes[index]["aim"], actions, row, raw, requested[index], mirror[index], centre[index])
        raw_next, rewards, _, _, infos = wrapper.step_indices(active, actions)
        decisions += len(active)
        following = tensor_dict(raw_next)
        for key in current:
            current[key][active] = following[key]
        for row, index in enumerate(active):
            episode, option, tracker = episodes[index], infos[row]["option"], wrapper.trackers[index]
            episode["rewards"].append(float(rewards[row]))
            episode["distances"].append(distances[row])
            episode["targetDamage"].append(float(option["metrics"]["targetDamage"]))
            results = infos[row].get("actionResults", [])
            episode["rejectedActions"] += sum(r.get("accepted") is False for r in results)
            episode["totalActions"] += len(results)
            raw = wrapper.environment.raw_observations[index]
            if episode["lateral"] is None and (tracker.decision >= cfg["lateralDecision"] or tracker.finished):
                living = [u["y"] for u in raw["allies"] if u["alive"]]
                episode["lateral"] = (side[index] * (float(np.mean(living)) - start_y[index])) if living else None
            if tracker.finished:
                episode.update(success=bool(option["success"]), failed=bool(option["failed"]),
                    timedOut=bool(option["timedOut"]), finalDecision=int(option["decision"]),
                    blueAliveAtEnd=any(u["alive"] and u["id"] in tracker.assigned_ids for u in raw["allies"]))
    rows = []
    for index, episode in enumerate(episodes):
        tracker = wrapper.trackers[index]
        allies = wrapper.environment.raw_observations[index]["allies"]
        alive = sum(1 for u in allies if u["alive"] and u["id"] in tracker.assigned_ids)
        rows.append(v1.episode_row(episode) | {"assignedUnits": len(tracker.assigned_ids), "blueAliveCount": alive,
            "unitsLostFraction": rb.units_lost_fraction(len(tracker.assigned_ids), alive)}
            | audit_fields(tracker, episode, requested[index], mirror[index], centre[index]))
    return rows, decisions


def record_aim(counts, actions, row, raw, requested, mirror, centre):
    action_type = np.asarray(actions["action_type"])[row]
    target = np.asarray(actions["target"])[row]
    for slot, unit in enumerate(raw["allies"]):
        if not unit["alive"] or int(action_type[slot]) != ACTION_THROW:
            continue
        world = (float(target[slot, 0]) * ARENA_HALF_EXTENT[0], float(target[slot, 1]) * ARENA_HALF_EXTENT[1])
        aimed = bearing_closest((unit["x"], unit["y"]), world, raw["enemies"])
        key = ("requested" if aimed in requested else "mirror" if aimed in mirror else "centre" if aimed in centre
               else "other")
        counts[key] += 1


def audit_fields(tracker, episode, requested, mirror, centre):
    requested_cross = min((tracker.crossed[i] for i in requested if i in tracker.crossed), default=None)
    mirror_cross = min((tracker.crossed[i] for i in mirror if i in tracker.crossed), default=None)
    centre_cross = min((tracker.crossed[i] for i in centre if i in tracker.crossed), default=None)
    crossings = sorted(tracker.crossed.items(), key=lambda item: item[1])
    first_decision = crossings[0][1] if crossings else None
    return {"requestedIds": list(requested), "mirrorIds": list(mirror), "centreIds": list(centre),
        "stopReason": tracker.stop_reason, "requestedCross": requested_cross, "mirrorCross": mirror_cross,
        "centreCross": centre_cross, "flankOrder": flank_order(requested_cross, mirror_cross) if mirror else None,
        "flankContrast": flank_contrast(sum(tracker.early_damage[i] for i in requested),
                                        sum(tracker.early_damage[i] for i in mirror)) if mirror else None,
        "firstOverallIds": [i for i, d in crossings if d == first_decision],
        "requestedFirstDamage": min((tracker.first_damage[i] for i in requested if i in tracker.first_damage),
                                    default=None),
        "executionSuccessDecision": tracker.execution_success_decision,
        "aim": episode["aim"], "lateralTowardRequested": episode["lateral"]}


def run_cell(client, cfg, worlds, *, requested_select, mirror_select, execution_selects, choose, source, account):
    rows = []
    for start in range(0, len(worlds), cfg["blockWorlds"]):
        block = worlds[start:start + cfg["blockWorlds"]]
        wrapper = CommandAuditBatch(v1.SelectiveBatchEnv(len(block), client=client, observation_version=3),
                                    gamma=cfg["gamma"])
        found, used = run_audit_block(wrapper, block, cfg, requested_select=requested_select,
            mirror_select=mirror_select, execution_selects=execution_selects[start:start + cfg["blockWorlds"]],
            choose=choose, source=source)
        account(used)
        rows.extend(found)
    return rows


def write_rows(directory, rows):
    rb.write_rows(Path(directory), rows)


def read_rows(directory):
    path = Path(directory) / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- Gate 1: runner regression (declaration §5.1) ------------------------------------------------------------


GATE_IGNORED_FIELDS = ("source",)


def gate_compare(rows, archived):
    def strip(row):
        return {k: v for k, v in row.items() if k not in GATE_IGNORED_FIELDS}
    mismatched = [r["seed"] for r in rows
                  if r["seed"] not in archived or strip({k: r[k] for k in archived[r["seed"]]}) != strip(
                      archived[r["seed"]])]
    return mismatched


def regression_gate(client, cfg, policies, root, account):
    worlds = [(i, seed, None) for i, seed in enumerate(s15.fidelity_seeds(cfg))]
    results = {}
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                if mode == "stochastic":
                    torch.manual_seed(s15.fidelity_torch_seed(cfg, cohort, policy))
                rows = run_cell(client, cfg, worlds, requested_select="nearest", mirror_select=None,
                    execution_selects=["nearest"] * len(worlds),
                    choose=model_chooser(policies[cohort][policy], mode == "deterministic"),
                    source=f"gate-c{cohort}-{policy}-{mode}", account=account)
                write_rows(Path(root) / "gate" / f"cohort-{cohort}" / f"{policy}-{mode}", rows)
                mismatched = gate_compare(rows, s15.archived_rows(cfg, cohort, policy, mode))
                results[f"{cohort}/{policy}-{mode}"] = {"worlds": len(rows), "mismatchedSeeds": mismatched,
                                                        "exact": not mismatched and len(rows) == len(worlds)}
    return results


# -- Teacher and learned cells (declaration §5.2, §3) ----------------------------------------------------------


def teacher_cells(client, cfg, root, account):
    worlds, shuffled = panel(cfg), shuffle_table(cfg)
    for condition in TEACHER_CONDITIONS:
        for task in TASKS:
            selects = [execution_select(condition, task, i, shuffled) for i, _, _ in worlds]
            rows = run_cell(client, cfg, worlds, requested_select=task, mirror_select=MIRROR[task],
                execution_selects=selects, choose=teacher_chooser, source=f"teacher-{condition}-{task}",
                account=account)
            write_rows(Path(root) / "teacher" / condition / task, rows)


def learned_cells(client, cfg, policies, root, account):
    worlds, shuffled = panel(cfg), shuffle_table(cfg)
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                for condition in CONDITIONS:
                    for task in TASKS:
                        model = policies[cohort][policy]
                        if condition == "zero-plan":
                            model = ZeroPlan(model)
                        if mode == "stochastic":
                            torch.manual_seed(stochastic_torch_seed(cfg, cohort, policy, condition, task))
                        selects = [execution_select(condition, task, i, shuffled) for i, _, _ in worlds]
                        rows = run_cell(client, cfg, worlds, requested_select=task, mirror_select=MIRROR[task],
                            execution_selects=selects, choose=model_chooser(model, mode == "deterministic"),
                            source=f"c{cohort}-{policy}-{mode}-{condition}-{task}", account=account)
                        write_rows(Path(root) / "learned" / f"cohort-{cohort}" / policy / mode / condition / task,
                                   rows)
            print(json.dumps({"stage": "learned", "cohort": cohort, "policy": policy}), flush=True)


# -- Analysis (declaration §6) ---------------------------------------------------------------------------------


def world_values(rows_by_task, metric, worlds):
    """Per-world mean over the two mirrored tasks."""
    by_task = {task: {r["seed"]: r for r in rows} for task, rows in rows_by_task.items()}
    for task, rows in by_task.items():
        if any(seed not in rows for _, seed, _ in worlds):
            raise RuntimeError(f"{task}: rows do not cover the requested worlds")
    return [float(np.mean([metric(by_task[task][seed]) for task in TASKS])) for _, seed, _ in worlds]


def order_metric(row):
    return row["flankOrder"]


def contrast_metric(row):
    return row["flankContrast"]


def informative(row):
    return float(row["requestedCross"] is not None or row["mirrorCross"] is not None)


def centre_first(row):
    return float(bool(row["centreIds"]) and row["centreIds"][0] in row["firstOverallIds"])


def requested_success(row):
    return float(row["requestedCross"] is not None)


def condition_summary(rows_by_task, worlds):
    rows = [r for task in TASKS for r in rows_by_task[task]]
    aim = {k: sum(r["aim"][k] for r in rows) for k in ("requested", "mirror", "centre", "other")}
    throws = sum(aim.values())
    lateral = [r["lateralTowardRequested"] for r in rows if r["lateralTowardRequested"] is not None]
    by_side = {task: float(np.mean([r["flankOrder"] for r in rows_by_task[task]])) for task in TASKS}
    return {"order": float(np.mean([r["flankOrder"] for r in rows])),
        "contrast": float(np.mean([r["flankContrast"] for r in rows])),
        "orderBySide": by_side, "informative": float(np.mean([informative(r) for r in rows])),
        "requestedSuccess": float(np.mean([requested_success(r) for r in rows])),
        "centreFirst": float(np.mean([centre_first(r) for r in rows])),
        "unitsLostFraction": float(np.mean([r["unitsLostFraction"] for r in rows])),
        "teamWipe": float(np.mean([r["blueAliveCount"] == 0 for r in rows])),
        "rejectedActionRate": (sum(r["rejectedActions"] for r in rows) / max(sum(r["totalActions"] for r in rows), 1)),
        "aimShares": {k: (v / throws if throws else None) for k, v in aim.items()}, "throwsInWindow": throws,
        "meanLateralTowardRequested": float(np.mean(lateral)) if lateral else None,
        "stopReasons": {reason: sum(r["stopReason"] == reason for r in rows) for reason in
                        ("requested-complete", "blue-wiped", "environment-done", "horizon")}}


def contrasts(cells, worlds, cfg, conditions):
    """correct minus each other condition, world-paired, on flank order and contrast, overall and per spread."""
    out = {}
    strata = {"all": [w for w in worlds]} | {f"d{d}": [w for w in worlds if w[2] == d] for d in cfg["spreads"]}
    for other in conditions:
        if other == "correct":
            continue
        out[other] = {}
        for name, subset in strata.items():
            out[other][name] = {label: fi.paired_difference(world_values(cells["correct"], metric, subset),
                world_values(cells[other], metric, subset), samples=cfg["bootstrapSamples"],
                seed=cfg["bootstrapSeed"]) for label, metric in (("order", order_metric),
                                                                ("contrast", contrast_metric))}
    return out


def mirror_identity(cells):
    """Deterministic execution of (other, r) replays (correct, m(r)), and (canonical, r) replays (canonical, m(r)),
    so o(other, w, r) = 1 - o(correct, w, m(r)) and o(canonical, w, r) = 1 - o(canonical, w, m(r)) up to batch
    effects (declaration §3). Returns the fraction of (world, task) pairs violating each identity."""
    def by_seed(rows):
        return {r["seed"]: r["flankOrder"] for r in rows}
    out = {}
    for name, left, right in (("otherVsCorrect", "other", "correct"), ("canonical", "canonical", "canonical")):
        if left not in cells or right not in cells:
            continue
        pairs = [(a, 1. - b[seed]) for task in TASKS
                 for a_rows, b in [(by_seed(cells[left][task]), by_seed(cells[right][MIRROR[task]]))]
                 for seed, a in a_rows.items()]
        out[name] = float(np.mean([abs(a - b) > 1e-12 for a, b in pairs]))
    return out


def averaged_cells(cell_sets):
    """Per-world mean over several policies' rows (cohort averaging): returns rows with averaged metrics."""
    averaged = {}
    for condition in cell_sets[0]:
        averaged[condition] = {}
        for task in TASKS:
            by_seed = [{r["seed"]: r for r in cells[condition][task]} for cells in cell_sets]
            averaged[condition][task] = [{"seed": seed, **{key: float(np.mean([b[seed][key] for b in by_seed]))
                                          for key in ("flankOrder", "flankContrast")}} for seed in by_seed[0]]
    return averaged


def classify(delta, informative_fraction, cfg):
    if informative_fraction < cfg["informativeMin"]:
        return "uninformative"
    low, high = delta["interval95"]
    if delta["mean"] >= cfg["controllableDelta"] and low > 0:
        return "controllable"
    if low > 0:
        return "sensitive"
    if high < 0:
        return "counter-sensitive"
    return "insensitive"


def load_cells(directory, conditions, worlds):
    cells = {condition: {task: read_rows(Path(directory) / condition / task) for task in TASKS}
             for condition in conditions}
    expected = sorted(seed for _, seed, _ in worlds)
    for condition, by_task in cells.items():
        for task, rows in by_task.items():
            if sorted(r["seed"] for r in rows) != expected:
                raise RuntimeError(f"{directory}/{condition}/{task}: rows do not cover the declared panel exactly")
    return cells


def teacher_gate(root, cfg):
    worlds = panel(cfg)
    cells = load_cells(Path(root) / "teacher", TEACHER_CONDITIONS, worlds)
    summaries = {c: condition_summary(cells[c], worlds) for c in TEACHER_CONDITIONS}
    delta = contrasts(cells, worlds, cfg, TEACHER_CONDITIONS)
    by_spread = {f"d{d}": float(np.mean(world_values(cells["correct"], order_metric,
                 [w for w in worlds if w[2] == d]))) for d in cfg["spreads"]}
    passed = (summaries["correct"]["order"] >= cfg["teacherOrderMin"]
              and delta["other"]["all"]["order"]["interval95"][0] >= cfg["teacherDeltaLowerMin"]
              and all(value >= cfg["teacherSpreadOrderMin"] for value in by_spread.values()))
    identity = mirror_identity(cells)
    return {"passed": passed, "summaries": summaries, "contrasts": delta, "correctOrderBySpread": by_spread,
            "mirrorIdentityViolation": identity,
            "mirrorIdentityWithinTolerance": all(v <= cfg["teacherIdentityTolerance"] for v in identity.values())}


def learned_analysis(root, cfg):
    worlds = panel(cfg)
    policies, cell_sets = {}, {}
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                key = f"c{cohort}-{policy}-{mode}"
                cells = load_cells(Path(root) / "learned" / f"cohort-{cohort}" / policy / mode, CONDITIONS, worlds)
                cell_sets.setdefault((policy, mode), []).append(cells)
                summaries = {c: condition_summary(cells[c], worlds) for c in CONDITIONS}
                delta = contrasts(cells, worlds, cfg, CONDITIONS)
                identity = mirror_identity(cells)
                policies[key] = {"summaries": summaries, "contrasts": delta,
                    "class": classify(delta["other"]["all"]["order"], summaries["correct"]["informative"], cfg),
                    "mirrorIdentityViolation": identity,
                    "mirrorIdentityWithinTolerance": None if mode == "stochastic" else all(
                        v <= cfg["learnedIdentityTolerance"] for v in identity.values())}
    averaged = {}
    for (policy, mode), sets in cell_sets.items():
        cells = averaged_cells(sets)
        averaged[f"{policy}-{mode}"] = contrasts(cells, worlds, cfg, CONDITIONS)
    return {"policies": policies, "cohortAveraged": averaged}


def prediction_scores(teacher, learned):
    deterministic = {k: v for k, v in learned["policies"].items() if k.endswith("-deterministic")}
    informative_keys = [k for k, v in deterministic.items() if v["class"] != "uninformative"]
    return {"1-teacherGatePasses": teacher["passed"],
        "2-noLearnedControllable": all(v["class"] != "controllable" for v in deterministic.values()),
        "3-someLearnedSensitive": any(v["class"] in ("sensitive", "controllable") for v in deterministic.values()),
        "4-canonicalCentreFirst": all(deterministic[k]["summaries"]["canonical"]["centreFirst"] >= .5
                                      for k in informative_keys) if informative_keys else None,
        "deterministicClasses": {k: v["class"] for k, v in deterministic.items()}}


# -- Declaration, run, archive -----------------------------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    runs = TRAINING / "runs"
    manifests = {"s14Training": dr.verify_sealed(runs / cfg["finalRun"]),
                 "s14Probe": dr.verify_sealed(runs / cfg["probeRun"]),
                 "s12": dr.verify_sealed(runs / cfg["sourceRun"]),
                 "s15": dr.verify_sealed(runs / cfg["s15Run"])}
    checkpoints = {str(c): {"final": file_digest(s15.final_checkpoint_path(cfg, c)),
                            "initializer": file_digest(s15.initializer_checkpoint_path(cfg, c))}
                   for c in cfg["cohorts"]}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "sourceManifests": manifests, "checkpointDigests": checkpoints,
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s16_declaration.md"),
        "implementationDigest": file_digest(Path(__file__).resolve()), "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, gate, teacher, learned, spent):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    gate_passed = bool(gate) and all(entry["exact"] for entry in gate.values())
    outcome = ("gate-failed" if not gate_passed else "teacher-gate-failed" if not teacher["passed"]
               else "complete")
    shuffled = shuffle_table(cfg)
    report = {"format": "snowgym.m8-s16-command-control-audit-report.v0", "outcome": outcome,
        "regressionGate": {"passed": gate_passed, "cells": gate}, "teacher": teacher, "learned": learned,
        "predictions": prediction_scores(teacher, learned) if learned else None,
        "shuffledMismatchRate": float(np.mean([shuffled[k] != k[1] for k in shuffled])),
        "torchThreads": torch.get_num_threads(), "blockWorlds": cfg["blockWorlds"],
        "simulatorDecisions": spent, "totalSimulatorDecisions": sum(spent.values()),
        "budgetBound": budget_bound(cfg), "withinBudgetCap": sum(spent.values()) <= cfg["budgetCap"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "authorizes": "nothing; a follow-up needs its own declaration"}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s16-manifest.v0")
    return report


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    spent = {"gate": 0, "teacher": 0, "learned": 0}

    def accountant(stage):
        def account(count):
            spent[stage] += count
            if sum(spent.values()) > cfg["budgetCap"]:
                raise ValueError("M8-S16 budget exceeded")
        return account

    teacher, learned = {"passed": False}, None
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        policies = {cohort: s15.load_policies(cfg, cohort) for cohort in cfg["cohorts"]}
        gate = regression_gate(client, cfg, policies, root, accountant("gate"))
        print(json.dumps({"stage": "gate", "passed": all(e["exact"] for e in gate.values()), "decisions": spent}),
              flush=True)
        if all(entry["exact"] for entry in gate.values()):
            teacher_cells(client, cfg, root, accountant("teacher"))
            teacher = teacher_gate(root, cfg)
            print(json.dumps({"stage": "teacher", "passed": teacher["passed"], "decisions": spent}), flush=True)
            if teacher["passed"]:
                learned_cells(client, cfg, policies, root, accountant("learned"))
                learned = learned_analysis(root, cfg)
    return aggregate(root, cfg, gate, teacher, learned, spent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps({"outcome": report["outcome"], "decisions": report["totalSimulatorDecisions"]}))
