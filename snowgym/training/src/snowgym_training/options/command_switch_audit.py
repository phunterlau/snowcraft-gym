"""M8-S18: mid-fight command switches -- do the frozen S14 fighters redirect when the target command changes?
(`reviews/m8_s18_declaration.md`)

From S16's three-singleton layout, every branch executes flank command A from reset to decision k; at k the `keep`
branch does nothing, `reactivate` activates a new plan with the same selector, and `switch` activates the mirror
selector B. Branches share an identical prefix (same seed, commands, block and torch seed), which is checked. Scoring
is always against B with a branch-independent stopping rule. Reuses `command_control_audit` (S16) helpers unchanged;
no existing module is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_THROW
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT
from ..ppo_collect import tensor_dict
from ..trainer import resolve_git_commit
from . import command_control_audit as s16
from . import death_rate_ppo as dr
from . import enemy_relative_throw_ppo_retention as s15
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from .engage_v1 import EngageOptionBatchV1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged
from .interventions import require_capabilities
from .reservoir import file_digest
from .supervised_probe import write_json

POLICIES, MODES, TASKS, MIRROR = s16.POLICIES, s16.MODES, s16.TASKS, s16.MIRROR
SWITCH_TIMES = (30, 50, 60)
BRANCHES = ("keep", "reactivate", "switch")
PLAN_KEYS = ("plan_groups", "plan_group_mask", "plan_unit_roles", "plan_role_state", "mission_progress")
REPLAY_FIELDS = ("success", "failed", "timedOut", "finalDecision", "blueAliveAtEnd", "rejectedActions", "totalActions",
                 "fold", "firstHitDecision", "finalTargetDamage", "minDistance", "meanDistance", "distanceAtFirstHit",
                 "assignedUnits", "blueAliveCount", "unitsLostFraction", "flankOrder", "requestedCross",
                 "mirrorCross", "stopReason")
CLASSES = ("redirects", "partial", "ignores", "counter", "insufficient")


def configuration():
    return {**s16.configuration(), "format": "snowgym.m8-s18-command-switch-audit-config.v0",
        "s16Run": "m8_s16_command_control_audit_v0", "s17Run": "m8_s17_command_control_qualification_v0",
        "worldSeedBase": 2750000, "worlds": 192, "spreads": (20, 30),
        "reservedQualificationSeeds": [2760000, 2760399],
        "switchTimes": SWITCH_TIMES, "postWindow": 40, "lateralWindow": 20, "minEligibleWorlds": 48,
        "teacherPostDeltaLowerMin": .80, "redirectFraction": .5,
        "torchSeedBase": 988000, "bootstrapSeed": 988001, "budgetCap": 10_450_000,
        "assistType": "none at learned-policy runtime; plan-aware teacher only as a separate positive control",
        "assistVersion": "snowgym.m8-s18.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon, worlds = cfg["commandHorizon"], cfg["worlds"]
    gate = len(cfg["cohorts"]) * len(POLICIES) * len(MODES) * s16.configuration()["worlds"] * horizon
    cells = len(TASKS) * (len(cfg["switchTimes"]) * len(BRANCHES) + 1)
    teacher = worlds * cells * horizon
    learned = worlds * cells * len(cfg["cohorts"]) * len(POLICIES) * len(MODES) * horizon
    return {"gate": gate, "teacher": teacher, "learned": learned, "total": gate + teacher + learned}


def stochastic_torch_seed(cfg, cohort, policy, k, task):
    """Shared by all three branches of a (cohort, policy, k, task) cell (common random numbers); k=None is the
    B-from-start reference."""
    k_index = len(cfg["switchTimes"]) if k is None else cfg["switchTimes"].index(k)
    return cfg["torchSeedBase"] + 100 * cohort + 20 * k_index + 2 * TASKS.index(task) + POLICIES.index(policy)


# -- Tracker and batch (declaration §5) ----------------------------------------------------------------------------


class SwitchAuditTracker(s16.CommandAuditTracker):
    """S16's tracker plus post-switch flank damage and an optional branch-independent stopping rule (both flanks
    crossed, blue wiped, environment done, or the horizon)."""

    def configure_switch(self, *, k, post_window, stop_rule):
        if stop_rule not in ("s16", "both-flanks"):
            raise ValueError(f"unknown stop rule {stop_rule!r}")
        self.switch_k, self.post_window, self.stop_rule = k, post_window, stop_rule
        self.post_damage = {i: 0. for i in self.initial_enemy_health}

    def update(self, observation, plan_observation, *, canonical_reward, gamma, environment_done=False):
        before = dict(self._health)
        step = super().update(observation, plan_observation, canonical_reward=canonical_reward, gamma=gamma,
                              environment_done=environment_done)
        if self.switch_k is not None and self.switch_k < self.decision <= self.switch_k + self.post_window:
            for unit_id, health in before.items():
                self.post_damage[unit_id] += max(0., health - self._health.get(unit_id, 0.))
        if self.stop_rule == "both-flanks":
            flanks = (*self.requested_ids, *self.mirror_ids)
            blue_alive = any(u["alive"] and u["id"] in self.assigned_ids for u in observation["allies"])
            reasons = [("both-flanks-crossed", all(i in self.crossed for i in flanks)),
                       ("blue-wiped", not blue_alive), ("environment-done", environment_done),
                       ("horizon", self.decision >= self.horizon)]
            self.stop_reason = next((name for name, hit in reasons if hit), None)
            self.finished = self.stop_reason is not None
        return step


class SwitchAuditBatch(EngageOptionBatchV1):
    def _install_trackers(self, indices, plans, specs, bodies):
        for index, plan, spec, body in zip(indices, plans, specs, bodies, strict=True):
            raw = self.environment.raw_observations[index]
            if raw is None:
                raise RuntimeError("missing activation observation")
            self.trackers[index] = SwitchAuditTracker(spec, plan, raw, body)


# -- Runner (declaration §3, §4, §6.2) -------------------------------------------------------------------------------


def flank_health(raw, ids):
    health = {u["id"]: (max(0., float(u["health"])) if u["alive"] else 0.) for u in raw["enemies"]}
    return [health.get(i, 0.) for i in ids]


def centroid_y(raw):
    living = [u["y"] for u in raw["allies"] if u["alive"]]
    return float(np.mean(living)) if living else None


def aimed_key(actions, row, raw, groups):
    """Per-unit aim classification for THROW actions (S16 §4's bearing-closest rule)."""
    action_type, target = np.asarray(actions["action_type"])[row], np.asarray(actions["target"])[row]
    keys = []
    for slot, unit in enumerate(raw["allies"]):
        if not unit["alive"] or int(action_type[slot]) != ACTION_THROW:
            continue
        world = (float(target[slot, 0]) * ARENA_HALF_EXTENT[0], float(target[slot, 1]) * ARENA_HALF_EXTENT[1])
        aimed = s16.bearing_closest((unit["x"], unit["y"]), world, raw["enemies"])
        keys.append(next((name for name, ids in groups.items() if aimed in ids), "other"))
    return keys


def activate_at_k(wrapper, active, seeds, select, expected_ids, source, current):
    """Activate `select` in the active worlds, assert grounding/version, and refresh every plan tensor key."""
    _, before = wrapper.environment.plan_observations(active)
    bodies = wrapper.environment.activate_plan_indices(active, [f"s18-{source}-k-{seeds[i]}" for i in active],
                                                       [s16.engage_plan(select)] * len(active))
    for row, index in enumerate(active):
        main = next(item for item in bodies[row]["activationObjectives"] if item["role"] == "main")
        if tuple(main["enemyIds"]) != expected_ids[index]:
            raise RuntimeError(f"world {seeds[index]}: activation grounded {main['enemyIds']}, preview "
                               f"{expected_ids[index]}")
        if bodies[row]["version"] != before[row]["version"] + 1:
            raise RuntimeError(f"world {seeds[index]}: plan version did not increment at activation")
    tensors, _ = wrapper.environment.plan_observations(active)
    refreshed = tensor_dict(tensors)
    if set(refreshed) != set(PLAN_KEYS) or not set(PLAN_KEYS) <= set(current):
        raise RuntimeError("plan tensor keys changed; cannot refresh the actor input")
    for key in PLAN_KEYS:
        current[key][active] = refreshed[key]
    return [int(b["version"]) for b in bodies]


def run_switch_block(wrapper, block, cfg, *, initial_select, requested_select, mirror_select, k, branch, choose,
                     source, stop_rule):
    count = len(block)
    seeds = [seed for _, seed, _ in block]
    if wrapper.batch_size != count:
        raise ValueError("block size must match the wrapper batch size")
    if branch not in BRANCHES or (k is None and branch != "keep"):
        raise ValueError("a branch other than keep needs a switch time")
    observation, _ = wrapper.reset(seeds, [s16.scenario_for(spread) for _, _, spread in block],
                                   [f"s18-{source}-{seed}" for seed in seeds],
                                   [s16.engage_plan(initial_select)] * count, [v1.engage_spec(cfg)] * count)
    requested = s16.preview_ids(wrapper, seeds, requested_select, source)
    mirror = s16.preview_ids(wrapper, seeds, mirror_select, source)
    centre = s16.preview_ids(wrapper, seeds, "nearest", source)
    for index in range(count):
        if not (len(requested[index]) == len(mirror[index]) == len(centre[index]) == 1
                and len({*requested[index], *mirror[index], *centre[index]}) == 3):
            raise RuntimeError(f"world {seeds[index]}: requested/mirror/centre do not ground to three singletons")
        tracker = wrapper.trackers[index]
        tracker.configure(requested[index], mirror[index], horizon=cfg["commandHorizon"],
                          threshold=cfg["thresholdFraction"], early_window=cfg["earlyWindow"])
        tracker.configure_switch(k=k, post_window=cfg["postWindow"], stop_rule=stop_rule)
    side = [float(np.sign(np.mean([u["y"] for u in wrapper.environment.raw_observations[i]["enemies"]
                                   if u["id"] in requested[i]]))) for i in range(count)]
    groups = [{"requested": requested[i], "mirror": mirror[i], "centre": centre[i]} for i in range(count)]
    current = tensor_dict(observation)
    episodes = [{"seed": seed, "source": source, "rewards": [], "distances": [], "targetDamage": [],
                 "rejectedActions": 0, "totalActions": 0, "prefix": {"reachedK": False}, "postAim": {
                 "requested": 0, "mirror": 0, "centre": 0, "other": 0}, "yAtK": None, "lateral": None,
                 "versions": None} for seed in seeds]
    decisions = 0
    while True:
        active = [i for i in range(count) if not wrapper.trackers[i].finished]
        if not active:
            break
        decision = wrapper.trackers[active[0]].decision
        if any(wrapper.trackers[i].decision != decision for i in active):
            raise RuntimeError("active worlds must share the decision count")
        if k is not None and decision == k:
            at_initial = s16.preview_ids(wrapper, seeds, initial_select, f"{source}-at-k")
            at_other = s16.preview_ids(wrapper, seeds, MIRROR[initial_select], f"{source}-at-k")
            for index in active:
                tracker, raw = wrapper.trackers[index], wrapper.environment.raw_observations[index]
                flank_ids = (*requested[index], *mirror[index])
                initial_ids = mirror[index] if initial_select == mirror_select else requested[index]
                other_ids = requested[index] if initial_select == mirror_select else mirror[index]
                healths = flank_health(raw, flank_ids)
                blue_alive = any(u["alive"] and u["id"] in tracker.assigned_ids for u in raw["allies"])
                eligible = (blue_alive and not any(i in tracker.crossed for i in flank_ids)
                            and at_initial[index] == initial_ids and at_other[index] == other_ids
                            and all(h >= tracker.initial_enemy_health[i] for h, i in zip(healths, flank_ids)))
                episodes[index]["prefix"] = {"reachedK": True, "stateHash": wrapper.environment.state_hashes[index],
                    "decision": tracker.decision, "crossed": sorted(tracker.crossed.items()),
                    "flankHealth": healths, "groundings": [list(at_initial[index]), list(at_other[index])],
                    "eligible": bool(eligible)}
                episodes[index]["yAtK"] = centroid_y(raw)
            if branch != "keep":
                select = initial_select if branch == "reactivate" else MIRROR[initial_select]
                expected = at_initial if branch == "reactivate" else at_other
                versions = activate_at_k(wrapper, active, seeds, select, expected, source, current)
                for row, index in enumerate(active):
                    episodes[index]["versions"] = versions[row]
        rows = {key: value[active] for key, value in current.items()}
        raws = [wrapper.environment.raw_observations[i] for i in active]
        distances = [v1.target_distance(raw, wrapper.trackers[i]) for raw, i in zip(raws, active)]
        actions = choose(wrapper, active, rows, raws)
        if k is not None and k <= decision < k + cfg["postWindow"]:
            for row, index in enumerate(active):
                for key in aimed_key(actions, row, raws[row], groups[index]):
                    episodes[index]["postAim"][key] += 1
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
            if (k is not None and episode["lateral"] is None and episode["yAtK"] is not None
                    and (tracker.decision >= k + cfg["lateralWindow"] or tracker.finished)):
                y = centroid_y(raw)
                episode["lateral"] = None if y is None else side[index] * (y - episode["yAtK"])
            if tracker.finished:
                episode.update(success=bool(option["success"]), failed=bool(option["failed"]),
                    timedOut=bool(option["timedOut"]), finalDecision=int(option["decision"]),
                    blueAliveAtEnd=any(u["alive"] and u["id"] in tracker.assigned_ids for u in raw["allies"]))
    out = []
    for index, episode in enumerate(episodes):
        tracker = wrapper.trackers[index]
        allies = wrapper.environment.raw_observations[index]["allies"]
        alive = sum(1 for u in allies if u["alive"] and u["id"] in tracker.assigned_ids)
        requested_cross = min((tracker.crossed[i] for i in requested[index] if i in tracker.crossed), default=None)
        mirror_cross = min((tracker.crossed[i] for i in mirror[index] if i in tracker.crossed), default=None)
        out.append(v1.episode_row(episode) | {"assignedUnits": len(tracker.assigned_ids), "blueAliveCount": alive,
            "unitsLostFraction": rb.units_lost_fraction(len(tracker.assigned_ids), alive),
            "requestedIds": list(requested[index]), "mirrorIds": list(mirror[index]), "centreIds": list(centre[index]),
            "stopReason": tracker.stop_reason, "requestedCross": requested_cross, "mirrorCross": mirror_cross,
            "flankOrder": s16.flank_order(requested_cross, mirror_cross),
            "postContrast": None if k is None else s16.flank_contrast(
                sum(tracker.post_damage[i] for i in requested[index]),
                sum(tracker.post_damage[i] for i in mirror[index])),
            "requestedLatency": None if (k is None or requested_cross is None) else requested_cross - k,
            "prefix": episode["prefix"], "postAim": episode["postAim"], "lateralTowardRequested": episode["lateral"],
            "planVersionAfterK": episode["versions"]})
    return out, decisions


def run_cell(client, cfg, worlds, *, account, **kwargs):
    rows = []
    for start in range(0, len(worlds), cfg["blockWorlds"]):
        block = worlds[start:start + cfg["blockWorlds"]]
        wrapper = SwitchAuditBatch(v1.SelectiveBatchEnv(len(block), client=client, observation_version=3),
                                   gamma=cfg["gamma"])
        found, used = run_switch_block(wrapper, block, cfg, **kwargs)
        account(used)
        rows.extend(found)
    return rows


def cell_path(root, group, task, k, branch):
    label = "reference" if k is None else f"k{k}"
    return Path(root) / group / task / label / branch


# -- Gate 1: S16 replay (declaration §6.1) ----------------------------------------------------------------------------


def replay_gate(client, cfg, policies, root, account):
    s16cfg = s16.configuration()
    worlds = s16.panel(s16cfg)
    archive = TRAINING / "runs" / cfg["s16Run"] / "learned"
    results = {}
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                if mode == "stochastic":
                    torch.manual_seed(s16.stochastic_torch_seed(s16cfg, cohort, policy, "correct", "leftmost"))
                rows = run_cell(client, cfg, worlds, account=account, initial_select="leftmost",
                    requested_select="leftmost", mirror_select="rightmost", k=None, branch="keep",
                    choose=s16.model_chooser(policies[cohort][policy], mode == "deterministic"),
                    source=f"s18-replay-c{cohort}-{policy}-{mode}", stop_rule="s16")
                s16.write_rows(Path(root) / "gate" / f"cohort-{cohort}" / f"{policy}-{mode}", rows)
                archived = {r["seed"]: r for r in s16.read_rows(
                    archive / f"cohort-{cohort}" / policy / mode / "correct" / "leftmost")}
                mismatched = [r["seed"] for r in rows if r["seed"] not in archived
                              or any(r[f] != archived[r["seed"]][f] for f in REPLAY_FIELDS)]
                results[f"{cohort}/{policy}-{mode}"] = {"worlds": len(rows), "mismatchedSeeds": mismatched,
                                                        "exact": not mismatched and len(rows) == len(worlds)}
    return results


# -- Teacher and learned cells -----------------------------------------------------------------------------------------


def cells_for(cfg):
    """(task, k, branch) for every switch cell, then the B-from-start reference per task."""
    switch = [(task, k, branch) for k in cfg["switchTimes"] for task in TASKS for branch in BRANCHES]
    return switch + [(task, None, "keep") for task in TASKS]


def run_group(client, cfg, root, group, choose_for, account, seed_for=None):
    worlds = s16.panel(cfg)
    for task, k, branch in cells_for(cfg):
        if seed_for is not None:
            seed_for(k, task)
        initial = task if k is not None else MIRROR[task]
        rows = run_cell(client, cfg, worlds, account=account, initial_select=initial,
            requested_select=MIRROR[task], mirror_select=task, k=k, branch=branch, choose=choose_for(),
            source=f"s18-{group.replace('/', '-')}-{task}-{k}-{branch}", stop_rule="both-flanks")
        s16.write_rows(cell_path(root, group, task, k, branch), rows)


def teacher_cells(client, cfg, root, account):
    run_group(client, cfg, root, "teacher", lambda: s16.teacher_chooser, account)


def learned_cells(client, cfg, policies, root, account):
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                group = f"learned/cohort-{cohort}/{policy}/{mode}"
                model = policies[cohort][policy]

                def seed_for(k, task, cohort=cohort, policy=policy, mode=mode):
                    if mode == "stochastic":
                        torch.manual_seed(stochastic_torch_seed(cfg, cohort, policy, k, task))

                run_group(client, cfg, root, group, lambda m=model, d=(mode == "deterministic"):
                          s16.model_chooser(m, d), account, seed_for=seed_for)
            print(json.dumps({"stage": "learned", "cohort": cohort, "policy": policy}), flush=True)


# -- Analysis (declaration §6.3, §7) -----------------------------------------------------------------------------------


def load_group(root, group, cfg):
    worlds = s16.panel(cfg)
    expected = sorted(seed for _, seed, _ in worlds)
    cells = {}
    for task, k, branch in cells_for(cfg):
        rows = s16.read_rows(cell_path(root, group, task, k, branch))
        if sorted(r["seed"] for r in rows) != expected:
            raise RuntimeError(f"{group} {task} {k} {branch}: rows do not cover the declared panel exactly")
        cells[(task, k, branch)] = {r["seed"]: r for r in rows}
    return cells


PREFIX_KEYS = ("reachedK", "stateHash", "decision", "crossed", "flankHealth", "groundings", "eligible")


def prefix_mismatches(cells, k):
    bad = 0
    for task in TASKS:
        keep = cells[(task, k, "keep")]
        for branch in ("reactivate", "switch"):
            other = cells[(task, k, branch)]
            for seed, row in keep.items():
                a, b = row["prefix"], other[seed]["prefix"]
                bad += any(a.get(key) != b.get(key) for key in PREFIX_KEYS)
    return bad


def eligible_pairs(cells, k):
    """(seed, task) pairs eligible at k (prefix-defined, identical across branches once prefix identity holds)."""
    return {(seed, task) for task in TASKS for seed, row in cells[(task, k, "keep")].items()
            if row["prefix"].get("eligible")}


def world_series(cells, k, branch, metric, pairs, worlds):
    """Per-world mean of `metric` over that world's pairs in `pairs`, for worlds with at least one pair."""
    values = []
    for _, seed, _ in worlds:
        tasks = [task for task in TASKS if (seed, task) in pairs]
        if tasks:
            values.append(float(np.mean([metric(cells[(task, k, branch)][seed]) for task in tasks])))
    return values


METRICS = {"postContrast": lambda r: r["postContrast"], "flankOrder": lambda r: r["flankOrder"],
           "unitsLostFraction": lambda r: r["unitsLostFraction"], "teamWipe": lambda r: float(r["blueAliveCount"] == 0)}


def contrast_set(cells, k, pairs, worlds, cfg):
    def diff(first, second, metric):
        return fi.paired_difference(world_series(cells, k, first, METRICS[metric], pairs, worlds),
                                    world_series(cells, k, second, METRICS[metric], pairs, worlds),
                                    samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
    return {f"{a}-{b}": {metric: diff(a, b, metric) for metric in METRICS}
            for a, b in (("switch", "reactivate"), ("reactivate", "keep"), ("switch", "keep"))}


def branch_summary(cells, k, branch, pairs):
    rows = [cells[(task, k, branch)][seed] for seed, task in pairs]
    aim = {key: sum(r["postAim"][key] for r in rows) for key in ("requested", "mirror", "centre", "other")}
    throws = sum(aim.values())
    lateral = [r["lateralTowardRequested"] for r in rows if r["lateralTowardRequested"] is not None]
    latency = [r["requestedLatency"] for r in rows if r["requestedLatency"] is not None]
    return {"pairs": len(rows), "postContrast": float(np.mean([r["postContrast"] for r in rows])) if rows else None,
            "flankOrder": float(np.mean([r["flankOrder"] for r in rows])) if rows else None,
            "unitsLostFraction": float(np.mean([r["unitsLostFraction"] for r in rows])) if rows else None,
            "teamWipe": float(np.mean([r["blueAliveCount"] == 0 for r in rows])) if rows else None,
            "aimShares": {key: (v / throws if throws else None) for key, v in aim.items()},
            "meanLateralTowardRequested": float(np.mean(lateral)) if lateral else None,
            "medianRequestedLatency": float(np.median(latency)) if latency else None,
            "requestedCrossedRate": float(np.mean([r["requestedCross"] is not None for r in rows])) if rows else None,
            "stopReasons": {reason: sum(r["stopReason"] == reason for r in rows) for reason in
                            ("both-flanks-crossed", "blue-wiped", "environment-done", "horizon")}}


def analyse_group(cells, cfg):
    worlds = s16.panel(cfg)
    out = {}
    for k in cfg["switchTimes"]:
        pairs = eligible_pairs(cells, k)
        eligible_worlds = len({seed for seed, _ in pairs})
        entry = {"eligiblePairs": len(pairs), "eligibleWorlds": eligible_worlds,
                 "reachedK": sum(cells[(t, k, "keep")][s]["prefix"]["reachedK"] for t in TASKS for _, s, _ in worlds),
                 "prefixMismatches": prefix_mismatches(cells, k),
                 "summaries": {branch: branch_summary(cells, k, branch, pairs) for branch in BRANCHES}}
        if eligible_worlds >= cfg["minEligibleWorlds"]:
            entry["contrasts"] = contrast_set(cells, k, pairs, worlds, cfg)
            entry["bySpread"] = {f"d{d}": contrast_set(cells, k, pairs, [w for w in worlds if w[2] == d], cfg)
                                 ["switch-reactivate"]["postContrast"] for d in cfg["spreads"]
                                 if len({s for s, _ in pairs} & {w[1] for w in worlds if w[2] == d}) >= 2}
            entry["bySide"] = {task: contrast_set(cells, k, {p for p in pairs if p[1] == task}, worlds, cfg)
                               ["switch-reactivate"]["postContrast"] for task in TASKS
                               if sum(p[1] == task for p in pairs) >= 2}
        out[f"k{k}"] = entry
    reference = [r for task in TASKS for r in cells[(task, None, "keep")].values()]
    out["reference"] = {"flankOrder": float(np.mean([r["flankOrder"] for r in reference])),
                        "unitsLostFraction": float(np.mean([r["unitsLostFraction"] for r in reference]))}
    return out


def teacher_validity(teacher, cfg):
    valid = {}
    for k in cfg["switchTimes"]:
        entry = teacher[f"k{k}"]
        delta = entry.get("contrasts", {}).get("switch-reactivate", {}).get("postContrast")
        valid[k] = bool(delta) and entry["eligibleWorlds"] >= cfg["minEligibleWorlds"] and \
            delta["interval95"][0] >= cfg["teacherPostDeltaLowerMin"]
    return valid


def classify(entry, teacher_mean, cfg):
    if entry["eligibleWorlds"] < cfg["minEligibleWorlds"] or "contrasts" not in entry:
        return "insufficient"
    delta = entry["contrasts"]["switch-reactivate"]["postContrast"]
    low, high = delta["interval95"]
    if delta["mean"] >= cfg["redirectFraction"] * teacher_mean and low > 0:
        return "redirects"
    if low > 0:
        return "partial"
    if high < 0:
        return "counter"
    return "ignores"


def averaged(cell_sets, cfg):
    """Cohort-averaged contrasts on (seed, task) pairs eligible for every policy in the set (declaration §4)."""
    worlds = s16.panel(cfg)
    out = {}
    for k in cfg["switchTimes"]:
        common = set.intersection(*(eligible_pairs(cells, k) for cells in cell_sets))
        entry = {"commonEligiblePairs": len(common), "commonEligibleWorlds": len({s for s, _ in common})}
        if entry["commonEligibleWorlds"] >= cfg["minEligibleWorlds"]:
            def series(branch, metric):
                per_policy = [world_series(cells, k, branch, METRICS[metric], common, worlds) for cells in cell_sets]
                return list(np.mean(per_policy, axis=0))
            entry["switch-reactivate"] = {metric: fi.paired_difference(series("switch", metric),
                series("reactivate", metric), samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
                for metric in METRICS}
            entry["reactivate-keep"] = {metric: fi.paired_difference(series("reactivate", metric),
                series("keep", metric), samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
                for metric in METRICS}
        out[f"k{k}"] = entry
    return out


def learned_analysis(root, cfg, teacher, valid):
    policies, sets = {}, {}
    for cohort in cfg["cohorts"]:
        for policy in POLICIES:
            for mode in MODES:
                cells = load_group(root, f"learned/cohort-{cohort}/{policy}/{mode}", cfg)
                sets.setdefault(f"{policy}-{mode}", []).append(cells)
                analysis = analyse_group(cells, cfg)
                for k in cfg["switchTimes"]:
                    entry = analysis[f"k{k}"]
                    teacher_mean = teacher[f"k{k}"].get("contrasts", {}).get("switch-reactivate", {}).get(
                        "postContrast", {}).get("mean")
                    entry["class"] = classify(entry, teacher_mean, cfg) if valid[k] else "teacher-invalid"
                policies[f"c{cohort}-{policy}-{mode}"] = analysis
    return {"policies": policies, "cohortAveraged": {key: averaged(value, cfg) for key, value in sets.items()}}


def prediction_scores(teacher, valid, learned, cfg):
    first, last = (f"k{cfg['switchTimes'][0]}", f"k{cfg['switchTimes'][-1]}")  # k = 30 and k = 60 (declaration §8)
    finals = {c: learned["policies"][f"c{c}-final-deterministic"] for c in cfg["cohorts"]}
    avg60 = learned["cohortAveraged"]["final-deterministic"].get(last, {}).get("switch-reactivate", {})
    return {"1-teacherValidAllK": all(valid.values()),
        "2-finalsRedirectAtK30": all(f[first]["class"] == "redirects" for f in finals.values()),
        "3-notAllFinalsRedirectAtK60": not all(f[last]["class"] == "redirects" for f in finals.values()),
        "4-reactivationSideEffectSmall": all(abs(f[f"k{k}"]["contrasts"]["reactivate-keep"]["postContrast"]["mean"])
            < .20 for f in finals.values() for k in cfg["switchTimes"] if valid[k] and "contrasts" in f[f"k{k}"]),
        "5-switchCostsUnitsAtK60": (avg60.get("unitsLostFraction", {}).get("mean", 0) > 0) if avg60 else None,
        "finalClasses": {c: {f"k{k}": f[f"k{k}"]["class"] for k in cfg["switchTimes"]} for c, f in finals.items()}}


# -- Declaration, run, archive ----------------------------------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    runs = TRAINING / "runs"
    reused = file_digest(Path(s16.__file__).resolve())
    sealed = json.loads((runs / cfg["s16Run"] / "declaration.json").read_text(encoding="utf-8"))
    if reused != sealed["implementationDigest"]:
        raise RuntimeError("command_control_audit.py differs from the implementation S16 sealed")
    manifests = {name: dr.verify_sealed(runs / cfg[key]) for name, key in (
        ("s17", "s17Run"), ("s16", "s16Run"), ("s15", "s15Run"), ("s14Training", "finalRun"),
        ("s14Probe", "probeRun"), ("s12", "sourceRun"))}
    checkpoints = {str(c): {"final": file_digest(s15.final_checkpoint_path(cfg, c)),
                            "initializer": file_digest(s15.initializer_checkpoint_path(cfg, c))}
                   for c in cfg["cohorts"]}
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "sourceManifests": manifests, "checkpointDigests": checkpoints,
        "reusedImplementationDigest": reused,
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s18_declaration.md"),
        "implementationDigest": file_digest(Path(__file__).resolve()), "pinnedE3Digests": pinned,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, gate, teacher, valid, learned, spent):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    gate_passed = bool(gate) and all(entry["exact"] for entry in gate.values())
    outcome = ("gate-failed" if not gate_passed else "teacher-invalid" if not any((valid or {}).values())
               else "complete")
    prefix = None
    if learned:
        prefix = {key: {k: entry[k]["prefixMismatches"] for k in entry if k.startswith("k")}
                  for key, entry in learned["policies"].items()}
        prefix["teacher"] = {k: teacher[k]["prefixMismatches"] for k in teacher if k.startswith("k")}
    report = {"format": "snowgym.m8-s18-command-switch-audit-report.v0", "outcome": outcome,
        "replayGate": {"passed": gate_passed, "cells": gate}, "teacher": teacher,
        "teacherValidByK": {f"k{k}": v for k, v in (valid or {}).items()}, "learned": learned,
        "prefixMismatches": prefix,
        "prefixIdentityHolds": None if prefix is None else all(v == 0 for g in prefix.values() for v in g.values()),
        "predictions": prediction_scores(teacher, valid, learned, cfg) if learned else None,
        "torchThreads": torch.get_num_threads(), "blockWorlds": cfg["blockWorlds"], "simulatorDecisions": spent,
        "totalSimulatorDecisions": sum(spent.values()), "budgetBound": budget_bound(cfg),
        "withinBudgetCap": sum(spent.values()) <= cfg["budgetCap"], "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "authorizes": "nothing; a follow-up needs its own declaration"}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s18-manifest.v0")
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
                raise ValueError("M8-S18 budget exceeded")
        return account

    teacher, valid, learned = {}, {}, None
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        policies = {cohort: s15.load_policies(cfg, cohort) for cohort in cfg["cohorts"]}
        gate = replay_gate(client, cfg, policies, root, accountant("gate"))
        print(json.dumps({"stage": "gate", "passed": all(e["exact"] for e in gate.values()), "decisions": spent}),
              flush=True)
        if all(entry["exact"] for entry in gate.values()):
            teacher_cells(client, cfg, root, accountant("teacher"))
            teacher = analyse_group(load_group(root, "teacher", cfg), cfg)
            valid = teacher_validity(teacher, cfg)
            print(json.dumps({"stage": "teacher", "validByK": valid, "decisions": spent}), flush=True)
            if any(valid.values()):
                learned_cells(client, cfg, policies, root, accountant("learned"))
                learned = learned_analysis(root, cfg, teacher, valid)
    return aggregate(root, cfg, gate, teacher, valid, learned, spent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args().output)
    print(json.dumps({"outcome": report["outcome"], "decisions": report["totalSimulatorDecisions"]}))
