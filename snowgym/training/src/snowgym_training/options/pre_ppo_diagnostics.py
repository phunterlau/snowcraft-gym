"""R1n-d: diagnostics before PPO for R1n-c's final imitation policies.

D1 splits the stochastic-execution gap across six sampling modes. D2 bounds the
attainable critic R^2 with branched rollouts that replay recorded stochastic
prefixes. D3 fits critic variants C0-C3 on new folds. No actor or PPO update
runs here. See `reviews/m7b_r1n_d_declaration.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..checkpoint import semantic_state_digest
from ..executor.full_authority_critics import EgocentricCritic
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, outcome_summary, write_episodes
from .interventions import require_capabilities
from .reservoir import file_digest
from .supervised_probe import write_json

MODES = {"det": (False, False, 1.), "type": (True, False, 1.), "cont1": (False, True, 1.),
         "full1": (True, True, 1.), "full05": (True, True, .5), "full025": (True, True, .25)}
VARIANTS = ("C0", "C1", "C2", "C3")


def configuration():
    seeds = [97101, 97102, 97103]
    return {**v1.configuration(), "format": "snowgym.pre-ppo-diagnostics-config.v0", "destination": "global",
        "policySeeds": seeds, "trainingRngs": seeds, "sourceRun": "runs/m7b_engage_r1n_c_v0",
        "policyCheckpoint": "fit-4.pt",
        "modes": {name: {"sampleType": t, "sampleContinuous": c, "sigmaScale": s} for name, (t, c, s) in MODES.items()},
        "d1Seeds": [680000, 680099], "evaluationBlockWorlds": 50,
        "d2SourceSeedBase": 690000, "d2SourceEpisodes": 24, "branchDecisions": [0, 25, 50, 75, 100, 125],
        "rolloutsPerState": 8,
        "trainSeedBase": 684000, "heldOutSeedBase": 687000, "seedBandStride": 1000,
        "warmStartTrainEpisodes": 256, "warmStartHeldOutEpisodes": 128, "validationFromIndex": 204,
        "criticBaselineEpochs": 10, "maxCriticEpochs": 200, "earlyStoppingPatience": 10, "egocentricHidden": 256,
        "decisionWindows": [[0, 49], [50, 99], [100, None]], "branchWindowRadius": 2,
        "samplingSeedBase": 975000, "rolloutSeedBase": 976000, "criticSeedBase": 97201,
        "bootstrapSeed": 974001, "bootstrapSamples": 10000,
        "sigmaScales": [1., .5, .25], "sigmaGapLimit": .10, "typeSamplingLimit": .10,
        "simulatorBudget": 1350000, "assistType": "none at runtime; policies from teacher-imitation training",
        "assistVersion": "snowgym.pre-ppo-diagnostics.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    policies, horizon = len(cfg["policySeeds"]), cfg["optionHorizon"]
    d1 = policies * len(cfg["modes"]) * (cfg["d1Seeds"][1] - cfg["d1Seeds"][0] + 1) * horizon
    episodes, states = cfg["d2SourceEpisodes"], len(cfg["branchDecisions"]) * cfg["rolloutsPerState"]
    d2 = policies * (episodes * horizon + episodes * states * horizon)
    d3 = policies * (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    return {"d1": d1, "d2": d2, "d3": d3, "total": d1 + d2 + d3}


# -- Sampling modes (declaration §2) ---------------------------------------------------


def sample_actions(model, observation, *, sample_type, sample_continuous, sigma_scale=1., latents=False):
    """`FullAuthorityPolicyV1.act`'s draws and decoding, with the type and continuous draws
    switched independently and all three standard deviations scaled by `sigma_scale`."""
    prediction = model(observation, with_value=False)
    offset = math.log(sigma_scale)
    move_std = (model.move_log_std + offset).exp()
    throw_std = (model.throw_log_std + offset).exp()
    power_std = (model.power_log_std + offset).exp()
    move_mean, throw_mean, power_mean = prediction["move_raw"], prediction["throw_raw"], prediction["power_raw"]
    action_type = (Categorical(logits=prediction["action_logits"]).sample() if sample_type
                   else prediction["action_logits"].argmax(-1))
    move_latent = Normal(move_mean, move_std).sample() if sample_continuous else move_mean
    throw_latent = Normal(throw_mean, throw_std).sample() if sample_continuous else throw_mean
    power_latent = Normal(power_mean, power_std).sample() if sample_continuous else power_mean
    raw = {"move": move_latent, "throw": throw_latent, "power": power_latent}
    moves, throws = action_type == ACTION_MOVE, action_type == ACTION_THROW
    move_latent = torch.where(moves[..., None], move_latent, torch.zeros_like(move_latent))
    throw_latent = torch.where(throws[..., None], throw_latent, torch.zeros_like(throw_latent))
    power_latent = torch.where(throws, power_latent, torch.zeros_like(power_latent))
    move_target = model.decode_move(observation, move_latent)
    target = torch.where(moves[..., None], move_target, torch.where(throws[..., None], torch.tanh(throw_latent),
                                                                    torch.zeros_like(move_target)))
    action = {"action_type": action_type, "target": target, "power": torch.sigmoid(power_latent)}
    return (action, raw) if latents else action


def mode_chooser(model, mode):
    sample_type, sample_continuous, scale = mode

    def choose(_wrapper, _active, rows, _raws):
        with torch.no_grad():
            return numpy_actions(sample_actions(model, rows, sample_type=sample_type,
                                                sample_continuous=sample_continuous, sigma_scale=scale))

    return choose


def collect_seeds(client, seeds, cfg, *, choose, source, block_worlds, account):
    episodes = []
    for start in range(0, len(seeds), block_worlds):
        block = seeds[start:start + block_worlds]
        found, _, used = v1.run_block(v1.make_wrapper(client, len(block), cfg["gamma"]), block, cfg, choose=choose,
                                      source=source)
        account(used)
        episodes.extend(found)
    return episodes


def summary(episodes, cfg):
    result = outcome_summary(episodes, cfg)
    result["timeoutFraction"] = sum(e["timedOut"] for e in episodes) / len(episodes) if episodes else None
    return result


def exploration_rules(d1, cfg):
    """Seed-averaged success gaps `det - mode` (fractions) and the §5 exploration rules."""
    seeds = [str(s) for s in cfg["policySeeds"]]
    gaps = {mode: float(np.mean([d1[s]["det"]["successFraction"] - d1[s][mode]["successFraction"] for s in seeds]))
            for mode in cfg["modes"]}
    full = {spec["sigmaScale"]: name for name, spec in cfg["modes"].items()
            if spec["sampleType"] and spec["sampleContinuous"]}
    limit = cfg["sigmaGapLimit"] + 1e-9
    recommended = next((scale for scale in cfg["sigmaScales"] if gaps[full[scale]] <= limit), None)
    return {"seedAveragedGaps": gaps, "recommendedSigmaScale": recommended,
            "typeSamplingDominates": bool(gaps["type"] > gaps["cont1"]
                                          and gaps["type"] > cfg["typeSamplingLimit"] + 1e-9),
            "perPolicyGaps": {s: {m: d1[s]["det"]["successFraction"] - d1[s][m]["successFraction"]
                                  for m in cfg["modes"]} for s in seeds}}


# -- Critic variants (declaration §4) --------------------------------------------------


def critic_values(critic, observation, chunk=4096):
    size = len(next(iter(observation.values())))
    with torch.no_grad():
        return torch.cat([critic({k: v[s:s + chunk] for k, v in observation.items()}) for s in range(0, size, chunk)])


def mean_squared_error(critic, data):
    return float((critic_values(critic, data["observation"]) - data["returns"]).square().mean())


def subset(data, mask):
    return {"observation": {k: v[mask] for k, v in data["observation"].items()},
            **{k: data[k][mask] for k in ("episode", "decision", "seed", "returns")}}


def validation_split(data, cfg, index):
    """By episode: train-fold seeds at or after `validationFromIndex` form the validation split."""
    first = cfg["trainSeedBase"] + cfg["seedBandStride"] * index + cfg["validationFromIndex"]
    held = data["seed"] >= first
    return subset(data, ~held), subset(data, held)


def train_critic(critic, train, cfg, *, epochs, seed, clip, validation=None):
    """Monte Carlo regression as in R1n-b's warm start. With `validation`, stop after
    `earlyStoppingPatience` epochs without improvement and restore the best epoch."""
    parameters = list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["learningRate"])
    generator = torch.Generator().manual_seed(seed)
    size = len(train["returns"])
    history, best, stale = [], None, 0
    for epoch in range(epochs):
        norms, squared = [], 0.
        for indices in torch.randperm(size, generator=generator).split(cfg["minibatchSize"]):
            value = critic({k: v[indices] for k, v in train["observation"].items()})
            loss = (value - train["returns"][indices]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite critic loss")
            squared += float(loss.detach()) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, math.inf if clip is None else clip,
                                                              error_if_nonfinite=True)))
            optimizer.step()
        # Running minibatch MSE (before each step); a full train pass would double the epoch cost.
        row = {"epoch": epoch, "trainMseRunning": squared / size,
               "gradientNormMedian": float(np.median(norms)), "clippedFraction": float(np.mean(np.asarray(norms) > .5))}
        if validation is not None:
            row["validationMse"] = mean_squared_error(critic, validation)
            if best is None or row["validationMse"] < best["validationMse"]:
                best, stale = {"epoch": epoch, "validationMse": row["validationMse"],
                               "state": copy.deepcopy(critic.state_dict())}, 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= cfg["earlyStoppingPatience"]:
            break
    if best is not None:
        critic.load_state_dict(best["state"])
    return {"history": history, "epochsRun": len(history), "bestEpoch": None if best is None else best["epoch"],
            "bestValidationMse": None if best is None else best["validationMse"]}


def window_r2(values, data, cfg):
    rows = {}
    for low, high in cfg["decisionWindows"]:
        mask = data["decision"] >= low
        if high is not None:
            mask &= data["decision"] <= high
        target = data["returns"][mask].double()
        variance = float(target.var(unbiased=False)) if len(target) > 1 else 0.
        rows[f"{low}-{'' if high is None else high}"] = (
            None if variance <= 1e-12 else 1 - float((values[mask].double() - target).square().mean()) / variance)
    return rows


def branch_window_r2(values, data, cfg):
    """Held-out R^2 on rows within `branchWindowRadius` decisions of each D2 branch decision, pooled
    and per k: the in-distribution comparison for D2's `capture` (amendment A9)."""
    radius = cfg["branchWindowRadius"]

    def r2(mask):
        target = data["returns"][mask].double()
        variance = float(target.var(unbiased=False)) if len(target) > 1 else 0.
        return None if variance <= 1e-12 else 1 - float((values[mask].double() - target).square().mean()) / variance

    near = {str(k): (data["decision"] - k).abs() <= radius for k in cfg["branchDecisions"]}
    pooled = torch.zeros_like(data["decision"], dtype=torch.bool)
    for mask in near.values():
        pooled |= mask
    return {"pooled": r2(pooled), "byDecision": {k: r2(mask) for k, mask in near.items()}}


def held_out_report(values, train, held, cfg):
    metrics = v1.critic_metrics(values, held["returns"], held["decision"], train["returns"], train["decision"],
                                horizon=cfg["optionHorizon"], bins=cfg["timeBins"])
    return {**metrics, "predictiveR2Interval95": v1.bootstrap_predictive_r2(
                values, held["returns"], held["episode"], samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]),
            "gatePassed": v1.critic_gate(metrics, cfg), "gateConditions": v1.gate_conditions(metrics, cfg),
            "decisionWindowR2": window_r2(values, held, cfg), "branchWindowR2": branch_window_r2(values, held, cfg)}


def fit_variants(model, train, held, cfg, index):
    """C0-C3 for policy `index`; returns the trained critics and their reports."""
    base = cfg["criticSeedBase"] + 10 * index
    fit, validation = validation_split(train, cfg, index)
    critics, reports = {}, {}
    for number, name in enumerate(VARIANTS):
        if name in ("C0", "C1"):
            critic = copy.deepcopy(model.critic).requires_grad_(True)
        else:
            torch.manual_seed(base)
            critic = EgocentricCritic(cfg["egocentricHidden"])
        if name == "C0":
            training = train_critic(critic, train, cfg, epochs=cfg["criticBaselineEpochs"], seed=base + number,
                                    clip=cfg["criticGradClip"])
        else:
            training = train_critic(critic, fit, cfg, epochs=cfg["maxCriticEpochs"], seed=base + number,
                                    clip=None if name == "C3" else cfg["criticGradClip"], validation=validation)
        values = critic_values(critic, held["observation"])
        critics[name] = critic
        reports[name] = {"training": training, "heldOut": held_out_report(values, train, held, cfg), "values": values}
    selected = min(("C1", "C2", "C3"), key=lambda name: reports[name]["training"]["bestValidationMse"])
    return critics, reports, selected


# -- Branched rollouts (declaration §3) ------------------------------------------------


def row_of(rows, index):
    return {key: value[index].clone() for key, value in rows.items()}


class SourceRecorder:
    """Samples at sigma x1 and records every action, every row digest, and the rows at branch decisions."""

    def __init__(self, model, count, branches):
        self.model, self.branches = model, set(branches)
        self.actions, self.digests = [[] for _ in range(count)], [[] for _ in range(count)]
        self.rows, self.steps = {}, [0] * count

    def __call__(self, _wrapper, active, rows, _raws):
        with torch.no_grad():
            action = numpy_actions(sample_actions(self.model, rows, sample_type=True, sample_continuous=True))
        for index, world in enumerate(active):
            step = self.steps[world]
            row = row_of(rows, index)
            self.actions[world].append({key: value[index].copy() for key, value in action.items()})
            self.digests[world].append(semantic_state_digest(row))
            if step in self.branches:
                self.rows[(world, step)] = row
            self.steps[world] += 1
        return action


class BranchReplayer:
    """World w replays the source's first `branch[w]` actions, must match the source row digest at
    that decision, and then samples at sigma x1."""

    def __init__(self, model, actions, digests, branches):
        self.model, self.actions, self.digests, self.branches = model, actions, digests, branches
        self.steps, self.identity = [0] * len(branches), [None] * len(branches)

    def __call__(self, _wrapper, active, rows, _raws):
        with torch.no_grad():
            action = numpy_actions(sample_actions(self.model, rows, sample_type=True, sample_continuous=True))
        for index, world in enumerate(active):
            step, branch = self.steps[world], self.branches[world]
            if step == branch:
                self.identity[world] = semantic_state_digest(row_of(rows, index)) == self.digests[step]
                if not self.identity[world]:
                    raise RuntimeError(f"replay identity mismatch at decision {step}")
            if step < branch:
                for key in action:
                    action[key][index] = self.actions[step][key]
            self.steps[world] += 1
        return action


def collect_source(client, model, seeds, cfg, *, account):
    recorder = SourceRecorder(model, len(seeds), cfg["branchDecisions"])
    episodes, _, used = v1.run_block(v1.make_wrapper(client, len(seeds), cfg["gamma"]), seeds, cfg, choose=recorder,
                                     source="d2-source")
    account(used)
    return episodes, recorder


def run_rollouts(client, model, episode, actions, digests, cfg, *, account, source):
    """All rollouts for one source episode, one world per (branch decision, rollout)."""
    plan = [(k, r) for k in cfg["branchDecisions"] if k < len(episode["rewards"])
            for r in range(cfg["rolloutsPerState"])]
    if not plan:
        return []
    replayer = BranchReplayer(model, actions, digests, [k for k, _ in plan])
    found, _, used = v1.run_block(v1.make_wrapper(client, len(plan), cfg["gamma"]), [episode["seed"]] * len(plan), cfg,
                                  choose=replayer, source=source)
    account(used)
    if not all(replayer.identity):
        raise RuntimeError("a rollout never reached its branch decision")
    return [{**v1.episode_row(e), "branchDecision": k, "rollout": r, "replayIdentity": True,
             "return": v1.monte_carlo_returns(e["rewards"], cfg["gamma"])[k]} for e, (k, r) in zip(found, plan)]


def ceiling_statistics(returns):
    """ANOVA decomposition for n states x m rollouts: V = B - W/m estimates Var(E[G|s])."""
    returns = np.asarray(returns, dtype=float)
    states, rollouts = returns.shape
    if states < 2 or rollouts < 2:
        return {"states": int(states), "rollouts": int(rollouts), "ceilingR2": None}
    within = float(returns.var(axis=1, ddof=1).mean())
    between = float(returns.mean(axis=1).var(ddof=1))
    value = between - within / rollouts
    total = value + within
    return {"states": int(states), "rollouts": int(rollouts), "withinVariance": within, "betweenVariance": between,
            "stateValueVariance": value, "totalVariance": total, "ceilingR2": value / total if total > 0 else None}


def branch_metrics(values, returns):
    """`rolloutR2` on individual rollout returns and the bias-corrected `capture` of Var(E[G|s])."""
    values, returns = np.asarray(values, dtype=float), np.asarray(returns, dtype=float)
    stats = ceiling_statistics(returns)
    spread = float(np.mean((returns - returns.mean()) ** 2))
    rollout_r2 = 1 - float(np.mean((returns - values[:, None]) ** 2)) / spread if spread > 0 else None
    capture = None
    if stats["ceilingR2"] is not None and stats["stateValueVariance"] > 0:
        error = float(np.mean((values - returns.mean(axis=1)) ** 2)) - stats["withinVariance"] / returns.shape[1]
        capture = 1 - error / stats["stateValueVariance"]
    return {"rolloutR2": rollout_r2, "capture": capture}


def cluster_interval(clusters, statistic, *, samples, seed):
    """95% percentile interval, resampling clusters (source episodes) with all their states."""
    generator = np.random.default_rng(seed)
    keys = sorted(clusters)
    estimates = []
    for _ in range(samples):
        pick = generator.integers(0, len(keys), len(keys))
        value = statistic(np.concatenate([clusters[keys[p]] for p in pick]))
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None
    return [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]


def ceiling_report(states, returns, critic_branch_values, cfg):
    """`states[i] = (episode, decision)`; `returns[i]` holds that state's m rollout returns."""
    returns = np.asarray(returns, dtype=float)
    episodes = np.asarray([episode for episode, _ in states])
    decisions = np.asarray([decision for _, decision in states])

    def clusters(mask):
        return {int(e): np.flatnonzero(mask & (episodes == e)) for e in np.unique(episodes[mask])}

    def interval(mask, statistic):
        return cluster_interval(clusters(mask), statistic, samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])

    everything = np.ones(len(states), dtype=bool)
    report = {"pooled": {**ceiling_statistics(returns),
                         "ceilingR2Interval95": interval(everything, lambda i: ceiling_statistics(returns[i])["ceilingR2"])},
              "byDecision": {}}
    for k in cfg["branchDecisions"]:
        mask = decisions == k
        if mask.sum() == 0:
            report["byDecision"][str(k)] = {"states": 0, "ceilingR2": None}
            continue
        report["byDecision"][str(k)] = {**ceiling_statistics(returns[mask]), "ceilingR2Interval95": interval(
            mask, lambda i: ceiling_statistics(returns[i])["ceilingR2"])}
    report["critics"] = {}
    for name, values in critic_branch_values.items():
        values = np.asarray(values, dtype=float)
        report["critics"][name] = {**branch_metrics(values, returns),
            "captureInterval95": interval(everything, lambda i: branch_metrics(values[i], returns[i])["capture"]),
            "rolloutR2Interval95": interval(everything, lambda i: branch_metrics(values[i], returns[i])["rolloutR2"])}
    return report


# -- Decision rules (declaration §5) ---------------------------------------------------


def critic_rules(ceilings, selections, cfg):
    gate = cfg["gateMinPredictiveR2"]
    rows = {}
    for seed in (str(s) for s in cfg["policySeeds"]):
        interval = ceilings[seed]["pooled"]["ceilingR2Interval95"]
        upper = None if interval is None else interval[1]
        lower = selections[seed]["heldOutPredictiveR2"]
        if lower is not None and lower >= gate:
            row = "repairable"
        elif upper is not None and upper < gate:
            row = "gate-unreachable"
        else:
            row = "bracketed"
        rows[seed] = {"row": row, "lowerBound": lower, "upperBound95": upper,
                      "ceilingR2": ceilings[seed]["pooled"]["ceilingR2"], "selected": selections[seed]["variant"]}
    kinds = {row["row"] for row in rows.values()}
    outcome = kinds.pop() if len(kinds) == 1 and next(iter(kinds)) != "bracketed" else "bracketed"
    recommendation = {
        "repairable": "R1n-e: use the selected critic variant and keep the 0.25 gate",
        "gate-unreachable": "R1n-e: re-declare the critic gate against the measured ceiling (e.g. capture) and "
                            "justify the advantage estimator",
        "bracketed": "R1n-e: use the selected variant, state the critic gate as capture of the measured ceiling, "
                     "and state D2's red-draw boundary"}[outcome]
    return {"perPolicy": rows, "outcome": outcome, "recommendation": recommendation,
            "authorizes": "nothing; R1n-e needs its own declaration"}


# -- Run -------------------------------------------------------------------------------


def verify_source_run(cfg):
    root = TRAINING / cfg["sourceRun"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifestDigest")
    if json_digest(manifest) != digest:
        raise RuntimeError("R1n-c manifest digest mismatch")
    inventory = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if inventory != set(manifest["artifacts"]):
        raise RuntimeError("R1n-c artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if file_digest(root / name) != expected:
            raise RuntimeError(f"R1n-c artifact digest mismatch: {name}")
    return {"manifestDigest": digest, "artifacts": len(inventory),
            "checkpoints": {str(s): manifest["artifacts"][f"seed-{s}/{cfg['policyCheckpoint']}"]
                            for s in cfg["policySeeds"]}}


def load_policy(cfg, seed):
    model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
                                  target_world_sigma=cfg["targetWorldSigma"],
                                  initial_power_log_std=cfg["initialPowerLogStd"])
    state = torch.load(TRAINING / cfg["sourceRun"] / f"seed-{seed}" / cfg["policyCheckpoint"], map_location="cpu")
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


def execute(output, cfg):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    torch.set_num_threads(1)
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    source_run = verify_source_run(cfg)
    bound = budget_bound(cfg)
    root.mkdir(parents=True)
    here = Path(__file__).resolve()
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(), "budgetBound": bound,
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_d_declaration.md"),
        "implementationDigest": file_digest(here),
        "criticImplementationDigest": file_digest(here.parents[1] / "executor/full_authority_critics.py"),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "policyImplementationDigest": file_digest(here.parents[1] / "executor/full_authority_ppo_v1.py"),
        "sourceRun": source_run, "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["simulatorBudget"]:
            raise ValueError("R1n-d simulator budget exceeded")

    d1, variants, selections, ceilings = {}, {}, {}, {}
    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        for index, seed in enumerate(cfg["policySeeds"]):
            key, model = str(seed), load_policy(cfg, seed)

            # D1: six sampling modes on the same worlds.
            d1[key] = {}
            worlds = list(range(cfg["d1Seeds"][0], cfg["d1Seeds"][1] + 1))
            for number, (name, spec) in enumerate(cfg["modes"].items()):
                torch.manual_seed(cfg["samplingSeedBase"] + 100 * index + number)
                mode = (spec["sampleType"], spec["sampleContinuous"], spec["sigmaScale"])
                found = collect_seeds(client, worlds, cfg, choose=mode_chooser(model, mode), source=f"d1-{seed}-{name}",
                                      block_worlds=cfg["evaluationBlockWorlds"], account=account)
                write_episodes(root / "d1" / f"policy-{seed}" / name, found)
                d1[key][name] = summary(found, cfg)
            print(json.dumps({"seed": seed, "d1": {m: s["successes"] for m, s in d1[key].items()},
                              "simulatorDecisions": steps}), flush=True)

            # D3: new stochastic folds and critic variants C0-C3.
            directory = root / "d3" / f"policy-{seed}"
            wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
            torch.manual_seed(cfg["samplingSeedBase"] + 500 + index)
            train_episodes, train, used = v1.collect_fold(wrapper, model, v1.fold_seeds(cfg, index, held_out=False), cfg,
                                                          source=f"d3-{seed}", fold="train")
            account(used)
            torch.manual_seed(cfg["samplingSeedBase"] + 600 + index)
            held_episodes, held, used = v1.collect_fold(wrapper, model, v1.fold_seeds(cfg, index, held_out=True), cfg,
                                                        source=f"d3-{seed}", fold="heldOut")
            account(used)
            write_episodes(directory / "train-fold", train_episodes)
            write_episodes(directory / "held-out-fold", held_episodes)
            critics, reports, selected = fit_variants(model, train, held, cfg, index)
            arrays = {"heldOutSeed": held["seed"].numpy(), "heldOutEpisode": held["episode"].numpy(),
                      "heldOutDecision": held["decision"].numpy(), "heldOutReturn": held["returns"].numpy()}
            variants[key] = {}
            for name in VARIANTS:
                torch.save({"critic": critics[name].state_dict(), "variant": name}, directory / f"critic-{name}.pt")
                arrays[f"heldOutValue{name}"] = reports[name].pop("values").numpy()
                write_json(directory / f"critic-{name}.json", reports[name])
                variants[key][name] = {"heldOutPredictiveR2": reports[name]["heldOut"]["predictiveR2"],
                    "predictiveR2Interval95": reports[name]["heldOut"]["predictiveR2Interval95"],
                    "timeOnlyR2": reports[name]["heldOut"]["timeOnlyR2"],
                    "clockSkillScore": reports[name]["heldOut"]["clockSkillScore"],
                    "decisionWindowR2": reports[name]["heldOut"]["decisionWindowR2"],
                    "branchWindowR2": reports[name]["heldOut"]["branchWindowR2"],
                    "epochsRun": reports[name]["training"]["epochsRun"], "bestEpoch": reports[name]["training"]["bestEpoch"],
                    "bestValidationMse": reports[name]["training"]["bestValidationMse"]}
            selections[key] = {"variant": selected, "heldOutPredictiveR2": variants[key][selected]["heldOutPredictiveR2"],
                               "validationMse": {n: variants[key][n]["bestValidationMse"] for n in ("C1", "C2", "C3")}}
            write_json(directory / "selection.json", selections[key])
            del train, held
            print(json.dumps({"seed": seed, "d3": {n: variants[key][n]["heldOutPredictiveR2"] for n in VARIANTS},
                              "selected": selected, "simulatorDecisions": steps}), flush=True)

            # D2: source episodes, then branched rollouts that replay recorded prefixes.
            directory = root / "d2" / f"policy-{seed}"
            seeds = [cfg["d2SourceSeedBase"] + cfg["seedBandStride"] * index + j for j in range(cfg["d2SourceEpisodes"])]
            torch.manual_seed(cfg["samplingSeedBase"] + 700 + index)
            source_episodes, recorder = collect_source(client, model, seeds, cfg, account=account)
            write_episodes(directory / "source", source_episodes)
            np.savez_compressed(directory / "source-actions.npz", **{
                f"{key_name}-{j}": np.stack([a[key_name] for a in recorder.actions[j]])
                for j in range(len(seeds)) for key_name in ("action_type", "target", "power")})
            states, grouped, rollout_rows, source_returns = [], [], [], []
            for j, episode in enumerate(source_episodes):
                torch.manual_seed(cfg["rolloutSeedBase"] + 100 * index + j)
                rows = run_rollouts(client, model, episode, recorder.actions[j], recorder.digests[j], cfg,
                                    account=account, source=f"d2-{seed}-{j}")
                returns = v1.monte_carlo_returns(episode["rewards"], cfg["gamma"])
                for k in cfg["branchDecisions"]:
                    chosen = [r for r in rows if r["branchDecision"] == k]
                    if chosen:
                        states.append((j, k))
                        grouped.append([r["return"] for r in sorted(chosen, key=lambda r: r["rollout"])])
                        source_returns.append(returns[k])
                rollout_rows.extend({**r, "sourceEpisode": j} for r in rows)
            (directory / "rollouts.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rollout_rows), encoding="utf-8")
            branch_rows = {name: torch.stack([recorder.rows[state][name] for state in states])
                           for name in recorder.rows[states[0]]}
            branch_values = {name: critic_values(critics[name], branch_rows).numpy() for name in VARIANTS}
            np.savez_compressed(directory / "branch-states.npz", episode=np.asarray([s[0] for s in states]),
                decision=np.asarray([s[1] for s in states]), returns=np.asarray(grouped),
                sourceReturn=np.asarray(source_returns),
                digests=np.asarray([recorder.digests[j][k] for j, k in states]),
                **{f"observation-{name}": value.numpy() for name, value in branch_rows.items()},
                **{f"value{name}": value for name, value in branch_values.items()})
            ceilings[key] = ceiling_report(states, grouped, branch_values, cfg)
            write_json(directory / "ceiling.json", ceilings[key])
            print(json.dumps({"seed": seed, "d2": {"states": len(states), "ceilingR2": ceilings[key]["pooled"]["ceilingR2"],
                              "interval": ceilings[key]["pooled"]["ceilingR2Interval95"]},
                              "simulatorDecisions": steps}), flush=True)
            del critics, recorder, branch_rows

    exploration = exploration_rules(d1, cfg)
    write_json(root / "d1/summary.json", {"modes": d1, "rules": exploration})
    rules = critic_rules(ceilings, selections, cfg)
    report = {"format": "snowgym.pre-ppo-diagnostics-report.v0", "capabilities": capabilities,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "d1": d1, "exploration": exploration, "d3": variants, "selections": selections, "d2": ceilings,
        "criticRules": rules, "budgetBound": bound, "simulatorDecisions": steps}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.pre-ppo-diagnostics-manifest.v0",
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
