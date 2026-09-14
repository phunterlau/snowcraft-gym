"""R1n-e: KL-anchored frozen-reward PPO from R1n-c's imitation policies.

Complete-episode rollouts at sigma x0.5 with pure Monte Carlo advantages, an
exact hybrid-policy KL anchor to the initializer, and a deterministic
death-rate primary test with success/timeout non-inferiority. Each policy is an
independent run; `aggregate` verifies the per-policy manifests and applies the
decision rules. See `reviews/m7b_r1n_e_declaration.md`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..ppo import living_unit_mask, ppo_loss
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, write_episodes
from .full_authority_imitation import paired_difference
from .interventions import require_capabilities
from .pre_ppo_diagnostics import MODES, collect_seeds, mode_chooser, summary, verify_source_run
from .reservoir import file_digest
from .supervised_probe import write_json

LOG_STDS = ("move_log_std", "throw_log_std", "power_log_std")


def configuration():
    runs = [97301, 97302, 97303]
    return {**v1.configuration(), "format": "snowgym.death-rate-ppo-config.v0", "destination": "global",
        "initializerSeeds": [97101, 97102, 97103], "runSeeds": runs, "trainingRngs": runs,
        "sourceRun": "runs/m7b_engage_r1n_c_v0", "policyCheckpoint": "fit-4.pt", "sigmaScale": .5,
        "updates": 200, "episodesPerUpdate": 64, "trainingSeedBase": 1400000, "trainingSeedStride": 50000,
        "epochs": 4, "criticEpochs": 4, "minibatchSize": 512, "learningRate": 3e-4, "actorLearningRate": 1e-5,
        "clipRatio": .2,
        "actorGradClip": .5, "criticGradClip": .5, "movementKlStop": .01, "entropyWeight": .01,
        "entropy": "categorical", "anchorWeight": .01, "gaeLambda": 1., "advantage": "monte-carlo",
        "checkpointUpdates": [50, 100, 150, 200],
        "trainSeedBase": 880000, "heldOutSeedBase": 885000, "seedBandStride": 1000,
        "warmStartTrainEpisodes": 256, "warmStartHeldOutEpisodes": 128, "warmStartEpochs": 10,
        "blockWorlds": 64, "criticSanityMinR2": 0.,
        "evaluationSeeds": [870000, 870399], "evaluationBlockWorlds": 50, "evaluationSeedBase": 977000,
        "bootstrapSeed": 978001, "bootstrapSamples": 10000,
        "deathThreshold": -.05, "successMargin": -.05, "timeoutMargin": .05, "rejectionRateMax": .001,
        "policySimulatorBudget": 3000000, "simulatorBudget": 9000000,
        "assistType": "none at runtime; initializers from teacher-imitation training",
        "assistVersion": "snowgym.death-rate-ppo.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    warm = (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    training = cfg["updates"] * cfg["episodesPerUpdate"] * horizon
    evaluation = 2 * 2 * (cfg["evaluationSeeds"][1] - cfg["evaluationSeeds"][0] + 1) * horizon
    policy = warm + training + evaluation
    return {"warmStart": warm, "training": training, "evaluation": evaluation, "perPolicy": policy,
            "total": policy * len(cfg["initializerSeeds"])}


def training_seeds(cfg, index, update):
    start = cfg["trainingSeedBase"] + cfg["trainingSeedStride"] * index + cfg["episodesPerUpdate"] * update
    return list(range(start, start + cfg["episodesPerUpdate"]))


# -- Policy preparation and the hybrid KL anchor (declaration §1, §3) -------------------


def prepare_policy(cfg, index):
    """Initializer at sigma x `sigmaScale` with frozen log-stds, plus a frozen reference copy."""
    model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
                                  target_world_sigma=cfg["targetWorldSigma"],
                                  initial_power_log_std=cfg["initialPowerLogStd"])
    path = TRAINING / cfg["sourceRun"] / f"seed-{cfg['initializerSeeds'][index]}" / cfg["policyCheckpoint"]
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    with torch.no_grad():
        for name in LOG_STDS:
            getattr(model, name).add_(math.log(cfg["sigmaScale"]))
    for name in LOG_STDS:
        getattr(model, name).requires_grad_(False)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    return model, reference


def hybrid_kl(model, reference, observation, *, prediction=None):
    """Exact KL(pi_model || pi_reference) for the hybrid policy, averaged over living units per row
    and then over rows. Both policies share the same frozen standard deviations."""
    p = prediction if prediction is not None else model(observation, with_value=False)
    with torch.no_grad():
        q = reference(observation, with_value=False)
    live = p["living"].to(p["action_logits"].dtype)
    log_p, log_q = torch.log_softmax(p["action_logits"], -1), torch.log_softmax(q["action_logits"], -1)
    probs = log_p.exp()
    categorical = (probs * (log_p - log_q)).sum(-1)
    move = ((p["move_raw"] - q["move_raw"]).square() / (2 * model.move_log_std.exp().square())).sum(-1)
    throw = ((p["throw_raw"] - q["throw_raw"]).square() / (2 * model.throw_log_std.exp().square())).sum(-1)
    power = (p["power_raw"] - q["power_raw"]).square() / (2 * model.power_log_std.exp().square())
    unit = categorical + probs[..., ACTION_MOVE] * move + probs[..., ACTION_THROW] * (throw + power)
    return ((unit * live).sum(-1) / live.sum(-1).clamp_min(1.)).mean()


# -- Complete-episode rollouts (declaration §3) -----------------------------------------


class RolloutRecorder:
    """`run_block` controller that samples with `act` and logs latents, log-probs and values per step."""

    def __init__(self, model):
        self.model, self.steps = model, []

    def __call__(self, _wrapper, _active, rows, _raws):
        with torch.no_grad():
            action, latent, logp, value = self.model.act(rows)
        self.steps.append({"action_type": action["action_type"], "moveLatent": latent["move"],
                           "throwLatent": latent["throw"], "powerLatent": latent["power"], "logp": logp,
                           "value": value})
        return numpy_actions(action)


def build_rollout(episodes, stored, recorder, cfg):
    if len(stored) != len(recorder.steps):
        raise RuntimeError("logged policy outputs are misaligned with stored rows")
    flat = v1.flatten_block(episodes, stored, cfg["gamma"])
    logged = {key: torch.cat([step[key] for step in recorder.steps]) for key in recorder.steps[0]}
    rewards = torch.tensor([episodes[int(w)]["rewards"][int(d)] for w, d in zip(
        torch.cat([s["worlds"] for s in stored]), flat["decision"])], dtype=torch.float32)
    return {**flat, **logged, "reward": rewards, "advantage": flat["returns"] - logged["value"]}


# -- Anchored PPO update (declaration §3) -----------------------------------------------


def anchored_actor_loss(model, reference, rollout, indices, cfg):
    """Clipped surrogate with categorical-only entropy and the hybrid KL anchor."""
    obs = {key: value[indices] for key, value in rollout["observation"].items()}
    latent = {"move": rollout["moveLatent"][indices], "throw": rollout["throwLatent"][indices],
              "power": rollout["powerLatent"][indices]}
    logp, prediction = model.evaluate_latents(obs, rollout["action_type"][indices], latent, with_value=False)
    live = living_unit_mask(obs)
    entropy = Categorical(logits=prediction["action_logits"]).entropy() * live
    placeholder = torch.zeros(len(indices))
    losses = ppo_loss(logp, rollout["logp"][indices], rollout["advantage"][indices], placeholder, placeholder,
                      entropy, v1.ppo_config(cfg), active_mask=live)
    anchor = (hybrid_kl(model, reference, obs, prediction=prediction) if cfg["anchorWeight"] > 0
              else torch.zeros(()))
    total = losses["policy"] - cfg["entropyWeight"] * losses["entropy"] + cfg["anchorWeight"] * anchor
    return total, {**losses, "anchorKl": anchor}


def anchored_ppo_update(model, reference, actor_optimizer, critic_optimizer, rollout, cfg):
    """`v1.ppo_update`'s structure: actor epochs with the approximate-KL stop, then decoupled critic epochs."""
    size = len(rollout["advantage"])
    actor_traces, critic_traces, stopped, stop_kl = [], [], False, None
    for epoch in range(cfg["epochs"]):
        for indices in torch.randperm(size).split(cfg["minibatchSize"]):
            loss, losses = anchored_actor_loss(model, reference, rollout, indices, cfg)
            if not torch.isfinite(loss):
                raise ValueError("non-finite anchored actor loss")
            if float(losses["approximate_kl"].detach()) > cfg["movementKlStop"]:
                stopped, stop_kl = True, float(losses["approximate_kl"].detach())
                break
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg["actorGradClip"], error_if_nonfinite=True)
            actor_optimizer.step()
            actor_traces.append({"epoch": epoch, "actorLoss": float(loss.detach()),
                "policy": float(losses["policy"].detach()), "entropy": float(losses["entropy"].detach()),
                "approximateKl": float(losses["approximate_kl"].detach()),
                "anchorKl": float(losses["anchorKl"].detach()), "clipFraction": float(losses["clip_fraction"].detach()),
                "actorGradientNorm": float(norm)})
        if stopped:
            break
    for epoch in range(cfg["criticEpochs"]):
        for indices in torch.randperm(size).split(cfg["minibatchSize"]):
            obs = {key: value[indices] for key, value in rollout["observation"].items()}
            loss = (model.critic(obs) - rollout["returns"][indices]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite critic loss")
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg["criticGradClip"], error_if_nonfinite=True)
            critic_optimizer.step()
            critic_traces.append({"epoch": epoch, "valueLoss": float(loss.detach()), "criticGradientNorm": float(norm)})
    return {"actorOptimizerSteps": len(actor_traces), "criticOptimizerSteps": len(critic_traces),
        "klStopped": stopped, "stopApproximateKl": stop_kl, "actorMinibatches": actor_traces,
        "criticMinibatches": critic_traces, "meanReward": float(rollout["reward"].mean())}


def parameter_distance(model, reference):
    pairs = [(p, dict(reference.named_parameters())[n]) for n, p in model.named_parameters()
             if not n.startswith("critic.")]
    return float(torch.sqrt(sum(((a.detach() - b.detach()) ** 2).sum() for a, b in pairs)))


# -- One policy run (declaration §3, §4) -------------------------------------------------


def death(row):
    return not row["blueAliveAtEnd"]


def update_summary(update, episodes, step, model, reference, rollout, rollout_gamma):
    rows = [v1.episode_row(e) for e in episodes]
    with torch.no_grad():
        anchor = float(hybrid_kl(model, reference, rollout["observation"]))
    actor = step["actorMinibatches"]
    return {"update": update, "episodes": len(rows), "decisions": int(len(rollout["advantage"])),
            "successes": sum(r["success"] for r in rows), "deaths": sum(death(r) for r in rows),
            "timeouts": sum(r["timedOut"] for r in rows),
            "meanUndiscountedReturn": float(np.mean([sum(e["rewards"]) for e in episodes])),
            "meanDiscountedReturn": float(np.mean([v1.monte_carlo_returns(e["rewards"], rollout_gamma)[0]
                                                   for e in episodes])),
            "rejectedActions": sum(r["rejectedActions"] for r in rows), "totalActions": sum(r["totalActions"] for r in rows),
            "actorOptimizerSteps": step["actorOptimizerSteps"], "klStopped": step["klStopped"],
            "stopApproximateKl": step["stopApproximateKl"],
            "meanApproximateKl": float(np.mean([t["approximateKl"] for t in actor])) if actor else None,
            "meanClipFraction": float(np.mean([t["clipFraction"] for t in actor])) if actor else None,
            "anchorKlAfterUpdate": anchor, "meanCriticLoss": float(np.mean([t["valueLoss"] for t in step["criticMinibatches"]])),
            "meanReward": step["meanReward"]}


def run_policy(root, cfg, index):
    """Train and evaluate policy `index` into `root/policy-{initializer}`; requires `root/declaration.json`."""
    root = Path(root)
    declaration = json.loads((root / "declaration.json").read_text(encoding="utf-8"))
    if json_digest(declaration["config"]) != json_digest(cfg):
        raise RuntimeError("configuration differs from the run declaration")
    seed = cfg["initializerSeeds"][index]
    directory = root / f"policy-{seed}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    torch.set_num_threads(1)
    directory.mkdir(parents=True)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["policySimulatorBudget"]:
            raise ValueError("R1n-e per-policy simulator budget exceeded")

    torch.manual_seed(cfg["runSeeds"][index])
    model, reference = prepare_policy(cfg, index)
    report = {"initializerSeed": seed, "runSeed": cfg["runSeeds"][index]}
    rejected, total = 0, 0
    with SnowGymBatchClient() as client:
        report["capabilities"] = require_capabilities(client)
        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        warm, arrays, warm_episodes, _, critic_optimizer = v1.warm_start_critic_mc(
            model, wrapper, cfg, index, source=f"r1n-e-{seed}-critic")
        account(warm["simulatorDecisions"])
        write_json(directory / "critic-warm-start.json", warm)
        np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
        write_episodes(directory / "critic-episodes", warm_episodes)
        report["criticWarmStart"] = {k: warm[k] for k in ("predictiveR2", "predictiveR2Interval95", "timeOnlyR2",
                                                         "clockSkillScore", "gatePassed")}
        predictive = warm["predictiveR2"]
        report["criticSanityStop"] = predictive is None or predictive < cfg["criticSanityMinR2"]
        if not report["criticSanityStop"]:
            actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=cfg["actorLearningRate"])
            history, rows = [], []
            for update in range(cfg["updates"]):
                recorder = RolloutRecorder(model)
                episodes, stored, used = v1.run_block(wrapper, training_seeds(cfg, index, update), cfg,
                                                      choose=recorder, source=f"r1n-e-{seed}-{update}",
                                                      keep_observations=True)
                account(used)
                rollout = build_rollout(episodes, stored, recorder, cfg)
                step = anchored_ppo_update(model, reference, actor_optimizer, critic_optimizer, rollout, cfg)
                history.append(update_summary(update + 1, episodes, step, model, reference, rollout, cfg["gamma"]))
                rows.extend({**v1.episode_row(e), "update": update + 1} for e in episodes)
                rejected += history[-1]["rejectedActions"]
                total += history[-1]["totalActions"]
                if update + 1 in cfg["checkpointUpdates"]:
                    torch.save({"model": model.state_dict(), "actorOptimizer": actor_optimizer.state_dict(),
                                "criticOptimizer": critic_optimizer.state_dict(), "update": update + 1},
                               directory / f"update-{update + 1:03d}.pt")
                print(json.dumps({"seed": seed, "update": update + 1, "successes": history[-1]["successes"],
                                  "deaths": history[-1]["deaths"], "klStopped": step["klStopped"],
                                  "actorSteps": step["actorOptimizerSteps"],
                                  "anchorKl": round(history[-1]["anchorKlAfterUpdate"], 4),
                                  "simulatorDecisions": steps}), flush=True)
                del rollout, stored, recorder
            write_json(directory / "training-history.json", history)
            (directory / "training-episodes.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
            report["parameterDistance"] = parameter_distance(model, reference)
            report["evaluation"] = {}
            worlds = list(range(cfg["evaluationSeeds"][0], cfg["evaluationSeeds"][1] + 1))
            for m, (name, policy) in enumerate((("initializer", reference), ("final", model))):
                for mode in ("deterministic", "stochastic"):
                    if mode == "stochastic":
                        torch.manual_seed(cfg["evaluationSeedBase"] + 10 * index + m)
                    choose = mode_chooser(policy, MODES["det" if mode == "deterministic" else "full1"])
                    found = collect_seeds(client, worlds, cfg, choose=choose, source=f"r1n-e-{seed}-{name}-{mode}",
                                          block_worlds=cfg["evaluationBlockWorlds"], account=account)
                    write_episodes(directory / "evaluation" / f"{name}-{mode}", found)
                    report["evaluation"][f"{name}-{mode}"] = summary(found, cfg)
                    rejected += sum(e["rejectedActions"] for e in found)
                    total += sum(e["totalActions"] for e in found)
    report.update(simulatorDecisions=steps, rejectedActions=rejected, totalActions=total,
                  rejectionRate=rejected / total if total else None)
    write_json(directory / "policy-report.json", report)
    seal(directory, "snowgym.death-rate-ppo-policy-manifest.v0")
    return report


# -- Aggregation, paired analysis and decision rules (declaration §4, §5) ----------------


def seal(directory, format_name):
    own = directory / "manifest.json"
    manifest = {"format": format_name, "artifacts": {str(p.relative_to(directory)): file_digest(p)
                for p in sorted(directory.rglob("*")) if p.is_file() and p != own}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(directory / "manifest.json", manifest)


def verify_sealed(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifestDigest")
    own = directory / "manifest.json"
    inventory = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != own}
    if json_digest(manifest) != digest or inventory != set(manifest["artifacts"]):
        raise RuntimeError(f"manifest mismatch in {directory}")
    for name, expected in manifest["artifacts"].items():
        if file_digest(directory / name) != expected:
            raise RuntimeError(f"artifact digest mismatch: {directory / name}")
    return digest


def outcomes(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {r["seed"]: {"success": float(r["success"]), "death": float(death(r)), "timeout": float(r["timedOut"])}
            for r in rows}


def paired_analysis(directories, cfg, mode):
    """Per policy and seed-averaged final - initializer differences, with worlds resampled."""
    per_policy, final, initial = {}, [], []
    worlds = None
    for directory in directories:
        before = outcomes(directory / "evaluation" / f"initializer-{mode}" / "episodes.jsonl")
        after = outcomes(directory / "evaluation" / f"final-{mode}" / "episodes.jsonl")
        worlds = sorted(before) if worlds is None else worlds
        if sorted(before) != worlds or sorted(after) != worlds:
            raise RuntimeError("evaluation worlds differ across policies")
        per_policy[directory.name] = {metric: paired_difference([after[w][metric] for w in worlds],
            [before[w][metric] for w in worlds], samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
            for metric in ("death", "success", "timeout")}
        final.append({w: after[w] for w in worlds})
        initial.append({w: before[w] for w in worlds})
    averaged = {metric: paired_difference(
        [np.mean([f[w][metric] for f in final]) for w in worlds],
        [np.mean([i[w][metric] for i in initial]) for w in worlds],
        samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]) for metric in ("death", "success", "timeout")}
    return {"perPolicy": per_policy, "seedAveraged": averaged, "worlds": len(worlds)}


def decision_rules(analysis, reports, cfg):
    if any(r["criticSanityStop"] for r in reports.values()):
        return {"outcome": "incomplete", "recommendation": "resolve the critic sanity failure first",
                "authorizes": "nothing; R1n-f needs its own declaration"}
    averaged = analysis["seedAveraged"]
    death_mean, (death_low, death_high) = averaged["death"]["mean"], averaged["death"]["interval95"]
    success_low, timeout_high = averaged["success"]["interval95"][0], averaged["timeout"]["interval95"][1]
    non_inferior = success_low > cfg["successMargin"] and timeout_high < cfg["timeoutMargin"]
    if not non_inferior or death_low > 0:
        outcome = "harm-or-avoidance"
    elif death_mean <= cfg["deathThreshold"] + 1e-12 and death_high < 0:
        outcome = "survival-improved"
    elif death_high < 0:
        outcome = "improved-below-threshold"
    else:
        outcome = "no-detectable-change"
    recommendation = {
        "survival-improved": "R1n-f: fresh replication (new training RNGs, new untouched split), then a second mission",
        "improved-below-threshold": "diagnose learning curves (KL stop, anchor KL, return trend) before longer runs",
        "no-detectable-change": "diagnose learning curves (KL stop, anchor KL, return trend) before longer runs",
        "harm-or-avoidance": "stop this PPO configuration; revisit sigma, the anchor or the objective"}[outcome]
    per_policy_death = {name: rows["death"]["mean"] for name, rows in analysis["perPolicy"].items()}
    rejection = [r["rejectionRate"] for r in reports.values()]
    return {"outcome": outcome, "recommendation": recommendation, "primaryPassed": outcome == "survival-improved",
            "nonInferiority": {"passed": non_inferior, "successLower95": success_low, "timeoutUpper95": timeout_high},
            "death": averaged["death"],
            "checks": {"parameterChange": all(r.get("parameterDistance", 0) > 0 for r in reports.values()),
                       "rejectionRateBelowMax": all(x is not None and x < cfg["rejectionRateMax"] for x in rejection),
                       "policiesWithLowerDeath": sum(v < 0 for v in per_policy_death.values()),
                       "perPolicyDeathDifference": per_policy_death},
            "authorizes": "nothing; R1n-f needs its own declaration"}


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    source_run = verify_source_run({**cfg, "policySeeds": cfg["initializerSeeds"]})
    root.mkdir(parents=True)
    here = Path(__file__).resolve()
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_e_declaration.md"),
        "implementationDigest": file_digest(here),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "diagnosticsImplementationDigest": file_digest(here.parent / "pre_ppo_diagnostics.py"),
        "policyImplementationDigest": file_digest(here.parents[1] / "executor/full_authority_ppo_v1.py"),
        "sourceRun": source_run, "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    directories = [root / f"policy-{seed}" for seed in cfg["initializerSeeds"]]
    digests = {d.name: verify_sealed(d) for d in directories}
    reports = {d.name: json.loads((d / "policy-report.json").read_text(encoding="utf-8")) for d in directories}
    complete = not any(r["criticSanityStop"] for r in reports.values())
    analysis = {mode: paired_analysis(directories, cfg, mode) for mode in ("deterministic", "stochastic")} if complete else None
    rules = decision_rules(analysis["deterministic"] if complete else None, reports, cfg)
    report = {"format": "snowgym.death-rate-ppo-report.v0", "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"], "policyManifests": digests,
        "policies": reports, "pairedDifferences": analysis, "decisionRules": rules,
        "simulatorDecisions": sum(r["simulatorDecisions"] for r in reports.values()), "budgetBound": budget_bound(cfg)}
    write_json(root / "report.json", report)
    seal(root, "snowgym.death-rate-ppo-manifest.v0")
    return report


def run(output):
    cfg = configuration()
    declare(output, cfg)
    for index in range(len(cfg["initializerSeeds"])):
        run_policy(output, cfg, index)
    return aggregate(output, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "declare", "policy", "aggregate"], default="all")
    parser.add_argument("--policy", type=int, choices=[0, 1, 2])
    arguments = parser.parse_args()
    if arguments.stage == "all":
        run(arguments.output)
    elif arguments.stage == "declare":
        declare(arguments.output, configuration())
    elif arguments.stage == "policy":
        if arguments.policy is None:
            raise SystemExit("--policy is required for --stage policy")
        run_policy(arguments.output, configuration(), arguments.policy)
    else:
        aggregate(arguments.output, configuration())
