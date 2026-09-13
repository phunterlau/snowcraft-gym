"""R1n (reviewer E3): from-scratch full-authority 1v1 Engage PPO.

No frozen source, no corrected shots, no teacher MOVE: the policy chooses
NOOP/HOLD/MOVE/THROW, movement destination, throw aim, and throw power
entirely on its own from a random initialization, exercising E2's winning
egocentric representation (`FullAuthorityPolicy`). Two arms differ only in
movement geometry (`local` vs `global` destination decoding). See
`reviews/m7b_r1n_declaration.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo import FullAuthorityPolicy, calibrated_target_log_std
from ..ppo import PPOConfig, generalized_advantage_estimate, living_unit_mask, ppo_loss
from ..ppo_collect import SeedSchedule, numpy_actions, tensor_dict
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from .interventions import require_capabilities
from .movement_train import TRAINING, make_wrapper
from .opportunity_audit import plain
from .plans import teacher_option_plan
from .reservoir import file_digest
from .supervised_probe import write_json

ARMS = ("local", "global")
ROSTER = 1  # 1v1; 2v2 is deferred (see declaration).


def scenario():
    return {"blueUnits": ROSTER, "redUnits": ROSTER, "arenaWidth": 100, "arenaHeight": 80,
        "maxTicks": 1800, "decisionHz": 10, "redDifficulty": "easy", "redController": "random"}


def configuration():
    return {"format": "snowgym.full-authority-config.v0", "arms": list(ARMS), "roster": ROSTER,
        "trainingRngs": [96001, 96002, 96003], "worlds": 64, "rolloutDecisions": 128,
        "updates": 244, "epochs": 4, "minibatchSize": 512, "learningRate": 3e-4,
        "clipRatio": .2, "actorGradClip": .5, "criticGradClip": .5, "movementKlStop": .01,
        "entropyWeight": .01, "localRadius": 8.0, "targetWorldSigma": 2.0,
        "initialPowerLogStd": -1.0, "gamma": .9976921765, "gaeLambda": .9885140204,
        "warmStartTrainDecisions": 16384, "warmStartHeldOutDecisions": 4096,
        "warmStartEpochs": 10, "warmStartMinRSquared": .25,
        "developmentSeeds": [600000, 600099], "floorSeeds": [610000, 610099],
        "simulatorBudget": 12500000, "bootstrapSeed": 960001, "bootstrapSamples": 10000,
        "assistType": "none", "assistVersion": "snowgym.full-authority.v0",
        "autonomousQualificationEligible": False}


class Collector:
    """Multi-world rollout collector for a full-authority policy. Adapted from
    `movement_collect.MovementCollector`: no `corrected_shots` (full authority),
    a fixed small-roster scenario instead of the 5v5 teacher option, and
    latents stored as three separate tensors instead of one."""

    def __init__(self, wrapper, model, schedule):
        self.wrapper, self.model, self.schedule = wrapper, model, schedule
        self.seeds, self.records, self.events = [], [], []
        self.observation, self.horizon = None, None

    def _reset(self, indices, seeds, *, full=False):
        plan, spec = teacher_option_plan("engage")
        args = (seeds, [scenario()] * len(indices), [f"full-authority-{seed}" for seed in seeds],
            [plan] * len(indices), [spec] * len(indices))
        observation, _ = self.wrapper.reset(*args) if full else self.wrapper.reset_indices(indices, *args)
        return tensor_dict(observation)

    def start(self, horizon):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("rollout horizon must be positive")
        self.horizon = horizon
        self.seeds = self.schedule.take(self.wrapper.batch_size)
        self.records, self.events = [], []
        self.observation = self._reset(list(range(len(self.seeds))), self.seeds, full=True)

    def advance(self, decisions=None):
        if self.observation is None:
            raise RuntimeError("start the collector first")
        count = self.horizon - len(self.records) if decisions is None else decisions
        if type(count) is not int or count < 0 or count > self.horizon - len(self.records):
            raise ValueError("invalid collection decision count")
        for _ in range(count):
            observation = {k: v.clone() for k, v in self.observation.items()}
            with torch.no_grad():
                action, latent, logp, value = self.model.act(observation)
            executed = numpy_actions(action)
            next_obs, rewards, terminated, truncated, infos = self.wrapper.step(executed)
            next_obs = tensor_dict(next_obs)
            with torch.no_grad():
                next_value = self.model(next_obs)["value"]
            final = len(self.records) + 1 == self.horizon
            cut = np.asarray(truncated) | (final & ~np.asarray(terminated))
            self.records.append({"observation": observation, "action_type": action["action_type"].clone(),
                "moveLatent": latent["move"].clone(), "throwLatent": latent["throw"].clone(),
                "powerLatent": latent["power"].clone(), "logp": logp.clone(), "value": value.clone(),
                "reward": torch.as_tensor(rewards, dtype=torch.float32).clone(),
                "terminated": torch.as_tensor(terminated).clone(), "truncated": torch.as_tensor(cut).clone(),
                "next_value": next_value.clone()})
            self.events.extend({"seed": self.seeds[i], "rolloutDecision": len(self.records),
                "world": i, **plain(info)} for i, info in enumerate(infos))
            self.observation = next_obs
            done = np.flatnonzero(np.asarray(terminated) | np.asarray(truncated)).tolist()
            if done:
                seeds = self.schedule.take(len(done))
                replacement = self._reset(done, seeds)
                for row, index in enumerate(done):
                    self.seeds[index] = seeds[row]
                    for key in self.observation:
                        self.observation[key][index] = replacement[key][row]
        return len(self.records) == self.horizon

    def rollout(self, *, gamma, gae_lambda):
        if len(self.records) != self.horizon:
            raise RuntimeError("cannot optimize an incomplete collection")
        stacked = {key: torch.stack([r[key] for r in self.records]) for key in self.records[0] if key != "observation"}
        advantage, returns = generalized_advantage_estimate(stacked["reward"], stacked["value"],
            stacked["next_value"], stacked["terminated"], stacked["truncated"], gamma=gamma, gae_lambda=gae_lambda)
        return {"observation": {k: torch.stack([r["observation"][k] for r in self.records]).flatten(0, 1)
                                for k in self.records[0]["observation"]},
            **{k: v.flatten(0, 1) for k, v in stacked.items()},
            "advantage": advantage.flatten(), "returns": returns.flatten()}


def warm_start_critic(model, client, cfg, seed):
    """Fit the critic alone (no actor gradient) on a fresh random-policy rollout;
    held-out R2 is measured on a disjoint seed range so it is a genuine episode-level holdout."""
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg["learningRate"])
    train_wrapper = make_wrapper(client, cfg["worlds"], cfg["gamma"])
    train_collector = Collector(train_wrapper, model, SeedSchedule(seed * 1000, seed * 1000 + 99999))
    train_collector.start(cfg["warmStartTrainDecisions"] // cfg["worlds"])
    train_collector.advance()
    train_rollout = train_collector.rollout(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"])
    held_wrapper = make_wrapper(client, cfg["worlds"], cfg["gamma"])
    held_collector = Collector(held_wrapper, model, SeedSchedule(seed * 1000 + 500000, seed * 1000 + 599999))
    held_collector.start(max(1, cfg["warmStartHeldOutDecisions"] // cfg["worlds"]))
    held_collector.advance()
    held_rollout = held_collector.rollout(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"])
    size = len(train_rollout["returns"])
    generator = torch.Generator().manual_seed(seed)
    for _ in range(cfg["warmStartEpochs"]):
        order = torch.randperm(size, generator=generator)
        for start in range(0, size, cfg["minibatchSize"]):
            indices = order[start:start + cfg["minibatchSize"]]
            obs = {k: v[indices] for k, v in train_rollout["observation"].items()}
            value = model.critic(obs)
            loss = (value - train_rollout["returns"][indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg["criticGradClip"], error_if_nonfinite=True)
            optimizer.step()
    with torch.no_grad():
        held_pred = model.critic(held_rollout["observation"])
    held_target = held_rollout["returns"]
    variance = float(held_target.var(unbiased=False))
    r_squared = 1 - float((held_pred - held_target).var(unbiased=False)) / variance if variance > 1e-12 else None
    decisions = len(train_rollout["returns"]) + len(held_rollout["returns"])
    return {"heldOutRSquared": r_squared, "trainRows": len(train_rollout["returns"]),
        "heldOutRows": len(held_rollout["returns"]), "simulatorDecisions": decisions}, optimizer


def ppo_update(model, actor_optimizer, critic_optimizer, rollout, cfg):
    config = PPOConfig(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"], clip_ratio=cfg["clipRatio"],
        entropy_weight=cfg["entropyWeight"], max_grad_norm=cfg["actorGradClip"])
    size = len(rollout["advantage"])
    traces, stopped, stop_kl = [], False, None
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(size)
        for indices in order.split(cfg["minibatchSize"]):
            obs = {k: v[indices] for k, v in rollout["observation"].items()}
            latent = {"move": rollout["moveLatent"][indices], "throw": rollout["throwLatent"][indices],
                "power": rollout["powerLatent"][indices]}
            logp, extra = model.evaluate_latents(obs, rollout["action_type"][indices], latent)
            live = living_unit_mask(obs)
            losses = ppo_loss(logp, rollout["logp"][indices], rollout["advantage"][indices],
                extra["value"], rollout["returns"][indices], extra["entropy"], config, active_mask=live)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise ValueError("non-finite full-authority PPO loss")
            if float(losses["approximate_kl"].detach()) > cfg["movementKlStop"]:
                stopped, stop_kl = True, float(losses["approximate_kl"].detach())
                break
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg["actorGradClip"], error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg["criticGradClip"], error_if_nonfinite=True)
            actor_optimizer.step()
            critic_optimizer.step()
            traces.append({"epoch": epoch, **{k: float(v.detach()) for k, v in losses.items()},
                "actorGradientNorm": float(actor_norm), "criticGradientNorm": float(critic_norm)})
        if stopped:
            break
    return {"optimizerSteps": len(traces), "klStopped": stopped, "stopApproximateKl": stop_kl,
        "minibatches": traces, "meanReward": float(rollout["reward"].mean())}


def evaluate(model, client, seeds, cfg, *, deterministic=True):
    wrapper = make_wrapper(client, 1, cfg["gamma"])
    plan, spec = teacher_option_plan("engage")
    rows = []
    for seed in seeds:
        observation, _ = wrapper.reset([seed], [scenario()], [f"full-authority-eval-{seed}"], [plan], [spec])
        observation = tensor_dict(observation)
        rejected = total = 0
        for _ in range(spec.horizon):
            with torch.no_grad():
                action, _, _, _ = model.act(observation, deterministic=deterministic)
            executed = numpy_actions(action)
            raw, _, terminated, truncated, infos = wrapper.step(executed)
            observation = tensor_dict(raw)
            results = infos[0].get("actionResults", [])
            rejected += sum(r.get("accepted") is False for r in results)
            total += len(results)
            if terminated[0] or truncated[0]:
                break
        final = infos[0]["option"]
        rows.append({"seed": seed, "success": bool(final["success"]), "progress": float(final["progress"]),
            "decisions": final["decision"], "rejectedActions": rejected, "totalActions": total,
            "simulatorDecisions": final["decision"]})
    return rows


def uniform_random_floor(client, seeds, cfg, *, rng_seed):
    """Floor control: sample a legal action type uniformly and a uniform-random
    target/power, independent of any learned parameters."""
    wrapper = make_wrapper(client, 1, cfg["gamma"])
    plan, spec = teacher_option_plan("engage")
    generator = np.random.default_rng(rng_seed)
    rows = []
    for seed in seeds:
        observation, _ = wrapper.reset([seed], [scenario()], [f"full-authority-floor-{seed}"], [plan], [spec])
        for _ in range(spec.horizon):
            mask = observation["unit_action_mask"][0].astype(bool)
            action_type = np.array([generator.choice(np.flatnonzero(row)) if row.any() else 0 for row in mask])
            target = generator.uniform(-1, 1, size=(mask.shape[0], 2)).astype(np.float32)
            power = generator.uniform(0, 1, size=mask.shape[0]).astype(np.float32)
            executed = {"action_type": action_type[None], "target": target[None], "power": power[None]}
            observation, _, terminated, truncated, infos = wrapper.step(executed)
            if terminated[0] or truncated[0]:
                break
        final = infos[0]["option"]
        rows.append({"seed": seed, "success": bool(final["success"]), "progress": float(final["progress"]),
            "decisions": final["decision"], "simulatorDecisions": final["decision"]})
    return rows


def train_run(client, cfg, root, seed, arm):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    if arm not in ARMS:
        raise ValueError("invalid full-authority arm")
    root.mkdir(parents=True)
    torch.manual_seed(seed)
    model = FullAuthorityPolicy(destination=arm, local_radius=cfg["localRadius"],
        initial_target_log_std=calibrated_target_log_std(arm, cfg["localRadius"],
            target_world_sigma=cfg["targetWorldSigma"]),
        initial_power_log_std=cfg["initialPowerLogStd"])
    warm_report, critic_optimizer = warm_start_critic(model, client, cfg, seed)
    write_json(root / "critic-warm-start.json", warm_report)
    if warm_report["heldOutRSquared"] is None or warm_report["heldOutRSquared"] < cfg["warmStartMinRSquared"]:
        report = {"arm": arm, "stoppedAtCriticGate": True, "criticWarmStart": warm_report,
            "simulatorDecisions": warm_report["simulatorDecisions"]}
        write_json(root / "training.json", report)
        return None, report
    actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=cfg["learningRate"])
    wrapper = make_wrapper(client, cfg["worlds"], cfg["gamma"])
    collector = Collector(wrapper, model, SeedSchedule(seed * 10_000_000, seed * 10_000_000 + 9_999_999))
    history, steps = [], warm_report["simulatorDecisions"]
    for update in range(cfg["updates"]):
        collector.start(cfg["rolloutDecisions"])
        collector.advance()
        rollout = collector.rollout(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"])
        step = ppo_update(model, actor_optimizer, critic_optimizer, rollout, cfg)
        decisions = cfg["worlds"] * cfg["rolloutDecisions"]
        steps += decisions
        step.update({"update": update + 1, "simulatorDecisions": decisions,
            "successes": sum(e["option"]["success"] for e in collector.events),
            "completedOptions": sum(e["option"]["success"] or e["option"]["failed"] for e in collector.events)})
        history.append(step)
        print(json.dumps({"arm": arm, "seed": seed, "update": update + 1, "optimizerSteps": step["optimizerSteps"],
            "klStopped": step["klStopped"], "meanReward": round(step["meanReward"], 4),
            "successes": step["successes"]}), flush=True)
    torch.save({"model": model.state_dict()}, root / "final-state.pt")
    report = {"arm": arm, "stoppedAtCriticGate": False, "criticWarmStart": warm_report,
        "history": history, "simulatorDecisions": steps}
    write_json(root / "training.json", report)
    return model, report


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    cfg = configuration()
    torch.set_num_threads(1)
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "implementationDigest": file_digest(Path(__file__)),
        "policyImplementationDigest": file_digest(Path(__file__).resolve().parents[1] / "executor/full_authority_ppo.py"),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_declaration.md"),
        "gitCommit": resolve_git_commit(), "assistType": cfg["assistType"], "assistVersion": cfg["assistVersion"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})
    steps = 0
    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["simulatorBudget"]:
            raise ValueError("full-authority simulator budget exceeded")
    development_seeds = list(range(cfg["developmentSeeds"][0], cfg["developmentSeeds"][1] + 1))
    floor_seeds = list(range(cfg["floorSeeds"][0], cfg["floorSeeds"][1] + 1))
    reports = {}
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        floor = uniform_random_floor(client, floor_seeds, cfg, rng_seed=cfg["bootstrapSeed"])
        account(sum(x["simulatorDecisions"] for x in floor))
        write_json(root / "floor.json", floor)
        print(json.dumps({"floorSuccesses": sum(x["success"] for x in floor), "n": len(floor)}), flush=True)
        for arm in ARMS:
            for seed in cfg["trainingRngs"]:
                directory = root / arm / str(seed)
                model, training = train_run(client, cfg, directory, seed, arm)
                account(training["simulatorDecisions"])
                if training["stoppedAtCriticGate"]:
                    reports.setdefault(arm, {})[str(seed)] = {"training": training}
                    print(json.dumps({"arm": arm, "seed": seed, "stoppedAtCriticGate": True}), flush=True)
                    continue
                evaluation = evaluate(model, client, development_seeds, cfg)
                account(sum(x["simulatorDecisions"] for x in evaluation))
                write_json(directory / "evaluation.json", evaluation)
                reports.setdefault(arm, {})[str(seed)] = {"training": {k: v for k, v in training.items() if k != "history"},
                    "developmentSuccesses": sum(x["success"] for x in evaluation),
                    "developmentMeanProgress": float(np.mean([x["progress"] for x in evaluation])),
                    "rejectedActionRate": sum(x["rejectedActions"] for x in evaluation) / max(1, sum(x["totalActions"] for x in evaluation))}
                print(json.dumps({"arm": arm, "seed": seed, "developmentSuccesses": reports[arm][str(seed)]["developmentSuccesses"],
                    "simulatorDecisions": steps}), flush=True)
    report = {"format": "snowgym.full-authority-report.v0", "assistType": cfg["assistType"],
        "assistVersion": cfg["assistVersion"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "floorSuccesses": sum(x["success"] for x in floor), "runs": reports, "simulatorDecisions": steps}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.full-authority-manifest.v0",
        "artifacts": {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    manifest["manifestDigest"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
