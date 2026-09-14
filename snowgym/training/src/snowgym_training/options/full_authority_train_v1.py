"""R1n-b: repaired full-authority training contract (B2-B4, B7).

Implements `reviews/m7b_r1n_b_declaration.md` section 2 without editing the
digest-pinned E3 modules:

- B2 `ppo_update`: actor and critic have separate losses, backward passes and
  optimizers; the actor stops on KL exactly as E3 did, while the critic runs
  all of its epochs on every update.
- B3 `critic_metrics`/`critic_gate`: predictive R^2 gates; explained variance,
  the mean residual and a 20-bin time-only baseline are reported.
- B4 `run_block`/`warm_start_critic_mc`: complete episodes with no replacement
  resets and exact Monte Carlo returns, so no target bootstraps from the critic.
- B7 `exploration_calibration`: offline commanded-exploration measurements.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from snowgym_client.batch import BatchOperationError, SnowGymBatchEnv
from snowgym_client.encoding import decode_action
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT
from ..ppo import PPOConfig, living_unit_mask, ppo_loss
from ..ppo_collect import merge_observations, numpy_actions, tensor_dict
from .definitions import FROZEN_OPTION_SPECS, OptionSpec
from .engage_v1 import OPTION_STATE_VERSION, EngageOptionBatchV1
from .full_authority_train import scenario
from .plans import teacher_option_plan

ENGAGE_HORIZON = FROZEN_OPTION_SPECS["engage"].horizon


def configuration():
    return {"format": "snowgym.full-authority-config.v1", "arms": ["local", "global"], "roster": 1,
        "trainingRngs": [98001, 98002, 98003], "blockWorlds": 64, "optionHorizon": ENGAGE_HORIZON,
        "epochs": 4, "criticEpochs": 4, "minibatchSize": 512, "learningRate": 3e-4, "clipRatio": .2,
        "actorGradClip": .5, "criticGradClip": .5, "movementKlStop": .01, "entropyWeight": .01,
        "localRadius": 8.0, "targetWorldSigma": 2.0, "initialPowerLogStd": -1.0,
        "gamma": .9976921765, "gaeLambda": .9885140204,
        "warmStartTrainEpisodes": 256, "warmStartHeldOutEpisodes": 128, "warmStartEpochs": 10,
        "trainSeedBase": 630000, "heldOutSeedBase": 640000, "seedBandStride": 1000,
        "gateMinPredictiveR2": .25, "gateTimeOnlyMargin": .05, "timeBins": 20,
        "bootstrapSeed": 981001, "bootstrapSamples": 10000,
        "calibrationRows": 1024, "calibrationDraws": 64, "calibrationSeed": 982001,
        "assistType": "none", "assistVersion": "snowgym.full-authority.v1",
        "autonomousQualificationEligible": False}


def engage_spec(cfg):
    return OptionSpec("engage", cfg["optionHorizon"])


def make_wrapper(client, count, gamma, *, scripted=False):
    kind = ScriptedEngageOptionBatch if scripted else EngageOptionBatchV1
    return kind(SelectiveBatchEnv(count, client=client, observation_version=3), gamma=gamma)


class SelectiveBatchEnv(SnowGymBatchEnv):
    """Index-selective scripted steps and teacher reads.

    Stepping a completed episode returns `episode_complete`, so blocks must skip
    finished worlds. The archived R1m-S1/S2/S3/S5/S6 declarations pin every
    `snowgym_client` source digest, so this lives in a subclass rather than in
    the client."""

    def _checked(self, indices):
        if not indices or len(set(indices)) != len(indices) or any(
                type(index) is not int or index < 0 or index >= self.batch_size for index in indices):
            raise ValueError("selected world indices are invalid")
        return list(indices)

    def step_scripted_indices(self, indices):
        selected = self._checked(indices)
        self._step_index += 1
        items = []
        for index in selected:
            if self.state_hashes[index] is None:
                raise RuntimeError("reset() must initialize every batch slot before a scripted step")
            items.append({"worldId": self.world_ids[index], "body": {"expectedStateHash": self.state_hashes[index],
                "idempotencyKey": f"batch-scripted-{self._step_index}-{self.world_ids[index]}"}})
        payloads = self._consume_results(self.client.request("stepScripted", items), selected)
        return (self._stack_observations(selected),
                np.asarray([p["reward"] for p in payloads], dtype=np.float32),
                np.asarray([p["terminated"] for p in payloads], dtype=np.bool_),
                np.asarray([p["truncated"] for p in payloads], dtype=np.bool_),
                [p["info"] for p in payloads])

    def plan_teacher_actions_indices(self, indices):
        selected = self._checked(indices)
        results = self.client.request("planTeacherAction", [{"worldId": self.world_ids[i]} for i in selected])
        if len(results) != len(selected):
            raise RuntimeError("batch plan teacher result count mismatch")
        if any(result.get("status") != 200 for result in results):
            raise BatchOperationError("one or more batch plan teachers failed", results)
        actions = []
        for index, result in zip(selected, results, strict=True):
            body = result.get("body")
            if not isinstance(body, dict) or not isinstance(body.get("action"), dict):
                raise RuntimeError("batch plan teacher payload is missing action")
            status = body.get("status")
            if not isinstance(status, dict) or status.get("stateHash") != self.state_hashes[index]:
                raise RuntimeError("batch plan teacher stateHash does not match world state")
            actions.append(body["action"])
        return actions

    def plan_teacher_tensor_actions_indices(self, indices):
        selected = self._checked(indices)
        decoded = [decode_action(action, self.raw_observations[index], self.max_team_units)
                   for index, action in zip(selected, self.plan_teacher_actions_indices(selected), strict=True)]
        return {name: np.stack([action[name] for action in decoded]) for name in ("action_type", "target", "power")}


class ScriptedEngageOptionBatch(EngageOptionBatchV1):
    """Engage scoring for worlds advanced by the native blue policy (`/step-scripted`)."""

    def step_scripted_indices(self, indices):
        if any(self.trackers[index] is None or self.trackers[index].finished for index in indices):
            raise RuntimeError("reset a completed option before stepping it")
        physical, canonical, terminated, truncated, infos = self.environment.step_scripted_indices(indices)
        tensors, bodies = self.environment.plan_observations(indices)
        steps, enriched = [], []
        for row, index in enumerate(indices):
            step = self.trackers[index].update(self.environment.raw_observations[index], bodies[row],
                canonical_reward=float(canonical[row]), gamma=self.gamma,
                environment_done=bool(terminated[row] or truncated[row]))
            steps.append(step)
            enriched.append({**infos[row], "optionStateVersion": OPTION_STATE_VERSION,
                "option": {"decision": step.decision, "success": step.success,
                    "failed": step.failed, "timedOut": step.timed_out, "progress": step.progress,
                    "rewards": {"mission": step.mission_reward, "combat": step.combat_reward,
                        "shaping": step.shaping_reward, "canonical": step.canonical_reward,
                        "executor": step.executor_reward}, "metrics": step.metrics}})
        return (self._augment(merge_observations(physical, tensors), indices),
                np.asarray([step.executor_reward for step in steps], dtype=np.float32),
                np.asarray([step.done for step in steps], dtype=bool),
                np.zeros(len(indices), dtype=bool), enriched)


# -- Complete-episode blocks (B4) ------------------------------------------------------


def target_distance(raw, tracker):
    """World distance from the nearest living assigned fighter to the nearest living
    activated-target member, or None when either side has no living unit."""
    allies = [u for u in raw["allies"] if u["alive"] and u["id"] in tracker.assigned_ids]
    targets = [u for u in raw["enemies"] if u["alive"] and u["id"] in tracker.activated_target_ids]
    if not allies or not targets:
        return None
    return min(math.hypot(a["x"] - t["x"], a["y"] - t["y"]) for a in allies for t in targets)


def run_block(wrapper, seeds, cfg, *, choose=None, source, keep_observations=False):
    """Reset one world per seed, then step only unfinished worlds until every option ends.

    `choose(wrapper, active, observation, raws)` returns numpy actions for the active
    rows; `choose=None` selects the native scripted policy. There are no replacement
    resets, so every episode is complete and no return needs a bootstrap."""
    count = len(seeds)
    if wrapper.batch_size != count:
        raise ValueError("block seeds must match the wrapper batch size")
    plan, _ = teacher_option_plan("engage")
    spec = engage_spec(cfg)
    observation, _ = wrapper.reset(list(seeds), [scenario()] * count, [f"r1n-b-{source}-{s}" for s in seeds],
                                   [plan] * count, [spec] * count)
    current = tensor_dict(observation)
    episodes = [{"seed": int(seed), "source": source, "rewards": [], "remainingFraction": [],
                 "distances": [], "targetDamage": [], "rejectedActions": 0, "totalActions": 0}
                for seed in seeds]
    stored, decisions = [], 0
    while True:
        active = [i for i in range(count) if not wrapper.trackers[i].finished]
        if not active:
            break
        rows = {key: value[active] for key, value in current.items()}
        raws = [wrapper.environment.raw_observations[i] for i in active]
        distances = [target_distance(raw, wrapper.trackers[i]) for raw, i in zip(raws, active)]
        if choose is None:
            raw_next, rewards, _, _, infos = wrapper.step_scripted_indices(active)
        else:
            raw_next, rewards, _, _, infos = wrapper.step_indices(active, choose(wrapper, active, rows, raws))
        decisions += len(active)
        if keep_observations:
            stored.append({"observation": rows, "worlds": torch.as_tensor(active),
                           "decision": torch.as_tensor([len(episodes[i]["rewards"]) for i in active])})
        following = tensor_dict(raw_next)
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
    return episodes, stored, decisions


def monte_carlo_returns(rewards, gamma):
    returns, following = [0.] * len(rewards), 0.
    for step in reversed(range(len(rewards))):
        following = rewards[step] + gamma * following
        returns[step] = following
    return returns


def episode_row(episode):
    """The persisted per-episode summary (declaration B6)."""
    first = next((t for t, damage in enumerate(episode["targetDamage"]) if damage > 0), None)
    known = [d for d in episode["distances"] if d is not None]
    return {key: episode[key] for key in ("seed", "source", "success", "failed", "timedOut", "finalDecision",
                                          "blueAliveAtEnd", "rejectedActions", "totalActions")} | {
        "fold": episode.get("fold"), "firstHitDecision": None if first is None else first + 1,
        "finalTargetDamage": episode["targetDamage"][-1],
        "minDistance": min(known) if known else None,
        "meanDistance": float(np.mean(known)) if known else None,
        "distanceAtFirstHit": None if first is None else episode["distances"][first]}


def flatten_block(episodes, stored, gamma, *, episode_offset=0):
    """Stack stored rows (in step order) with their episode id, decision and Monte Carlo return."""
    returns = {i: monte_carlo_returns(e["rewards"], gamma) for i, e in enumerate(episodes)}
    observation = {key: torch.cat([s["observation"][key] for s in stored]) for key in stored[0]["observation"]}
    worlds = torch.cat([s["worlds"] for s in stored])
    decision = torch.cat([s["decision"] for s in stored])
    target = torch.tensor([returns[int(w)][int(d)] for w, d in zip(worlds, decision)], dtype=torch.float32)
    return {"observation": observation, "episode": worlds + episode_offset, "decision": decision,
            "seed": torch.tensor([episodes[int(w)]["seed"] for w in worlds]), "returns": target}


def concatenate(parts):
    return {"observation": {key: torch.cat([p["observation"][key] for p in parts]) for key in parts[0]["observation"]},
            **{key: torch.cat([p[key] for p in parts]) for key in ("episode", "decision", "seed", "returns")}}


def collect_fold(wrapper, model, seeds, cfg, *, source, fold):
    episodes, parts = [], []
    decisions = 0

    def choose(_wrapper, _active, rows, _raws):
        with torch.no_grad():
            action, _, _, _ = model.act(rows)
        return numpy_actions(action)

    for start in range(0, len(seeds), wrapper.batch_size):
        block_seeds = seeds[start:start + wrapper.batch_size]
        block, stored, used = run_block(wrapper, block_seeds, cfg, choose=choose, source=source, keep_observations=True)
        for episode in block:
            episode["fold"] = fold
        parts.append(flatten_block(block, stored, cfg["gamma"], episode_offset=len(episodes)))
        episodes.extend(block)
        decisions += used
    return episodes, concatenate(parts), decisions


# -- Critic metrics and gate (B3) ------------------------------------------------------


def time_bins(decision, horizon, bins):
    return torch.clamp(decision.long() * bins // horizon, max=bins - 1)


def critic_metrics(prediction, target, decision, train_target, train_decision, *, horizon, bins):
    prediction, target = prediction.double(), target.double()
    variance = float(target.var(unbiased=False))
    residual = prediction - target
    if variance <= 1e-12:
        return {"predictiveR2": None, "explainedVariance": None, "meanResidual": float(residual.mean()),
                "timeOnlyR2": None, "heldOutTargetVariance": variance}
    train_bin, held_bin = time_bins(train_decision, horizon, bins), time_bins(decision, horizon, bins)
    table = torch.full((bins,), float(train_target.double().mean()), dtype=torch.float64)
    for index in range(bins):
        members = train_target[train_bin == index]
        if len(members):
            table[index] = members.double().mean()
    time_prediction = table[held_bin]
    clock_error = float((time_prediction - target).square().mean())
    return {"predictiveR2": 1 - float(residual.square().mean()) / variance,
            "explainedVariance": 1 - float(residual.var(unbiased=False)) / variance,
            "meanResidual": float(residual.mean()),
            "timeOnlyR2": 1 - clock_error / variance,
            # Reported (A8): > 0 when the critic beats the clock lookup, ~0 when it only matches it.
            "clockSkillScore": 1 - float(residual.square().mean()) / clock_error if clock_error > 1e-12 else None,
            "heldOutTargetVariance": variance}


def gate_conditions(metrics, cfg):
    predictive, time_only = metrics["predictiveR2"], metrics["timeOnlyR2"]
    if predictive is None or time_only is None:
        return {"absolutePassed": False, "clockRelativePassed": False,
                "bindingThreshold": None}
    return {"absolutePassed": predictive >= cfg["gateMinPredictiveR2"],
            "clockRelativePassed": predictive >= time_only - cfg["gateTimeOnlyMargin"],
            "bindingThreshold": max(cfg["gateMinPredictiveR2"], time_only - cfg["gateTimeOnlyMargin"])}


def critic_gate(metrics, cfg):
    conditions = gate_conditions(metrics, cfg)
    return conditions["absolutePassed"] and conditions["clockRelativePassed"]


def bootstrap_predictive_r2(prediction, target, episode, *, samples, seed):
    """95% interval for predictive R^2, resampling held-out episodes with replacement."""
    prediction, target = prediction.double().numpy(), target.double().numpy()
    ids, inverse = np.unique(episode.numpy(), return_inverse=True)
    rows = np.bincount(inverse).astype(float)
    sse = np.bincount(inverse, (prediction - target) ** 2)
    first = np.bincount(inverse, target)
    second = np.bincount(inverse, target ** 2)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(ids), size=(samples, len(ids)))
    counts = np.stack([np.bincount(row, minlength=len(ids)) for row in draws]).astype(float)
    # einsum rather than matmul: avoids spurious BLAS matmul warnings on macOS.
    n = np.einsum("se,e->s", counts, rows)
    variance = np.einsum("se,e->s", counts, second) / n - (np.einsum("se,e->s", counts, first) / n) ** 2
    valid = variance > 1e-12
    values = 1 - (np.einsum("se,e->s", counts, sse) / n)[valid] / variance[valid]
    if not len(values):
        return None
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def predict_values(model, observation, chunk=4096):
    size = len(next(iter(observation.values())))
    with torch.no_grad():
        return torch.cat([model.critic({k: v[s:s + chunk] for k, v in observation.items()})
                          for s in range(0, size, chunk)])


def fold_seeds(cfg, rng_index, *, held_out):
    base = cfg["heldOutSeedBase" if held_out else "trainSeedBase"] + cfg["seedBandStride"] * rng_index
    count = cfg["warmStartHeldOutEpisodes" if held_out else "warmStartTrainEpisodes"]
    return list(range(base, base + count))


def warm_start_critic_mc(model, wrapper, cfg, rng_index, *, source):
    """Fit the critic alone on complete-episode Monte Carlo returns, then measure it
    on a disjoint held-out seed band (declaration B3/B4)."""
    train_episodes, train, train_decisions = collect_fold(
        wrapper, model, fold_seeds(cfg, rng_index, held_out=False), cfg, source=source, fold="train")
    held_episodes, held, held_decisions = collect_fold(
        wrapper, model, fold_seeds(cfg, rng_index, held_out=True), cfg, source=source, fold="heldOut")
    before_train, before_held = predict_values(model, train["observation"]), predict_values(model, held["observation"])
    untrained = critic_metrics(before_held, held["returns"], held["decision"], train["returns"], train["decision"],
                               horizon=cfg["optionHorizon"], bins=cfg["timeBins"])
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg["learningRate"])
    size = len(train["returns"])
    generator = torch.Generator().manual_seed(cfg["trainingRngs"][rng_index])
    for _ in range(cfg["warmStartEpochs"]):
        order = torch.randperm(size, generator=generator)
        for indices in order.split(cfg["minibatchSize"]):
            value = model.critic({k: v[indices] for k, v in train["observation"].items()})
            loss = (value - train["returns"][indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg["criticGradClip"], error_if_nonfinite=True)
            optimizer.step()
    after_train, after_held = predict_values(model, train["observation"]), predict_values(model, held["observation"])
    metrics = critic_metrics(after_held, held["returns"], held["decision"], train["returns"], train["decision"],
                             horizon=cfg["optionHorizon"], bins=cfg["timeBins"])
    report = {**metrics, "untrainedPredictiveR2": untrained["predictiveR2"],
        "predictiveR2Interval95": bootstrap_predictive_r2(after_held, held["returns"], held["episode"],
            samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]),
        "gatePassed": critic_gate(metrics, cfg), "gateConditions": gate_conditions(metrics, cfg),
        "trainRows": size, "heldOutRows": len(held["returns"]),
        "trainEpisodes": len(train_episodes), "heldOutEpisodes": len(held_episodes),
        "simulatorDecisions": train_decisions + held_decisions}
    arrays = {}
    for fold, data, before, after in (("train", train, before_train, after_train), ("heldOut", held, before_held, after_held)):
        arrays.update({f"{fold}Seed": data["seed"].numpy(), f"{fold}Episode": data["episode"].numpy(),
            f"{fold}Decision": data["decision"].numpy(), f"{fold}OptionState": data["observation"]["option_state"].numpy(),
            f"{fold}Return": data["returns"].numpy(), f"{fold}ValueBefore": before.numpy(),
            f"{fold}ValueAfter": after.numpy()})
    return report, arrays, train_episodes + held_episodes, held, optimizer


# -- Decoupled PPO update (B2) ---------------------------------------------------------


def ppo_config(cfg):
    return PPOConfig(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"], clip_ratio=cfg["clipRatio"],
        entropy_weight=cfg["entropyWeight"], max_grad_norm=cfg["actorGradClip"])


def actor_loss(model, rollout, indices, cfg):
    """Clipped policy term minus entropy; no value term. Advantage normalization stays
    minibatch-level inside `ppo_loss`, exactly as in E3. The value arguments are
    gradient-free placeholders, and the critic is not evaluated."""
    obs = {key: value[indices] for key, value in rollout["observation"].items()}
    latent = {"move": rollout["moveLatent"][indices], "throw": rollout["throwLatent"][indices],
              "power": rollout["powerLatent"][indices]}
    logp, extra = model.evaluate_latents(obs, rollout["action_type"][indices], latent, with_value=False)
    placeholder = torch.zeros(len(indices))
    losses = ppo_loss(logp, rollout["logp"][indices], rollout["advantage"][indices], placeholder, placeholder,
                      extra["entropy"], ppo_config(cfg), active_mask=living_unit_mask(obs))
    return losses["policy"] - cfg["entropyWeight"] * losses["entropy"], losses


def ppo_update(model, actor_optimizer, critic_optimizer, rollout, cfg):
    size = len(rollout["advantage"])
    actor_traces, critic_traces, stopped, stop_kl = [], [], False, None
    for epoch in range(cfg["epochs"]):
        for indices in torch.randperm(size).split(cfg["minibatchSize"]):
            loss, losses = actor_loss(model, rollout, indices, cfg)
            if not torch.isfinite(loss):
                raise ValueError("non-finite full-authority actor loss")
            if float(losses["approximate_kl"].detach()) > cfg["movementKlStop"]:
                stopped, stop_kl = True, float(losses["approximate_kl"].detach())
                break
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg["actorGradClip"], error_if_nonfinite=True)
            actor_optimizer.step()
            actor_traces.append({"epoch": epoch, "actorLoss": float(loss.detach()),
                "policy": float(losses["policy"].detach()), "entropy": float(losses["entropy"].detach()),
                "approximateKl": float(losses["approximate_kl"].detach()), "actorGradientNorm": float(norm)})
        if stopped:
            break
    for epoch in range(cfg["criticEpochs"]):
        for indices in torch.randperm(size).split(cfg["minibatchSize"]):
            obs = {key: value[indices] for key, value in rollout["observation"].items()}
            loss = (model.critic(obs) - rollout["returns"][indices]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite full-authority critic loss")
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg["criticGradClip"], error_if_nonfinite=True)
            critic_optimizer.step()
            critic_traces.append({"epoch": epoch, "valueLoss": float(loss.detach()), "criticGradientNorm": float(norm)})
    return {"actorOptimizerSteps": len(actor_traces), "criticOptimizerSteps": len(critic_traces),
        "klStopped": stopped, "stopApproximateKl": stop_kl, "actorMinibatches": actor_traces,
        "criticMinibatches": critic_traces, "meanReward": float(rollout["reward"].mean())}


# -- Offline exploration calibration (B7) ----------------------------------------------


def exploration_calibration(model, observation, cfg):
    """Commanded (not realized) exploration at the initial distributions, in world units."""
    rows = min(cfg["calibrationRows"], len(observation["allies"]))
    obs = {key: value[:rows] for key, value in observation.items()}
    draws = cfg["calibrationDraws"]
    generator = torch.Generator().manual_seed(cfg["calibrationSeed"])
    scale = torch.tensor(ARENA_HALF_EXTENT)
    with torch.no_grad():
        prediction = model(obs, with_value=False)
        live = prediction["living"][:, 0]
        own = obs["allies"][live, :1].float()
        move_mean, throw_mean = prediction["move_raw"][live, :1], prediction["throw_raw"][live, :1]
        states = len(own)
        repeated = {"allies": own.repeat_interleave(draws, 0)}
        noise = torch.randn((states * draws, 1, 2), generator=generator)
        move_latent = move_mean.repeat_interleave(draws, 0) + model.move_log_std.exp() * noise
        destination = model.decode_move(repeated, move_latent)
        centre = model.decode_move(repeated, move_mean.repeat_interleave(draws, 0))
        displacement = ((destination - centre) * scale).norm(dim=-1).flatten()
        clamped = (destination.abs() >= .999).any(-1).flatten()
        if model.destination == "global":
            saturated = (torch.tanh(move_latent).abs() > .99).any(-1).flatten()
        else:
            saturated = (torch.tanh(move_latent.norm(dim=-1)) > .99).flatten()
        noise = torch.randn((states * draws, 1, 2), generator=generator)
        throw_latent = throw_mean.repeat_interleave(draws, 0) + model.throw_log_std.exp() * noise
        position = repeated["allies"][..., 2:4] * scale
        aim = torch.tanh(throw_latent) * scale - position
        mean_aim = torch.tanh(throw_mean.repeat_interleave(draws, 0)) * scale - position
        defined = (mean_aim.norm(dim=-1) > 1e-3).flatten()
        cosine = torch.nn.functional.cosine_similarity(aim, mean_aim, dim=-1).flatten().clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cosine))[defined]

    def quantiles(values):
        return None if not len(values) else {"median": float(values.median()), "p90": float(torch.quantile(values, .9))}

    return {"states": states, "draws": draws, "moveDisplacementWorld": quantiles(displacement),
            "moveClampedFraction": float(clamped.float().mean()) if states else None,
            "moveSaturatedFraction": float(saturated.float().mean()) if states else None,
            "throwAngularDeviationDegrees": quantiles(angle), "throwAngleDefinedSamples": int(defined.sum())}
