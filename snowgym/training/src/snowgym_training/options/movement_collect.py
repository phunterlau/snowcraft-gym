"""Exact-resumable, teacher-assisted movement-only option rollouts."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from ..ppo import generalized_advantage_estimate
from ..ppo_collect import SeedSchedule, numpy_actions, tensor_dict
from ..trajectory import json_digest
from .opportunity_audit import plain
from .plans import teacher_option_plan, teacher_option_scenario
from .throw_channels import recommend_shots

ASSIST_VERSION = "snowgym.r1h-corrected-shots.v0"
ASSIST_FIELDS = {"assistType": "teacher-shot-direction-and-power", "assistVersion": ASSIST_VERSION,
                 "autonomousQualificationEligible": False}


def corrected_shots(action, raw_states):
    """Override only selected throws; legality/readiness is not inferred from labels."""
    result = {key: value.copy() for key, value in action.items()}
    for index, raw in enumerate(raw_states):
        shot = recommend_shots(raw, result["action_type"].shape[1])
        selected = result["action_type"][index] == 2
        if (selected & ~shot["valid"][0]).any():
            raise ValueError("selected throw lacks a geometric recommendation")
        result["target"][index, selected] = shot["target"][0, selected]
        result["power"][index, selected] = shot["power"][0, selected]
    return result


def world_identities(wrapper):
    base = wrapper.environment
    _, bodies = base.plan_observations()
    return [{"physical": base.state_hashes[i], "plan": json_digest(plain(bodies[i])),
             "option": json_digest(plain(vars(tracker)))} for i, tracker in enumerate(wrapper.trackers)]


class MovementCollector:
    def __init__(self, wrapper, model, schedule: SeedSchedule):
        self.wrapper, self.model, self.schedule = wrapper, model, schedule
        self.seeds, self.prefixes, self.records, self.events = [], [], [], []
        self.observation = None
        self.horizon = None

    def _reset(self, indices, seeds, *, full=False):
        plan, spec = teacher_option_plan("engage")
        args = (seeds, [teacher_option_scenario("engage")] * len(indices),
                [f"movement-{seed}" for seed in seeds], [plan] * len(indices), [spec] * len(indices))
        observation, _ = self.wrapper.reset(*args) if full else self.wrapper.reset_indices(indices, *args)
        return tensor_dict(observation)

    def start(self, horizon):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("rollout horizon must be positive")
        self.horizon = horizon
        self.seeds = self.schedule.take(self.wrapper.batch_size)
        self.prefixes = [[] for _ in self.seeds]
        self.records, self.events = [], []
        self.observation = self._reset(list(range(len(self.seeds))), self.seeds, full=True)

    def advance(self, decisions=None):
        if self.observation is None:
            raise RuntimeError("start or restore the collector first")
        count = self.horizon-len(self.records) if decisions is None else decisions
        if type(count) is not int or count < 0 or count > self.horizon-len(self.records):
            raise ValueError("invalid collection decision count")
        for _ in range(count):
            observation = {k: v.clone() for k, v in self.observation.items()}
            with torch.no_grad():
                action, latent, logp, value = self.model.act(observation)
            executed = corrected_shots(numpy_actions(action), self.wrapper.environment.raw_observations)
            for i, prefix in enumerate(self.prefixes):
                prefix.append({k: v[i].tolist() for k, v in executed.items()})
            next_obs, rewards, terminated, truncated, infos = self.wrapper.step(executed)
            next_obs = tensor_dict(next_obs)
            with torch.no_grad():
                next_value = self.model(next_obs)["value"]
            final = len(self.records)+1 == self.horizon
            # Artificial rollout cuts bootstrap; option timeout is terminal.
            cut = np.asarray(truncated) | (final & ~terminated)
            self.records.append({"observation": observation, "action_type": action["action_type"].clone(),
                "latent": latent.clone(), "logp": logp.clone(), "value": value.clone(),
                "reward": torch.as_tensor(rewards).clone(), "terminated": torch.as_tensor(terminated).clone(),
                "truncated": torch.as_tensor(cut).clone(), "next_value": next_value.clone()})
            self.events.extend({"seed": self.seeds[i], "rolloutDecision": len(self.records),
                                "world": i, **plain(info)} for i, info in enumerate(infos))
            self.observation = next_obs
            done = np.flatnonzero(terminated | truncated).tolist()
            if done:
                seeds = self.schedule.take(len(done))
                replacement = self._reset(done, seeds)
                for row, index in enumerate(done):
                    self.seeds[index], self.prefixes[index] = seeds[row], []
                    for key in self.observation:
                        self.observation[key][index] = replacement[key][row]
        return len(self.records) == self.horizon

    def snapshot(self):
        if self.observation is None:
            raise RuntimeError("collector is uninitialized")
        return deepcopy({"format": "snowgym.movement-collection.v0", **ASSIST_FIELDS,
            "horizon": self.horizon, "seeds": self.seeds, "prefixes": self.prefixes,
            "records": self.records, "events": self.events, "observation": self.observation,
            "identities": world_identities(self.wrapper), "schedule": self.schedule.state(),
            "torchRng": torch.get_rng_state()})

    def restore(self, snapshot):
        if snapshot.get("format") != "snowgym.movement-collection.v0" or any(
            snapshot.get(k) != v for k, v in ASSIST_FIELDS.items()
        ):
            raise ValueError("collection format or assistance identity mismatch")
        if len(snapshot["seeds"]) != self.wrapper.batch_size:
            raise ValueError("collection batch size mismatch")
        state = deepcopy(snapshot)
        self.horizon, self.seeds, self.prefixes = state["horizon"], state["seeds"], state["prefixes"]
        self.observation = self._reset(list(range(len(self.seeds))), self.seeds, full=True)
        for decision in range(max(map(len, self.prefixes), default=0)):
            indices = [i for i, prefix in enumerate(self.prefixes) if decision < len(prefix)]
            action = {key: np.asarray([self.prefixes[i][decision][key] for i in indices],
                                      dtype=np.int64 if key == "action_type" else np.float32)
                      for key in ("action_type", "target", "power")}
            selected, _, _, _, _ = self.wrapper.step_indices(indices, action)
            for key, values in tensor_dict(selected).items():
                self.observation[key][indices] = values
        if world_identities(self.wrapper) != state["identities"] or any(
            not torch.equal(self.observation[k], v) for k, v in state["observation"].items()
        ):
            raise ValueError("prefix reconstruction physical/plan/option identity mismatch")
        cursor = state["schedule"]
        self.schedule = SeedSchedule(cursor["minimum"], cursor["maximum"], cursor["nextSeed"])
        self.records, self.events = state["records"], state["events"]
        torch.set_rng_state(state["torchRng"])

    def rollout(self, *, gamma, gae_lambda):
        if len(self.records) != self.horizon:
            raise RuntimeError("cannot optimize an incomplete collection")
        stacked = {key: torch.stack([r[key] for r in self.records]) for key in self.records[0]
                   if key != "observation"}
        advantage, returns = generalized_advantage_estimate(stacked["reward"], stacked["value"],
            stacked["next_value"], stacked["terminated"], stacked["truncated"], gamma=gamma, gae_lambda=gae_lambda)
        return {"observation": {k: torch.stack([r["observation"][k] for r in self.records]).flatten(0, 1)
                                for k in self.records[0]["observation"]},
                **{k: v.flatten(0, 1) for k, v in stacked.items()},
                "advantage": advantage.flatten(), "returns": returns.flatten()}
