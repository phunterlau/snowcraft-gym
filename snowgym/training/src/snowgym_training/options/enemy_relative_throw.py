"""M8-S12: throw-decoder redesign, enemy-relative heading (`reviews/m8_s12_declaration.md`).

`FullAuthorityPolicyV1EnemyThrow` predicts which living enemy to throw at and an angular offset from that
enemy's bearing, instead of an absolute arena coordinate (`ThrowSystem.tryThrow` uses only direction; distance
is discarded, established in S9). Action type, movement and power are the same computation as
`FullAuthorityPolicyV1` (bound, unedited methods); `fi.collect`/`fi.Labeler`/`mi.collect_mixture`/
`mi.warm_start_critic_mc_mixture`/`fi.imitation_parameters` all run against this class unchanged, since none of
them depend on how `target` is computed internally. Only the throw pathway and the imitation loss/fit loop are
new; no existing module is edited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT, FullAuthorityPolicy
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1, FEATURES
from ..executor.movement_ppo import OptionCentralCritic
from ..executor.model import ModelConfig
from ..ppo import living_unit_mask
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from . import death_rate_ppo as dr
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from . import mixture_imitation as mi
from . import roster_baseline as rb
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, write_episodes
from .interventions import require_capabilities
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

COHORTS = (1, 2, 3)
DECODERS = ("old", "new")


def configuration():
    return {**mi.configuration(), "format": "snowgym.m8-s12-enemy-relative-throw-config.v0", "roster": rb.ROSTER,
        "cohorts": COHORTS, "optimizerSeeds": (98401, 98402, 98403), "initialOffsetLogStd": -1.0,
        "pairedEvalWorlds": 100, "pairedEvalArms": ("normal", "easy"),
        "assistType": "none; ordinary autonomous imitation training", "assistVersion": "snowgym.m8-s12.v0",
        "budgetCap": 1_800_000, "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    round_zero = cfg["roundEpisodes"] * horizon
    learner_rounds = (cfg["fits"] - 1) * cfg["roundEpisodes"] * horizon
    critic = (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    paired = len(cfg["pairedEvalArms"]) * cfg["pairedEvalWorlds"] * horizon
    per_run = round_zero + learner_rounds + critic + paired
    return {"perRun": per_run, "total": per_run * len(cfg["cohorts"]) * len(DECODERS)}


def seed_bands(cfg):
    bands = []
    for cohort in cfg["cohorts"]:
        round_base = int(f"5{cohort}00000")
        for round_index in range(cfg["fits"]):
            low = round_base + cfg["roundSeedStride"] * round_index
            bands.append((f"cohort-{cohort}-round-{round_index}", low, low + cfg["roundEpisodes"] - 1))
        train_base, held_base = int(f"5{cohort}10000"), int(f"5{cohort}15000")
        bands.append((f"cohort-{cohort}-critic-train", train_base, train_base + cfg["warmStartTrainEpisodes"] - 1))
        bands.append((f"cohort-{cohort}-critic-held", held_base, held_base + cfg["warmStartHeldOutEpisodes"] - 1))
    return bands


def paired_eval_seeds(cfg):
    base = rb.configuration()["evaluationSeedBase"]
    return list(range(base, base + cfg["pairedEvalWorlds"]))


def round_seeds(cfg, cohort, round_index):
    base = int(f"5{cohort}00000") + cfg["roundSeedStride"] * round_index
    seeds = list(range(base, base + cfg["roundEpisodes"]))
    half = cfg["roundEpisodes"] // 2
    return {"random": seeds[:half], "easy": seeds[half:]}


# -- Geometry (declaration section 2; batched torch) ----------------------------------------


def safe_normalize(vectors, eps=1e-6):
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def rotate(base_unit, offset_unit):
    """Compose two unit vectors by complex multiplication: rotates `base_unit` by the angle of `offset_unit`."""
    bx, by = base_unit[..., 0], base_unit[..., 1]
    ox, oy = offset_unit[..., 0], offset_unit[..., 1]
    return torch.stack([bx * ox - by * oy, bx * oy + by * ox], dim=-1)


def enemy_relative_world(observation):
    """Each enemy slot's position relative to the acting unit, in WORLD units (the arena is 100x80, not square,
    so bearings must be computed here, not from the observation's per-axis-normalized delta)."""
    scale = torch.tensor(ARENA_HALF_EXTENT, dtype=torch.float32)
    delta_normalized = FullAuthorityPolicy.pair_features(None, observation, "enemies")[..., 2:4].float()
    return delta_normalized * scale


def enemy_living_mask(observation):
    return observation["enemy_mask"].bool() & (observation["enemies"][..., 1] > 0.5)


# -- Model (declaration section 2) -----------------------------------------------------------


class FullAuthorityPolicyV1EnemyThrow(nn.Module):
    pair_features = FullAuthorityPolicy.pair_features
    features = FullAuthorityPolicy.features
    decode_move = FullAuthorityPolicy.decode_move

    def __init__(self, *, destination: str, local_radius: float = 8.0, target_world_sigma: float = 2.0,
                 initial_power_log_std: float = -1.0, initial_offset_log_std: float = -1.0,
                 throw_radius_world: float = 9.0, critic_config: ModelConfig | None = None):
        super().__init__()
        from ..executor.full_authority_ppo import DESTINATIONS, calibrated_target_log_std
        if destination not in DESTINATIONS:
            raise ValueError("unknown full-authority destination geometry")
        if local_radius <= 0:
            raise ValueError("local destination radius must be positive")
        self.destination, self.local_radius, self.throw_radius_world = destination, float(local_radius), float(throw_radius_world)
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            for name, size in (("allies", 21), ("enemies", 21), ("projectiles", 9), ("obstacles", 9))
        })
        self.action_head = nn.Linear(FEATURES, 4)
        self.move_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 2))
        self.power_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 1))
        self.enemy_score_head = nn.Sequential(nn.Linear(21, 32), nn.ReLU(), nn.Linear(32, 1))
        self.throw_offset_head = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(), nn.Linear(64, 2))
        self.move_log_std = nn.Parameter(torch.tensor(
            calibrated_target_log_std(destination, local_radius, target_world_sigma=target_world_sigma)))
        self.throw_offset_log_std = nn.Parameter(torch.full((2,), float(initial_offset_log_std)))
        self.power_log_std = nn.Parameter(torch.full((1,), float(initial_power_log_std)))
        self.critic = OptionCentralCritic(critic_config or ModelConfig(observation_version=3))

    def actor_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad and not name.startswith("critic.")]

    def forward(self, observation, *, with_value: bool = True):
        features = self.features(observation)
        live = living_unit_mask(observation)
        action_logits = self.action_head(features).masked_fill(~live[..., None], -1e9)
        action_logits = action_logits.masked_fill(~observation["unit_action_mask"].bool(), -1e9)
        enemy_relative = self.pair_features(observation, "enemies").float()
        enemy_logits = self.enemy_score_head(enemy_relative).squeeze(-1)
        # enemy_mask is (batch, slots) -- one shared enemy list per world, not per acting unit.
        enemy_mask = enemy_living_mask(observation)
        enemy_logits = enemy_logits.masked_fill(~enemy_mask[:, None, :], -1e9)
        prediction = {"action_logits": action_logits, "living": live, "move_raw": self.move_head(features),
                      "power_raw": self.power_head(features).squeeze(-1), "enemy_logits": enemy_logits,
                      "throw_offset_raw": self.throw_offset_head(features), "enemy_mask": enemy_mask,
                      "enemy_world": enemy_relative_world(observation)}
        if with_value:
            prediction["value"] = self.critic(observation)
        return prediction

    def _compose_direction(self, prediction, enemy_choice, offset_latent):
        chosen = prediction["enemy_world"].gather(2, enemy_choice[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        bearing_unit = safe_normalize(chosen)
        offset_unit = safe_normalize(offset_latent)
        return rotate(bearing_unit, offset_unit)

    def act(self, observation, *, deterministic: bool = False):
        prediction = self(observation)
        move_mean, power_mean = prediction["move_raw"], prediction["power_raw"]
        move_std, power_std = self.move_log_std.exp(), self.power_log_std.exp()
        offset_mean, offset_std = prediction["throw_offset_raw"], self.throw_offset_log_std.exp()
        action_type = (prediction["action_logits"].argmax(-1) if deterministic
                       else Categorical(logits=prediction["action_logits"]).sample())
        move_latent = move_mean if deterministic else Normal(move_mean, move_std).sample()
        power_latent = power_mean if deterministic else Normal(power_mean, power_std).sample()
        offset_latent = offset_mean if deterministic else Normal(offset_mean, offset_std).sample()
        enemy_choice = (prediction["enemy_logits"].argmax(-1) if deterministic
                        else Categorical(logits=prediction["enemy_logits"]).sample())
        moves, throws = action_type == ACTION_MOVE, action_type == ACTION_THROW
        move_latent = torch.where(moves[..., None], move_latent, torch.zeros_like(move_latent))
        power_latent = torch.where(throws, power_latent, torch.zeros_like(power_latent))
        move_target = self.decode_move(observation, move_latent)
        direction = self._compose_direction(prediction, enemy_choice, offset_latent)
        scale = torch.tensor(ARENA_HALF_EXTENT, dtype=torch.float32)
        own_world = observation["allies"][..., 2:4].float() * scale
        throw_target_world = own_world + self.throw_radius_world * direction
        throw_target = torch.clamp(throw_target_world / scale, -1.0, 1.0)
        target = torch.where(moves[..., None], move_target, torch.where(throws[..., None], throw_target,
            torch.zeros_like(move_target)))
        action = {"action_type": action_type, "target": target, "power": torch.sigmoid(power_latent)}
        return action, {"move": move_latent, "power": power_latent, "offset": offset_latent, "enemy": enemy_choice}, \
            torch.zeros_like(power_mean), prediction["value"]


# -- Loss (declaration section 3) ------------------------------------------------------------


def imitation_loss_enemy_relative(model, observation, labels, cfg):
    prediction = model(observation, with_value=False)
    live = prediction["living"]
    teacher_type = labels["action_type"].long()
    legal = observation["unit_action_mask"].bool().gather(-1, teacher_type[..., None]).squeeze(-1)
    valid = live & legal
    zero = prediction["action_logits"].sum() * 0
    type_loss = F.cross_entropy(prediction["action_logits"][valid], teacher_type[valid]) if valid.any() else zero

    scale = torch.tensor(ARENA_HALF_EXTENT, dtype=torch.float32)
    own = observation["allies"][..., 2:4].float() * scale
    teacher_world = labels["target"].float() * scale

    moves = valid & (teacher_type == ACTION_MOVE)
    learned_move = model.decode_move(observation, prediction["move_raw"]) * scale
    to_teacher, to_learned = teacher_world - own, learned_move - own
    distance = to_teacher.norm(dim=-1)
    far, near = moves & (distance > cfg["slowRadius"]), moves & (distance <= cfg["slowRadius"])
    heading = ((1 - F.cosine_similarity(to_learned[far], to_teacher[far], dim=-1, eps=1e-6)).mean()
               if far.any() else zero)
    endpoint = ((learned_move - teacher_world).norm(dim=-1).clamp(max=cfg["endpointClip"])[near].square().mean()
                if near.any() else zero)

    throws = valid & (teacher_type == ACTION_THROW)
    teacher_direction = teacher_world - own
    aimed = throws & (teacher_direction.norm(dim=-1) > 1e-3)
    # prediction["enemy_mask"] is (batch, slots): whether any enemy exists is a per-world fact, broadcast to units.
    has_enemy = prediction["enemy_mask"].any(dim=-1)[:, None].expand_as(aimed)
    usable = aimed & has_enemy

    cosine_to_enemies = F.cosine_similarity(teacher_direction[..., None, :], prediction["enemy_world"], dim=-1, eps=1e-6)
    cosine_to_enemies = cosine_to_enemies.masked_fill(~prediction["enemy_mask"][:, None, :], -2.0)
    teacher_enemy = cosine_to_enemies.argmax(-1)

    enemy_loss = (F.cross_entropy(prediction["enemy_logits"][usable], teacher_enemy[usable])
                 if usable.any() else zero)

    chosen_world = prediction["enemy_world"].gather(2, teacher_enemy[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    bearing_unit = safe_normalize(chosen_world)
    offset_unit = safe_normalize(prediction["throw_offset_raw"])
    composed = rotate(bearing_unit, offset_unit)
    teacher_unit = safe_normalize(teacher_direction)
    angle_loss = ((1 - F.cosine_similarity(composed[usable], teacher_unit[usable], dim=-1, eps=1e-6)).mean()
                 if usable.any() else zero)

    power = ((torch.sigmoid(prediction["power_raw"][throws]) - labels["power"].float()[throws]).square().mean()
             if throws.any() else zero)
    total = type_loss + heading + endpoint + enemy_loss + angle_loss + power
    return {"total": total, "type": type_loss, "moveHeading": heading, "moveEndpoint": endpoint,
            "enemyLoss": enemy_loss, "angleLoss": angle_loss, "power": power,
            "illegalLabels": int((live & ~legal).sum()), "moveFar": int(far.sum()), "moveNear": int(near.sum()),
            "throwUnits": int(throws.sum()), "throwUsable": int(usable.sum())}


def fit_enemy_relative(model, optimizer, data, cfg, *, seed):
    generator = torch.Generator().manual_seed(seed)
    parameters = fi.imitation_parameters(model)
    history, step = [], 0
    while step < cfg["stepsPerFit"]:
        for indices in torch.randperm(len(data), generator=generator).split(cfg["imitationMinibatch"]):
            if step >= cfg["stepsPerFit"]:
                break
            observation, labels = data.batch(indices)
            losses = imitation_loss_enemy_relative(model, observation, labels, cfg)
            if not torch.isfinite(losses["total"]):
                raise ValueError("non-finite imitation loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["imitationGradClip"], error_if_nonfinite=True)
            optimizer.step()
            if step % cfg["historyEvery"] == 0 or step == cfg["stepsPerFit"] - 1:
                history.append({"step": step, "gradientNorm": float(norm), **{k: float(v.detach()) if torch.is_tensor(v)
                                else v for k, v in losses.items()}})
            step += 1
    return history


# -- Training orchestration: one run per (cohort, decoder) -----------------------------------


def build_model(decoder, cfg, seed):
    torch.manual_seed(seed)
    if decoder == "old":
        return FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
            target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
    return FullAuthorityPolicyV1EnemyThrow(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"],
        initial_offset_log_std=cfg["initialOffsetLogStd"])


def fit_one(decoder, model, optimizer, data, cfg, *, seed):
    if decoder == "old":
        return fi.fit(model, optimizer, data, cfg, seed=seed)
    return fit_enemy_relative(model, optimizer, data, cfg, seed=seed)


def train_one(client, cfg, decoder, cohort, seed, root, account):
    directory = Path(root) / f"cohort-{cohort}" / decoder
    directory.mkdir(parents=True)
    seeds0 = round_seeds(cfg, cohort, 0)
    round_zero, round_zero_part = mi.collect_mixture(client, cfg, None, seeds0, source=f"c{cohort}-{decoder}-round-0",
                                                      block_worlds=cfg["roundBlockWorlds"], account=account)
    write_episodes(directory / "round-0-teacher", round_zero)

    model = build_model(decoder, cfg, seed)
    optimizer = torch.optim.Adam(fi.imitation_parameters(model), lr=cfg["imitationLearningRate"])
    data, history = fi.Aggregate(), {}
    data.add(round_zero_part)
    for fit_index in range(cfg["fits"]):
        history[str(fit_index)] = fit_one(decoder, model, optimizer, data, cfg, seed=seed + 1000 * fit_index)
        torch.save({"model": model.state_dict(), "fit": fit_index}, directory / f"fit-{fit_index}.pt")
        if fit_index == cfg["fits"] - 1:
            break
        seeds_k = round_seeds(cfg, cohort, fit_index + 1)
        found, part = mi.collect_mixture(client, cfg, model, seeds_k, source=f"c{cohort}-{decoder}-round-{fit_index + 1}",
                                         block_worlds=cfg["roundBlockWorlds"], account=account)
        write_episodes(directory / f"round-{fit_index + 1}", found)
        data.add(part)
        del part
    write_json(directory / "fit-history.json", history)
    model.eval()

    critic_index = 0
    wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
    # trainingRngs[0] seeds the critic minibatch generator (mi.warm_start_critic_mc_mixture); set per (cohort,
    # decoder) so the critic fit's randomness is not silently shared across every run in this step (S4's own
    # declaration flagged exactly this class of undeclared shared randomness).
    critic_report_cfg = {**cfg, "criticTrainSeedBaseByCondition": {"M": int(f"5{cohort}10000")},
                         "criticHeldSeedBaseByCondition": {"M": int(f"5{cohort}15000")},
                         "trainingRngs": [seed + (0 if decoder == "old" else 500000)]}
    report, arrays, episodes, _, _ = mi.warm_start_critic_mc_mixture(
        model, wrapper, critic_report_cfg, critic_index, source=f"c{cohort}-{decoder}-critic", condition="M")
    account(report["simulatorDecisions"])
    write_json(directory / "critic-warm-start.json", report)
    np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)

    paired = {}
    for arm in cfg["pairedEvalArms"]:
        rows = rb.collect_cell(client, paired_eval_seeds(cfg), cfg, arm, model=model,
                               mode="deterministic", account=account)
        rb.write_rows(directory / "paired-eval" / arm, rows)
        paired[arm] = rb.summarize(rows, cfg)
    result = {"pairedEval": paired, "critic": {k: report[k] for k in
        ("predictiveR2", "timeOnlyR2", "untrainedPredictiveR2", "gatePassed")}}
    write_json(directory / "result.json", result)
    del model, optimizer, arrays, episodes
    return result


# -- Declaration, run, archive ----------------------------------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    here = Path(__file__).resolve()
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg), "seedBands": seed_bands(cfg),
        "s4Manifest": dr.verify_sealed(TRAINING / "runs" / "m8_s4_roster_baseline_v0"),
        "s5Manifest": dr.verify_sealed(TRAINING / "runs" / "m8_s5_roster_imitation_v0"),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s12_declaration.md"),
        "implementationDigest": file_digest(here), "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, results, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    report = {"format": "snowgym.m8-s12-enemy-relative-throw-report.v0", "cohorts": results,
        "simulatorDecisions": decisions, "withinBudgetBound": decisions <= bound["total"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s12-manifest.v0")
    return report


def run(output, cfg=None):
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["budgetCap"]:
            raise ValueError("M8-S12 budget exceeded")

    results = {}
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            for cohort, seed in zip(cfg["cohorts"], cfg["optimizerSeeds"]):
                results[str(cohort)] = {}
                for decoder in DECODERS:
                    results[str(cohort)][decoder] = train_one(client, cfg, decoder, cohort, seed, root, account)
    return aggregate(root, cfg, results, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
