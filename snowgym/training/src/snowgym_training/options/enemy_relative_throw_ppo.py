"""M8-S14: KL-anchored, Monte Carlo PPO continuation from the M8-S12 enemy-relative throw initializer, at 3v3.
(`reviews/m8_s14_declaration.md`)

Continues R1n-e's recipe (`death_rate_ppo.py`, the only PPO-from-imitation precedent in this repo) to S12's
`FullAuthorityPolicyV1EnemyThrow` checkpoints. `FullAuthorityPolicyV1EnemyThrowPPO` is a new subclass supplying
real log-probabilities (S12's `act()` returns a placeholder zero logp); `hybrid_kl_enemy_relative` extends the KL
anchor to the enemy-categorical + offset-Gaussian throw representation. `ppo.ppo_loss`, `full_authority_train_v1`'s
collection/critic-warm-start machinery, and `roster_baseline.collect_cell` are reused unchanged. No existing module
is edited, including S12's sealed `enemy_relative_throw.py`."""

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
from ..ppo import living_unit_mask, ppo_loss
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import death_rate_ppo as dr
from . import enemy_relative_throw as ert
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from . import roster_baseline as rb
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, write_episodes
from .interventions import require_capabilities
from .opponent_transfer import scenario_override
from .reservoir import file_digest
from .supervised_probe import write_json

COHORTS = (1, 2, 3)
LOG_STDS = ("move_log_std", "power_log_std")  # throw_offset_log_std is set absolutely, not shifted (see prepare_policy)
LR_CANDIDATES = (3e-4, 1e-4, 3e-5, 1e-5, 3e-6)
SIGMA_FALLBACK = (0.5, 0.25)  # tried in order after the 1x candidate, per declaration §6


def configuration():
    return {**ert.configuration(), "format": "snowgym.m8-s14-ppo-continuation-config.v0",
        "cohorts": COHORTS, "sourceRun": "m8_s12_enemy_relative_throw_v0", "policyCheckpoint": "fit-4.pt",
        "sigmaScale": .5, "offsetLogStdTarget": -2.0, "angularExplorationTargetDegrees": 2.5,
        "runSeedBase": 98500, "sigmaProbeTorchSeedBase": 98600, "lrProbeTorchSeedBase": 98700,
        "updates": 200, "episodesPerUpdate": 64,
        "trainingSeedBaseByCohort": {c: int(f"5{c}70000") for c in COHORTS},
        "sigmaProbeSeedBaseByCohort": {c: int(f"5{c}60000") for c in COHORTS},
        "lrProbeSeedBaseByCohort": {c: int(f"5{c}60064") for c in COHORTS},
        "probeEpisodes": 64, "epochs": 4, "criticEpochs": 4, "minibatchSize": 512,
        "actorGradClip": .5, "criticGradClip": .5, "movementKlStop": .01, "entropyWeight": .01,
        "anchorWeight": .01, "gaeLambda": 1., "advantage": "monte-carlo", "checkpointUpdates": [50, 100, 150, 200],
        "trainSeedBase": 5090000, "heldOutSeedBase": 5095000, "seedBandStride": 100000,
        "warmStartTrainEpisodes": 256, "warmStartHeldOutEpisodes": 128, "warmStartEpochs": 10,
        "criticLearningRate": 3e-4, "criticSanityMinR2": 0., "movementFloorAnchorKl": .01,
        "evaluationSeedBase": 2600000, "evaluationWorlds": 400,  # rb.collect_cell chunks by blockWorlds, not a
        # separate evaluationBlockWorlds key -- both training and evaluation share cfg["blockWorlds"] (64).
        "stochasticEvalSeedBase": 984500, "bootstrapSeed": 984001, "bootstrapSamples": 10000,
        "sigmaProbeSuccessGapMin": -.15, "sigmaProbeEntropyMargin": .1,
        "deathThreshold": -.05, "successMargin": -.05, "timeoutMargin": .05,
        # Pinned after the probe stage runs and its results are reviewed (declaration §12 amendment) -- None
        # here on purpose; `--stage cohort` refuses to run while either is unset, so a real launch cannot
        # silently skip the review step. `run()`'s in-process convenience path (tests) probes and fills these
        # in for itself instead of reading them from cfg.
        "sigmaScaleByCohort": {c: None for c in COHORTS}, "actorLearningRate": None,
        "probeBudgetCap": 300_000, "trainingBudgetCap": 9_000_000, "perCohortTrainingBudgetCap": 3_000_000,
        "assistType": "none at runtime; initializer from teacher-imitation training",
        "assistVersion": "snowgym.m8-s14-ppo-continuation.v0", "autonomousQualificationEligible": False}


def probe_budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    sigma = 2 * cfg["probeEpisodes"] * horizon  # deterministic + stochastic, 1x candidate
    sigma_fallback = len(SIGMA_FALLBACK) * 2 * cfg["probeEpisodes"] * horizon
    lr = cfg["probeEpisodes"] * horizon  # one rollout, reused for every lr candidate
    per_cohort = sigma + sigma_fallback + lr
    return {"sigma1x": sigma, "sigmaFallbackWorstCase": sigma_fallback, "lr": lr, "perCohortWorstCase": per_cohort,
            "total": per_cohort * len(cfg["cohorts"])}


def training_budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    warm = (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    training = cfg["updates"] * cfg["episodesPerUpdate"] * horizon
    evaluation = 2 * 2 * cfg["evaluationWorlds"] * horizon
    per_cohort = warm + training + evaluation
    return {"warmStart": warm, "training": training, "evaluation": evaluation, "perCohort": per_cohort,
            "total": per_cohort * len(cfg["cohorts"])}


def training_seeds(cfg, cohort, update):
    base = cfg["trainingSeedBaseByCohort"][cohort] + cfg["episodesPerUpdate"] * update
    return list(range(base, base + cfg["episodesPerUpdate"]))


def sigma_probe_seeds(cfg, cohort):
    base = cfg["sigmaProbeSeedBaseByCohort"][cohort]
    return list(range(base, base + cfg["probeEpisodes"]))


def lr_probe_seeds(cfg, cohort):
    base = cfg["lrProbeSeedBaseByCohort"][cohort]
    return list(range(base, base + cfg["probeEpisodes"]))


def evaluation_seeds(cfg):
    return list(range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + cfg["evaluationWorlds"]))


# -- Policy preparation (declaration §4) ------------------------------------------------


def prepare_policy(cfg, cohort, *, sigma_scale=None):
    """Load the S12 checkpoint into the PPO-capable subclass, apply exploration scale, freeze log-stds."""
    scale = cfg["sigmaScale"] if sigma_scale is None else sigma_scale
    path = TRAINING / "runs" / cfg["sourceRun"] / f"cohort-{cohort}" / "new" / cfg["policyCheckpoint"]
    model = FullAuthorityPolicyV1EnemyThrowPPO(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"],
        initial_offset_log_std=cfg["initialOffsetLogStd"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    with torch.no_grad():
        for name in LOG_STDS:
            getattr(model, name).add_(math.log(scale))
        # throw_offset_log_std is set absolutely (declaration §4): S12 left it at an uncalibrated placeholder,
        # so there is no meaningful "imitation-calibrated value" to shift multiplicatively.
        offset_target = cfg["offsetLogStdTarget"] + math.log(scale / cfg["sigmaScale"])
        model.throw_offset_log_std.fill_(offset_target)
    for name in (*LOG_STDS, "throw_offset_log_std"):
        getattr(model, name).requires_grad_(False)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    return model, reference


# -- The PPO-capable model (declaration §3) ----------------------------------------------


class FullAuthorityPolicyV1EnemyThrowPPO(ert.FullAuthorityPolicyV1EnemyThrow):
    """Adds real log-probabilities to S12's model, required for PPO's importance-sampling ratio."""

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
        scale = torch.tensor(ert.ARENA_HALF_EXTENT, dtype=torch.float32)
        own_world = observation["allies"][..., 2:4].float() * scale
        throw_target_world = own_world + self.throw_radius_world * direction
        throw_target = torch.clamp(throw_target_world / scale, -1.0, 1.0)
        target = torch.where(moves[..., None], move_target, torch.where(throws[..., None], throw_target,
            torch.zeros_like(move_target)))
        action = {"action_type": action_type, "target": target, "power": torch.sigmoid(power_latent)}
        latent = {"move": move_latent, "enemy": enemy_choice, "offset": offset_latent, "power": power_latent}
        logp, _ = self.evaluate_latents(observation, action_type, latent, prediction=prediction)
        return action, latent, logp, prediction["value"]

    def evaluate_latents(self, observation, action_type, latent, *, prediction=None, with_value=True):
        prediction = prediction if prediction is not None else self(observation, with_value=with_value)
        live = prediction["living"]
        moves, throws = live & (action_type == ACTION_MOVE), live & (action_type == ACTION_THROW)
        move_normal = Normal(prediction["move_raw"], self.move_log_std.exp())
        offset_normal = Normal(prediction["throw_offset_raw"], self.throw_offset_log_std.exp())
        power_normal = Normal(prediction["power_raw"], self.power_log_std.exp())
        action_categorical = Categorical(logits=prediction["action_logits"])
        enemy_categorical = Categorical(logits=prediction["enemy_logits"])
        type_logp, type_entropy = action_categorical.log_prob(action_type) * live, action_categorical.entropy() * live
        move_logp = move_normal.log_prob(latent["move"]).sum(-1) * moves
        enemy_logp = enemy_categorical.log_prob(latent["enemy"]) * throws
        offset_logp = offset_normal.log_prob(latent["offset"]).sum(-1) * throws
        power_logp = power_normal.log_prob(latent["power"]) * throws
        per_unit_logp = type_logp + move_logp + enemy_logp + offset_logp + power_logp
        probabilities = action_categorical.probs
        move_probability, throw_probability = probabilities[..., ACTION_MOVE], probabilities[..., ACTION_THROW]
        # Entropy bonus is categorical-only on action_type (declaration §8, matching R1n-e's amendment A2):
        # enemy/offset/power entropy is deliberately excluded, so per_unit_entropy carries only type_entropy.
        return per_unit_logp, {"entropy": type_entropy, **prediction}


# -- KL anchor (declaration §7) -----------------------------------------------------------


def hybrid_kl_enemy_relative(model, reference, observation, *, prediction=None):
    """Exact KL(pi_model || pi_reference) for the enemy-relative hybrid policy, averaged over living units per
    row and then over rows. THROW, enemy choice, offset and power are conditionally independent given THROW,
    since the offset head does not condition on the chosen enemy."""
    p = prediction if prediction is not None else model(observation, with_value=False)
    with torch.no_grad():
        q = reference(observation, with_value=False)
    live = p["living"].to(p["action_logits"].dtype)
    log_p, log_q = torch.log_softmax(p["action_logits"], -1), torch.log_softmax(q["action_logits"], -1)
    probs = log_p.exp()
    categorical = (probs * (log_p - log_q)).sum(-1)

    enemy_mask = p["enemy_mask"]  # (batch, slots); identical for p and q, a property of the observation
    enemy_log_p = torch.log_softmax(p["enemy_logits"], -1)
    enemy_log_q = torch.log_softmax(q["enemy_logits"], -1)
    enemy_probs = enemy_log_p.exp()
    # masked slots: both distributions are uniform over them relative to each other where masked out identically,
    # contributing a finite, small value; the (batch, slots) mask broadcasts to (batch, units, slots).
    enemy_kl_per_unit = (enemy_probs * (enemy_log_p - enemy_log_q)).sum(-1)

    move = ((p["move_raw"] - q["move_raw"]).square() / (2 * model.move_log_std.exp().square())).sum(-1)
    offset = ((p["throw_offset_raw"] - q["throw_offset_raw"]).square()
              / (2 * model.throw_offset_log_std.exp().square())).sum(-1)
    power = (p["power_raw"] - q["power_raw"]).square() / (2 * model.power_log_std.exp().square())
    unit = categorical + probs[..., ACTION_MOVE] * move + probs[..., ACTION_THROW] * (enemy_kl_per_unit + offset + power)
    return ((unit * live).sum(-1) / live.sum(-1).clamp_min(1.)).mean()


KL_CHUNK_ROWS = 512  # built in from the start (declaration §14): the per-row tensor here is larger than R1n-e's


def rollout_anchor_kl_enemy_relative(model, reference, observation, chunk=KL_CHUNK_ROWS):
    rows = len(next(iter(observation.values())))
    total = 0.
    with torch.no_grad():
        for start in range(0, rows, chunk):
            part = {key: value[start:start + chunk] for key, value in observation.items()}
            total += float(hybrid_kl_enemy_relative(model, reference, part)) * len(next(iter(part.values())))
    return total / rows


# -- Rollouts and update (declaration §8) --------------------------------------------------


class RolloutRecorderEnemyRelative:
    def __init__(self, model):
        self.model, self.steps = model, []

    def __call__(self, _wrapper, _active, rows, _raws):
        with torch.no_grad():
            action, latent, logp, value = self.model.act(rows)
        self.steps.append({"action_type": action["action_type"], "moveLatent": latent["move"],
                           "enemyLatent": latent["enemy"], "offsetLatent": latent["offset"],
                           "powerLatent": latent["power"], "logp": logp, "value": value})
        return numpy_actions(action)


def build_rollout_enemy_relative(episodes, stored, recorder, cfg):
    if len(stored) != len(recorder.steps):
        raise RuntimeError("logged policy outputs are misaligned with stored rows")
    flat = v1.flatten_block(episodes, stored, cfg["gamma"])
    logged = {key: torch.cat([step[key] for step in recorder.steps]) for key in recorder.steps[0]}
    rewards = torch.tensor([episodes[int(w)]["rewards"][int(d)] for w, d in zip(
        torch.cat([s["worlds"] for s in stored]), flat["decision"])], dtype=torch.float32)
    return {**flat, **logged, "reward": rewards, "advantage": flat["returns"] - logged["value"]}


def anchored_actor_loss_enemy_relative(model, reference, rollout, indices, cfg):
    obs = {key: value[indices] for key, value in rollout["observation"].items()}
    latent = {"move": rollout["moveLatent"][indices], "enemy": rollout["enemyLatent"][indices],
              "offset": rollout["offsetLatent"][indices], "power": rollout["powerLatent"][indices]}
    logp, prediction = model.evaluate_latents(obs, rollout["action_type"][indices], latent, with_value=False)
    live = living_unit_mask(obs)
    losses = ppo_loss(logp, rollout["logp"][indices], rollout["advantage"][indices],
                      torch.zeros(len(indices)), torch.zeros(len(indices)), prediction["entropy"],
                      v1.ppo_config({**cfg, "gaeLambda": cfg["gaeLambda"]}), active_mask=live)
    anchor = (hybrid_kl_enemy_relative(model, reference, obs, prediction=prediction) if cfg["anchorWeight"] > 0
              else torch.zeros(()))
    total = losses["policy"] - cfg["entropyWeight"] * losses["entropy"] + cfg["anchorWeight"] * anchor
    return total, {**losses, "anchorKl": anchor}


def anchored_ppo_update_enemy_relative(model, reference, actor_optimizer, critic_optimizer, rollout, cfg):
    size = len(rollout["advantage"])
    actor_traces, critic_traces, stopped, stop_kl = [], [], False, None
    for epoch in range(cfg["epochs"]):
        for indices in torch.randperm(size).split(cfg["minibatchSize"]):
            loss, losses = anchored_actor_loss_enemy_relative(model, reference, rollout, indices, cfg)
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


def units_lost_fraction_from_wrapper(wrapper, index):
    assigned = wrapper.trackers[index].assigned_ids
    final = wrapper.environment.raw_observations[index]["allies"]
    alive = sum(1 for u in final if u["alive"] and u["id"] in assigned)
    return len(assigned), alive, rb.units_lost_fraction(len(assigned), alive)


# -- Pre-collection probes (declaration §6, §8) --------------------------------------------


def probe_rollout(client, seeds, cfg, *, model, deterministic, account, source):
    """One block, keeping observations for entropy diagnostics and computing per-world L the way
    `roster_baseline.collect_cell` does (needs wrapper state, unlike `full_authority_imitation.collect`)."""
    wrapper = v1.make_wrapper(client, len(seeds), cfg["gamma"])

    def choose(_wrapper, _active, rows, _raws):
        with torch.no_grad():
            action, _, _, _ = model.act(rows, deterministic=deterministic)
        return numpy_actions(action)

    with scenario_override(rb.arm_scenario("normal")):
        episodes, stored, used = v1.run_block(wrapper, seeds, cfg, choose=choose, source=source,
                                              keep_observations=True)
    account(used)
    rows = []
    for index, episode in enumerate(episodes):
        row = v1.episode_row(episode)
        assigned, alive, lost = units_lost_fraction_from_wrapper(wrapper, index)
        rows.append(row | {"assignedUnits": assigned, "blueAliveCount": alive, "unitsLostFraction": lost})
    if any(r["assignedUnits"] != cfg["roster"] for r in rows):
        # the test-fixture bug (a missing scenario_override) ran a whole rollout at roster 1 with no error
        # anywhere downstream; the probe is the first real-scale run and gets pinned, so guard it too.
        raise RuntimeError(f"expected roster {cfg['roster']} assigned units, got "
                           f"{[r['assignedUnits'] for r in rows]}")
    observation = {key: torch.cat([s["observation"][key] for s in stored]) for key in stored[0]["observation"]}
    return rows, observation, episodes, stored


def sigma_candidates(cfg):
    """Absolute sigma-scale values: {1x, 0.5x, 0.25x} of the declared cfg["sigmaScale"] itself (declaration
    §6), NOT {1.0, 0.5, 0.25} in absolute terms -- `prepare_policy`'s `sigma_scale` is the absolute multiplier
    on the imitation-calibrated log-std, so "1x of the declared sigma" means cfg["sigmaScale"] * 1, not 1.0."""
    return [cfg["sigmaScale"] * multiplier for multiplier in (1.0, *SIGMA_FALLBACK)]


def chunked_enemy_entropy(model, observation, chunk=KL_CHUNK_ROWS):
    """Mean enemy-choice entropy and its per-row ceiling (log of the living-enemy count), over THROW-labelled
    rows with AT LEAST TWO living enemies -- a row with exactly one living enemy has entropy = max-entropy =
    ln(1) = 0 by construction (there is only one legal choice), which is not evidence the categorical is
    confident; pooling those rows in with genuinely-ambiguous ones can make a perfectly usable categorical read
    as a false failure (e.g. 30% of rows with 2+ enemies and a real 0.3-nat gap, pooled with 70% single-enemy
    rows at a forced 0-nat gap, averages to 0.09 -- below the 0.1 margin even though the categorical is fine).
    Computed in chunks so the whole stochastic probe rollout is never forwarded in one pass (declaration §14's
    chunking requirement, extended to this probe)."""
    rows = len(next(iter(observation.values())))
    entropy_total, max_entropy_total, considered = 0.0, 0.0, 0
    for start in range(0, rows, chunk):
        part = {key: value[start:start + chunk] for key, value in observation.items()}
        with torch.no_grad():
            prediction = model(part, with_value=False)
        live = living_unit_mask(part)
        throws = live & (prediction["action_logits"].argmax(-1) == ACTION_THROW)  # descriptive only
        living_enemy_count = prediction["enemy_mask"].sum(-1).float().clamp_min(1.)
        entropy_per_row = Categorical(probs=torch.softmax(prediction["enemy_logits"], dim=-1)).entropy()
        max_entropy = living_enemy_count.log()
        ambiguous = (living_enemy_count > 1.5)[:, None].expand_as(throws)
        rows_considered = throws & prediction["enemy_mask"].any(-1)[:, None].expand_as(throws) & ambiguous
        entropy_total += float(entropy_per_row[rows_considered].sum())
        max_entropy_total += float(max_entropy[:, None].expand_as(rows_considered)[rows_considered].sum())
        considered += int(rows_considered.sum())
    if considered == 0:
        return None, None
    return entropy_total / considered, max_entropy_total / considered


def sigma_probe(client, cfg, cohort, account, *, directory=None):
    """Deterministic vs. stochastic success/L on the same worlds, plus enemy-choice entropy, at each sigma
    candidate in turn, stopping at the first that passes (declaration §6). The success-gap rule fires only on
    CONFIDENT degradation (upper CI bound below zero AND the point estimate below the margin) -- at 64 worlds
    and R1n-d's 20-41% discordance the paired-success SE is around 0.07-0.08, so a bare lower-bound-only rule
    fires on noise roughly half the time. Enemy-choice entropy does not depend on sigma (sigma scales only the
    Gaussian move/offset/power heads, not the categorical enemy-score head), so an entropy failure stops the
    probe immediately rather than cycling through the fallback grid for no reason. If `directory` is given, the
    per-seed rows behind each candidate's decision are written there for audit (declaration §14/§12: the probe
    is itself a real-scale run and its results get pinned into the declaration)."""
    candidates = sigma_candidates(cfg)
    seeds = sigma_probe_seeds(cfg, cohort)
    torch.manual_seed(cfg["sigmaProbeTorchSeedBase"] + cohort)  # covers the stochastic rollout's sampling
    report = {"candidates": []}
    for scale in candidates:
        model, _ = prepare_policy(cfg, cohort, sigma_scale=scale)
        model.eval()
        det_rows, det_obs, _, _ = probe_rollout(client, seeds, cfg, model=model, deterministic=True,
            account=account, source=f"s14-c{cohort}-sigma-det-{scale}")
        sto_rows, sto_obs, _, _ = probe_rollout(client, seeds, cfg, model=model, deterministic=False,
            account=account, source=f"s14-c{cohort}-sigma-sto-{scale}")
        if directory is not None:
            rb.write_rows(directory / f"sigma-{scale}-deterministic", det_rows)
            rb.write_rows(directory / f"sigma-{scale}-stochastic", sto_rows)
        det_by_seed = {r["seed"]: r for r in det_rows}
        sto_by_seed = {r["seed"]: r for r in sto_rows}
        ordered = sorted(det_by_seed)
        success_gap = fi.paired_difference([sto_by_seed[s]["success"] for s in ordered],
            [det_by_seed[s]["success"] for s in ordered], samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
        l_gap = fi.paired_difference([sto_by_seed[s]["unitsLostFraction"] for s in ordered],
            [det_by_seed[s]["unitsLostFraction"] for s in ordered], samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
        mean_entropy, mean_max_entropy = chunked_enemy_entropy(model, sto_obs)
        entropy_ok = (mean_entropy is None or mean_max_entropy is None
                      or mean_entropy < mean_max_entropy - cfg["sigmaProbeEntropyMargin"])
        confident_degradation = (success_gap["interval95"][1] < 0
                                  and success_gap["mean"] < cfg["sigmaProbeSuccessGapMin"])
        gap_ok = not confident_degradation
        passed = gap_ok and entropy_ok
        report["candidates"].append({"sigmaScale": scale, "successGap": success_gap, "unitsLostGap": l_gap,
            "meanEnemyEntropy": mean_entropy, "meanMaxEnemyEntropy": mean_max_entropy,
            "successGapOk": gap_ok, "entropyOk": entropy_ok, "passed": passed})
        if passed:
            report["selectedSigmaScale"] = scale
            break
        if not entropy_ok:
            report["selectedSigmaScale"] = None
            report["entropyFailed"] = True
            break
    else:
        report["selectedSigmaScale"] = None
    return report


def chunked_approximate_kl(model, rollout, indices, cfg, chunk=KL_CHUNK_ROWS):
    """`ppo_loss`'s approximate-KL, from `evaluate_latents` alone (no reference forward pass needed -- unlike
    the anchor term, this doesn't require the reference policy), in row chunks, as a row-weighted mean equal to
    the full-batch value."""
    total, count = 0.0, 0
    for start in range(0, len(indices), chunk):
        part = indices[start:start + chunk]
        obs = {key: value[part] for key, value in rollout["observation"].items()}
        latent = {"move": rollout["moveLatent"][part], "enemy": rollout["enemyLatent"][part],
                  "offset": rollout["offsetLatent"][part], "power": rollout["powerLatent"][part]}
        with torch.no_grad():
            logp, _ = model.evaluate_latents(obs, rollout["action_type"][part], latent, with_value=False)
        active = living_unit_mask(obs).float()
        counts = active.sum(-1).clamp_min(1.0)
        log_ratio = (logp - rollout["logp"][part]) * active
        ratio = log_ratio.exp()
        unit_kl = (ratio - 1 - log_ratio) * active
        total += float((unit_kl.sum(-1) / counts).sum())
        count += len(part)
    return total / count


def lr_probe(cfg, cohort, sigma_scale, account):
    """R1n-e amendment A1's exact procedure: for each candidate lr, take ONE seeded Adam step on ONE minibatch
    of a calibration rollout (on a fresh copy of the model, from the same initializer state each time), then
    measure the approximate KL that ONE step produced, evaluated over the WHOLE calibration rollout against the
    rollout's own logged (pre-step) log-probabilities (declaration §8). This is deliberately NOT "does a full
    4-epoch anchored update ever trigger the KL stop" -- at real scale a 64-episode rollout is thousands of rows
    and tens of minibatches, so nearly any lr would eventually trip a 0.01 stop somewhere; that measures whether
    the WHOLE update accumulates too much movement, not whether the FIRST step already overshoots, which is the
    quantity R1n-e's own rule (and the quoted table in the declaration) is about."""
    model, reference = prepare_policy(cfg, cohort, sigma_scale=sigma_scale)
    model.eval()
    seeds = lr_probe_seeds(cfg, cohort)
    torch.manual_seed(cfg["lrProbeTorchSeedBase"] + cohort)  # covers the calibration rollout's sampling
    with SnowGymBatchClient() as client:
        recorder = RolloutRecorderEnemyRelative(model)
        wrapper = v1.make_wrapper(client, len(seeds), cfg["gamma"])
        with scenario_override(rb.arm_scenario("normal")):
            episodes, stored, used = v1.run_block(wrapper, seeds, cfg, choose=recorder,
                source=f"s14-c{cohort}-lrprobe", keep_observations=True)
        account(used)
        assigned = [units_lost_fraction_from_wrapper(wrapper, i)[0] for i in range(len(episodes))]
        if any(a != cfg["roster"] for a in assigned):
            raise RuntimeError(f"expected roster {cfg['roster']} assigned units, got {assigned}")
    rollout = build_rollout_enemy_relative(episodes, stored, recorder, cfg)
    size = len(rollout["advantage"])
    all_indices = torch.arange(size)
    report = {"candidates": [], "rolloutRows": size}
    for lr in LR_CANDIDATES:
        probe_model = copy.deepcopy(model)
        actor_optimizer = torch.optim.Adam(probe_model.actor_parameters(), lr=lr)
        torch.manual_seed(cfg["runSeedBase"] + cohort)  # identical minibatch draw across every candidate
        step_indices = torch.randperm(size)[:min(cfg["minibatchSize"], size)]
        loss, _ = anchored_actor_loss_enemy_relative(probe_model, reference, rollout, step_indices, cfg)
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss in the lr-calibration probe's single step")
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe_model.actor_parameters(), cfg["actorGradClip"], error_if_nonfinite=True)
        actor_optimizer.step()
        first_step_kl = chunked_approximate_kl(probe_model, rollout, all_indices, cfg)
        report["candidates"].append({"lr": lr, "firstStepKl": first_step_kl})
        del probe_model, actor_optimizer
    passing = [c for c in report["candidates"] if c["firstStepKl"] <= cfg["movementKlStop"]]
    report["selectedLr"] = max((c["lr"] for c in passing), default=None)
    return report


def select_shared_lr(lr_reports, cfg):
    """Declaration §8: one shared lr, the largest candidate whose first-step KL is at most the movement stop
    for ALL THREE cohorts, not a separate lr per cohort."""
    passing = set(LR_CANDIDATES)
    for report in lr_reports.values():
        passing &= {c["lr"] for c in report["candidates"] if c["firstStepKl"] <= cfg["movementKlStop"]}
    return max(passing, default=None)


# -- Aggregation, paired analysis and decision rules (declaration §9, §10) ------------------


def paired_analysis_enemy_relative(directories, cfg):
    per_cohort, final, initial = {}, [], []
    worlds = None
    for directory in directories:
        before = {json.loads(line)["seed"]: json.loads(line) for line in
                  (directory / "evaluation" / "initializer-deterministic" / "episodes.jsonl").read_text().splitlines()}
        after = {json.loads(line)["seed"]: json.loads(line) for line in
                 (directory / "evaluation" / "final-deterministic" / "episodes.jsonl").read_text().splitlines()}
        ordered = sorted(before)
        if sorted(after) != ordered:
            raise RuntimeError("evaluation worlds differ across cohorts")
        worlds = ordered if worlds is None else worlds
        if ordered != worlds:
            raise RuntimeError("evaluation worlds differ across cohorts")
        per_cohort[directory.name] = {metric: fi.paired_difference(
            [after[w][metric] for w in worlds], [before[w][metric] for w in worlds],
            samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])
            for metric in ("success", "unitsLostFraction", "timedOut")}
        final.append({w: after[w] for w in worlds})
        initial.append({w: before[w] for w in worlds})
    averaged = {metric: fi.paired_difference(
        [np.mean([f[w][metric] for f in final]) for w in worlds],
        [np.mean([i[w][metric] for i in initial]) for w in worlds],
        samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]) for metric in
        ("success", "unitsLostFraction", "timedOut")}
    return {"perCohort": per_cohort, "cohortAveraged": averaged, "worlds": len(worlds)}


def decision_rules(analysis, reports, cfg):
    if any(r.get("criticSanityStop") or r.get("outcome") in
           ("sigma-probe-failed", "lr-probe-failed", "incomplete") for r in reports.values()):
        return {"outcome": "incomplete", "recommendation": "resolve the failing precondition first",
                "authorizes": "nothing; a follow-up needs its own declaration"}
    averaged = analysis["cohortAveraged"]
    l_mean, (l_low, l_high) = averaged["unitsLostFraction"]["mean"], averaged["unitsLostFraction"]["interval95"]
    success_low = averaged["success"]["interval95"][0]
    timeout_high = averaged["timedOut"]["interval95"][1]
    non_inferior = success_low > cfg["successMargin"] and timeout_high < cfg["timeoutMargin"]
    if not non_inferior or l_low > 0:
        outcome = "harm-or-avoidance"
    elif l_mean <= cfg["deathThreshold"] + 1e-12 and l_high < 0:
        outcome = "improved"
    elif l_high < 0:
        outcome = "improved-below-threshold"
    elif np.median([r["finalAnchorKl"] for r in reports.values() if "finalAnchorKl" in r]) < cfg["movementFloorAnchorKl"]:
        outcome = "no-effective-training"
    else:
        outcome = "no-detectable-change"
    recommendation = {
        "improved": "fresh replication (new training RNGs, new untouched split), then a hard-arm generalization check",
        "improved-below-threshold": "diagnose learning curves (KL stop, anchor KL, return trend) before a longer run",
        "no-effective-training": "the lr-selection procedure was too conservative for the update budget; "
                                 "a longer run or a less conservative lr band needs its own declaration",
        "no-detectable-change": "diagnose learning curves (KL stop, anchor KL, return trend) before a longer run",
        "harm-or-avoidance": "stop this PPO configuration; revisit sigma, the anchor, or the objective"}[outcome]
    per_cohort_l = {name: rows["unitsLostFraction"]["mean"] for name, rows in analysis["perCohort"].items()}
    return {"outcome": outcome, "recommendation": recommendation, "primaryPassed": outcome == "improved",
            "nonInferiority": {"passed": non_inferior, "successLower95": success_low, "timeoutUpper95": timeout_high},
            "unitsLostFraction": averaged["unitsLostFraction"],
            "checks": {"parameterChange": all(r.get("parameterDistance", 0) > 0 for r in reports.values()),
                       "cohortsWithLowerL": sum(v < 0 for v in per_cohort_l.values()),
                       "finalAnchorKl": {name: r.get("finalAnchorKl") for name, r in reports.items()},
                       "perCohortUnitsLostDifference": per_cohort_l},
            "authorizes": "nothing; a follow-up needs its own declaration"}


# -- Declaration, run, archive ---------------------------------------------------------------


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
        "probeBudgetBound": probe_budget_bound(cfg), "trainingBudgetBound": training_budget_bound(cfg),
        "s12Manifest": dr.verify_sealed(TRAINING / "runs" / cfg["sourceRun"]),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s14_declaration.md"),
        "implementationDigest": file_digest(here), "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, reports, probe_decisions, training_decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    directories = [root / f"cohort-{c}" for c in cfg["cohorts"]]
    digests = {d.name: dr.verify_sealed(d) for d in directories}
    complete = not any(r.get("criticSanityStop") or r.get("outcome") in
                       ("sigma-probe-failed", "lr-probe-failed", "incomplete") for r in reports.values())
    analysis = paired_analysis_enemy_relative(directories, cfg) if complete else None
    rules = decision_rules(analysis, reports, cfg)
    report = {"format": "snowgym.m8-s14-ppo-continuation-report.v0", "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"], "cohortManifests": digests,
        "cohorts": reports, "pairedDifferences": analysis, "decisionRules": rules,
        "probeSimulatorDecisions": probe_decisions, "trainingSimulatorDecisions": training_decisions,
        "probeBudgetBound": probe_budget_bound(cfg), "trainingBudgetBound": training_budget_bound(cfg),
        "withinProbeBudget": probe_decisions <= cfg["probeBudgetCap"],
        "withinTrainingBudget": training_decisions <= cfg["trainingBudgetCap"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s14-manifest.v0")
    return report


# -- Stage 1: outcome-blind probes for every cohort, before any real training (declaration §6, §8, §12) --------


def probe_all_cohorts(root, cfg, account_probe):
    """Sigma probe per cohort, then the lr-calibration probe per cohort at its selected sigma, then ONE shared
    lr selected across all three (declaration §8: "one shared lr", not a separate lr per cohort). Writes
    `probe-report.json`. This must run, and its results be inspected and amended into the declaration, before
    any cohort's real training stage (declaration §12) -- `run()` does this inline for convenience (tests, a
    single-process run), but the real launch should run this stage on its own first."""
    root = Path(root)
    declaration = json.loads((root / "declaration.json").read_text(encoding="utf-8"))
    if json_digest(declaration["config"]) != json_digest(cfg):
        raise RuntimeError("configuration differs from the run declaration")
    sigma_reports, lr_reports = {}, {}
    for cohort in cfg["cohorts"]:
        # "probe-cohort-N", not "cohort-N": `run()`'s convenience path shares one root between this stage and
        # `run_cohort`, which creates its own "cohort-N" directory and refuses to see it pre-exist.
        cohort_directory = root / f"probe-cohort-{cohort}"
        cohort_directory.mkdir(parents=True, exist_ok=True)
        with SnowGymBatchClient() as client:
            require_capabilities(client)
            sigma_reports[str(cohort)] = sigma_probe(client, cfg, cohort, account_probe, directory=cohort_directory)
        selected_sigma = sigma_reports[str(cohort)]["selectedSigmaScale"]
        if selected_sigma is not None:
            lr_reports[str(cohort)] = lr_probe(cfg, cohort, selected_sigma, account_probe)
    shared_lr = select_shared_lr(lr_reports, cfg) if len(lr_reports) == len(cfg["cohorts"]) else None
    report = {"sigma": sigma_reports, "lr": lr_reports, "selectedSigmaByCohort":
        {c: sigma_reports[c]["selectedSigmaScale"] for c in sigma_reports}, "selectedSharedLr": shared_lr}
    write_json(root / "probe-report.json", report)
    # Not sealed here: `run()`'s convenience path shares one root between this stage and `run_cohort`/
    # `aggregate`, which seals that same root later, and `write_json` refuses to overwrite an existing
    # manifest.json. The real, separate-root usage (declaration §12) seals explicitly via `--stage probe`.
    return report


# -- Stage 2: one cohort's critic re-warm-start -> training -> evaluation, at the probed sigma/lr
# (declaration §2, §5, §8, §9) --------------------------------------------------------------------


def run_cohort(root, cfg, cohort, sigma_scale, actor_lr, account_training):
    """Critic re-warm-start at `sigma_scale` -> training at the shared `actor_lr` -> evaluation. Both come from
    the probe stage (`probe_all_cohorts`), already amended into the declaration before this stage runs."""
    root = Path(root)
    declaration = json.loads((root / "declaration.json").read_text(encoding="utf-8"))
    if json_digest(declaration["config"]) != json_digest(cfg):
        raise RuntimeError("configuration differs from the run declaration")
    directory = root / f"cohort-{cohort}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    spent = [0]

    def account_and_count(count):
        spent[0] += count
        account_training(count)
    torch.manual_seed(cfg["runSeedBase"] + cohort)
    report = {"cohort": cohort, "sigmaScale": sigma_scale, "actorLearningRate": actor_lr}

    with SnowGymBatchClient() as client:
        require_capabilities(client)
        model, reference = prepare_policy(cfg, cohort, sigma_scale=sigma_scale)
        model.train()
        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        # `warm_start_critic_mc(model, wrapper, cfg, rng_index, ...)` uses the SAME rng_index for both the world
        # seed band (fold_seeds, needs rng_index=cohort to hit this cohort's declared 5{cohort}90000/5{cohort}95000
        # bands) and `trainingRngs[rng_index]`, so the list must be long enough to index at `cohort`, not a
        # 1-element list.
        training_rngs = [0] * (max(cfg["cohorts"]) + 1)
        training_rngs[cohort] = cfg["runSeedBase"] + cohort
        critic_cfg = {**cfg, "trainingRngs": training_rngs, "learningRate": cfg["criticLearningRate"]}
        with scenario_override(rb.arm_scenario("normal")):
            warm, arrays, warm_episodes, _, critic_optimizer = v1.warm_start_critic_mc(
                model, wrapper, critic_cfg, cohort, source=f"s14-c{cohort}-critic")
        account_and_count(warm["simulatorDecisions"])
        write_json(directory / "critic-warm-start.json", warm)
        np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
        write_episodes(directory / "critic-episodes", warm_episodes)
        report["criticWarmStart"] = {k: warm[k] for k in
            ("predictiveR2", "predictiveR2Interval95", "timeOnlyR2", "gatePassed")}
        report["criticSanityStop"] = dr.critic_sanity_stop(warm, cfg)
        if report["criticSanityStop"]:
            report["outcome"] = "incomplete"
            report["simulatorDecisions"] = spent[0]
            write_json(directory / "cohort-report.json", report)
            dr.seal(directory, "snowgym.m8-s14-cohort-manifest.v0")
            return report

        actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=actor_lr)
        train_cfg = {**cfg, "actorLearningRate": actor_lr}
        history, rows = [], []
        with scenario_override(rb.arm_scenario("normal")):
            for update in range(cfg["updates"]):
                recorder = RolloutRecorderEnemyRelative(model)
                episodes, stored, used = v1.run_block(wrapper, training_seeds(cfg, cohort, update), train_cfg,
                    choose=recorder, source=f"s14-c{cohort}-{update}", keep_observations=True)
                account_and_count(used)
                if update == 0:
                    # the test-fixture bug (a missing scenario_override) ran a whole rollout at roster 1 with
                    # no error anywhere downstream; guard the real run against the same silent failure.
                    assigned = [units_lost_fraction_from_wrapper(wrapper, i)[0] for i in range(len(episodes))]
                    if any(a != cfg["roster"] for a in assigned):
                        raise RuntimeError(f"expected roster {cfg['roster']} assigned units, got {assigned}")
                rollout = build_rollout_enemy_relative(episodes, stored, recorder, train_cfg)
                step = anchored_ppo_update_enemy_relative(model, reference, actor_optimizer, critic_optimizer,
                    rollout, train_cfg)
                anchor = rollout_anchor_kl_enemy_relative(model, reference, rollout["observation"])
                rows_summary = [v1.episode_row(e) for e in episodes]
                lost_fractions = [units_lost_fraction_from_wrapper(wrapper, i)[2] for i in range(len(episodes))]
                entry = {"update": update + 1, "episodes": len(episodes), "decisions": int(len(rollout["advantage"])),
                    "successes": sum(r["success"] for r in rows_summary),
                    "timeouts": sum(r["timedOut"] for r in rows_summary),
                    "meanUnitsLostFraction": float(np.mean(lost_fractions)),
                    "meanUndiscountedReturn": float(np.mean([sum(e["rewards"]) for e in episodes])),
                    "actorOptimizerSteps": step["actorOptimizerSteps"], "klStopped": step["klStopped"],
                    "stopApproximateKl": step["stopApproximateKl"],
                    "meanApproximateKl": float(np.mean([t["approximateKl"] for t in step["actorMinibatches"]]))
                        if step["actorMinibatches"] else None,
                    "anchorKlAfterUpdate": anchor,
                    "meanCriticLoss": float(np.mean([t["valueLoss"] for t in step["criticMinibatches"]])),
                    "rejectedActions": sum(r["rejectedActions"] for r in rows_summary),
                    "totalActions": sum(r["totalActions"] for r in rows_summary)}
                history.append(entry)
                rows.extend({**r, "update": update + 1, "unitsLostFraction": lost}
                            for r, lost in zip(rows_summary, lost_fractions))
                if update + 1 in cfg["checkpointUpdates"]:
                    torch.save({"model": model.state_dict(), "actorOptimizer": actor_optimizer.state_dict(),
                                "criticOptimizer": critic_optimizer.state_dict(), "update": update + 1},
                               directory / f"update-{update + 1:03d}.pt")
                print(json.dumps({"cohort": cohort, "update": update + 1, "successes": entry["successes"],
                                  "meanUnitsLostFraction": round(entry["meanUnitsLostFraction"], 4),
                                  "klStopped": step["klStopped"], "actorSteps": step["actorOptimizerSteps"],
                                  "anchorKl": round(anchor, 4)}), flush=True)
                del rollout, stored, recorder
        write_json(directory / "training-history.json", history)
        (directory / "training-episodes.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
        report["finalAnchorKl"] = history[-1]["anchorKlAfterUpdate"]
        report["parameterDistance"] = dr.parameter_distance(model, reference)

        model.eval()
        report["evaluation"] = {}
        worlds = evaluation_seeds(cfg)
        for m, (name, policy) in enumerate((("initializer", reference), ("final", model))):
            for mode in ("deterministic", "stochastic"):
                if mode == "stochastic":
                    torch.manual_seed(cfg["stochasticEvalSeedBase"] + 10 * cohort + m)
                eval_rows = rb.collect_cell(client, worlds, cfg, "normal", model=policy, mode=mode,
                    account=account_and_count)
                rb.write_rows(directory / "evaluation" / f"{name}-{mode}", eval_rows)
                report["evaluation"][f"{name}-{mode}"] = rb.summarize(eval_rows, cfg)
                assert eval_rows[0]["assignedUnits"] == cfg["roster"], \
                    f"expected roster {cfg['roster']} assigned units, got {eval_rows[0]['assignedUnits']}"

    report["simulatorDecisions"] = spent[0]
    write_json(directory / "cohort-report.json", report)
    dr.seal(directory, "snowgym.m8-s14-cohort-manifest.v0")
    return report


# -- Stage 3: aggregation, and the single-process convenience path ------------------------------------------


def run(output, cfg=None):
    """Declares, then runs the probe stage inline (for a single-process run: tests, or a real launch that has
    already reviewed and pinned the probe results into the declaration). A real launch normally runs
    `probe_all_cohorts` as its own stage first (declaration §12) rather than trusting an in-process probe."""
    root = Path(output)
    cfg = cfg or configuration()
    declare(root, cfg)
    probe_steps, training_steps = 0, 0

    def account_probe(count):
        nonlocal probe_steps
        probe_steps += count
        if probe_steps > cfg["probeBudgetCap"]:
            raise ValueError("M8-S14 probe budget exceeded")

    def account_training(count):
        nonlocal training_steps
        training_steps += count
        if training_steps > cfg["trainingBudgetCap"]:
            raise ValueError("M8-S14 training budget exceeded")

    probes = probe_all_cohorts(root, cfg, account_probe)
    reports = {}
    for cohort in cfg["cohorts"]:
        sigma_scale = probes["selectedSigmaByCohort"].get(str(cohort))
        if sigma_scale is None:
            reports[str(cohort)] = {"cohort": cohort, "outcome": "sigma-probe-failed"}
            continue
        if probes["selectedSharedLr"] is None:
            reports[str(cohort)] = {"cohort": cohort, "outcome": "lr-probe-failed"}
            continue
        reports[str(cohort)] = run_cohort(root, cfg, cohort, sigma_scale, probes["selectedSharedLr"], account_training)
    return aggregate(root, cfg, reports, probe_steps, training_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "declare", "probe", "cohort", "aggregate"), default="all")
    parser.add_argument("--cohort", type=int, default=None)
    args = parser.parse_args()
    cfg = configuration()

    if args.stage == "all":
        run(args.output, cfg)
    elif args.stage == "declare":
        declare(args.output, cfg)
    elif args.stage == "probe":
        steps = 0

        def account(count, _steps=[0]):
            _steps[0] += count
            if _steps[0] > cfg["probeBudgetCap"]:
                raise ValueError("M8-S14 probe budget exceeded")

        probe_all_cohorts(args.output, cfg, account)
        dr.seal(args.output, "snowgym.m8-s14-probe-manifest.v0")
    elif args.stage == "cohort":
        if args.cohort is None:
            raise SystemExit("--stage cohort requires --cohort")
        # Reads sigma/lr from cfg itself, not from probe-report.json: they must already be pinned into
        # configuration() (declaration §12 amendment, reviewed and committed) before this stage can run at all,
        # so a real launch cannot silently skip that review step.
        sigma_scale = cfg["sigmaScaleByCohort"].get(args.cohort)
        actor_lr = cfg["actorLearningRate"]
        if sigma_scale is None or actor_lr is None:
            raise SystemExit("sigmaScaleByCohort/actorLearningRate are not pinned in configuration() yet -- "
                              "run --stage probe, amend the declaration, and pin the results first")

        def account(count, _steps=[0]):
            _steps[0] += count
            if _steps[0] > cfg["perCohortTrainingBudgetCap"]:
                raise ValueError("M8-S14 per-cohort training budget exceeded")

        run_cohort(args.output, cfg, args.cohort, sigma_scale, actor_lr, account)
    elif args.stage == "aggregate":
        # The probe stage runs in a SEPARATE root (declaration §12 amendment process) and its decisions were
        # already accounted and capped there; this root's aggregate only covers what `run_cohort` itself spent,
        # which each cohort-report.json already carries in full via "simulatorDecisions" (critic warm-start +
        # every training update + evaluation, all counted by run_cohort's own `account_and_count`).
        reports = {}
        for cohort in cfg["cohorts"]:
            path = Path(args.output) / f"cohort-{cohort}" / "cohort-report.json"
            reports[str(cohort)] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
                {"cohort": cohort, "outcome": "incomplete"}
        training_decisions = sum(r.get("simulatorDecisions", 0) for r in reports.values())
        aggregate(args.output, cfg, reports, 0, training_decisions)
