"""R1n-c: artifact-retaining full-authority imitation of the plan teacher.

BC then DAgger from `PlanAwareTeamController` labels into a global-decoder
`FullAuthorityPolicyV1` on 1v1 Engage, followed by closed-loop evaluation,
held-out label error and the Monte Carlo critic precondition for R1n-d.
No PPO update runs here. See `reviews/m7b_r1n_c_declaration.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_THROW
from ..checkpoint import semantic_state_digest
from ..executor.full_authority_ppo import ARENA_HALF_EXTENT
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..ppo_collect import numpy_actions
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import (TRAINING, e3_digests_unchanged, outcome_summary, uniform_floor,
                                         write_episodes)
from .geometry_representation_probe import SLOW_RADIUS
from .interventions import require_capabilities
from .reservoir import file_digest
from .supervised_probe import write_json

ACTION_NAMES = ("noop", "move", "throw", "hold")
ACTOR_KEYS = ("allies", "enemies", "projectiles", "obstacles", "ally_mask", "enemy_mask", "projectile_mask",
              "obstacle_mask", "unit_action_mask", "plan_unit_roles", "plan_groups", "plan_group_mask",
              "plan_role_state")
LABEL_KEYS = ("action_type", "target", "power")


def configuration():
    seeds = [97101, 97102, 97103]
    return {**v1.configuration(), "format": "snowgym.full-authority-imitation-config.v0", "destination": "global",
        "optimizerSeeds": seeds, "trainingRngs": seeds, "roundEpisodes": 128, "roundBlockWorlds": 64,
        "roundSeedBase": 650000, "roundSeedStride": 1000, "fits": 5, "stepsPerFit": 3000,
        "imitationMinibatch": 256, "imitationLearningRate": 3e-4, "imitationGradClip": .5,
        "slowRadius": SLOW_RADIUS, "endpointClip": 3.0, "historyEvery": 25,
        "developmentSplits": {"A": [600000, 600099], "B": [601000, 601099]}, "stochasticSplit": "A",
        "labelErrorSplit": "B", "evaluationBlockWorlds": 50, "floorRng": 981002,
        "trainSeedBase": 660000, "heldOutSeedBase": 670000, "bootstrapSeed": 973001, "bootstrapSamples": 10000,
        "nearCeilingMargin": .20, "contactMinFraction": .25, "successMinFraction": .25,
        "throwRecallFlag": .5, "executionGapFlag": .20, "seedSpreadFlag": .30,
        "simulatorBudget": 900000, "assistType": "teacher-imitation training; none at runtime",
        "assistVersion": "snowgym.full-authority-imitation.v0", "autonomousQualificationEligible": False}


def split_seeds(cfg, name):
    low, high = cfg["developmentSplits"][name]
    return list(range(low, high + 1))


def round_seeds(cfg, round_index):
    base = cfg["roundSeedBase"] + cfg["roundSeedStride"] * round_index
    return list(range(base, base + cfg["roundEpisodes"]))


def imitation_parameters(model):
    """Encoders and heads only: the calibrated log-stds and the critic stay frozen (declaration §3)."""
    return [p for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("critic.") and not name.endswith("log_std")]


# -- Loss (declaration §3) ----------------------------------------------------------


def imitation_loss(model, observation, labels, cfg):
    prediction = model(observation, with_value=False)
    live = prediction["living"]
    teacher_type = labels["action_type"].long()
    legal = observation["unit_action_mask"].bool().gather(-1, teacher_type[..., None]).squeeze(-1)
    valid = live & legal
    zero = prediction["action_logits"].sum() * 0
    type_loss = F.cross_entropy(prediction["action_logits"][valid], teacher_type[valid]) if valid.any() else zero
    scale = prediction["move_raw"].new_tensor(ARENA_HALF_EXTENT)
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
    aim_teacher = teacher_world - own
    aimed = throws & (aim_teacher.norm(dim=-1) > 1e-3)
    aim_learned = torch.tanh(prediction["throw_raw"]) * scale - own
    aim = ((1 - F.cosine_similarity(aim_learned[aimed], aim_teacher[aimed], dim=-1, eps=1e-6)).mean()
           if aimed.any() else zero)
    power = ((torch.sigmoid(prediction["power_raw"][throws]) - labels["power"].float()[throws]).square().mean()
             if throws.any() else zero)
    total = type_loss + heading + endpoint + aim + power
    return {"total": total, "type": type_loss, "moveHeading": heading, "moveEndpoint": endpoint,
            "throwAim": aim, "power": power, "illegalLabels": int((live & ~legal).sum()),
            "moveFar": int(far.sum()), "moveNear": int(near.sum()), "throwUnits": int(throws.sum())}


# -- Labeled collection (declaration §2, §4) -----------------------------------------


class Labeler:
    """`run_block` controller: the teacher labels every visited state; the teacher or the
    learner acts. One label entry per step keeps labels aligned with stored rows."""

    def __init__(self, model=None, *, deterministic=True, record=True):
        self.model, self.deterministic, self.record = model, deterministic, record
        self.labels = []

    def __call__(self, wrapper, active, rows, _raws):
        if self.record or self.model is None:
            labels = wrapper.environment.plan_teacher_tensor_actions_indices(active)
            if self.record:
                self.labels.append({key: torch.as_tensor(labels[key]).clone() for key in LABEL_KEYS})
            if self.model is None:
                return labels
        with torch.no_grad():
            action, _, _, _ = self.model.act(rows, deterministic=self.deterministic)
        return numpy_actions(action)


def collect(client, seeds, cfg, *, model, source, block_worlds, account, deterministic=True, record=True):
    """Complete-episode blocks with teacher labels at every visited state."""
    episodes, observation_parts, label_parts = [], [], []
    for start in range(0, len(seeds), block_worlds):
        block = seeds[start:start + block_worlds]
        labeler = Labeler(model, deterministic=deterministic, record=record)
        wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
        found, stored, used = v1.run_block(wrapper, block, cfg, choose=labeler, source=source, keep_observations=record)
        account(used)
        episodes.extend(found)
        if record:
            if len(stored) != len(labeler.labels):
                raise RuntimeError("teacher labels are misaligned with stored rows")
            observation_parts.append({key: torch.cat([s["observation"][key] for s in stored]) for key in ACTOR_KEYS})
            label_parts.append({key: torch.cat([entry[key] for entry in labeler.labels]) for key in LABEL_KEYS})
    if not record:
        return episodes, None
    return episodes, {"observation": {key: torch.cat([p[key] for p in observation_parts]) for key in ACTOR_KEYS},
                      "labels": {key: torch.cat([p[key] for p in label_parts]) for key in LABEL_KEYS}}


def part_summary(part, round_index, seeds):
    live = part["observation"]["ally_mask"].bool() & (part["observation"]["allies"][..., 1] > .5)
    types = part["labels"]["action_type"][live]
    legal = part["observation"]["unit_action_mask"].bool().gather(-1, part["labels"]["action_type"].long()[..., None])
    return {"round": round_index, "seeds": [seeds[0], seeds[-1]], "rows": len(types),
        "labelCounts": {name: int((types == index).sum()) for index, name in enumerate(ACTION_NAMES)},
        "illegalLabels": int((live & ~legal.squeeze(-1)).sum()),
        "observationDigest": semantic_state_digest(part["observation"]),
        "labelDigest": semantic_state_digest(part["labels"])}


class Aggregate:
    def __init__(self):
        self.observation, self.labels = None, None

    def add(self, part):
        if self.observation is None:
            self.observation, self.labels = dict(part["observation"]), dict(part["labels"])
        else:
            self.observation = {k: torch.cat([self.observation[k], part["observation"][k]]) for k in ACTOR_KEYS}
            self.labels = {k: torch.cat([self.labels[k], part["labels"][k]]) for k in LABEL_KEYS}

    def __len__(self):
        return 0 if self.labels is None else len(self.labels["action_type"])

    def batch(self, indices):
        return ({k: v[indices] for k, v in self.observation.items()}, {k: v[indices] for k, v in self.labels.items()})


def fit(model, optimizer, data, cfg, *, seed):
    generator = torch.Generator().manual_seed(seed)
    parameters = imitation_parameters(model)
    history, step = [], 0
    while step < cfg["stepsPerFit"]:
        for indices in torch.randperm(len(data), generator=generator).split(cfg["imitationMinibatch"]):
            if step >= cfg["stepsPerFit"]:
                break
            observation, labels = data.batch(indices)
            losses = imitation_loss(model, observation, labels, cfg)
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


# -- Held-out label error (declaration §5) -------------------------------------------


def label_error(model, part, cfg, chunk=2048):
    obs, labels = part["observation"], part["labels"]
    size = len(labels["action_type"])
    predicted, move_far, move_near, aim, power = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, size, chunk):
            o = {k: v[start:start + chunk] for k, v in obs.items()}
            lab = {k: v[start:start + chunk] for k, v in labels.items()}
            prediction = model(o, with_value=False)
            live = prediction["living"]
            kind = lab["action_type"].long()
            predicted.append(torch.stack([prediction["action_logits"].argmax(-1)[live], kind[live]], -1))
            scale = prediction["move_raw"].new_tensor(ARENA_HALF_EXTENT)
            own = o["allies"][..., 2:4].float() * scale
            teacher = lab["target"].float() * scale
            moves = live & (kind == ACTION_MOVE)
            learned = model.decode_move(o, prediction["move_raw"]) * scale
            distance = (teacher - own).norm(dim=-1)
            far, near = moves & (distance > cfg["slowRadius"]), moves & (distance <= cfg["slowRadius"])
            cosine = F.cosine_similarity(learned - own, teacher - own, dim=-1, eps=1e-6).clamp(-1, 1)
            move_far.append(torch.rad2deg(torch.acos(cosine))[far])
            move_near.append((learned - teacher).norm(dim=-1)[near])
            throws = live & (kind == ACTION_THROW)
            aim_teacher = teacher - own
            aimed = throws & (aim_teacher.norm(dim=-1) > 1e-3)
            aim_cos = F.cosine_similarity(torch.tanh(prediction["throw_raw"]) * scale - own, aim_teacher, dim=-1,
                                          eps=1e-6).clamp(-1, 1)
            aim.append(torch.rad2deg(torch.acos(aim_cos))[aimed])
            power.append((torch.sigmoid(prediction["power_raw"]) - lab["power"].float()).abs()[throws])
    pairs = torch.cat(predicted)
    guess, truth = pairs[:, 0], pairs[:, 1]

    def mean(values):
        values = torch.cat(values)
        return float(values.mean()) if len(values) else None

    per_type = {}
    for index, name in enumerate(ACTION_NAMES):
        support, chosen = int((truth == index).sum()), int((guess == index).sum())
        hit = int(((guess == index) & (truth == index)).sum())
        per_type[name] = {"support": support, "recall": hit / support if support else None,
                          "precision": hit / chosen if chosen else None}
    return {"units": len(truth), "typeAccuracy": float((guess == truth).float().mean()) if len(truth) else None,
            "perType": per_type, "moveHeadingErrorDegrees": mean(move_far), "moveEndpointErrorWorld": mean(move_near),
            "throwAimHeadingErrorDegrees": mean(aim), "powerMeanAbsoluteError": mean(power)}


# -- Comparisons, rules and flags (declaration §5, §6) --------------------------------


def paired_difference(first, second, *, samples, seed):
    """Mean of first − second over paired seeds, with a 95% bootstrap interval."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    generator = np.random.default_rng(seed)
    means = difference[generator.integers(0, len(difference), size=(samples, len(difference)))].mean(axis=1)
    return {"mean": float(difference.mean()), "interval95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def success_by_seed(episodes):
    return {e["seed"]: float(e["success"]) for e in episodes}


def decision_rules(evaluations, ceilings, critic, label_errors, cfg):
    splits = {}
    for split in cfg["developmentSplits"]:
        rows = [evaluations[str(s)][f"{split}-deterministic"] for s in cfg["optimizerSeeds"]]
        success = float(np.mean([r["successFraction"] for r in rows]))
        contact = float(np.mean([r["contactFraction"] for r in rows]))
        ceiling = ceilings[split]["successFraction"]
        # Rows overlap only when the ceiling is below 45%; the conservative row wins (amendment A3).
        if contact < cfg["contactMinFraction"] or success < cfg["successMinFraction"]:
            outcome = "imitation-failure"
        elif success >= ceiling - cfg["nearCeilingMargin"]:
            outcome = "near-ceiling"
        else:
            outcome = "ppo-headroom"
        splits[split] = {"meanSuccess": success, "meanContact": contact, "ceiling": ceiling, "outcome": outcome,
                         "seedSpread": float(max(r["successFraction"] for r in rows) - min(r["successFraction"] for r in rows))}
    severity = {"imitation-failure": 0, "ppo-headroom": 1, "near-ceiling": 2}
    conservative = min((s["outcome"] for s in splits.values()), key=severity.get)
    stochastic = cfg["stochasticSplit"]
    gaps = [evaluations[str(s)][f"{stochastic}-deterministic"]["successFraction"]
            - evaluations[str(s)][f"{stochastic}-stochastic"]["successFraction"] for s in cfg["optimizerSeeds"]]
    recalls = [label_errors[str(s)]["perType"]["throw"]["recall"] for s in cfg["optimizerSeeds"]]
    recommendation = {"near-ceiling": "R1n-d: continuous primary test (return, death rate, completion time) "
                      "or a harder task (second mission or 2v2); R1's +20 success gate is infeasible",
                      "ppo-headroom": "R1n-d: KL-anchored frozen-reward PPO with R1's +20 gate, powered at the measured success",
                      "imitation-failure": "diagnose per-head held-out errors before any PPO"}[conservative]
    return {"splits": splits, "outcome": conservative, "recommendation": recommendation,
        "flags": {"criticHealthy": all(critic[str(s)]["gatePassed"] for s in cfg["optimizerSeeds"]),
                  "throwCollapse": any(r is None or r < cfg["throwRecallFlag"] for r in recalls),
                  "executionModeGap": any(g > cfg["executionGapFlag"] for g in gaps),
                  "seedInstability": any(s["seedSpread"] > cfg["seedSpreadFlag"] for s in splits.values())},
        "executionGaps": gaps, "throwRecalls": recalls, "authorizes": "nothing; R1n-d needs its own declaration"}


def evaluation_summary(episodes, cfg):
    summary = outcome_summary(episodes, cfg)
    summary["timeoutFraction"] = sum(e["timedOut"] for e in episodes) / len(episodes) if episodes else None
    return summary


# -- Run -------------------------------------------------------------------------------


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
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_c_declaration.md"),
        "imitationImplementationDigest": file_digest(here),
        "trainImplementationDigest": file_digest(here.parent / "full_authority_train_v1.py"),
        "policyImplementationDigest": file_digest(here.parents[1] / "executor/full_authority_ppo_v1.py"),
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"], "assistVersion": cfg["assistVersion"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})
    steps = 0

    def account(count):
        nonlocal steps
        steps += count
        if steps > cfg["simulatorBudget"]:
            raise ValueError("R1n-c simulator budget exceeded")

    def teacher(wrapper, active, _rows, _raws):
        return wrapper.environment.plan_teacher_tensor_actions_indices(active)

    with SnowGymBatchClient() as client:
        capabilities = require_capabilities(client)
        ceilings, floors = {}, {}
        for split in cfg["developmentSplits"]:
            ceiling, _ = collect(client, split_seeds(cfg, split), cfg, model=None, source=f"ceiling-{split}",
                                 block_worlds=cfg["evaluationBlockWorlds"], account=account, record=False)
            write_episodes(root / "controls" / f"ceiling-{split}", ceiling)
            floor = []
            seeds = split_seeds(cfg, split)
            for start in range(0, len(seeds), cfg["evaluationBlockWorlds"]):
                block = seeds[start:start + cfg["evaluationBlockWorlds"]]
                wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
                found, _, used = v1.run_block(wrapper, block, cfg, choose=uniform_floor(cfg["floorRng"] + start),
                                              source=f"floor-{split}")
                account(used)
                floor.extend(found)
            write_episodes(root / "controls" / f"floor-{split}", floor)
            ceilings[split], floors[split] = evaluation_summary(ceiling, cfg), evaluation_summary(floor, cfg)
            ceilings[split]["bySeed"] = success_by_seed(ceiling)
        write_json(root / "controls/summary.json", {"ceiling": ceilings, "floor": floors})
        print(json.dumps({"ceiling": {s: c["successes"] for s, c in ceilings.items()},
                          "floor": {s: f["successes"] for s, f in floors.items()}}), flush=True)

        teacher_episodes, round_zero = collect(client, round_seeds(cfg, 0), cfg, model=None, source="round-0-teacher",
                                               block_worlds=cfg["roundBlockWorlds"], account=account)
        write_episodes(root / "round-0-teacher", teacher_episodes)
        round_zero_summary = part_summary(round_zero, 0, round_seeds(cfg, 0))
        write_json(root / "round-0-teacher/dataset.json", round_zero_summary)

        evaluations, label_errors, critic, histories = {}, {}, {}, {}
        for seed_index, seed in enumerate(cfg["optimizerSeeds"]):
            directory = root / f"seed-{seed}"
            directory.mkdir()
            torch.manual_seed(seed)
            model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
                target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
            optimizer = torch.optim.Adam(imitation_parameters(model), lr=cfg["imitationLearningRate"])
            data, rounds, history = Aggregate(), [round_zero_summary], {}
            data.add(round_zero)
            for fit_index in range(cfg["fits"]):
                history[str(fit_index)] = fit(model, optimizer, data, cfg, seed=seed + 1000 * fit_index)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "fit": fit_index,
                            "rows": len(data)}, directory / f"fit-{fit_index}.pt")
                print(json.dumps({"seed": seed, "fit": fit_index, "rows": len(data),
                                  "finalLoss": history[str(fit_index)][-1]["total"], "simulatorDecisions": steps}), flush=True)
                if fit_index == cfg["fits"] - 1:
                    break
                seeds = round_seeds(cfg, fit_index + 1)
                found, part = collect(client, seeds, cfg, model=model, source=f"seed-{seed}-round-{fit_index + 1}",
                                      block_worlds=cfg["roundBlockWorlds"], account=account)
                write_episodes(directory / f"round-{fit_index + 1}", found)
                rounds.append(part_summary(part, fit_index + 1, seeds))
                data.add(part)
                del part
            write_json(directory / "fit-history.json", history)
            write_json(directory / "dataset.json", {"rounds": rounds, "aggregateRows": len(data)})
            del data
            histories[str(seed)] = {k: v[-1] for k, v in history.items()}

            model.eval()
            evaluations[str(seed)] = {}
            for split in cfg["developmentSplits"]:
                wants_labels = split == cfg["labelErrorSplit"]
                found, part = collect(client, split_seeds(cfg, split), cfg, model=model, source=f"seed-{seed}-{split}-det",
                                      block_worlds=cfg["evaluationBlockWorlds"], account=account, record=wants_labels)
                write_episodes(directory / f"eval-{split}-deterministic", found)
                evaluations[str(seed)][f"{split}-deterministic"] = {**evaluation_summary(found, cfg),
                                                                   "bySeed": success_by_seed(found)}
                if wants_labels:
                    label_errors[str(seed)] = label_error(model, part, cfg)
                    write_json(directory / "label-error.json", label_errors[str(seed)])
                    del part
            split = cfg["stochasticSplit"]
            found, _ = collect(client, split_seeds(cfg, split), cfg, model=model, source=f"seed-{seed}-{split}-sto",
                               block_worlds=cfg["evaluationBlockWorlds"], account=account, deterministic=False, record=False)
            write_episodes(directory / f"eval-{split}-stochastic", found)
            evaluations[str(seed)][f"{split}-stochastic"] = evaluation_summary(found, cfg)

            wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
            report, arrays, episodes, _, _ = v1.warm_start_critic_mc(model, wrapper, cfg, seed_index,
                                                                     source=f"seed-{seed}-critic")
            account(report["simulatorDecisions"])
            report["behaviorOutcomes"] = evaluation_summary(episodes, cfg)
            critic[str(seed)] = report
            write_json(directory / "critic-warm-start.json", report)
            np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
            torch.save({"model": model.state_dict(), "note": "final imitation policy with warm-started critic"},
                       directory / "critic-policy.pt")
            write_episodes(directory / "critic-episodes", episodes)
            print(json.dumps({"seed": seed, "A": evaluations[str(seed)]["A-deterministic"]["successes"],
                "B": evaluations[str(seed)]["B-deterministic"]["successes"],
                "A-stochastic": evaluations[str(seed)]["A-stochastic"]["successes"],
                "criticGate": report["gatePassed"], "predictiveR2": report["predictiveR2"],
                "clockSkill": report["clockSkillScore"], "simulatorDecisions": steps}), flush=True)
            del model, optimizer, arrays, episodes

    comparisons = {}
    for split in cfg["developmentSplits"]:
        ceiling = ceilings[split]["bySeed"]
        keys = sorted(ceiling)
        per_seed = {str(s): paired_difference([evaluations[str(s)][f"{split}-deterministic"]["bySeed"][k] for k in keys],
                                              [ceiling[k] for k in keys], samples=cfg["bootstrapSamples"],
                                              seed=cfg["bootstrapSeed"]) for s in cfg["optimizerSeeds"]}
        pooled = [np.mean([evaluations[str(s)][f"{split}-deterministic"]["bySeed"][k] for s in cfg["optimizerSeeds"]])
                  for k in keys]
        comparisons[split] = {"perOptimizerSeed": per_seed, "seedAveraged": paired_difference(
            pooled, [ceiling[k] for k in keys], samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"])}
    rules = decision_rules(evaluations, ceilings, critic, label_errors, cfg)
    report = {"format": "snowgym.full-authority-imitation-report.v0", "capabilities": capabilities,
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "controls": {"ceiling": ceilings, "floor": floors}, "roundZero": round_zero_summary,
        "finalFitLosses": histories, "evaluations": evaluations, "imitationMinusCeiling": comparisons,
        "labelError": label_errors, "criticPrecondition": critic, "decisionRules": rules, "simulatorDecisions": steps}
    write_json(root / "report.json", report)
    manifest = {"format": "snowgym.full-authority-imitation-manifest.v0",
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
