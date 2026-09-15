"""R1n-h: mixture-imitation curriculum, with a single-opponent control, testing
transfer to a held-out opponent.

Two conditions train side by side on fresh seed bands: a mixture condition (M,
every round split 64/64 between `RandomAgent` and `ScriptedAiAgent` easy) and a
single-opponent control (C, 100% `RandomAgent`) — isolating the effect of
opponent exposure during training from the effect of using different training
worlds than R1n-c. Both are evaluated on the same three paired splits (random,
scripted easy, scripted normal). The primary test is the mixture-minus-control
success gap on scripted normal, an opponent neither condition ever trains on.
See `reviews/m7b_r1n_h_declaration.md`. No PPO update runs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import death_rate_ppo as dr
from . import full_authority_imitation as fi
from . import full_authority_train_v1 as v1
from .full_authority_diagnostics import TRAINING, e3_digests_unchanged, uniform_floor, write_episodes
from .interventions import require_capabilities
from .opponent_transfer import scenario_override, world_paired_difference
from .reservoir import file_digest
from .supervised_probe import write_json

CONDITIONS = ("M", "C")
EVAL_ARMS = {"random": {"redController": "random"},
             "easy": {"redController": "scripted", "redDifficulty": "easy"},
             "normal": {"redController": "scripted", "redDifficulty": "normal"}}
MIXTURE_ARMS = {"random": EVAL_ARMS["random"], "easy": EVAL_ARMS["easy"]}


def configuration():
    return {**fi.configuration(), "format": "snowgym.r1n-h-mixture-imitation-config.v0",
        "roundSeedBaseByCondition": {"M": 400000, "C": 410000}, "roundSeedStride": 1000,
        "roundEpisodes": 128, "roundBlockWorlds": 64,
        "evalSeedBase": {"random": 420000, "easy": 421000, "normal": 422000}, "evaluationEpisodes": 100,
        "evaluationBlockWorlds": 50,
        "stochasticEvalSeedBase": 423000, "floorSeedBase": 424000,
        "criticTrainSeedBaseByCondition": {"M": 430000, "C": 440000},
        "criticHeldSeedBaseByCondition": {"M": 435000, "C": 445000}, "criticSeedBandStride": 1000,
        "warmStartTrainEpisodes": 256, "warmStartHeldOutEpisodes": 128, "warmStartEpochs": 10,
        "floorRng": 981003,
        "bootstrapSeed": 973001, "bootstrapSamples": 10000,
        "failThreshold": -.50,
        "throwRecallFlag": .5, "executionGapFlag": .20, "seedSpreadFlag": .30,
        "budgetCap": 2_200_000,
        "assistType": "teacher-imitation training; none at runtime",
        "assistVersion": "snowgym.r1n-h-mixture-imitation.v0", "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    round_zero = 2 * cfg["roundEpisodes"] * horizon
    learner_rounds = 2 * len(cfg["optimizerSeeds"]) * 4 * cfg["roundEpisodes"] * horizon
    deterministic_eval = 2 * len(EVAL_ARMS) * len(cfg["optimizerSeeds"]) * cfg["evaluationEpisodes"] * horizon
    stochastic_eval = 2 * len(cfg["optimizerSeeds"]) * cfg["evaluationEpisodes"] * horizon
    ceiling = len(EVAL_ARMS) * cfg["evaluationEpisodes"] * horizon
    floor = cfg["evaluationEpisodes"] * horizon
    critic = 2 * len(cfg["optimizerSeeds"]) * (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    total = round_zero + learner_rounds + deterministic_eval + stochastic_eval + ceiling + floor + critic
    return {"roundZero": round_zero, "learnerRounds": learner_rounds, "deterministicEval": deterministic_eval,
            "stochasticEval": stochastic_eval, "ceiling": ceiling, "floor": floor, "critic": critic, "total": total}


# -- Seed helpers (declaration §5) ---------------------------------------------------------


def round_seeds_by_arm(cfg, condition, round_index):
    """The declared round range, split in half by arm for M (first half random, second
    half easy) or given entirely to random for C — one declared contiguous range per
    round, not separate bases per arm (declaration §5)."""
    base = cfg["roundSeedBaseByCondition"][condition] + cfg["roundSeedStride"] * round_index
    seeds = list(range(base, base + cfg["roundEpisodes"]))
    if condition == "M":
        half = cfg["roundEpisodes"] // 2
        return {"random": seeds[:half], "easy": seeds[half:]}
    return {"random": seeds, "easy": []}


def eval_seeds(cfg, split_name):
    base = cfg["evalSeedBase"][split_name]
    return list(range(base, base + cfg["evaluationEpisodes"]))


def stochastic_eval_seeds(cfg):
    return list(range(cfg["stochasticEvalSeedBase"], cfg["stochasticEvalSeedBase"] + cfg["evaluationEpisodes"]))


def floor_seeds(cfg):
    return list(range(cfg["floorSeedBase"], cfg["floorSeedBase"] + cfg["evaluationEpisodes"]))


def critic_fold_seeds_by_arm(cfg, condition, rng_index, *, held_out):
    """The same range `v1.fold_seeds` would compute for this condition's critic bands,
    split in half by arm for M or given entirely to random for C (declaration §4/§5)."""
    base = cfg["criticHeldSeedBaseByCondition" if held_out else "criticTrainSeedBaseByCondition"][condition]
    base = base + cfg["criticSeedBandStride"] * rng_index
    count = cfg["warmStartHeldOutEpisodes"] if held_out else cfg["warmStartTrainEpisodes"]
    seeds = list(range(base, base + count))
    if condition == "M":
        half = count // 2
        return {"random": seeds[:half], "easy": seeds[half:]}
    return {"random": seeds, "easy": []}


def validate_critic_fold_sizes(cfg):
    """`v1.collect_fold` reuses one fixed-size wrapper across every chunk of a fold and
    requires each chunk's seed count to exactly equal `wrapper.batch_size` (confirmed by a
    live probe: `run_block` raises "block seeds must match the wrapper batch size"
    otherwise). Since `collect_fold_mixture` runs one arm's whole seed list as a single
    call, every arm's fold slice must be an exact multiple of `blockWorlds` — checked here,
    at declare time, rather than left to surface mid-run after a full seed's training
    already completed (declaration §10)."""
    for condition in CONDITIONS:
        for held_out in (False, True):
            for rng_index in range(len(cfg["optimizerSeeds"])):
                arms = critic_fold_seeds_by_arm(cfg, condition, rng_index, held_out=held_out)
                for arm, seeds in arms.items():
                    if seeds and len(seeds) % cfg["blockWorlds"] != 0:
                        raise ValueError(f"critic fold condition={condition} arm={arm} heldOut={held_out} "
                                         f"has {len(seeds)} seeds, not a multiple of blockWorlds={cfg['blockWorlds']}")


# -- Mixture collection (declaration §2) ---------------------------------------------------


def collect_mixture(client, cfg, model, seeds_by_arm, *, source, block_worlds, account,
                     deterministic=True, record=True):
    """`fi.collect` (unedited), called once per non-empty arm under `scenario_override`,
    then concatenated. For condition C, `seeds_by_arm["easy"]` is empty and this reduces to
    a single `fi.collect` call."""
    all_episodes, parts = [], []
    for arm, seeds in seeds_by_arm.items():
        if not seeds:
            continue
        with scenario_override(MIXTURE_ARMS[arm]):
            episodes, part = fi.collect(client, seeds, cfg, model=model, source=f"{source}-{arm}",
                                        block_worlds=block_worlds, account=account,
                                        deterministic=deterministic, record=record)
        all_episodes.extend(episodes)
        if record:
            parts.append(part)
    if not record:
        return all_episodes, None
    merged = {"observation": {key: torch.cat([p["observation"][key] for p in parts]) for key in fi.ACTOR_KEYS},
              "labels": {key: torch.cat([p["labels"][key] for p in parts]) for key in fi.LABEL_KEYS}}
    return all_episodes, merged


def collect_fold_mixture(wrapper, model, seeds_by_arm, cfg, *, source, fold):
    """`v1.collect_fold` (unedited), called once per non-empty arm under
    `scenario_override`, then merged via `v1.concatenate` (unedited) — but each arm's
    `episode` id column is re-offset first, since `v1.collect_fold` starts that column at 0
    independently for every call and a naive concatenation would collide two different
    arms' episode 0, 1, 2, ... into the same id, corrupting `bootstrap_predictive_r2`'s
    per-episode resampling (declaration §4)."""
    all_episodes, parts, decisions, offset = [], [], 0, 0
    for arm, seeds in seeds_by_arm.items():
        if not seeds:
            continue
        with scenario_override(MIXTURE_ARMS[arm]):
            episodes, part, used = v1.collect_fold(wrapper, model, seeds, cfg, source=f"{source}-{arm}", fold=fold)
        part = {**part, "episode": part["episode"] + offset}
        offset += len(episodes)
        all_episodes.extend(episodes)
        parts.append(part)
        decisions += used
    return all_episodes, v1.concatenate(parts), decisions


def warm_start_critic_mc_mixture(model, wrapper, cfg, rng_index, *, source, condition):
    """Controlled reimplementation of `v1.warm_start_critic_mc`'s body (not edited): both
    folds are collected per-arm via `collect_fold_mixture` instead of `v1.collect_fold`
    directly, then `predict_values`/`critic_metrics`/`critic_gate`/`gate_conditions`/
    `bootstrap_predictive_r2` all run unchanged on the merged data (declaration §4). For
    condition C this still runs through the mixture machinery (one arm, empty easy split)
    rather than calling `v1.warm_start_critic_mc` directly, so both conditions share
    exactly one code path here."""
    train_episodes, train, train_decisions = collect_fold_mixture(
        wrapper, model, critic_fold_seeds_by_arm(cfg, condition, rng_index, held_out=False), cfg,
        source=source, fold="train")
    held_episodes, held, held_decisions = collect_fold_mixture(
        wrapper, model, critic_fold_seeds_by_arm(cfg, condition, rng_index, held_out=True), cfg,
        source=source, fold="heldOut")
    before_train, before_held = v1.predict_values(model, train["observation"]), v1.predict_values(model, held["observation"])
    untrained = v1.critic_metrics(before_held, held["returns"], held["decision"], train["returns"], train["decision"],
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
    after_train, after_held = v1.predict_values(model, train["observation"]), v1.predict_values(model, held["observation"])
    metrics = v1.critic_metrics(after_held, held["returns"], held["decision"], train["returns"], train["decision"],
                                horizon=cfg["optionHorizon"], bins=cfg["timeBins"])
    report = {**metrics, "untrainedPredictiveR2": untrained["predictiveR2"],
        "predictiveR2Interval95": v1.bootstrap_predictive_r2(after_held, held["returns"], held["episode"],
            samples=cfg["bootstrapSamples"], seed=cfg["bootstrapSeed"]),
        "gatePassed": v1.critic_gate(metrics, cfg), "gateConditions": v1.gate_conditions(metrics, cfg),
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


# -- Per-condition training and evaluation (declaration §2, §3) ---------------------------


def train_condition(client, cfg, condition, root, account):
    condition_root = root / f"condition-{condition}"
    condition_root.mkdir(parents=True)
    seeds0 = round_seeds_by_arm(cfg, condition, 0)
    round_zero, round_zero_part = collect_mixture(client, cfg, None, seeds0, source=f"{condition}-round-0-teacher",
                                                   block_worlds=cfg["roundBlockWorlds"], account=account)
    write_episodes(condition_root / "round-0-teacher", round_zero)
    all_seeds0 = seeds0["random"] + seeds0["easy"]
    round_zero_summary = fi.part_summary(round_zero_part, 0, all_seeds0)
    write_json(condition_root / "round-0-teacher/dataset.json", round_zero_summary)

    evaluations, label_errors, critic, histories = {}, {}, {}, {}
    for seed_index, seed in enumerate(cfg["optimizerSeeds"]):
        directory = condition_root / f"seed-{seed}"
        directory.mkdir()
        torch.manual_seed(seed)
        model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
            target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
        optimizer = torch.optim.Adam(fi.imitation_parameters(model), lr=cfg["imitationLearningRate"])
        data, rounds, history = fi.Aggregate(), [round_zero_summary], {}
        data.add(round_zero_part)
        for fit_index in range(cfg["fits"]):
            history[str(fit_index)] = fi.fit(model, optimizer, data, cfg, seed=seed + 1000 * fit_index)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "fit": fit_index,
                        "rows": len(data)}, directory / f"fit-{fit_index}.pt")
            if fit_index == cfg["fits"] - 1:
                break
            seeds_k = round_seeds_by_arm(cfg, condition, fit_index + 1)
            found, part = collect_mixture(client, cfg, model, seeds_k,
                source=f"{condition}-seed-{seed}-round-{fit_index + 1}",
                block_worlds=cfg["roundBlockWorlds"], account=account)
            write_episodes(directory / f"round-{fit_index + 1}", found)
            rounds.append(fi.part_summary(part, fit_index + 1, seeds_k["random"] + seeds_k["easy"]))
            data.add(part)
            del part
        write_json(directory / "fit-history.json", history)
        write_json(directory / "dataset.json", {"rounds": rounds, "aggregateRows": len(data)})
        del data
        histories[str(seed)] = {k: v[-1] for k, v in history.items()}

        model.eval()
        evaluations[str(seed)] = {}
        for split_name in ("random", "easy", "normal"):
            wants_labels = split_name == "normal"
            with scenario_override(EVAL_ARMS[split_name]):
                found, part = fi.collect(client, eval_seeds(cfg, split_name), cfg, model=model,
                    source=f"{condition}-seed-{seed}-{split_name}-det",
                    block_worlds=cfg["evaluationBlockWorlds"], account=account, record=wants_labels)
            write_episodes(directory / f"eval-{split_name}-deterministic", found)
            evaluations[str(seed)][split_name] = {**fi.evaluation_summary(found, cfg), "bySeed": fi.success_by_seed(found)}
            if wants_labels:
                label_errors[str(seed)] = fi.label_error(model, part, cfg)
                write_json(directory / "label-error.json", label_errors[str(seed)])
                del part
        with scenario_override(EVAL_ARMS["random"]):
            found, _ = fi.collect(client, stochastic_eval_seeds(cfg), cfg, model=model,
                source=f"{condition}-seed-{seed}-random-sto", block_worlds=cfg["evaluationBlockWorlds"],
                account=account, deterministic=False, record=False)
        write_episodes(directory / "eval-random-stochastic", found)
        evaluations[str(seed)]["random-stochastic"] = fi.evaluation_summary(found, cfg)

        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        report, arrays, episodes, _, _ = warm_start_critic_mc_mixture(model, wrapper, cfg, seed_index,
            source=f"{condition}-seed-{seed}-critic", condition=condition)
        account(report["simulatorDecisions"])
        report["behaviorOutcomes"] = fi.evaluation_summary(episodes, cfg)
        critic[str(seed)] = report
        write_json(directory / "critic-warm-start.json", report)
        np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
        torch.save({"model": model.state_dict(), "note": "final mixture-imitation policy with warm-started critic"},
                   directory / "critic-policy.pt")
        write_episodes(directory / "critic-episodes", episodes)
        del model, optimizer, arrays, episodes

    write_json(condition_root / "condition-report.json", {"evaluations": evaluations, "labelError": label_errors,
        "criticPrecondition": critic, "finalFitLosses": histories})
    return {"evaluations": evaluations, "labelError": label_errors, "criticPrecondition": critic}


# -- Decision rules (declaration §6; authorizes nothing) -----------------------------------


def condition_outcomes_by_seed(root, condition, split_name):
    outcomes = {}
    for seed in [97101, 97102, 97103]:
        path = root / f"condition-{condition}" / f"seed-{seed}" / f"eval-{split_name}-deterministic" / "episodes.jsonl"
        outcomes[seed] = dr.outcomes(path)
    return outcomes


def condition_seed_averaged_by_world(outcomes_by_seed, metric):
    worlds = sorted(next(iter(outcomes_by_seed.values())))
    return {w: {metric: float(np.mean([outcomes_by_seed[s][w][metric] for s in outcomes_by_seed]))} for w in worlds}


def precondition_gap(root, cfg):
    """gap(M, eval-easy): M's seed-averaged success minus the teacher's, world-paired
    (declaration §6). Must not be more negative than `failThreshold` before the primary is
    interpreted. Reads the ceiling straight from its own `episodes.jsonl` via `dr.outcomes`
    (unedited) — not from `controls/summary.json`'s `bySeed`, whose world keys come back as
    JSON-object-key strings after a round trip through `write_json`/`json.loads`, while
    `dr.outcomes`'s keys stay int (parsed from a JSON *value*, not a key); mixing the two
    would make `world_paired_difference`'s world-set comparison fail every time."""
    m_outcomes = condition_outcomes_by_seed(root, "M", "easy")
    teacher_outcomes = dr.outcomes(root / "controls" / "ceiling-easy" / "episodes.jsonl")
    teacher_by_world = {w: {"success": o["success"]} for w, o in teacher_outcomes.items()}
    return world_paired_difference(m_outcomes, teacher_by_world, cfg, "success")


def primary_comparison(root, cfg):
    """M minus C on eval-normal, directly — the shared teacher term cancels out of
    gap(M) - gap(C), so this reuses `world_paired_difference` unchanged with C's
    seed-averaged outcomes standing in for the "reference" argument (declaration §6)."""
    m_outcomes = condition_outcomes_by_seed(root, "M", "normal")
    c_outcomes = condition_outcomes_by_seed(root, "C", "normal")
    c_by_world = condition_seed_averaged_by_world(c_outcomes, "success")
    return world_paired_difference(m_outcomes, c_by_world, cfg, "success")


def primary_outcome(precondition, primary, cfg):
    if precondition["mean"] <= cfg["failThreshold"]:
        return "precondition-failed"
    lower, upper = primary["interval95"]
    if lower > 0:
        return "transfers"
    if upper < 0:
        return "regresses"
    return "no-detected-transfer"


RECOMMENDATION = {
    "precondition-failed": "Mixture training did not reach a usable floor against the opponent it trained on "
        "(scripted easy); the held-out comparison is reported but not interpretable as evidence about transfer. "
        "Diagnose the mixture round/fit schedule before rerunning.",
    "transfers": "Scripted-easy exposure improves transfer to scripted-normal, an opponent neither condition "
        "trained on. The mixture curriculum is worth extending (more opponents, or M8's MARL scaffolding) rather "
        "than abandoned.",
    "no-detected-transfer": "No evidence the mixture generalizes past its own training opponents at this scale. "
        "Check the secondary label-error comparison before concluding the curriculum failed outright — a partial "
        "aim-mechanism improvement with no success-rate movement would still be informative.",
    "regresses": "Mixture training makes the held-out opponent worse than the single-opponent control. This needs "
        "its own explanation (e.g. interference between the two training opponents) before any further curriculum "
        "work.",
}


def secondary_label_error_comparison(root, cfg):
    """M vs C's label_error on eval-normal, pooled across the 3 optimizer seeds — the
    direct follow-up to R1n-g's finding (declaration §6)."""
    out = {}
    for condition in CONDITIONS:
        errors = json.loads((root / f"condition-{condition}" / "condition-report.json").read_text(encoding="utf-8"))["labelError"]
        aim = [errors[str(s)]["throwAimHeadingErrorDegrees"] for s in cfg["optimizerSeeds"]
               if errors[str(s)]["throwAimHeadingErrorDegrees"] is not None]
        recall = [errors[str(s)]["perType"]["throw"]["recall"] for s in cfg["optimizerSeeds"]
                  if errors[str(s)]["perType"]["throw"]["recall"] is not None]
        out[condition] = {"meanThrowAimHeadingErrorDegrees": float(np.mean(aim)) if aim else None,
                          "meanThrowRecall": float(np.mean(recall)) if recall else None}
    return out


def flags(root, cfg):
    out = {}
    for condition in CONDITIONS:
        report = json.loads((root / f"condition-{condition}" / "condition-report.json").read_text(encoding="utf-8"))
        recalls = [report["labelError"][str(s)]["perType"]["throw"]["recall"] for s in cfg["optimizerSeeds"]]
        gaps = [report["evaluations"][str(s)]["random"]["successFraction"]
                - report["evaluations"][str(s)]["random-stochastic"]["successFraction"] for s in cfg["optimizerSeeds"]]
        spreads = {split: float(max(report["evaluations"][str(s)][split]["successFraction"] for s in cfg["optimizerSeeds"])
                                - min(report["evaluations"][str(s)][split]["successFraction"] for s in cfg["optimizerSeeds"]))
                  for split in ("random", "easy", "normal")}
        out[condition] = {"criticHealthy": all(report["criticPrecondition"][str(s)]["gatePassed"] for s in cfg["optimizerSeeds"]),
            "throwCollapse": any(r is None or r < cfg["throwRecallFlag"] for r in recalls),
            "executionModeGap": any(g > cfg["executionGapFlag"] for g in gaps),
            "seedInstability": any(s > cfg["seedSpreadFlag"] for s in spreads.values())}
    return out


# -- Declaration, run, aggregate (declaration §8, §9) ---------------------------------------


def declare(root, cfg):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    pinned = e3_digests_unchanged()
    if not all(entry["match"] for entry in pinned.values()):
        raise RuntimeError("E3 source digests no longer match the archived run")
    here = Path(__file__).resolve()
    imitation_path = here.parent / "full_authority_imitation.py"
    train_path = here.parent / "full_authority_train_v1.py"
    imitation_digest, train_digest = file_digest(imitation_path), file_digest(train_path)
    g_declaration = json.loads((TRAINING / "runs/m7b_engage_r1n_g_v0/declaration.json").read_text(encoding="utf-8"))
    f_declaration = json.loads((TRAINING / "runs/m7b_engage_r1n_f_v0/declaration.json").read_text(encoding="utf-8"))
    if imitation_digest != g_declaration["imitationImplementationDigest"]:
        raise RuntimeError("full_authority_imitation.py digest no longer matches R1n-g's declaration.json")
    if train_digest != f_declaration["trainImplementationDigest"]:
        raise RuntimeError("full_authority_train_v1.py digest no longer matches R1n-f's declaration.json")
    validate_critic_fold_sizes(cfg)
    root.mkdir(parents=True)
    write_json(root / "declaration.json", {"config": cfg, "gitCommit": resolve_git_commit(),
        "budgetBound": budget_bound(cfg),
        "declarationDigest": file_digest(TRAINING / "reviews/m7b_r1n_h_declaration.md"),
        "implementationDigest": file_digest(here),
        "imitationImplementationDigest": imitation_digest, "trainImplementationDigest": train_digest,
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"], "assistVersion": cfg["assistVersion"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def collect_controls(client, cfg, root, account):
    ceilings, floors = {}, {}
    for split_name in ("random", "easy", "normal"):
        with scenario_override(EVAL_ARMS[split_name]):
            ceiling, _ = fi.collect(client, eval_seeds(cfg, split_name), cfg, model=None,
                source=f"ceiling-{split_name}", block_worlds=cfg["evaluationBlockWorlds"],
                account=account, record=False)
        write_episodes(root / "controls" / f"ceiling-{split_name}", ceiling)
        ceilings[split_name] = {**fi.evaluation_summary(ceiling, cfg), "bySeed": fi.success_by_seed(ceiling)}
    with scenario_override(EVAL_ARMS["normal"]):
        seeds = floor_seeds(cfg)
        floor_episodes = []
        for start in range(0, len(seeds), cfg["evaluationBlockWorlds"]):
            block = seeds[start:start + cfg["evaluationBlockWorlds"]]
            wrapper = v1.make_wrapper(client, len(block), cfg["gamma"])
            found, _, used = v1.run_block(wrapper, block, cfg, choose=uniform_floor(cfg["floorRng"] + start),
                                          source="floor-normal")
            account(used)
            floor_episodes.extend(found)
    write_episodes(root / "controls" / "floor-normal", floor_episodes)
    floors["normal"] = fi.evaluation_summary(floor_episodes, cfg)
    write_json(root / "controls" / "summary.json", {"ceiling": ceilings, "floor": floors})
    return ceilings, floors


def aggregate(root, cfg):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    precondition = precondition_gap(root, cfg)
    primary = primary_comparison(root, cfg)
    outcome = primary_outcome(precondition, primary, cfg)
    report = {"format": "snowgym.r1n-h-mixture-imitation-report.v0",
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"],
        "precondition": precondition, "primary": primary, "outcome": outcome,
        "recommendation": RECOMMENDATION[outcome],
        "secondaryLabelError": secondary_label_error_comparison(root, cfg),
        "flags": flags(root, cfg)}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.r1n-h-mixture-imitation-manifest.v0")
    return report


# -- Budget ledger spanning stages run as separate processes ------------------------------


def load_ledger(root):
    path = Path(root) / "budget-ledger.json"
    return json.loads(path.read_text(encoding="utf-8"))["simulatorDecisions"] if path.exists() else 0


def make_account(root, cfg):
    """`write_json` (unedited) refuses to overwrite, by design, for sealed artifacts — but
    this ledger is a mutable running counter updated on every collected block, across
    however many separate process invocations a staged run uses, so it is written directly
    rather than through that guard. It is picked up like any other file when `aggregate`
    seals the run."""
    path = Path(root) / "budget-ledger.json"
    steps = load_ledger(root)

    def account(count):
        nonlocal steps
        steps += count
        path.write_text(json.dumps({"simulatorDecisions": steps}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if steps > cfg["budgetCap"]:
            raise ValueError("R1n-h budget exceeded")

    return account


def run_controls(output):
    cfg = configuration()
    root = Path(output)
    account = make_account(root, cfg)
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        collect_controls(client, cfg, root, account)


def run_condition(output, condition):
    cfg = configuration()
    root = Path(output)
    account = make_account(root, cfg)
    with SnowGymBatchClient() as client:
        train_condition(client, cfg, condition, root, account)


def run(output):
    cfg = configuration()
    root = Path(output)
    declare(root, cfg)
    account = make_account(root, cfg)
    with SnowGymBatchClient() as client:
        require_capabilities(client)
        collect_controls(client, cfg, root, account)
        for condition in CONDITIONS:
            train_condition(client, cfg, condition, root, account)
    return aggregate(root, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "declare", "controls", "condition", "aggregate"], default="all")
    parser.add_argument("--condition", choices=list(CONDITIONS))
    arguments = parser.parse_args()
    if arguments.stage == "all":
        run(arguments.output)
    elif arguments.stage == "declare":
        declare(arguments.output, configuration())
    elif arguments.stage == "controls":
        run_controls(arguments.output)
    elif arguments.stage == "condition":
        if arguments.condition is None:
            raise SystemExit("--condition is required for --stage condition")
        run_condition(arguments.output, arguments.condition)
    else:
        aggregate(arguments.output, configuration())
