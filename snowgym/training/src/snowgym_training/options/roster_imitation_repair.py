"""M8-S10: autonomous repair attempt 1, upweighting the throw-aim loss term (`reviews/m8_s10_declaration.md`).

`full_authority_imitation.imitation_loss` sums five already-isolated per-type mean losses with an implicit
coefficient of 1 each. This module changes only the coefficient on the throw-aim term; every loss component
computation is reused unchanged from `full_authority_imitation`. `mixture_imitation.train_condition` hardcodes a
call to the unweighted loss, so the training orchestration is reimplemented here, reusing every other piece
(`collect_mixture`, `round_seeds_by_arm`, `warm_start_critic_mc_mixture`, `Aggregate`, `part_summary`) unchanged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
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

CONDITION = "M"
S4_RUN, S5_RUN, S7_RUN, S8_RUN, S9_RUN = (
    "m8_s4_roster_baseline_v0", "m8_s5_roster_imitation_v0", "m8_s7_trace_diagnosis_v0",
    "m8_s8_channel_intervention_v0", "m8_s9_target_split_v0")
EVAL_ARMS = ("normal", "easy")


def configuration():
    return {**mi.configuration(), "format": "snowgym.m8-s10-throw-aim-repair-config.v0", "roster": rb.ROSTER,
        "aimWeight": 5.0,
        "roundSeedBaseByCondition": {"M": 4200000}, "roundSeedStride": 1000,
        "criticTrainSeedBaseByCondition": {"M": 4210000}, "criticHeldSeedBaseByCondition": {"M": 4215000},
        "developmentNormalSeedBase": 4220000, "pairedEvalSeedBase": rb.configuration()["evaluationSeedBase"],
        "pairedEvalWorlds": 100, "pairedEvalArms": EVAL_ARMS, "gainThreshold": 0.10, "easyRegressionMargin": 0.15,
        "budgetCap": 950_000,
        "assistType": "none; ordinary autonomous imitation training", "assistVersion": "snowgym.m8-s10.v0",
        "autonomousQualificationEligible": False}


def budget_bound(cfg):
    horizon = cfg["optionHorizon"]
    seeds = len(cfg["optimizerSeeds"])
    round_zero = cfg["roundEpisodes"] * horizon
    learner_rounds = seeds * (cfg["fits"] - 1) * cfg["roundEpisodes"] * horizon
    development = seeds * cfg["evaluationEpisodes"] * horizon
    critic = seeds * (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * horizon
    paired = seeds * len(cfg["pairedEvalArms"]) * cfg["pairedEvalWorlds"] * horizon
    return {"roundZero": round_zero, "learnerRounds": learner_rounds, "development": development,
            "critic": critic, "pairedEval": paired,
            "total": round_zero + learner_rounds + development + critic + paired}


def seed_bands(cfg):
    bands = []
    for round_index in range(cfg["fits"]):
        low = cfg["roundSeedBaseByCondition"][CONDITION] + cfg["roundSeedStride"] * round_index
        bands.append((f"round-{round_index}", low, low + cfg["roundEpisodes"] - 1))
    for index in range(len(cfg["optimizerSeeds"])):
        stride = cfg["criticSeedBandStride"] * index
        low = cfg["criticTrainSeedBaseByCondition"][CONDITION] + stride
        bands.append((f"critic-train-{index}", low, low + cfg["warmStartTrainEpisodes"] - 1))
        low = cfg["criticHeldSeedBaseByCondition"][CONDITION] + stride
        bands.append((f"critic-held-{index}", low, low + cfg["warmStartHeldOutEpisodes"] - 1))
    bands.append(("development-normal", cfg["developmentNormalSeedBase"],
                  cfg["developmentNormalSeedBase"] + cfg["evaluationEpisodes"] - 1))
    return bands


# -- The one change: a loss with a different linear combination (declaration section 0) ------------


def weighted_imitation_loss(model, observation, labels, cfg):
    """`fi.imitation_loss`, unedited, called for its per-type components; only `total`'s
    coefficients differ. At `aimWeight = 1` this is bit-identical to `fi.imitation_loss`
    (test_roster_imitation_repair.py's regression guard)."""
    losses = fi.imitation_loss(model, observation, labels, cfg)
    weighted_total = (losses["type"] + losses["moveHeading"] + losses["moveEndpoint"]
                      + cfg["aimWeight"] * losses["throwAim"] + losses["power"])
    return {**losses, "total": weighted_total, "unweightedTotal": losses["total"]}


def fit_weighted(model, optimizer, data, cfg, *, seed):
    """`fi.fit`'s loop, reimplemented because it hardcodes a call to the unweighted loss; every other
    piece (parameters, minibatching, clipping) is `fi.fit`'s own, called here, not copied by value."""
    generator = torch.Generator().manual_seed(seed)
    parameters = fi.imitation_parameters(model)
    history, step = [], 0
    while step < cfg["stepsPerFit"]:
        for indices in torch.randperm(len(data), generator=generator).split(cfg["imitationMinibatch"]):
            if step >= cfg["stepsPerFit"]:
                break
            observation, labels = data.batch(indices)
            losses = weighted_imitation_loss(model, observation, labels, cfg)
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


# -- Training orchestration (mirrors mi.train_condition's body; see declaration section 1) ----------


def round_seeds(cfg, round_index):
    base = cfg["roundSeedBaseByCondition"][CONDITION] + cfg["roundSeedStride"] * round_index
    seeds = list(range(base, base + cfg["roundEpisodes"]))
    half = cfg["roundEpisodes"] // 2
    return {"random": seeds[:half], "easy": seeds[half:]}


def development_normal_seeds(cfg):
    return list(range(cfg["developmentNormalSeedBase"], cfg["developmentNormalSeedBase"] + cfg["evaluationEpisodes"]))


def paired_eval_seeds(cfg):
    return list(range(cfg["pairedEvalSeedBase"], cfg["pairedEvalSeedBase"] + cfg["pairedEvalWorlds"]))


def train_and_measure(client, cfg, root, account):
    condition_root = Path(root) / f"condition-{CONDITION}"
    condition_root.mkdir(parents=True)
    seeds0 = round_seeds(cfg, 0)
    round_zero, round_zero_part = mi.collect_mixture(client, cfg, None, seeds0, source="round-0-teacher",
                                                      block_worlds=cfg["roundBlockWorlds"], account=account)
    write_episodes(condition_root / "round-0-teacher", round_zero)
    all_seeds0 = seeds0["random"] + seeds0["easy"]
    round_zero_summary = fi.part_summary(round_zero_part, 0, all_seeds0)
    write_json(condition_root / "round-0-teacher/dataset.json", round_zero_summary)

    seeds_out = {}
    for seed in cfg["optimizerSeeds"]:
        directory = condition_root / f"seed-{seed}"
        directory.mkdir()
        torch.manual_seed(seed)
        model = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
            target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
        optimizer = torch.optim.Adam(fi.imitation_parameters(model), lr=cfg["imitationLearningRate"])
        data, rounds, history = fi.Aggregate(), [round_zero_summary], {}
        data.add(round_zero_part)
        for fit_index in range(cfg["fits"]):
            history[str(fit_index)] = fit_weighted(model, optimizer, data, cfg, seed=seed + 1000 * fit_index)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "fit": fit_index,
                        "rows": len(data)}, directory / f"fit-{fit_index}.pt")
            if fit_index == cfg["fits"] - 1:
                break
            seeds_k = round_seeds(cfg, fit_index + 1)
            found, part = mi.collect_mixture(client, cfg, model, seeds_k, source=f"seed-{seed}-round-{fit_index + 1}",
                                             block_worlds=cfg["roundBlockWorlds"], account=account)
            write_episodes(directory / f"round-{fit_index + 1}", found)
            rounds.append(fi.part_summary(part, fit_index + 1, seeds_k["random"] + seeds_k["easy"]))
            data.add(part)
            del part
        write_json(directory / "fit-history.json", history)
        write_json(directory / "dataset.json", {"rounds": rounds, "aggregateRows": len(data)})
        del data
        model.eval()

        with scenario_override(mi.EVAL_ARMS["normal"]):
            found, part = fi.collect(client, development_normal_seeds(cfg), cfg, model=model,
                source=f"seed-{seed}-normal-det", block_worlds=cfg["evaluationBlockWorlds"], account=account, record=True)
        write_episodes(directory / "eval-normal-deterministic", found)
        development = {"normal": {**fi.evaluation_summary(found, cfg), "bySeed": fi.success_by_seed(found)}}
        label_errors = fi.label_error(model, part, cfg)
        write_json(directory / "label-error.json", label_errors)
        del part

        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        report, arrays, episodes, _, _ = mi.warm_start_critic_mc_mixture(
            model, wrapper, cfg, list(cfg["optimizerSeeds"]).index(seed), source=f"seed-{seed}-critic", condition=CONDITION)
        account(report["simulatorDecisions"])
        report["behaviorOutcomes"] = fi.evaluation_summary(episodes, cfg)
        write_json(directory / "critic-warm-start.json", report)
        np.savez_compressed(directory / "critic-warm-start-arrays.npz", **arrays)
        torch.save({"model": model.state_dict(), "note": "S10 throw-aim-weighted imitation policy with warm-started critic"},
                   directory / "critic-policy.pt")
        write_episodes(directory / "critic-episodes", episodes)

        paired = {}
        for arm in cfg["pairedEvalArms"]:
            rows = rb.collect_cell(client, paired_eval_seeds(cfg), cfg, arm, model=model,
                                   mode="deterministic", account=account)
            rb.write_rows(directory / "paired-eval" / arm, rows)
            paired[arm] = rb.summarize(rows, cfg)
        seeds_out[str(seed)] = {"development": development, "labelError": label_errors, "critic":
            {k: report[k] for k in ("predictiveR2", "timeOnlyR2", "untrainedPredictiveR2", "gatePassed")},
            "pairedEval": paired}
        del model, optimizer, arrays, episodes
    write_json(condition_root / "condition-report.json", seeds_out)
    return seeds_out


# -- Comparison against S5's own archived paired evaluation (declaration section 2) ------------------


def s5_rows(arm, seed, worlds):
    path = TRAINING / "runs" / S5_RUN / "paired-eval" / f"seed-{seed}" / arm / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:worlds]


def score(seeds_out, cfg):
    gains, easy_ok, critic_ok = {}, {}, {}
    for seed_str, entry in seeds_out.items():
        seed = int(seed_str)
        baseline_normal = rb.summarize(s5_rows("normal", seed, cfg["pairedEvalWorlds"]), cfg)["success"]
        baseline_easy = rb.summarize(s5_rows("easy", seed, cfg["pairedEvalWorlds"]), cfg)["success"]
        gains[seed_str] = entry["pairedEval"]["normal"]["success"] - baseline_normal
        easy_ok[seed_str] = entry["pairedEval"]["easy"]["success"] >= baseline_easy - cfg["easyRegressionMargin"]
        critic_ok[seed_str] = entry["critic"]["gatePassed"]
    aim_seeds = [s for s in gains if int(s) in (97102, 97103)]
    other_seeds = [s for s in gains if int(s) not in (97102, 97103)]
    p1 = all(gains[s] >= cfg["gainThreshold"] for s in aim_seeds) if aim_seeds else None
    p4 = (None if not aim_seeds or not other_seeds else
          all(gains[s] <= min(gains[a] for a in aim_seeds) for s in other_seeds))
    return {"normalSuccessGainVsS5": gains, "easyDoesNotRegress": easy_ok, "criticGatePassed": critic_ok,
            "P1_aimSeedsGainAtLeastThreshold": p1, "P2_noEasyRegression": all(easy_ok.values()),
            "P3_criticGatePasses": all(critic_ok.values()), "P4_descriptive_otherSeedsGainNoMoreThanAimSeeds": p4}


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
        "s4Manifest": dr.verify_sealed(TRAINING / "runs" / S4_RUN), "s5Manifest": dr.verify_sealed(TRAINING / "runs" / S5_RUN),
        "s7Manifest": dr.verify_sealed(TRAINING / "runs" / S7_RUN), "s8Manifest": dr.verify_sealed(TRAINING / "runs" / S8_RUN),
        "s9Manifest": dr.verify_sealed(TRAINING / "runs" / S9_RUN),
        "declarationDigest": file_digest(TRAINING / "reviews/m8_s10_declaration.md"), "implementationDigest": file_digest(here),
        "sourceDigests": {name: file_digest(here.parent / f"{name}.py") for name in
                          ("full_authority_imitation", "mixture_imitation", "full_authority_train_v1", "roster_baseline",
                           "roster_imitation")},
        "pinnedE3Digests": pinned, "assistType": cfg["assistType"],
        "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]})


def aggregate(root, cfg, seeds_out, decisions):
    root = Path(root)
    if (root / "report.json").exists():
        raise FileExistsError("run already aggregated")
    bound = budget_bound(cfg)
    report = {"format": "snowgym.m8-s10-throw-aim-repair-report.v0", "learners": seeds_out, "score": score(seeds_out, cfg),
        "simulatorDecisions": decisions, "withinBudgetBound": decisions <= bound["total"],
        "assistType": cfg["assistType"], "autonomousQualificationEligible": cfg["autonomousQualificationEligible"]}
    write_json(root / "report.json", report)
    dr.seal(root, "snowgym.m8-s10-manifest.v0")
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
            raise ValueError("M8-S10 budget exceeded")

    with SnowGymBatchClient() as client:
        require_capabilities(client)
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            seeds_out = train_and_measure(client, cfg, root, account)
    return aggregate(root, cfg, seeds_out, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
