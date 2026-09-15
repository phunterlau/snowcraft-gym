import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options import mixture_imitation as mi


def tiny(**overrides):
    return {**mi.configuration(), "optionHorizon": 8,
            "roundEpisodes": 8, "fits": 2, "stepsPerFit": 20, "imitationMinibatch": 32,
            "evaluationEpisodes": 4,
            "blockWorlds": 4, "warmStartTrainEpisodes": 16, "warmStartHeldOutEpisodes": 8, "warmStartEpochs": 2,
            "bootstrapSamples": 50, "budgetCap": 400_000, **overrides}


# -- budget -----------------------------------------------------------------------------


def test_budget_bound_matches_declared_figures():
    cfg = mi.configuration()
    bound = mi.budget_bound(cfg)
    assert bound == {"roundZero": 51_200, "learnerRounds": 614_400, "deterministicEval": 360_000,
                     "stochasticEval": 120_000, "ceiling": 60_000, "floor": 20_000, "critic": 460_800,
                     "total": 1_686_400}
    assert cfg["budgetCap"] == 2_200_000


# -- seed bands ---------------------------------------------------------------------------


def all_seed_ranges(cfg):
    ranges = []
    for condition in mi.CONDITIONS:
        for round_index in range(5):
            arms = mi.round_seeds_by_arm(cfg, condition, round_index)
            for arm, seeds in arms.items():
                if seeds:
                    ranges.append((f"round-{condition}-{round_index}-{arm}", min(seeds), max(seeds)))
    for split_name in ("random", "easy", "normal"):
        seeds = mi.eval_seeds(cfg, split_name)
        ranges.append((f"eval-{split_name}", min(seeds), max(seeds)))
    ranges.append(("stochastic", min(mi.stochastic_eval_seeds(cfg)), max(mi.stochastic_eval_seeds(cfg))))
    ranges.append(("floor", min(mi.floor_seeds(cfg)), max(mi.floor_seeds(cfg))))
    for condition in mi.CONDITIONS:
        for rng_index in range(3):
            for held_out in (False, True):
                arms = mi.critic_fold_seeds_by_arm(cfg, condition, rng_index, held_out=held_out)
                for arm, seeds in arms.items():
                    if seeds:
                        ranges.append((f"critic-{condition}-{rng_index}-{held_out}-{arm}", min(seeds), max(seeds)))
    return ranges


def test_seed_bands_do_not_overlap():
    ranges = all_seed_ranges(mi.configuration())
    bad = [(n1, n2) for i, (n1, l1, h1) in enumerate(ranges) for n2, l2, h2 in ranges[i + 1:]
           if l1 <= h2 and l2 <= h1]
    assert bad == []


def test_seed_bands_stay_within_the_reserved_block_and_clear_of_r1n_b():
    ranges = all_seed_ranges(mi.configuration())
    lows, highs = [r[1] for r in ranges], [r[2] for r in ranges]
    assert min(lows) >= 400_000 and max(highs) <= 449_999
    assert not any(low <= 630_119 and high >= 630_000 for _, low, high in ranges)


def test_round_seeds_by_arm_splits_evenly_for_mixture_and_gives_control_all_random():
    cfg = mi.configuration()
    m = mi.round_seeds_by_arm(cfg, "M", 0)
    assert len(m["random"]) == len(m["easy"]) == 64
    assert m["random"][-1] + 1 == m["easy"][0]  # contiguous overall span
    c = mi.round_seeds_by_arm(cfg, "C", 0)
    assert len(c["random"]) == 128 and c["easy"] == []


def test_validate_critic_fold_sizes_passes_for_the_real_configuration():
    mi.validate_critic_fold_sizes(mi.configuration())  # must not raise


def test_validate_critic_fold_sizes_passes_for_tiny():
    mi.validate_critic_fold_sizes(tiny())  # must not raise


def test_validate_critic_fold_sizes_rejects_a_mismatched_block_worlds():
    cfg = {**mi.configuration(), "blockWorlds": 63}  # 256/2=128 is not a multiple of 63
    with pytest.raises(ValueError, match="not a multiple of blockWorlds"):
        mi.validate_critic_fold_sizes(cfg)


# -- collect_mixture: plumbing, no live simulator ------------------------------------------


def test_collect_mixture_splits_by_arm_and_concatenates(monkeypatch):
    calls = []

    def fake_collect(client, seeds, cfg, *, model, source, block_worlds, account, deterministic=True, record=True):
        calls.append({"seeds": list(seeds), "redController": v1.scenario()["redController"],
                      "redDifficulty": v1.scenario().get("redDifficulty")})
        account(len(seeds))
        episodes = [{"seed": s} for s in seeds]
        if not record:
            return episodes, None
        n = len(seeds)
        part = {"observation": {key: torch.arange(n) for key in fi.ACTOR_KEYS},
                "labels": {key: torch.arange(n) for key in fi.LABEL_KEYS}}
        return episodes, part

    monkeypatch.setattr(mi.fi, "collect", fake_collect)
    steps = []
    episodes, part = mi.collect_mixture(None, {}, None, {"random": [1, 2, 3], "easy": [4, 5]},
        source="test", block_worlds=8, account=lambda n: steps.append(n))
    assert [e["seed"] for e in episodes] == [1, 2, 3, 4, 5]
    assert len(calls) == 2
    assert calls[0]["redController"] == "random"
    assert calls[1]["redController"] == "scripted" and calls[1]["redDifficulty"] == "easy"
    assert sum(steps) == 5
    for key in fi.ACTOR_KEYS:
        assert part["observation"][key].shape[0] == 5


def test_collect_mixture_control_condition_is_a_single_arm_call(monkeypatch):
    calls = []

    def fake_collect(client, seeds, cfg, *, model, source, block_worlds, account, deterministic=True, record=True):
        calls.append(list(seeds))
        account(len(seeds))
        return [{"seed": s} for s in seeds], None

    monkeypatch.setattr(mi.fi, "collect", fake_collect)
    episodes, part = mi.collect_mixture(None, {}, None, {"random": [1, 2, 3], "easy": []},
        source="test", block_worlds=8, account=lambda n: None, record=False)
    assert len(calls) == 1
    assert part is None
    assert [e["seed"] for e in episodes] == [1, 2, 3]


# -- collect_fold_mixture: the episode-id offset fix ---------------------------------------


def test_collect_fold_mixture_reoffsets_episode_ids_to_avoid_collision(monkeypatch):
    def fake_collect_fold(wrapper, model, seeds, cfg, *, source, fold):
        n = len(seeds)
        episodes = [{"seed": s} for s in seeds]
        part = {"observation": {"x": torch.arange(n)}, "episode": torch.arange(n),
                "decision": torch.zeros(n), "seed": torch.tensor(seeds), "returns": torch.zeros(n)}
        return episodes, part, n * 3

    monkeypatch.setattr(mi.v1, "collect_fold", fake_collect_fold)
    episodes, merged, decisions = mi.collect_fold_mixture(None, None, {"random": [10, 11, 12], "easy": [20, 21]},
        {}, source="test", fold="train")
    assert len(episodes) == 5
    assert decisions == 3 * 3 + 2 * 3
    # Naive concatenation would give [0,1,2,0,1] (both arms' collect_fold independently start
    # their own episode column at 0) -- the offset fix must produce five distinct ids.
    assert merged["episode"].tolist() == [0, 1, 2, 3, 4]
    assert len(set(merged["episode"].tolist())) == 5


def test_collect_fold_mixture_single_arm_matches_the_unmerged_call(monkeypatch):
    def fake_collect_fold(wrapper, model, seeds, cfg, *, source, fold):
        n = len(seeds)
        part = {"observation": {"x": torch.arange(n)}, "episode": torch.arange(n),
                "decision": torch.zeros(n), "seed": torch.tensor(seeds), "returns": torch.zeros(n)}
        return [{"seed": s} for s in seeds], part, n

    monkeypatch.setattr(mi.v1, "collect_fold", fake_collect_fold)
    episodes, merged, decisions = mi.collect_fold_mixture(None, None, {"random": [1, 2], "easy": []},
        {}, source="test", fold="train")
    assert merged["episode"].tolist() == [0, 1]
    assert decisions == 2


# -- decision rules -------------------------------------------------------------------------


def test_primary_outcome_precondition_failed():
    cfg = mi.configuration()
    precondition = {"mean": -0.6, "interval95": [-0.7, -0.5]}
    primary = {"mean": 0.1, "interval95": [-0.05, 0.25]}
    assert mi.primary_outcome(precondition, primary, cfg) == "precondition-failed"


def test_primary_outcome_transfers():
    cfg = mi.configuration()
    precondition = {"mean": -0.1, "interval95": [-0.2, 0.0]}
    primary = {"mean": 0.15, "interval95": [0.05, 0.25]}
    assert mi.primary_outcome(precondition, primary, cfg) == "transfers"


def test_primary_outcome_regresses():
    cfg = mi.configuration()
    precondition = {"mean": -0.1, "interval95": [-0.2, 0.0]}
    primary = {"mean": -0.15, "interval95": [-0.25, -0.05]}
    assert mi.primary_outcome(precondition, primary, cfg) == "regresses"


def test_primary_outcome_no_detected_transfer():
    cfg = mi.configuration()
    precondition = {"mean": -0.1, "interval95": [-0.2, 0.0]}
    primary = {"mean": 0.02, "interval95": [-0.05, 0.09]}
    assert mi.primary_outcome(precondition, primary, cfg) == "no-detected-transfer"


def test_recommendation_covers_every_outcome():
    assert set(mi.RECOMMENDATION) == {"precondition-failed", "transfers", "no-detected-transfer", "regresses"}


# -- live: the declared regression guard for warm_start_critic_mc_mixture -----------------


def test_warm_start_critic_mc_mixture_reproduces_the_original_at_an_all_random_ratio():
    cfg = tiny()
    v1_cfg = {**cfg, "trainSeedBase": cfg["criticTrainSeedBaseByCondition"]["C"],
              "heldOutSeedBase": cfg["criticHeldSeedBaseByCondition"]["C"],
              "seedBandStride": cfg["criticSeedBandStride"]}

    torch.manual_seed(555001)
    model_a = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])
    torch.manual_seed(555001)
    model_b = FullAuthorityPolicyV1(destination=cfg["destination"], local_radius=cfg["localRadius"],
        target_world_sigma=cfg["targetWorldSigma"], initial_power_log_std=cfg["initialPowerLogStd"])

    with SnowGymBatchClient() as client:
        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        torch.manual_seed(555002)
        report_a, arrays_a, *_ = mi.warm_start_critic_mc_mixture(model_a, wrapper, cfg, 0,
            source="regression-a", condition="C")
        torch.manual_seed(555002)
        report_b, arrays_b, *_ = v1.warm_start_critic_mc(model_b, wrapper, v1_cfg, 0, source="regression-b")

    for key in ("predictiveR2", "explainedVariance", "trainRows", "heldOutRows", "trainEpisodes",
                "heldOutEpisodes", "gatePassed", "simulatorDecisions"):
        assert report_a[key] == pytest.approx(report_b[key]) if isinstance(report_a[key], float) else \
            report_a[key] == report_b[key]
    for key in arrays_a:
        assert (arrays_a[key] == arrays_b[key]).all(), key


# -- live: tiny end-to-end -------------------------------------------------------------------


def test_tiny_end_to_end_declare_controls_conditions_and_aggregate(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    mi.declare(root, cfg)
    with pytest.raises(FileExistsError):
        mi.declare(root, cfg)

    with SnowGymBatchClient() as client:
        account = mi.make_account(root, cfg)
        mi.require_capabilities(client)
        mi.collect_controls(client, cfg, root, account)
        for condition in mi.CONDITIONS:
            mi.train_condition(client, cfg, condition, root, account)

    for condition in mi.CONDITIONS:
        for seed in cfg["optimizerSeeds"]:
            directory = root / f"condition-{condition}" / f"seed-{seed}"
            assert (directory / "fit-history.json").exists()
            assert (directory / "label-error.json").exists()
            assert (directory / "critic-warm-start.json").exists()
            for split_name in ("random", "easy", "normal"):
                assert (directory / f"eval-{split_name}-deterministic" / "episodes.jsonl").exists()

    report = mi.aggregate(root, cfg)
    assert report["outcome"] in mi.RECOMMENDATION
    assert "primary" in report and "precondition" in report
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "report.json" in manifest["artifacts"]
    assert "condition-M/seed-97101/critic-warm-start.json" in manifest["artifacts"]

    with pytest.raises(FileExistsError):
        mi.aggregate(root, cfg)

    # Tamper detection reuses death_rate_ppo's seal/verify_sealed, unchanged.
    path = root / "condition-C" / "seed-97101" / "eval-normal-deterministic" / "episodes.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        dr.verify_sealed(root)


def test_budget_ledger_persists_across_separate_process_style_calls(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    root.mkdir()
    account1 = mi.make_account(root, cfg)
    account1(1000)
    assert mi.load_ledger(root) == 1000
    account2 = mi.make_account(root, cfg)  # a fresh closure, as a new process invocation would create
    account2(500)
    assert mi.load_ledger(root) == 1500
    with pytest.raises(ValueError, match="budget exceeded"):
        account2(cfg["budgetCap"])
