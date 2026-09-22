import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options import roster_imitation_repair as rp
from snowgym_training.options.opponent_transfer import scenario_override
from snowgym_training.options.plans import teacher_option_plan


def tiny(**overrides):
    return {**rp.configuration(), "optionHorizon": 8, "roundEpisodes": 8, "fits": 2, "stepsPerFit": 6,
            "imitationMinibatch": 16, "evaluationEpisodes": 4, "blockWorlds": 4, "roundBlockWorlds": 4,
            "evaluationBlockWorlds": 4, "warmStartTrainEpisodes": 16, "warmStartHeldOutEpisodes": 8,
            "warmStartEpochs": 2, "bootstrapSamples": 50, "pairedEvalWorlds": 4, "optimizerSeeds": (97101,),
            "trainingRngs": (97101,), "budgetCap": 400_000, **overrides}


def make_part(client, seeds, cfg):
    from snowgym_training.options import full_authority_train_v1 as v1
    plan, spec = teacher_option_plan("engage")
    wrapper = v1.make_wrapper(client, len(seeds), cfg["gamma"])
    with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
        _, part, _ = v1.collect_fold(wrapper, None, seeds, cfg, source="test", fold="t") \
            if False else fi.collect(client, seeds, cfg, model=None, source="test", block_worlds=len(seeds),
                                     account=lambda n: None)
    return part


def test_weighted_loss_is_bit_identical_to_unweighted_at_aim_weight_one(client=None):
    torch.manual_seed(0)
    model = rp.FullAuthorityPolicyV1(destination="global")
    cfg = tiny(aimWeight=1.0)
    with SnowGymBatchClient() as c:
        seeds = list(range(9401, 9401 + cfg["evaluationEpisodes"]))
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER, **{}}):
            found, part = fi.collect(c, seeds, cfg, model=model, source="t", block_worlds=len(seeds),
                                     account=lambda n: None, record=True)
    plain = fi.imitation_loss(model, part["observation"], part["labels"], cfg)
    weighted = rp.weighted_imitation_loss(model, part["observation"], part["labels"], cfg)
    torch.testing.assert_close(plain["total"], weighted["total"])
    assert float(weighted["unweightedTotal"]) == pytest.approx(float(plain["total"]))


def test_weighted_loss_amplifies_only_the_aim_term():
    torch.manual_seed(0)
    model = rp.FullAuthorityPolicyV1(destination="global")
    with SnowGymBatchClient() as c:
        cfg = tiny()
        seeds = list(range(9405, 9409))
        with scenario_override({"blueUnits": rb.ROSTER, "redUnits": rb.ROSTER}):
            found, part = fi.collect(c, seeds, cfg, model=model, source="t", block_worlds=len(seeds),
                                     account=lambda n: None, record=True)
    plain = fi.imitation_loss(model, part["observation"], part["labels"], cfg)
    weighted = rp.weighted_imitation_loss(model, {**part["observation"]}, part["labels"], {**cfg, "aimWeight": 5.0})
    for key in ("type", "moveHeading", "moveEndpoint", "power"):
        torch.testing.assert_close(plain[key], weighted[key])
    expected = plain["type"] + plain["moveHeading"] + plain["moveEndpoint"] + 5.0 * plain["throwAim"] + plain["power"]
    torch.testing.assert_close(weighted["total"], expected)


def test_budget_bound_matches_the_declaration():
    bound = rp.budget_bound(rp.configuration())
    assert bound == {"roundZero": 25600, "learnerRounds": 307200, "development": 60000, "critic": 230400,
                     "pairedEval": 120000, "total": 743200}
    assert bound["total"] <= rp.configuration()["budgetCap"]


def test_seed_bands_are_disjoint_and_unused_elsewhere_in_the_repository_json():
    from snowgym_training.options.full_authority_diagnostics import TRAINING
    cfg = rp.configuration()
    ranges = [(low, high) for _, low, high in rp.seed_bands(cfg)]
    ordered = sorted(ranges)
    for (_, high), (low, _) in zip(ordered, ordered[1:]):
        assert high < low

    def integers(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from integers(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from integers(item)

    collisions = []
    for path in TRAINING.parents[1].rglob("*.json"):
        # S11 deliberately reuses S10's exact bands (its declaration section 0), so it must be excluded too.
        if any(part in {".git", "node_modules", ".venv", "dist", "m8_s10_throw_aim_repair_v0",
                        "m8_s11_throw_aim_isolation_v0"} for part in path.parts) \
                or path.stat().st_size > 5_000_000:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if any(low <= n <= high for n in integers(data) for low, high in ranges):
            collisions.append(str(path))
    assert collisions == []


def test_score_applies_the_declared_gain_and_regression_rules():
    cfg = rp.configuration()

    def entry(normal_success, easy_success, critic_ok=True):
        return {"pairedEval": {"normal": {"success": normal_success}, "easy": {"success": easy_success}},
                "critic": {"gatePassed": critic_ok}}

    seeds_out = {"97101": entry(0.05, 0.60), "97102": entry(0.20, 0.85), "97103": entry(0.20, 0.85)}

    class FakeRow(list):
        pass

    def fake_s5_rows(arm, seed, worlds):
        # 97101 baseline: normal 0.00, easy 0.73; 97102/97103: normal 0.00, easy 0.89/0.91 (S5's own archived values)
        table = {97101: {"normal": 0.00, "easy": 0.73}, 97102: {"normal": 0.00, "easy": 0.89},
                 97103: {"normal": 0.00, "easy": 0.91}}
        fraction = table[seed][arm]
        wins = round(fraction * worlds)

        def row(success):
            return {"success": success, "assignedUnits": 3, "unitsLostFraction": 0.0,
                    "blueAliveCount": 3 if success else 0, "timedOut": not success, "totalActions": 100,
                    "rejectedActions": 0, "finalDecision": 100}

        return [row(True) for _ in range(wins)] + [row(False) for _ in range(worlds - wins)]

    original = rp.s5_rows
    rp.s5_rows = fake_s5_rows
    try:
        result = rp.score(seeds_out, cfg)
    finally:
        rp.s5_rows = original
    assert result["normalSuccessGainVsS5"] == pytest.approx({"97101": 0.05, "97102": 0.20, "97103": 0.20})
    assert result["P1_aimSeedsGainAtLeastThreshold"] is True   # both 97102 and 97103 gain >= 0.10
    assert result["P2_noEasyRegression"] is True
    assert result["P3_criticGatePasses"] is True
    assert result["P4_descriptive_otherSeedsGainNoMoreThanAimSeeds"] is True   # 97101's 0.05 <= min(0.20, 0.20)

    seeds_out["97102"] = entry(0.05, 0.85)   # weaken one aim seed below threshold
    rp.s5_rows = fake_s5_rows
    try:
        result = rp.score(seeds_out, cfg)
    finally:
        rp.s5_rows = original
    assert result["P1_aimSeedsGainAtLeastThreshold"] is False

    seeds_out["97101"] = entry(0.50, 0.20)   # 97101 regresses easy hard and beats the aim seeds
    rp.s5_rows = fake_s5_rows
    try:
        result = rp.score(seeds_out, cfg)
    finally:
        rp.s5_rows = original
    assert result["easyDoesNotRegress"]["97101"] is False and result["P2_noEasyRegression"] is False


# -- live -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def test_tiny_end_to_end_run_trains_one_seed_measures_and_seals(tmp_path, client):
    cfg = tiny()
    root = tmp_path / "run"
    report = rp.run(root, cfg)
    assert set(report["learners"]) == {"97101"}
    entry = report["learners"]["97101"]
    assert set(entry["pairedEval"]) == {"normal", "easy"}
    assert entry["pairedEval"]["normal"]["episodes"] == 4
    assert "throwAimHeadingErrorDegrees" in entry["labelError"]
    assert set(entry["critic"]) == {"predictiveR2", "timeOnlyR2", "untrainedPredictiveR2", "gatePassed"}
    assert (root / "condition-M" / "seed-97101" / "fit-1.pt").exists()
    assert report["withinBudgetBound"]
    score = report["score"]
    assert "P1_aimSeedsGainAtLeastThreshold" in score and score["P1_aimSeedsGainAtLeastThreshold"] is None  # no aim seed in this tiny run
    assert dr.verify_sealed(root)
    with pytest.raises(FileExistsError):
        rp.declare(root, cfg)
