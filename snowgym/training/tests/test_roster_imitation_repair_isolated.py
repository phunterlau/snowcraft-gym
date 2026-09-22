import json

import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_imitation_repair as rp
from snowgym_training.options import roster_imitation_repair_isolated as ri


def tiny(**overrides):
    return {**ri.configuration(), "optionHorizon": 8, "roundEpisodes": 8, "fits": 2, "stepsPerFit": 6,
            "imitationMinibatch": 16, "evaluationEpisodes": 4, "blockWorlds": 4, "roundBlockWorlds": 4,
            "evaluationBlockWorlds": 4, "warmStartTrainEpisodes": 16, "warmStartHeldOutEpisodes": 8,
            "warmStartEpochs": 2, "bootstrapSamples": 50, "pairedEvalWorlds": 4, "optimizerSeeds": (97101,),
            "trainingRngs": (97101,), "budgetCap": 400_000, **overrides}


def test_configuration_matches_s10_exactly_except_the_aim_weight():
    s10, s11 = rp.configuration(), ri.configuration()
    assert s11["aimWeight"] == 1.0 and s10["aimWeight"] == 5.0
    differing = {k for k in s10 if s10.get(k) != s11.get(k)} | {k for k in s11 if k not in s10}
    assert differing == {"format", "aimWeight"}


def test_classify_uses_the_declared_thresholds():
    assert ri.classify(0.00, 0.51) == "loss-weight-driven"
    assert ri.classify(0.20, 0.51) == "loss-weight-driven"   # boundary is inclusive on the low side
    assert ri.classify(0.35, 0.51) == "data-draw-driven"
    assert ri.classify(0.51, 0.51) == "data-draw-driven"
    assert ri.classify(0.25, 0.51) == "mixed-or-inconsistent"
    assert ri.classify(0.60, 0.51) == "mixed-or-inconsistent"   # exceeds S10 itself


def test_score_computes_the_three_pairwise_effects_and_their_sum(monkeypatch):
    cfg = ri.configuration()
    seeds_out = {"97101": {"pairedEval": {"normal": {"success": 0.10}, "easy": {"success": 0.80}},
                           "critic": {"gatePassed": True}}}
    monkeypatch.setattr(ri, "s10_success", lambda seed, arm, cfg: {"normal": 0.51, "easy": 0.94}[arm])
    monkeypatch.setattr(ri, "s5_success", lambda seed, arm, cfg: {"normal": 0.00, "easy": 0.74}[arm])
    result = ri.score(seeds_out, cfg)
    cell = result["byArm"]["normal"]["97101"]
    assert cell == {"s11": 0.10, "s10": 0.51, "s5": 0.00, "s11MinusS5_dataEffect": pytest.approx(0.10),
                    "s10MinusS11_lossEffect": pytest.approx(0.41), "s10MinusS5_combined": pytest.approx(0.51)}
    check = result["decompositionCheck"]
    assert check["sumOfParts"] == pytest.approx(check["combined"])
    assert result["seed97101Classification"] == "loss-weight-driven"   # 0.10 <= 0.20


def test_score_reports_easy_against_both_s10_and_s5_separately(monkeypatch):
    cfg = ri.configuration()
    # 97101 (S10 easy 0.94, S5 easy 0.74): S11 comes back near S5's original, below S10's -> restored is False-ish,
    # backNearS5 True since it recovers close to where S5 started.
    seeds_out = {"97101": {"pairedEval": {"normal": {"success": 0.10}, "easy": {"success": 0.72}},
                           "critic": {"gatePassed": False}}}
    monkeypatch.setattr(ri, "s10_success", lambda seed, arm, cfg: {"normal": 0.51, "easy": 0.94}[arm])
    monkeypatch.setattr(ri, "s5_success", lambda seed, arm, cfg: {"normal": 0.00, "easy": 0.74}[arm])
    result = ri.score(seeds_out, cfg)
    easy = result["easyComparedToS10AndS5"]["97101"]
    assert easy == {"s11": 0.72, "s10": 0.94, "s5": 0.74, "restoredVsS10": False, "backNearS5": True}
    assert result["criticGatePassed"]["97101"] is False


def test_budget_bound_matches_s10s():
    assert ri.budget_bound(ri.configuration()) == rp.budget_bound(rp.configuration())


def test_s10_rows_and_s5_rows_read_the_real_archives_for_the_first_few_worlds():
    rows = ri.s10_rows(97101, "normal", 5)
    assert len(rows) == 5 and all("success" in r for r in rows)
    rows5 = rp.s5_rows("normal", 97101, 5)
    assert len(rows5) == 5


# -- live -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def test_tiny_end_to_end_run_trains_at_aim_weight_one_and_seals(tmp_path, client):
    cfg = tiny()
    report = ri.run(tmp_path / "run", cfg)
    assert report["withinBudgetBound"]
    entry = report["learners"]["97101"]
    assert entry["pairedEval"]["normal"]["episodes"] == 4
    score = report["score"]
    assert score["seed97101Classification"] in {"loss-weight-driven", "data-draw-driven", "mixed-or-inconsistent"}
    assert "decompositionCheck" in score
    assert dr.verify_sealed(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ri.declare(tmp_path / "run", cfg)
