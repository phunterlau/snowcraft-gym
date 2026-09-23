import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw as ert
from snowgym_training.options import enemy_relative_throw_random_red as s13
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import roster_baseline as rb


def test_configuration_and_budget():
    cfg = s13.configuration()
    assert cfg["format"] == "snowgym.m8-s13-random-red-eval-config.v0"
    assert cfg["evalArm"] == "random"
    assert cfg["sourceRun"] == "m8_s12_enemy_relative_throw_v0"
    assert cfg["autonomousQualificationEligible"] is False
    bound = s13.budget_bound(cfg)
    assert bound["perRun"] == cfg["pairedEvalWorlds"] * cfg["optionHorizon"]
    assert bound["total"] == bound["perRun"] * len(cfg["cohorts"]) * len(ert.DECODERS)
    assert bound["total"] <= cfg["budgetCap"]


def row(seed, success, lost):
    return {"seed": seed, "success": success, "unitsLostFraction": lost}


def test_paired_summary_pairs_by_seed_not_list_position():
    cfg = s13.configuration()
    # deliberately out-of-order and differently-ordered between new/old to prove pairing is by seed, not index
    new_rows = [row(3, True, 0.0), row(1, True, 0.33), row(2, False, 1.0)]
    old_rows = [row(1, False, 1.0), row(2, False, 1.0), row(3, False, 0.67)]
    summary = s13.paired_summary(new_rows, old_rows, cfg)
    # seed 1: new success True(1)-old False(0) = +1; seed 2: 0-0=0; seed 3: 1-0=+1 -> mean = 2/3
    assert summary["deltaSuccess"]["mean"] == pytest.approx(2 / 3)
    # seed1: 0.33-1.0=-0.67; seed2: 1.0-1.0=0.0; seed3: 0.0-0.67=-0.67 -> mean = -1.34/3
    assert summary["deltaUnitsLostFraction"]["mean"] == pytest.approx((-0.67 + 0.0 - 0.67) / 3, abs=1e-6)


def test_paired_summary_raises_on_mismatched_seed_sets():
    cfg = s13.configuration()
    new_rows = [row(1, True, 0.0), row(2, True, 0.0)]
    old_rows = [row(1, False, 1.0), row(3, False, 1.0)]
    with pytest.raises(ValueError):
        s13.paired_summary(new_rows, old_rows, cfg)


# -- live -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def test_load_checkpoint_recovers_the_right_model_class_for_each_decoder():
    cfg = s13.configuration()
    old = s13.load_checkpoint(1, "old", cfg)
    new = s13.load_checkpoint(1, "new", cfg)
    assert isinstance(old, ert.FullAuthorityPolicyV1)
    assert isinstance(new, ert.FullAuthorityPolicyV1EnemyThrow)
    assert old.training is False and new.training is False


def test_evaluate_one_against_the_real_cohort_1_checkpoints_on_random_red(tmp_path, client):
    cfg = {**s13.configuration(), "pairedEvalWorlds": 4, "cohorts": (1,)}
    rows_old, summary_old = s13.evaluate_one(client, cfg, "old", 1, tmp_path, account=lambda n: None)
    rows_new, summary_new = s13.evaluate_one(client, cfg, "new", 1, tmp_path, account=lambda n: None)
    assert len(rows_old) == len(rows_new) == 4
    assert {r["seed"] for r in rows_old} == {r["seed"] for r in rows_new} == set(ert.paired_eval_seeds(cfg))
    assert summary_old["episodes"] == summary_new["episodes"] == 4
    paired = s13.paired_summary(rows_new, rows_old, cfg)
    assert set(paired) == {"deltaSuccess", "deltaUnitsLostFraction"}
    assert -1.0 <= paired["deltaSuccess"]["mean"] <= 1.0


def test_tiny_end_to_end_run_evaluates_one_cohort_on_random_and_seals(tmp_path, client):
    cfg = {**s13.configuration(), "pairedEvalWorlds": 4, "cohorts": (1,)}
    report = s13.run(tmp_path / "run", cfg)
    assert set(report["cohorts"]) == {"1"}
    entry = report["cohorts"]["1"]
    assert set(entry) == {"old", "new", "paired"}
    assert entry["old"]["episodes"] == entry["new"]["episodes"] == 4
    assert set(entry["paired"]) == {"deltaSuccess", "deltaUnitsLostFraction"}
    assert report["withinBudgetBound"]
    assert dr.verify_sealed(tmp_path / "run")
    with pytest.raises(FileExistsError):
        s13.declare(tmp_path / "run", cfg)
