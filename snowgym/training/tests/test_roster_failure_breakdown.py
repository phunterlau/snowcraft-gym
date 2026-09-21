import pytest

from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_failure_breakdown as fb


def row(success=False, alive=0, damage=0.0, timed_out=False, decision=100, min_distance=5.0, first_hit=None,
        distance_at_hit=None):
    return {"success": success, "blueAliveCount": alive, "assignedUnits": 3, "timedOut": timed_out,
            "finalTargetDamage": damage, "finalDecision": decision, "minDistance": min_distance,
            "firstHitDecision": first_hit, "distanceAtFirstHit": distance_at_hit}


def test_damage_bands_partition_the_range_at_the_declared_edges():
    assert [fb.damage_band(d) for d in (0, 1, 99.9, 100, 199.9, 200, 239.9, 240, 300)] == [
        "none", "lt100", "lt100", "100-199", "100-199", "200-239", "200-239", "ge240", "ge240"]


def test_outcomes_are_a_partition_in_priority_order():
    assert fb.outcome(row(success=True, alive=3, damage=250)) == "success"
    assert fb.outcome(row(alive=0, damage=250, timed_out=True)) == "wipe"
    assert fb.outcome(row(alive=2, damage=150, timed_out=True)) == "timeout-ge100"
    assert fb.outcome(row(alive=2, damage=50, timed_out=True)) == "timeout-lt100"
    assert fb.outcome(row(alive=2, damage=50, timed_out=False)) == "other"


def test_summarize_cell_reports_contact_bands_composition_and_exchange():
    rows = [row(success=True, alive=3, damage=250), row(alive=0, damage=120), row(alive=0, damage=0),
            row(alive=2, damage=210, timed_out=True, first_hit=40, distance_at_hit=6.0)]
    summary = fb.summarize_cell(rows)
    assert summary["contact"] == 0.75 and summary["failures"] == 3
    assert sum(summary["outcomes"].values()) == pytest.approx(1.0)
    assert summary["outcomes"]["wipe"] == 0.5 and summary["outcomes"]["timeout-ge100"] == 0.25
    assert summary["failureDamageBands"]["none"] == pytest.approx(1 / 3)
    assert summary["failureDamageBands"]["200-239"] == pytest.approx(1 / 3)
    assert summary["unitsLost"] == 0 + 3 + 3 + 1
    assert summary["damagePerUnitLost"] == pytest.approx((250 + 120 + 0 + 210) / 7)
    assert summary["medianDistanceAtFirstHit"] == 6.0
    with pytest.raises(ValueError):
        fb.summarize_cell([])


def test_a_cell_with_no_losses_has_no_exchange_ratio_and_no_failures_has_no_bands():
    summary = fb.summarize_cell([row(success=True, alive=3, damage=250)])
    assert summary["damagePerUnitLost"] is None and summary["failureDamageBands"]["none"] is None


def test_real_archives_align_by_seed_and_the_outcome_partition_sums_to_one():
    for arm in fb.ARMS:
        learner, teacher = fb.load_paired(arm, "97101")
        assert len(learner) == len(teacher) == 400
        assert sum(fb.summarize_cell(learner)["outcomes"].values()) == pytest.approx(1.0)


def test_run_seals_a_zero_decision_archive(tmp_path):
    root = tmp_path / "breakdown"
    report = fb.run(root)
    assert report["simulatorDecisions"] == 0 and set(report["learners"]) == set(fb.SEEDS)
    assert dr.verify_sealed(root)
    with pytest.raises(FileExistsError):
        fb.declare(root)
