import numpy as np
import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_NOOP, ACTION_THROW
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options import roster_intervention as rv
from snowgym_training.options import roster_trace as rt
from snowgym_training.options.full_authority_diagnostics import TRAINING


def arrays(types, targets, powers):
    return {"action_type": np.array([types], dtype=np.int64), "target": np.array([targets], dtype=np.float32),
            "power": np.array([powers], dtype=np.float32)}


LIVING = np.array([[True, True, True]])
# learner: unit0 THROW, unit1 THROW, unit2 MOVE. teacher: unit0 THROW, unit1 NOOP, unit2 THROW.
LEARNER = arrays([ACTION_THROW, ACTION_THROW, ACTION_MOVE], [[.1, .1], [.2, .2], [.3, .3]], [.5, .5, 0.])
TEACHER = arrays([ACTION_THROW, ACTION_NOOP, ACTION_THROW], [[.9, .9], [0, 0], [.7, .7]], [.9, 0., .8])
HEAD_TARGET = np.full((1, 3, 2), .55, dtype=np.float32)
HEAD_POWER = np.full((1, 3), .66, dtype=np.float32)


def apply(rule, living=LIVING):
    return rv.substitute(rule, LEARNER, TEACHER, HEAD_TARGET, HEAD_POWER, living)


def test_none_changes_nothing_and_never_mutates_the_inputs():
    before = {k: v.copy() for k, v in LEARNER.items()}
    action, counts = apply("none")
    assert all(np.array_equal(action[k], LEARNER[k]) for k in LEARNER) and counts["changed"] == 0
    assert all(np.array_equal(LEARNER[k], before[k]) for k in LEARNER)


def test_aim_replaces_only_the_target_where_both_throw():
    action, counts = apply("aim")
    assert np.allclose(action["target"][0, 0], [.9, .9])          # both throw -> teacher's aim
    assert np.allclose(action["target"][0, 1], [.2, .2])          # teacher not throwing -> unchanged
    assert np.array_equal(action["action_type"], LEARNER["action_type"]) and np.array_equal(action["power"], LEARNER["power"])
    assert counts["comparableThrows"] == 1 and counts["learnerThrows"] == 2 and counts["changed"] == 1


def test_power_replaces_only_the_power_where_both_throw_and_aim_plus_power_does_both():
    action, _ = apply("power")
    assert action["power"][0, 0] == pytest.approx(.9) and action["power"][0, 1] == pytest.approx(.5)
    assert np.array_equal(action["target"], LEARNER["target"])
    both, _ = apply("aim+power")
    assert np.allclose(both["target"][0, 0], [.9, .9]) and both["power"][0, 0] == pytest.approx(.9)


def test_timing_makes_the_unit_throw_exactly_when_the_teacher_does():
    action, _ = apply("timing")
    # unit1: learner throws, teacher does not -> adopts the teacher's whole action (NOOP)
    assert action["action_type"][0, 1] == ACTION_NOOP
    # unit2: teacher throws, learner moves -> throws with the learner's own throw-head target and power
    assert action["action_type"][0, 2] == ACTION_THROW
    assert np.allclose(action["target"][0, 2], .55) and action["power"][0, 2] == pytest.approx(.66)
    # unit0: both throw -> learner's own aim and power are kept in timing-only
    assert np.allclose(action["target"][0, 0], [.1, .1]) and action["power"][0, 0] == pytest.approx(.5)


def test_throw_all_adds_the_teachers_aim_and_power_on_every_teacher_throw():
    action, _ = apply("throw-all")
    assert np.allclose(action["target"][0, 0], [.9, .9]) and action["power"][0, 0] == pytest.approx(.9)
    assert action["action_type"][0, 2] == ACTION_THROW and np.allclose(action["target"][0, 2], [.7, .7])
    assert action["power"][0, 2] == pytest.approx(.8)
    assert action["action_type"][0, 1] == ACTION_NOOP


def test_move_replaces_only_the_move_target_where_both_move():
    move_teacher = arrays([ACTION_MOVE, ACTION_NOOP, ACTION_MOVE], [[.4, .4], [0, 0], [-.6, .6]], [0, 0, 0])
    action, counts = rv.substitute("move", LEARNER, move_teacher, HEAD_TARGET, HEAD_POWER, LIVING)
    assert np.allclose(action["target"][0, 2], [-.6, .6]) and np.array_equal(action["action_type"], LEARNER["action_type"])
    assert counts["comparableMoves"] == 1
    assert np.allclose(action["target"][0, 0], LEARNER["target"][0, 0])


def test_all_takes_the_teachers_action_for_every_living_unit_and_dead_units_are_never_touched():
    action, _ = apply("all")
    assert all(np.array_equal(action[k], TEACHER[k]) for k in TEACHER)
    dead = np.array([[True, False, True]])
    action, counts = apply("throw-all", living=dead)
    assert action["action_type"][0, 1] == ACTION_THROW and np.array_equal(action["target"][0, 1], LEARNER["target"][0, 1])
    assert counts["unitDecisions"] == 2


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        apply("everything")


def test_recovery_is_the_fraction_of_the_yield_gap_closed_and_undefined_without_a_gap():
    assert rv.recovery(10.0, 4.0, 16.0) == pytest.approx(0.5)
    assert rv.recovery(4.0, 4.0, 16.0) == 0.0 and rv.recovery(20.0, 4.0, 16.0) > 1.0
    assert rv.recovery(1.0, 4.0, 4.0) is None and rv.recovery(None, 4.0, 16.0) is None


def cell(recovery, success):
    return {"yieldRecovery": recovery, "summary": {"success": success}}


def test_prediction_scoring_needs_all_three_seeds_and_reports_undefined():
    seed = lambda: {"none": cell(0.0, 0.0), "aim": cell(0.6, .1), "power": cell(0.1, 0), "timing": cell(0.1, 0),
                    "throw-all": cell(0.9, 0.4), "move": cell(0.0, 0.0)}
    report = {"learners": {"97101": seed(), "97102": seed(), "97103": seed()}}
    assert all(v["holds"] for v in rv.score_predictions(report).values())
    report["learners"]["97102"]["aim"] = cell(0.4, .1)
    report["learners"]["97103"]["throw-all"] = cell(None, .4)
    scored = rv.score_predictions(report)
    assert scored["P1"]["holds"] is False and scored["P4"]["holds"] is None and scored["P5"]["holds"] is True


def test_budget_bound_matches_the_declaration():
    bound = rv.budget_bound(rv.configuration())
    assert (bound["arms"], bound["allControls"], bound["total"]) == (420000, 40000, 460000)
    assert bound["total"] <= rv.configuration()["budgetCap"]


# -- live controls -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def cfg50():
    return {**rv.configuration(), "interventionWorlds": 50, "bootstrapSamples": 50}


def test_none_arm_with_teacher_queries_reproduces_the_s5_rows_exactly(client):
    """The hard gate: querying the teacher every decision must not perturb the simulation."""
    cfg = cfg50()
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / rv.S5_RUN, 97101)
    rows, traces, totals = rv.collect_cell(client, cfg, model, "none", account=lambda n: None)
    gate = rt.reproduction_gate(rows, rt.archived_rows("learner", "normal", 97101, 50))
    assert gate["passed"], gate["mismatches"][:3]
    assert totals["changed"] == 0 and totals["learnerThrows"] > 0 and totals["teacherThrows"] > 0


def test_an_intervention_arm_changes_actions_and_the_outcome_is_recorded(client):
    cfg = cfg50()
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / rv.S5_RUN, 97101)
    rows, traces, totals = rv.collect_cell(client, cfg, model, "throw-all", account=lambda n: None)
    assert totals["changed"] > 0 and len(traces) == 50
    assert all(r["assignedUnits"] == 3 and 0 <= r["unitsLostFraction"] <= 1 for r in rows)
    assert rv.coverage(totals)["throwCoverage"] is not None


def test_all_arm_versus_the_s4_teacher_rows_is_recorded(client):
    """Not a hard gate (declaration section 1): records whether executing decoded teacher tensors matches the native step."""
    cfg = cfg50()
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / rv.S5_RUN, 97101)
    rows, _, _ = rv.collect_cell(client, cfg, model, "all", account=lambda n: None)
    gate = rt.reproduction_gate(rows, rt.archived_rows("teacher", "normal", None, 50))
    print("ALL-ARM vs S4 teacher:", gate["passed"], gate["mismatches"][:3])
    assert isinstance(gate["passed"], bool)


def test_tiny_end_to_end_run_has_the_controls_reference_and_seals(tmp_path):
    cfg = {**rv.configuration(), "interventionWorlds": 2, "blockWorlds": 2, "learnerSeeds": (97101,),
           "rules": ("none", "aim"), "bootstrapSamples": 50}
    report = rv.run(tmp_path / "run", cfg)
    assert set(report["allControls"]) == {"normal", "random"} and report["withinBudgetBound"]
    assert report["tensorPathTeacherYield"] == report["allControls"]["normal"]["traceMetrics"]["blueYield"]["mean"]
    assert set(report["learners"]["97101"]) == {"none", "aim"} and "P1" in report["predictions"]
    assert report["learners"]["97101"]["none"]["coverage"]["changed"] == 0
    assert dr.verify_sealed(tmp_path / "run")


def test_every_rule_can_be_collected_including_names_that_are_not_valid_plan_ids(client):
    """The first real run crashed at `aim+power`: the rule name became part of the plan id, and '+' is invalid there."""
    cfg = {**rv.configuration(), "interventionWorlds": 2, "blockWorlds": 2, "bootstrapSamples": 50}
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / rv.S5_RUN, 97101)
    for rule in (*rv.RULES, rv.ALL_ARM):
        rows, traces, totals = rv.collect_cell(client, cfg, model, rule, account=lambda n: None)
        assert len(rows) == 2 and len(traces) == 2, rule
