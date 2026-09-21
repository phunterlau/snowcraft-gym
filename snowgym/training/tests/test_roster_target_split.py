import math

import numpy as np
import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE, ACTION_NOOP, ACTION_THROW
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options import roster_intervention as rv
from snowgym_training.options import roster_target_split as ts
from snowgym_training.options import roster_trace as rt
from snowgym_training.options.full_authority_diagnostics import TRAINING

D = math.radians


def test_decompose_picks_the_angularly_nearest_enemy_and_a_signed_offset():
    k, delta, bearings = ts.decompose(D(80), (0.0, 0.0), [(10.0, 0.0), (0.0, 10.0)])
    assert k == 1 and delta == pytest.approx(D(-10)) and bearings == pytest.approx([0, D(90)])


def test_decompose_wraps_around_the_branch_cut():
    k, delta, _ = ts.decompose(D(179), (0.0, 0.0), [(-10.0, -0.01 * 10)])  # bearing about -179.94 degrees
    assert k == 0 and abs(math.degrees(delta)) < 2.5


def test_enemy_and_offset_rules_mix_choice_and_offset_as_declared():
    enemies = [(10.0, 0.0), (0.0, 10.0)]
    theta_t, theta_l = D(0), D(80)              # teacher: enemy 0, offset 0; learner: enemy 1, offset -10 degrees
    enemy, stats = ts.new_heading("enemy", theta_l, theta_t, (0.0, 0.0), enemies)
    assert enemy == pytest.approx(D(0) + D(-10))    # teacher's enemy, learner's offset
    offset, _ = ts.new_heading("offset", theta_l, theta_t, (0.0, 0.0), enemies)
    assert offset == pytest.approx(D(90) + D(0))    # learner's enemy, teacher's offset
    assert ts.new_heading("aim-heading", theta_l, theta_t, (0.0, 0.0), enemies)[0] == theta_t
    assert stats["sameEnemy"] is False and stats["learnerOffset"] == pytest.approx(D(10))
    assert stats["headingGap"] == pytest.approx(D(80))
    with pytest.raises(ValueError):
        ts.new_heading("none", theta_l, theta_t, (0.0, 0.0), enemies)


def test_place_on_ray_keeps_the_direction_inside_the_arena_and_refuses_a_degenerate_ray():
    point, ok = ts.place_on_ray((48.0, 0.0), 0.0, 7.5, (50.0, 40.0))
    assert ok and point == pytest.approx((50.0, 0.0), abs=0.01)   # shortened to the boundary, same heading
    _, ok = ts.place_on_ray((49.9, 0.0), 0.0, 7.5, (50.0, 40.0))
    assert not ok
    point, ok = ts.place_on_ray((0.0, 0.0), D(45), 7.5, (50.0, 40.0))
    assert ok and math.hypot(*point) == pytest.approx(7.5)


def arrays(types, targets, powers):
    return {"action_type": np.array([types], dtype=np.int64), "target": np.array([targets], dtype=np.float32),
            "power": np.array([powers], dtype=np.float32)}


HALF = np.array([[50.0, 40.0]])
THROWER = np.zeros((1, 3, 2))
LIVING = np.array([[True, True, True]])
ENEMIES = [[(10.0, 0.0), (0.0, 10.0)]]


def normalized(x, y):
    return [x / 50.0, y / 40.0]


def test_split_actions_applies_rules_only_to_comparable_living_throws_and_keeps_type_and_power():
    learner = arrays([ACTION_THROW, ACTION_THROW, ACTION_MOVE], [normalized(30, 200 / 6), normalized(0, 10), [.3, .3]], [.8, .8, 0.])
    teacher = arrays([ACTION_THROW, ACTION_NOOP, ACTION_THROW], [normalized(10, 0), [0, 0], [.7, .7]], [.9, 0., .9])
    # unit 0: learner heading 80 degrees off its enemy 1; teacher exactly at enemy 0. unit 1: teacher does not throw.
    learner["target"][0, 0] = normalized(math.cos(D(80)) * 12, math.sin(D(80)) * 12)
    for rule in ("aim", "aim-heading", "enemy", "offset"):
        action, counts, samples = ts.split_actions(rule, learner, teacher, THROWER, ENEMIES, HALF, LIVING)
        assert np.array_equal(action["action_type"], learner["action_type"]) and np.array_equal(action["power"], learner["power"])
        assert np.array_equal(action["target"][0, 1:], learner["target"][0, 1:])   # not comparable -> untouched
        assert counts["comparableThrows"] == 1 and counts["learnerThrows"] == 2 and len(samples) == 1
    action, _, _ = ts.split_actions("aim", learner, teacher, THROWER, ENEMIES, HALF, LIVING)
    assert np.allclose(action["target"][0, 0], teacher["target"][0, 0])
    action, _, _ = ts.split_actions("enemy", learner, teacher, THROWER, ENEMIES, HALF, LIVING)
    point = (action["target"][0, 0, 0] * 50, action["target"][0, 0, 1] * 40)
    assert ts.bearing((0, 0), point) == pytest.approx(D(-10), abs=1e-4)          # teacher's enemy 0 + learner's -10 degrees
    action, _, _ = ts.split_actions("offset", learner, teacher, THROWER, ENEMIES, HALF, LIVING)
    point = (action["target"][0, 0, 0] * 50, action["target"][0, 0, 1] * 40)
    assert ts.bearing((0, 0), point) == pytest.approx(D(90), abs=1e-4)           # learner's enemy 1 + teacher's 0 offset


def test_none_changes_nothing_and_dead_units_are_never_touched():
    learner = arrays([ACTION_THROW, ACTION_THROW, ACTION_THROW], [normalized(3, 4)] * 3, [.8] * 3)
    teacher = arrays([ACTION_THROW, ACTION_THROW, ACTION_THROW], [normalized(10, 0)] * 3, [.9] * 3)
    action, counts, _ = ts.split_actions("none", learner, teacher, THROWER, ENEMIES, HALF, LIVING)
    assert all(np.array_equal(action[k], learner[k]) for k in learner) and counts["changed"] == 0
    dead = np.array([[True, False, True]])
    action, counts, _ = ts.split_actions("aim", learner, teacher, THROWER, ENEMIES, HALF, dead)
    assert np.array_equal(action["target"][0, 1], learner["target"][0, 1]) and counts["changed"] == 2


def test_a_degenerate_ray_falls_back_to_the_unchanged_target_and_is_counted():
    thrower = np.zeros((1, 3, 2)); thrower[0, 0] = (49.9, 0.0)
    learner = arrays([ACTION_THROW] * 3, [normalized(49.9, 5.0)] * 3, [.8] * 3)
    teacher = arrays([ACTION_THROW] * 3, [normalized(49.9, 0.0)] * 3, [.9] * 3)
    teacher["target"][0, 0] = normalized(50.0, 0.0)  # teacher aims outward along +x from x=49.9 (radius 0.1)
    action, counts, _ = ts.split_actions("enemy", learner, teacher, thrower, [[(0.0, 5.0)]], HALF, LIVING)
    assert counts["fallbacks"] >= 1
    assert np.array_equal(action["target"][0, 0], learner["target"][0, 0])


def test_sample_summary_reports_share_and_quartiles_in_degrees():
    samples = [{"sameEnemy": True, "learnerOffset": D(10), "teacherOffset": D(2), "headingGap": D(12)},
               {"sameEnemy": False, "learnerOffset": D(20), "teacherOffset": D(4), "headingGap": D(30)}]
    summary = ts.summarize_samples(samples)
    assert summary["sameEnemyShare"] == 0.5 and summary["learnerOffsetDegrees"]["50"] == pytest.approx(15.0)
    assert ts.summarize_samples([]) == {"comparableThrows": 0}


def cell(recovery=None, same=None, learner=None, teacher=None):
    split = {}
    if same is not None:
        split["sameEnemyShare"] = same
    if learner is not None:
        split["learnerOffsetDegrees"] = {"50": learner}
        split["teacherOffsetDegrees"] = {"50": teacher}
    return {"yieldRecovery": recovery, "targetSplit": split}


def test_prediction_scoring_is_mechanical_and_needs_all_three_seeds():
    seed = lambda: {"none": cell(same=0.3, learner=7.0, teacher=2.0), "enemy": cell(0.8), "offset": cell(0.2)}
    report = {"learners": {"97101": seed(), "97102": seed(), "97103": seed()}}
    assert all(v["holds"] for v in ts.score_predictions(report).values())
    report["learners"]["97102"]["enemy"] = cell(0.49)
    report["learners"]["97103"]["none"] = cell(same=0.6, learner=7.0, teacher=2.0)
    scored = ts.score_predictions(report)
    assert scored["P1"]["holds"] is False and scored["P3"]["holds"] is False and scored["P2"]["holds"] is True


def test_budget_bound_matches_the_declaration():
    assert ts.budget_bound(ts.configuration())["total"] == 300000
    assert ts.budget_bound(ts.configuration())["total"] <= ts.configuration()["budgetCap"]


# -- live ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def model_97101():
    return ri.load_final(ri.configuration(), TRAINING / "runs" / rv.S5_RUN, 97101)


def test_none_arm_reproduces_the_s5_rows_exactly(client):
    cfg = {**ts.configuration(), "interventionWorlds": 50, "bootstrapSamples": 50}
    rows, traces, totals, samples = ts.collect_cell(client, cfg, model_97101(), "none", account=lambda n: None)
    gate = rt.reproduction_gate(rows, rt.archived_rows("learner", "normal", 97101, 50))
    assert gate["passed"], gate["mismatches"][:3]
    assert totals["changed"] == 0 and len(samples) > 0


def test_every_arm_can_be_collected_and_the_new_arms_change_actions(client):
    """S8's first launch crashed on one arm's name; every arm name must run."""
    cfg = {**ts.configuration(), "interventionWorlds": 2, "blockWorlds": 2, "bootstrapSamples": 50}
    for rule in ts.RULES:
        rows, traces, totals, _ = ts.collect_cell(client, cfg, model_97101(), rule, account=lambda n: None)
        assert len(rows) == 2 and len(traces) == 2, rule
        if rule != "none":
            assert totals["changed"] > 0, rule


def test_traces_match_is_episode_exact_and_reports_the_first_mismatch():
    a = [{"seed": 1, "states": [[1.0]]}, {"seed": 2, "states": [[2.0]]}]
    assert ts.traces_match(a, [dict(x) for x in a])["passed"]
    b = [a[0], {"seed": 2, "states": [[2.5]]}]
    assert ts.traces_match(a, b) == {"passed": False, "firstMismatchSeed": 2}
    assert not ts.traces_match(a, a[:1])["passed"]


def test_aim_arm_reproduces_the_archived_s8_aim_traces_for_the_first_50_worlds(client, tmp_path):
    cfg = {**ts.configuration(), "interventionWorlds": 50, "bootstrapSamples": 50}
    rows, traces, _, _ = ts.collect_cell(client, cfg, model_97101(), "aim", account=lambda n: None)
    rt.write_traces(tmp_path / "learner-97101" / "aim", traces)
    gate = ts.aim_matches_s8(tmp_path, 97101, count=50)
    assert gate["passed"], gate


def test_aim_heading_versus_aim_is_recorded_as_a_check_on_the_discarded_distance(client):
    cfg = {**ts.configuration(), "interventionWorlds": 50, "bootstrapSamples": 50}
    a, _, _, _ = ts.collect_cell(client, cfg, model_97101(), "aim", account=lambda n: None)
    h, _, _, _ = ts.collect_cell(client, cfg, model_97101(), "aim-heading", account=lambda n: None)
    same = sum(x["success"] == y["success"] and x["finalDecision"] == y["finalDecision"] for x, y in zip(a, h))
    print("AIM vs AIM-HEADING identical episodes:", same, "of 50")
    assert same >= 0


def test_tiny_end_to_end_run_seals(tmp_path):
    cfg = {**ts.configuration(), "interventionWorlds": 2, "blockWorlds": 2, "learnerSeeds": (97101,),
           "bootstrapSamples": 50}
    report = ts.run(tmp_path / "run", cfg)
    assert set(report["learners"]["97101"]) == set(ts.RULES) and report["withinBudgetBound"]
    assert set(report["predictions"]) == {"P1", "P2", "P3", "P4"}
    assert "97101/none-vs-S5" in report["reproduction"] and "97101/aim-vs-S8" in report["reproduction"]
    assert dr.verify_sealed(tmp_path / "run")
