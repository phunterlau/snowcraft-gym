import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options import roster_trace as rt
from snowgym_training.options.full_authority_diagnostics import TRAINING

CFG = {**rt.configuration(), "bootstrapSamples": 100}


def unit(x, y, health=100.0, alive=True, stun=0.0, phase=0.0):
    return [int(alive), health if alive else 0.0, x, y, 0.0, 0.0, stun, phase, 0.0]


def state(blue, red, proj=()):
    return {"blue": blue, "red": red, "proj": [list(p) for p in proj]}


def synthetic():
    """3v3, 4 decisions. Blue at x=0 in a line (spacing 2 and 4); red at x=6.
    t0->t1: red unit 0 takes 60 damage (first hit = decision 1). t1->t2: red 0 dies (full 100) and blue 2 is hit.
    t2->t3: blue unit 1 dies; red 1 takes 20. Final: red 0 dead, red 1 at 80 health."""
    red = lambda h0, h1, h2: [unit(6, 0, h0, alive=h0 > 0), unit(6, 3, h1, alive=h1 > 0), unit(6, -3, h2)]
    s0 = state([unit(0, 0), unit(0, 2), unit(0, 4)], red(100, 100, 100), [])
    s1 = state([unit(0, 0), unit(0, 2), unit(0, 4)], red(40, 100, 100), [(1, 4, "red", 5, 0, -8, 0), (2, 1, "blue", 1, 0, 8, 0)])
    s2 = state([unit(1, 0), unit(1, 2, stun=2.0), unit(1, 4, health=70.0)], red(0, 100, 100),
               [(1, 4, "red", 5, 0, -8, 0), (3, 2, "blue", 1, 0, 8, 0)])
    s3 = state([unit(2, 0), unit(1, 2, alive=False), unit(2, 4, health=70.0, phase=1.0)], red(0, 80, 100), [])
    acts = [[[1, "move", True], [2, "throw", True], [3, "noop", True]]] * 3
    return {"seed": 1, "states": [s0, s1, s2, s3], "acts": acts}


def test_synthetic_trace_gives_the_hand_computed_measures():
    m = rt.episode_metrics(synthetic(), CFG)
    assert m["firstHitDecision"] == 1
    assert m["firstDeathAfterHit"] == 2 and m["wipeAfterHit"] is None
    assert m["redKills"] == 1
    # damage: red0 100 (dead), red1 20 (alive), red2 0 -> wasted = 20 / 120
    assert m["wastedDamageShare"] == pytest.approx(20 / 120)
    # blue lost: unit1 dead (100) + unit2 30 = 130; red shots: id 1 only; blue shots: ids 2 and 3
    assert m["redYield"] == pytest.approx(130 / 1) and m["blueYield"] == pytest.approx(120 / 2)
    # spacing at the pre-hit state (s0): pairwise 2, 4, 2 -> mean 8/3
    assert m["spacingAtFirstHit"] == pytest.approx(8 / 3)
    # window = decisions 0..2; living blue per decision 3,3,3 -> 9 unit-decisions; blue shots first seen >= 1: ids 2 (k=1), 3 (k=2)
    assert m["blueShotsPerUnitDecision"] == pytest.approx(2 / 9)
    # red living at decisions 0, 1, 2: 3 (all), 3 (red0 still alive at 40 health), 2 = 8; red shot id 1 first seen at k=1
    assert m["redShotsPerUnitDecision"] == pytest.approx(1 / 8)
    # incapacitated: decision 1 unit 1 stunned (stun 2.0) -> 1 of 9
    assert m["incapacitatedShare"] == pytest.approx(1 / 9)
    assert m["throwActionShare"] == pytest.approx(1 / 3) and m["moveActionShare"] == pytest.approx(1 / 3)


def test_displacement_is_actual_position_change_split_by_projectile_proximity():
    m = rt.episode_metrics(synthetic(), CFG)
    # red projectile id 1 sits at (5, 0) in s1 and s2, within 9 of every blue unit there; s0 has none.
    # decision 0: no threat, blue units barely move (0); decision 1: threat, units move by 1 in x.
    assert m["displacementNoThreat"] is not None and m["displacementUnderThreat"] is not None
    assert m["displacementUnderThreat"] > m["displacementNoThreat"]


def test_a_trace_with_no_damage_has_undefined_postcontact_measures():
    s = state([unit(0, 0), unit(0, 2), unit(0, 4)], [unit(6, 0), unit(6, 3), unit(6, -3)])
    m = rt.episode_metrics({"seed": 1, "states": [s, s], "acts": [[]]}, CFG)
    assert m["firstHitDecision"] is None and m["spacingAtFirstHit"] is None and m["wastedDamageShare"] is None


def test_cell_summary_counts_defined_episodes_and_brackets_the_mean():
    good = rt.episode_metrics(synthetic(), CFG)
    empty = rt.episode_metrics({"seed": 2, "states": [state([unit(0, 0)] * 3, [unit(6, 0)] * 3)] * 2, "acts": [[]]}, CFG)
    summary = rt.cell_summary([good, good, empty], CFG)
    assert summary["episodes"] == 3 and summary["wastedDamageShare"]["defined"] == 2
    lo, hi = summary["wastedDamageShare"]["interval"]
    assert lo <= summary["wastedDamageShare"]["mean"] <= hi


def cell(**values):
    keys = ("wastedDamageShare", "spacingAtFirstHit", "blueShotsPerUnitDecision", "displacementUnderThreat", "redYield",
            "incapacitatedShare", "nearestRedDistance", "redShotsPerUnitDecision")
    return {k: {"mean": values.get(k), "defined": 1, "interval": [0, 1]} for k in keys}


def test_prediction_scoring_is_mechanical_and_needs_all_three_seeds():
    teacher = {"normal": cell(wastedDamageShare=0.1, spacingAtFirstHit=10.0, blueShotsPerUnitDecision=0.4,
                              displacementUnderThreat=1.0, redYield=10.0, incapacitatedShare=0.1),
               "random": cell(nearestRedDistance=6.0, redShotsPerUnitDecision=0.2)}
    holding = cell(wastedDamageShare=0.6, spacingAtFirstHit=7.0, blueShotsPerUnitDecision=0.2, displacementUnderThreat=0.9,
                   redYield=13.0, incapacitatedShare=0.2)
    random_holding = cell(nearestRedDistance=4.0, redShotsPerUnitDecision=0.22)
    learners = {str(s): {"normal": holding, "random": random_holding} for s in rt.SEEDS}
    scored = rt.score_predictions({"teacher": teacher, "learners": learners})
    assert all(scored[p]["holds"] is True for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"))
    learners["97102"] = {"normal": cell(wastedDamageShare=0.35, spacingAtFirstHit=7.0, blueShotsPerUnitDecision=0.2,
        displacementUnderThreat=0.5, redYield=13.0, incapacitatedShare=0.2), "random": random_holding}
    scored = rt.score_predictions({"teacher": teacher, "learners": learners})
    assert scored["P1"]["holds"] is False and scored["P4"]["holds"] is False and scored["P2"]["holds"] is True
    teacher["normal"]["redYield"]["mean"] = 0
    assert rt.score_predictions({"teacher": teacher, "learners": learners})["P5"]["holds"] is None


def test_reproduction_gate_reports_each_mismatching_field():
    row = {"seed": 1, "success": True, "blueAliveCount": 3, "finalDecision": 70, "finalTargetDamage": 240.0,
           "firstHitDecision": 50}
    assert rt.reproduction_gate([row], [dict(row)])["passed"]
    gate = rt.reproduction_gate([row], [{**row, "blueAliveCount": 2}])
    assert not gate["passed"] and gate["mismatches"][0]["key"] == "blueAliveCount"


def test_traces_round_trip_through_deterministic_gzip(tmp_path):
    traces = [synthetic()]
    rt.write_traces(tmp_path / "a", traces)
    rt.write_traces(tmp_path / "b", traces)
    assert (tmp_path / "a" / "traces.jsonl.gz").read_bytes() == (tmp_path / "b" / "traces.jsonl.gz").read_bytes()
    assert rt.read_traces(tmp_path / "a" / "traces.jsonl.gz")[0]["states"][0]["blue"][0][2] == 0
    with pytest.raises(FileExistsError):
        rt.write_traces(tmp_path / "a", traces)


def test_budget_bound_matches_the_declaration():
    bound = rt.budget_bound(rt.configuration())
    assert (bound["learners"], bound["teacher"], bound["total"]) == (120000, 40000, 160000)
    assert bound["total"] <= rt.configuration()["budgetCap"]


# -- live ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def test_teacher_traces_are_structurally_consistent_and_reproduce_the_s4_rows(client):
    cfg = {**rt.configuration(), "traceWorlds": 4, "bootstrapSamples": 50}
    rows, traces = rt.collect_cell(client, rt.trace_seeds(cfg), cfg, "normal", model=None, account=lambda n: None)
    assert rt.reproduction_gate(rows, rt.archived_rows("teacher", "normal", None, 4))["passed"]
    for row, trace in zip(rows, traces):
        assert len(trace["states"]) == len(trace["acts"]) + 1 == row["finalDecision"] + 1
        assert all(len(s["blue"]) == 3 and len(s["red"]) == 3 for s in trace["states"])
        m = rt.episode_metrics(trace, cfg)
        assert m["firstHitDecision"] == row["firstHitDecision"]
        assert sum(1 for u in trace["states"][-1]["blue"] if u[0] == 1) == row["blueAliveCount"]


def test_a_learner_cell_reproduces_its_s5_paired_evaluation_at_the_same_block_size(client):
    cfg = {**rt.configuration(), "traceWorlds": 50, "bootstrapSamples": 50}
    model = ri.load_final(ri.configuration(), TRAINING / "runs" / rt.S5_RUN, 97101)
    rows, traces = rt.collect_cell(client, rt.trace_seeds(cfg), cfg, "normal", model=model, account=lambda n: None)
    gate = rt.reproduction_gate(rows, rt.archived_rows("learner", "normal", 97101, 50))
    assert gate["passed"], gate["mismatches"][:3]
    assert len(traces) == 50


def test_tiny_end_to_end_run_seals(tmp_path):
    cfg = {**rt.configuration(), "traceWorlds": 2, "learnerSeeds": (97101,), "arms": ("normal",),
           "bootstrapSamples": 50, "blockWorlds": 2}
    report = rt.run(tmp_path / "run", cfg)
    assert set(report["predictions"]) == {f"P{i}" for i in range(1, 9)} and report["withinBudgetBound"]
    assert dr.verify_sealed(tmp_path / "run")
