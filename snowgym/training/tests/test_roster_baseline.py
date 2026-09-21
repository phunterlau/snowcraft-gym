import json
from pathlib import Path

import numpy as np
import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options.full_authority_diagnostics import TRAINING


def tiny(**overrides):
    return {**rb.configuration(), "teacherWorlds": 3, "floorWorlds": 2, "initSeeds": (98101,), "blockWorlds": 3,
            "bootstrapSamples": 200, **overrides}


def row(success, alive, *, timed_out=False, decisions=50, actions=100, rejected=0):
    return {"success": success, "timedOut": timed_out, "blueAliveCount": alive,
            "unitsLostFraction": rb.units_lost_fraction(3, alive), "finalDecision": decisions,
            "totalActions": actions, "rejectedActions": rejected}


def test_units_lost_fraction_generalizes_the_1v1_death_indicator():
    assert [rb.units_lost_fraction(1, alive) for alive in (1, 0)] == [0.0, 1.0]
    assert [rb.units_lost_fraction(3, alive) for alive in (3, 2, 1, 0)] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    for assigned, alive in ((0, 0), (3, 4), (3, -1)):
        with pytest.raises(ValueError):
            rb.units_lost_fraction(assigned, alive)


def test_wilson_interval_brackets_the_proportion_and_stays_in_unit_range():
    low, high = rb.wilson(90, 100)
    assert low < 0.9 < high and 0 <= low and high <= 1
    assert rb.wilson(0, 100)[0] == pytest.approx(0.0, abs=1e-12)
    assert rb.wilson(100, 100)[1] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        rb.wilson(0, 0)


def test_bootstrap_mean_is_seeded_and_brackets_the_mean():
    values = [0, 0, 1 / 3, 1, 2 / 3, 0, 1, 0]
    first = rb.bootstrap_mean(values, samples=500, seed=1)
    assert first == rb.bootstrap_mean(values, samples=500, seed=1)
    assert first[0] <= np.mean(values) <= first[1]


def test_summarize_reports_success_wipe_timeout_loss_and_rejections_separately():
    rows = [row(True, 3), row(True, 1), row(False, 0), row(False, 2, timed_out=True)]
    summary = rb.summarize(rows, tiny())
    assert summary["success"] == 0.5 and summary["teamWipe"] == 0.25 and summary["timeout"] == 0.25
    assert summary["meanUnitsLostFraction"] == pytest.approx((0 + 2 / 3 + 1 + 1 / 3) / 4)
    assert summary["rejectedActionRate"] == 0
    with pytest.raises(ValueError):
        rb.summarize([], tiny())


def test_gates_use_the_declared_thresholds_and_only_the_scripted_arms_for_floors():
    cfg = tiny()
    cells = {f"teacher/{arm}": {"success": 0.95} for arm in cfg["arms"]}
    cells |= {f"floor-98101-deterministic/{arm}": {"success": 0.0} for arm in cfg["arms"]}
    cells["floor-98101-deterministic/random"] = {"success": 0.9}  # the random arm is not a floor gate
    gates = rb.evaluate_gates(cells, cfg)
    assert gates["teacherAchievable"] and gates["floorsShowNoFreeSignal"]
    cells["teacher/normal"] = {"success": 0.89}
    cells["floor-98101-deterministic/easy"] = {"success": 0.11}
    gates = rb.evaluate_gates(cells, cfg)
    assert not gates["teacherAchievable"] and not gates["floorsShowNoFreeSignal"]


def test_budget_bound_matches_the_declaration():
    bound = rb.budget_bound(rb.configuration())
    assert (bound["teacher"], bound["floors"], bound["total"]) == (240000, 360000, 600000)


def test_seed_band_is_unused_by_every_archived_declaration_and_every_other_json():
    cfg = rb.configuration()
    band = range(cfg["evaluationSeedBase"], cfg["evaluationSeedBase"] + cfg["teacherWorlds"])
    lo, hi = band.start, band.stop - 1

    def integers(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from integers(item)
        elif isinstance(value, list):
            for item in value:
                yield from integers(item)

    collisions = []
    for path in TRAINING.parents[1].rglob("*.json"):
        if any(part in {".git", "node_modules", ".venv", "dist", "m8_s4_roster_baseline_v0", "m8_s5_roster_imitation_v0",
               "m8_s6_breakdown_v0", "m8_s7_trace_diagnosis_v0"}
               for part in path.parts) or path.stat().st_size > 5_000_000:
            continue  # S4's own archive and S5's declared reuse of these worlds are the only intended users
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if any(lo <= n <= hi for n in integers(data)):
            collisions.append(str(path))
    assert collisions == []


# -- live ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def test_teacher_cell_records_end_state_consistent_with_the_trackers_own_flags(client):
    cfg = tiny()
    rows = rb.collect_cell(client, rb.world_seeds(cfg, 3), cfg, "normal", model=None, mode="deterministic",
                           account=lambda n: None)
    assert [r["seed"] for r in rows] == rb.world_seeds(cfg, 3)
    for r in rows:
        assert r["assignedUnits"] == 3 and 0 <= r["blueAliveCount"] <= 3
        assert r["unitsLostFraction"] == pytest.approx((3 - r["blueAliveCount"]) / 3)
        assert r["blueAliveAtEnd"] == (r["blueAliveCount"] > 0)
        assert r["success"] or r["failed"]


def test_floor_cell_is_reproducible_for_a_fixed_init_seed_and_sampling_stream(client):
    import torch
    cfg = tiny()
    def cell():
        model = rb.floor_model(98101)
        torch.manual_seed(5)
        return rb.collect_cell(client, rb.world_seeds(cfg, 2), cfg, "easy", model=model, mode="stochastic",
                               account=lambda n: None)
    assert cell() == cell()


def test_tiny_end_to_end_run_seals_and_verifies_an_archive(client, tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    rb.declare(root, cfg)
    steps = []
    cells = rb.run_cells(root, cfg, client, steps.append)
    report = rb.aggregate(root, cfg, cells, sum(steps))
    assert len(cells) == 3 + 2 * 3 and report["withinBudgetBound"]
    assert dr.verify_sealed(root)
    with pytest.raises(FileExistsError):
        rb.declare(root, cfg)
    assert json.loads((root / "declaration.json").read_text())["autonomousQualificationEligible"] is False
