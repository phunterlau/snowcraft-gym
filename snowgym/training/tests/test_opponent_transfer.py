import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options import opponent_transfer as ot
from snowgym_training.options.pre_ppo_diagnostics import MODES, mode_chooser


def tiny(**overrides):
    return {**ot.configuration(), "optionHorizon": 8, "initializerSeeds": [97101],
            "evaluationSeeds": [871900, 871903], "evaluationBlockWorlds": 4,
            "reproductionSeeds": [870000, 870001], "reproductionInitializer": 97101,
            "bootstrapSamples": 50, "budgetCap": 20000, **overrides}


# -- scenario_override --------------------------------------------------------------------


def test_scenario_override_applies_and_restores():
    original = v1.scenario
    baseline = original()
    assert baseline["redController"] == "random"
    with ot.scenario_override({"redController": "scripted", "redDifficulty": "easy"}):
        overridden = v1.scenario()
        assert overridden["redController"] == "scripted" and overridden["redDifficulty"] == "easy"
        assert v1.scenario is not original
    assert v1.scenario is original
    assert v1.scenario() == baseline


def test_scenario_override_restores_even_on_error():
    original = v1.scenario
    with pytest.raises(RuntimeError):
        with ot.scenario_override({"redController": "scripted"}):
            raise RuntimeError("boom")
    assert v1.scenario is original


# -- shots_from_log: the projectile-attribution unit that had the health-index bug --------


def make_log(*rows):
    """Each row: (bHealth, rHealth, bx, by, bvx, bvy, rx, ry, rvx, rvy, projectiles)."""
    return list(rows)


def test_shots_from_log_attributes_a_mid_episode_red_hit_to_blue_not_red_health():
    # Red projectile 1 spawns at decision 0 near blue, "lands" (health drop + vanish) at
    # decision 1. Red's own health is untouched throughout: a naive same-team health check
    # would find no drop and mis-attribute nothing, or worse, attribute against the wrong
    # side entirely.
    log = make_log(
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((1, "red", 5, 0),)),
        (80, 100, 0, 0, 0, 0, 5, 0, 0, 0, ()),
    )
    row = {"blueAliveAtEnd": True, "success": False}
    shots = ot.shots_from_log(row, log, "red")
    assert len(shots) == 1
    assert shots[0]["hit"] is True
    assert shots[0]["spawnDecision"] == 0
    assert shots[0]["spawnDistance"] == pytest.approx(5.0)


def test_shots_from_log_does_not_attribute_a_miss():
    log = make_log(
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((1, "red", 5, 0),)),
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ()),  # projectile vanished, blue health unchanged
    )
    row = {"blueAliveAtEnd": True, "success": False}
    shots = ot.shots_from_log(row, log, "red")
    assert len(shots) == 1 and shots[0]["hit"] is False


def test_shots_from_log_handles_the_terminal_hit_with_no_following_state():
    # The projectile is still logged at the final decision (blue dies exactly then, so the
    # loop never records a post-hit state); the terminal fallback must still attribute it.
    log = make_log(
        (20, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((1, "red", 5, 0),)),
    )
    row = {"blueAliveAtEnd": False, "success": False}
    shots = ot.shots_from_log(row, log, "red")
    assert len(shots) == 1 and shots[0]["hit"] is True


def test_shots_from_log_blue_projectiles_use_reds_health_as_the_victim():
    log = make_log(
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((2, "blue", 4, 0),)),
        (100, 80, 0, 0, 0, 0, 5, 0, 0, 0, ()),
    )
    row = {"blueAliveAtEnd": True, "success": False}
    shots = ot.shots_from_log(row, log, "blue")
    assert len(shots) == 1 and shots[0]["hit"] is True


def test_shots_from_log_reports_target_displacement_during_flight():
    # Displacement is measured over the frames the projectile is actually visible for
    # (decisions 0-1), not including the frame after it lands.
    log = make_log(
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((1, "red", 5, 0),)),
        (100, 100, 3, 4, 0, 0, 5, 0, 0, 0, ((1, "red", 5, 0),)),
        (80, 100, 6, 8, 0, 0, 5, 0, 0, 0, ()),
    )
    row = {"blueAliveAtEnd": True, "success": False}
    shots = ot.shots_from_log(row, log, "red")
    assert len(shots) == 1
    assert shots[0]["displacement"] == pytest.approx(5.0)  # (0,0) -> (3,4)
    assert shots[0]["flightDecisions"] == 2


def test_shots_from_log_only_the_nearest_vanishing_projectile_gets_the_hit():
    # Two red projectiles vanish at the same decision, at different distances from blue's
    # position at that moment (blue is at (0, 0)); only the nearer one, id 1 at x=1, is
    # credited with the single health drop. `first.items()` preserves insertion order, so
    # `shots[0]` is projectile 1 and `shots[1]` is projectile 2.
    log = make_log(
        (100, 100, 0, 0, 0, 0, 5, 0, 0, 0, ((1, "red", 1, 0), (2, "red", 9, 0))),
        (80, 100, 0, 0, 0, 0, 5, 0, 0, 0, ()),
    )
    row = {"blueAliveAtEnd": True, "success": False}
    shots = ot.shots_from_log(row, log, "red")
    assert len(shots) == 2
    assert shots[0]["hit"] is True
    assert shots[1]["hit"] is False


# -- health_drop_events ---------------------------------------------------------------------


def test_health_drop_events_counts_mid_episode_and_terminal_drops():
    log = make_log(
        (100, 100, 0, 0, 0, 0, 0, 0, 0, 0, ()),
        (80, 100, 0, 0, 0, 0, 0, 0, 0, 0, ()),
        (20, 100, 0, 0, 0, 0, 0, 0, 0, 0, ()),
    )
    dead = {"blueAliveAtEnd": False, "success": False}
    alive = {"blueAliveAtEnd": True, "success": False}
    assert ot.health_drop_events(dead, log, "blue") == 3  # 2 within-log drops + the terminal one
    assert ot.health_drop_events(alive, log, "blue") == 2
    assert ot.health_drop_events(dead, log, "red") == 0


# -- failure_label ----------------------------------------------------------------------


def test_failure_label_covers_the_six_declared_categories():
    base = {"success": False, "blueAliveAtEnd": True, "timedOut": False, "firstHitDecision": None}
    assert ot.failure_label({**base, "success": True, "blueAliveAtEnd": True}) == "win"
    assert ot.failure_label({**base, "success": True, "blueAliveAtEnd": False}) == "win-but-dead"
    assert ot.failure_label({**base, "blueAliveAtEnd": False}) == "death"
    assert ot.failure_label({**base, "timedOut": True, "firstHitDecision": 5}) == "timeout-with-hits"
    assert ot.failure_label({**base, "timedOut": True, "firstHitDecision": None}) == "timeout-no-hits"
    assert ot.failure_label(base) == "unresolved"


# -- arm_outcome: the declared §5 precedence ---------------------------------------------


def analysis(*, teacher=1.0, init_success=.5, final_success=.5, gap=0.0):
    return {"teacherSuccess": teacher, "initializerSuccessMean": init_success,
            "finalSuccessMean": final_success, "finalSuccessGapToTeacher": {"mean": gap}}


def test_arm_outcome_precedence():
    cfg = ot.configuration()
    assert ot.arm_outcome(analysis(teacher=.5), cfg) == "invalid"  # order 1: teacher too weak
    assert ot.arm_outcome(analysis(init_success=.02, final_success=.02, gap=-.9), cfg) == "no-transfer-floor"
    assert ot.arm_outcome(analysis(init_success=.3, final_success=.02, gap=-.9), cfg) == "fails"  # only final at floor
    assert ot.arm_outcome(analysis(gap=-.6), cfg) == "fails"
    assert ot.arm_outcome(analysis(gap=-.2), cfg) == "partial"
    assert ot.arm_outcome(analysis(gap=-.05), cfg) == "transfers"
    assert ot.arm_outcome(analysis(gap=.1), cfg) == "transfers"


# -- budget -------------------------------------------------------------------------------


def test_budget_bound_matches_declared_figures():
    cfg = ot.configuration()
    bound = ot.budget_bound(cfg)
    assert bound["reproduction"] == 10_000
    assert bound["total"] == pytest.approx(1_690_000)
    assert cfg["budgetCap"] == 2_000_000


# -- live: reproduction check, one small arm, and tamper detection ------------------------


def test_reproduction_check_matches_the_archived_r1n_e_episodes():
    cfg = {**ot.configuration(), "reproductionSeeds": [870000, 870004]}
    with SnowGymBatchClient() as client:
        check = ot.reproduction_check(client, cfg)
    assert check == {"worlds": 5, "decisions": check["decisions"], "mismatches": [], "passed": True}


def test_tiny_end_to_end_declare_arm_and_aggregate(tmp_path):
    # The reproduction check needs the real (full-horizon) configuration to match the
    # archive; it is exercised separately in
    # test_reproduction_check_matches_the_archived_r1n_e_episodes. This test covers the
    # declare/run_arm/aggregate pipeline and tamper detection at a tiny scale.
    cfg = tiny()
    root = tmp_path / "run"
    ot.declare(root, cfg)
    tables = ot.run_arm(root, cfg, "R")
    assert set(tables) == {f"init-{s}" for s in cfg["initializerSeeds"]} | \
        {f"final-{s}" for s in cfg["initializerSeeds"]} | {"teacher"}
    for name in tables:
        directory = root / "arm-R" / name
        assert (directory / "episodes.jsonl").exists()
        assert (directory / "shots.npz").exists()
    with pytest.raises(FileExistsError):
        ot.run_arm(root, cfg, "R")
    # aggregate needs all declared arms; run the remaining two at reduced scale
    ot.run_arm(root, cfg, "E")
    ot.run_arm(root, cfg, "N")
    report = ot.aggregate(root, cfg)
    assert set(report["outcomes"]) == {"E", "N"}
    assert report["outcomes"]["E"] in {"invalid", "no-transfer-floor", "fails", "partial", "transfers"}
    manifest = json.loads((root / "manifest.json").read_text())
    assert "arm-R/manifest.json" in manifest["artifacts"] and "report.json" in manifest["artifacts"]
    # tamper detection reuses death_rate_ppo's seal/verify_sealed
    rows = root / "arm-R" / "teacher" / "episodes.jsonl"
    rows.write_text(rows.read_text() + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        dr.verify_sealed(root / "arm-R")


def test_declare_refuses_to_overwrite(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    ot.declare(root, cfg)
    with pytest.raises(FileExistsError):
        ot.declare(root, cfg)
