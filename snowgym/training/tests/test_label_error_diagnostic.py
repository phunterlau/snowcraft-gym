import json
import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import label_error_diagnostic as led


def tiny(**overrides):
    return {**led.configuration(), "optionHorizon": 8, "initializerSeeds": [97101, 97102, 97103],
            "armSeeds": {"R": [994000, 994001], "E": [995000, 995001], "N": [996000, 996001]},
            "collectionBlockWorlds": 2, "budgetCap": 60000, **overrides}


# -- budget ---------------------------------------------------------------------------------


def test_budget_bound_matches_declared_figures():
    cfg = led.configuration()
    bound = led.budget_bound(cfg)
    assert bound == {"cells": 18, "episodesPerArm": 128, "assumedDecisionsPerEpisode": 110, "total": 253_440}
    assert cfg["budgetCap"] == 400_000


def test_configuration_seed_bands_do_not_collide_with_r1n_b():
    cfg = led.configuration()
    for lo, hi in cfg["armSeeds"].values():
        assert not (lo <= 630119 and hi >= 630000)
    assert not (cfg["trainSeedBase"] <= 630119 and cfg["trainSeedBase"] >= 630000)
    assert not (cfg["heldOutSeedBase"] <= 630119 and cfg["heldOutSeedBase"] >= 630000)


# -- label_error reuse-site regression guard (declaration §8) -------------------------------


class FakeLabelErrorModel:
    """Deterministic stand-in for `FullAuthorityPolicyV1`, exercising only the interface
    `label_error` calls: `__call__` and `decode_move`. Row 0 is teacher-labelled MOVE, row 1
    THROW; the model's own action-type guess matches the teacher exactly at both rows, so
    recall/precision reduce to 1.0 and the interesting content is the geometry."""

    def __call__(self, o, with_value=False):
        return {"living": torch.ones(2, 1, dtype=torch.bool),
                "action_logits": torch.tensor([[[0., 10., 0., 0.]], [[0., 0., 10., 0.]]]),
                "move_raw": torch.zeros(2, 1, 2),
                "throw_raw": torch.tensor([[[0., 0.]], [[math.atanh(0.5), 0.]]]),
                "power_raw": torch.zeros(2, 1)}

    def decode_move(self, o, move_raw):
        # Constant (0, 1) direction; only row 0 (the far-bucket MOVE row) reads this.
        return torch.tensor([[[0., 1.]], [[0., 1.]]])


def test_label_error_reproduces_a_hand_computed_case():
    cfg = led.configuration()
    obs = {"allies": torch.zeros(2, 1, 4)}
    labels = {"action_type": torch.tensor([[1], [2]]),  # MOVE, THROW (full_authority_imitation.ACTION_NAMES order)
              "target": torch.tensor([[[0.5, 0.0]], [[0.0, 0.5]]]),
              "power": torch.tensor([[0.0], [0.3]])}
    part = {"observation": obs, "labels": labels}
    result = fi.label_error(FakeLabelErrorModel(), part, cfg)

    assert result["units"] == 2
    assert result["typeAccuracy"] == pytest.approx(1.0)
    assert result["perType"]["move"] == {"support": 1, "recall": 1.0, "precision": 1.0}
    assert result["perType"]["throw"] == {"support": 1, "recall": 1.0, "precision": 1.0}
    assert result["perType"]["noop"] == {"support": 0, "recall": None, "precision": None}
    assert result["perType"]["hold"] == {"support": 0, "recall": None, "precision": None}
    # Own at (0,0); teacher move target world = (25, 0) (ARENA_HALF_EXTENT=(50,40), distance
    # 25 > slowRadius, so the "far"/heading-error bucket); the fake decode_move points world
    # direction (0, +1) — orthogonal to (1, 0) — for exactly 90 degrees regardless of scale.
    assert result["moveHeadingErrorDegrees"] == pytest.approx(90.0, abs=1e-3)
    assert result["moveEndpointErrorWorld"] is None
    # Teacher throw target world = (0, 20); tanh(atanh(0.5)) * 50 = 25 gives aim world (25, 0)
    # — orthogonal to (0, 1) — again exactly 90 degrees.
    assert result["throwAimHeadingErrorDegrees"] == pytest.approx(90.0, abs=1e-3)
    # sigmoid(0) = 0.5, label power = 0.3, |0.5 - 0.3| = 0.2.
    assert result["powerMeanAbsoluteError"] == pytest.approx(0.2, abs=1e-6)


# -- cross-checks and localization (pure, synthetic cells) ----------------------------------


def make_cell(*, type_accuracy, throw_recall, aim_degrees, power_error, move_degrees=5.0,
              endpoint_world=None, throw_support=40):
    return {"typeAccuracy": type_accuracy,
            "perType": {"noop": {"support": 10, "recall": 1.0, "precision": 1.0},
                        "move": {"support": 50, "recall": .9, "precision": .9},
                        "throw": {"support": throw_support, "recall": throw_recall, "precision": .8},
                        "hold": {"support": 5, "recall": .8, "precision": .8}},
            "moveHeadingErrorDegrees": move_degrees, "moveEndpointErrorWorld": endpoint_world,
            "throwAimHeadingErrorDegrees": aim_degrees, "powerMeanAbsoluteError": power_error}


def test_throw_recall_flags_detects_a_scripted_arm_collapse():
    cfg = led.configuration()
    cells_by_arm = {
        "R": {name: make_cell(type_accuracy=.95, throw_recall=.85, aim_degrees=3.0, power_error=.02)
              for name in led.COMPARATOR_NAMES},
        "E": {name: make_cell(type_accuracy=.90, throw_recall=.10, aim_degrees=4.0, power_error=.03)
              for name in led.COMPARATOR_NAMES},
        "N": {name: make_cell(type_accuracy=.90, throw_recall=.60, aim_degrees=4.0, power_error=.03)
              for name in led.COMPARATOR_NAMES},
    }
    flags = led.throw_recall_flags(cells_by_arm, cfg)
    assert flags["E"]["meanThrowRecall"] == pytest.approx(.10)
    assert flags["E"]["belowThrowRecallFlag"] is True
    assert flags["N"]["belowThrowRecallFlag"] is False


def test_metric_deltas_reports_arm_minus_reference():
    cells_by_arm = {
        "R": {name: make_cell(type_accuracy=.95, throw_recall=.85, aim_degrees=3.0, power_error=.02)
              for name in led.COMPARATOR_NAMES},
        "E": {name: make_cell(type_accuracy=.90, throw_recall=.85, aim_degrees=33.0, power_error=.20)
              for name in led.COMPARATOR_NAMES},
        "N": {name: make_cell(type_accuracy=.95, throw_recall=.85, aim_degrees=3.0, power_error=.02)
              for name in led.COMPARATOR_NAMES},
    }
    deltas = led.metric_deltas(cells_by_arm)
    assert deltas["E"]["throwAimHeadingErrorDegrees"]["delta"] == pytest.approx(30.0)
    assert deltas["E"]["throwAimHeadingErrorDegrees"]["cells"] == len(led.COMPARATOR_NAMES)
    assert deltas["N"]["throwAimHeadingErrorDegrees"]["delta"] == pytest.approx(0.0)
    assert set(deltas["E"]["throwSupport"]) == set(led.COMPARATOR_NAMES)


def test_archive_cross_check_matches_the_real_r1n_c_archive():
    cfg = led.configuration()
    arm_r_cells = {f"init-{seed}": json.loads(
        (led.TRAINING / cfg["sourceRun"] / f"seed-{seed}" / "label-error.json").read_text(encoding="utf-8"))
        for seed in cfg["initializerSeeds"]}
    rows = led.archive_cross_check(cfg, arm_r_cells)
    for seed in cfg["initializerSeeds"]:
        row = rows[str(seed)]
        # Comparing the archive against itself: every delta must be exactly zero. This alone
        # would pass even if every seed read the same file (x - x == 0 regardless of x), so
        # it is not sufficient by itself — see the distinctness check below.
        assert row["typeAccuracyDelta"] == pytest.approx(0.0)
        assert row["throwRecallDelta"] == pytest.approx(0.0)
        assert row["throwAimHeadingErrorDegreesDelta"] == pytest.approx(0.0)
    # The three archived files are not identical, so a path/seed-mapping bug that reads the
    # same file three times (or the wrong seed's file) would leave every "archived" entry
    # equal; catch that by requiring at least one metric to differ pairwise.
    archived = {seed: rows[str(seed)]["archived"] for seed in cfg["initializerSeeds"]}
    seeds = cfg["initializerSeeds"]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = archived[seeds[i]], archived[seeds[j]]
            assert (a["typeAccuracy"], a["perType"]["throw"]["support"]) != \
                   (b["typeAccuracy"], b["perType"]["throw"]["support"])


# -- live: tiny end-to-end declare/arm/aggregate and tamper detection -----------------------


def test_tiny_end_to_end_declare_arm_and_aggregate(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    led.declare(root, cfg)
    declaration = json.loads((root / "declaration.json").read_text(encoding="utf-8"))
    assert declaration["budgetBound"]["total"] == 253_440

    with pytest.raises(FileExistsError):
        led.declare(root, cfg)

    for arm in cfg["armOrder"]:
        cells = led.run_arm(root, cfg, arm)
        assert set(cells) == set(led.COMPARATOR_NAMES)
        for name in led.COMPARATOR_NAMES:
            directory = root / f"arm-{arm}" / name
            assert (directory / "episodes.jsonl").exists()
            assert (directory / "label-error.json").exists()
    with pytest.raises(FileExistsError):
        led.run_arm(root, cfg, "R")

    report = led.aggregate(root, cfg)
    assert set(report["cells"]) == {"R", "E", "N"}
    assert set(report["archiveCrossCheck"]) == {str(s) for s in cfg["initializerSeeds"]}
    assert set(report["throwRecallFlags"]) == {"E", "N"}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "arm-R/manifest.json" in manifest["artifacts"] and "report.json" in manifest["artifacts"]

    # Tamper detection reuses death_rate_ppo's seal/verify_sealed, unchanged.
    rows = root / "arm-R" / "init-97101" / "episodes.jsonl"
    rows.write_text(rows.read_text(encoding="utf-8") + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        dr.verify_sealed(root / "arm-R")
