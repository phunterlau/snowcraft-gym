import json

import numpy as np
import pytest

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import mixture_imitation as mi
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options.full_authority_diagnostics import TRAINING


def tiny(**overrides):
    return {**ri.configuration(), "optionHorizon": 8, "roundEpisodes": 8, "fits": 2, "stepsPerFit": 20,
            "imitationMinibatch": 32, "evaluationEpisodes": 4, "blockWorlds": 4, "roundBlockWorlds": 4,
            "evaluationBlockWorlds": 4, "warmStartTrainEpisodes": 16, "warmStartHeldOutEpisodes": 8,
            "warmStartEpochs": 2, "bootstrapSamples": 50, "budgetCap": 400_000, "pairedEvalWorlds": 4,
            "optimizerSeeds": (97101,), **overrides}


def cell(success, lost=0.0):
    return {"success": success, "unitsLostFraction": lost}


# -- pure ---------------------------------------------------------------------------------


def test_the_declared_bands_are_disjoint_and_the_paired_worlds_are_the_s4_band():
    cfg = ri.configuration()
    bands = ri.seed_bands(cfg)
    ordered = sorted((low, high, name) for name, low, high in bands)
    for (_, high, first), (low, _, second) in zip(ordered, ordered[1:]):
        assert high < low, (first, second)
    paired = next(b for b in bands if b[0] == "paired-eval")
    assert (paired[1], paired[2]) == (2100000, 2100399)
    assert all(name != "paired-eval" for name, _, _ in ri.training_bands(cfg))


def test_no_training_band_is_used_anywhere_else_in_the_repository_json():
    cfg = ri.configuration()
    ranges = [(low, high) for _, low, high in ri.training_bands(cfg)]

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
        if any(part in {".git", "node_modules", ".venv", "dist", "m8_s5_roster_imitation_v0"}
               for part in path.parts) or path.stat().st_size > 5_000_000:
            continue  # this run's own archive necessarily records its bands
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if any(low <= n <= high for n in integers(data) for low, high in ranges):
            collisions.append(str(path))
    assert collisions == []


def test_critic_fold_sizes_are_consistent_for_the_real_configuration():
    mi.validate_critic_fold_sizes(ri.configuration())


def test_budget_bound_stays_under_the_declared_cap_and_matches_its_parts():
    cfg = ri.configuration()
    bound = ri.budget_bound(cfg)
    assert bound["total"] == sum(v for k, v in bound.items() if k != "total")
    assert bound["total"] <= cfg["budgetCap"]


def test_paired_difference_is_learner_minus_teacher_by_world_and_refuses_mismatched_worlds():
    learner = [{"seed": 1, "s": 1}, {"seed": 2, "s": 0}, {"seed": 3, "s": 0}, {"seed": 4, "s": 0}]
    teacher = [{"seed": 1, "s": 1}, {"seed": 2, "s": 1}, {"seed": 3, "s": 1}, {"seed": 4, "s": 1}]
    difference = ri.paired_difference(learner, teacher, lambda r: r["s"], tiny())
    assert difference["mean"] == pytest.approx(-0.75)
    assert difference["interval"][0] <= -0.75 <= difference["interval"][1]
    with pytest.raises(ValueError):
        ri.paired_difference(learner, list(reversed(teacher)), lambda r: r["s"], tiny())


def test_readings_apply_the_declared_thresholds():
    cfg = ri.configuration()
    good = {"97101": {"normal": cell(0.9)}, "97102": {"normal": cell(0.4)}, "97103": {"normal": cell(0.3)}}
    assert ri.readings(good, cfg)["outcome"] == "viable-and-reliable"
    mixed = {"97101": {"normal": cell(1.0)}, "97102": {"normal": cell(0.0)}, "97103": {"normal": cell(0.14)}}
    assert ri.readings(mixed, cfg)["outcome"] == "viable-but-unreliable"  # mean 0.38, like R1n-h's M
    weak = {"97101": {"normal": cell(0.0)}, "97102": {"normal": cell(0.1)}, "97103": {"normal": cell(0.2)}}
    assert ri.readings(weak, cfg)["outcome"] == "not-viable"


def test_teacher_rows_come_from_the_s4_archive_in_seed_order():
    rows = ri.teacher_rows("normal", 5)
    assert [r["seed"] for r in rows] == list(range(2100000, 2100005))
    with pytest.raises(ValueError):
        ri.teacher_rows("normal", 10_000)


# -- live: a tiny end-to-end run at 3v3 --------------------------------------------------


def test_tiny_end_to_end_run_trains_measures_pairs_and_seals(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    report = ri.run(root, cfg)
    assert set(report["pairedEval"]) == {"97101"}
    for arm in rb.ARMS:
        summary = report["pairedEval"]["97101"][arm]
        assert summary["episodes"] == 4 and "successMinusTeacher" in summary and "unitsLostMinusTeacher" in summary
        rows = [json.loads(l) for l in (root / "paired-eval" / "seed-97101" / arm / "episodes.jsonl").read_text().splitlines()]
        assert [r["seed"] for r in rows] == list(range(2100000, 2100004)) and rows[0]["assignedUnits"] == 3
    assert report["readings"]["outcome"] in {"viable-and-reliable", "viable-but-unreliable", "not-viable"}
    assert set(report["criticWarmStart"]["97101"]) == {"predictiveR2", "timeOnlyR2", "untrainedPredictiveR2", "gatePassed"}
    assert dr.verify_sealed(root)
    label_error = json.loads((root / "condition-M" / "seed-97101" / "label-error.json").read_text())
    assert label_error["units"] % 3 == 0 and label_error["units"] > 0, "three units per decision reach the labeler"
    with pytest.raises(FileExistsError):
        ri.declare(root, cfg)
