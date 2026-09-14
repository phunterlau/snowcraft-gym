import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import full_authority_diagnostics as d
from snowgym_training.options import full_authority_train_v1 as v1


def test_configuration_matches_the_declared_budget_seeds_and_horizon():
    cfg = d.configuration()
    assert cfg["optionHorizon"] == 200 and cfg["autonomousQualificationEligible"] is False
    calibration = cfg["calibrationSeeds"][1] - cfg["calibrationSeeds"][0] + 1
    per_config = (cfg["warmStartTrainEpisodes"] + cfg["warmStartHeldOutEpisodes"]) * cfg["optionHorizon"]
    bound = (cfg["preconditionDecisions"] + 3 * calibration * cfg["optionHorizon"]
             + len(cfg["arms"]) * len(cfg["trainingRngs"]) * per_config)
    assert bound <= 521_000 <= cfg["simulatorBudget"] == 600_000
    bands = [range(cfg["calibrationSeeds"][0], cfg["calibrationSeeds"][1] + 1), *(v1.fold_seeds(cfg, i, held_out=h) for i in range(3) for h in (False, True))]
    reserved = [range(600000, 600100), range(610000, 610100)]  # E3 development seeds and floor
    seeds = [set(band) for band in bands]
    assert all(not a & b for i, a in enumerate(seeds) for b in seeds[i + 1:])
    assert all(not s & set(r) for s in seeds for r in reserved)
    assert cfg["trainingRngs"] == [98001, 98002, 98003] and cfg["bootstrapSeed"] == 981001


def test_teacher_precondition_records_a_failure_instead_of_raising(monkeypatch):
    def broken(*_args, **_kwargs):
        raise ValueError("activation objective is not an enemy cluster")

    monkeypatch.setattr(v1, "make_wrapper", broken)
    result = d.teacher_precondition(None, d.configuration())
    assert result["passed"] is False and "enemy cluster" in result["error"] and result["decisions"] == 0


def test_teacher_precondition_passes_for_the_plan_teacher_at_roster_one():
    with SnowGymBatchClient() as client:
        result = d.teacher_precondition(client, d.configuration())
    assert result == {"passed": True, "decisions": 5, "rejectedActions": 0, "error": None}


def test_potential_statistics_report_declared_and_distance_dependent_fractions():
    cfg = d.configuration()
    episodes = [{"distances": [30., 30., 20., 10.], "targetDamage": [0., 0., 0., 5.]},
                {"distances": [5., 5., 5.], "targetDamage": [0., 0., 0.]}]
    stats = d.potential_statistics(episodes, 10., cfg)
    # Episode 1 has three pre-contact transitions (the hit arrives on the fourth step); episode 2 has two.
    assert stats["preContactTransitions"] == 5
    assert stats["nonzeroShapingFraction"] == pytest.approx(3 / 5)  # gamma < 1 makes a constant nonzero Phi shape
    assert stats["potentialChangeFraction"] == pytest.approx(2 / 5)
    assert stats["earlyPotentialContactPointBiserial"] == pytest.approx(-1.)
    assert d.potential_statistics(episodes, None, cfg)["computed"] is False


def test_decision_rules_follow_the_declared_table():
    cfg = d.configuration()
    passed = {"passed": True}
    teacher = {"successFraction": .9, "rejectedActionRate": 0.}
    audit = {"moveBeyondLocalReachFraction": .5}
    warm = {"local": {"1": {"gatePassed": True}}, "global": {"1": {"gatePassed": False}}}
    pooled = {"local": {"contactFraction": .1}, "global": {"contactFraction": .9}}
    rules = d.decision_rules(passed, teacher, audit, warm, pooled, cfg)
    assert rules["teacherUsable"] and rules["labelAuditRequiresDecoderStatement"]
    assert rules["arms"]["local"]["recommendation"] == "D3"
    assert rules["arms"]["global"]["recommendation"].startswith("no PPO branch")
    no_teacher = d.decision_rules({"passed": False}, None, None, warm, pooled, cfg)
    assert no_teacher["arms"]["local"]["recommendation"] == "D2 or a separately declared teacher"
    healthy = {"local": {"1": {"gatePassed": True}}, "global": {"1": {"gatePassed": True}}}
    assert d.decision_rules(passed, teacher, audit, healthy, pooled, cfg)["arms"]["global"]["recommendation"] == \
        "D1 informative; D3 also open"


def tiny(**overrides):
    return {**d.configuration(), "optionHorizon": 6, "blockWorlds": 2, "diagnosticBlockWorlds": 2,
            "calibrationSeeds": [925000, 925002], "trainingRngs": [31], "warmStartTrainEpisodes": 2,
            "warmStartHeldOutEpisodes": 2, "warmStartEpochs": 1, "minibatchSize": 8, "bootstrapSamples": 20,
            "calibrationRows": 4, "calibrationDraws": 3, "trainSeedBase": 926000, "heldOutSeedBase": 927000,
            "simulatorBudget": 1000, **overrides}


def test_tiny_end_to_end_run_writes_the_declared_artifacts(tmp_path):
    cfg = tiny()
    report = d.execute(tmp_path / "run", cfg)
    root = tmp_path / "run"
    assert report["c1aPrecondition"]["passed"] is True
    assert report["simulatorDecisions"] <= cfg["simulatorBudget"]
    assert set(report["c3WarmStart"]) == {"local", "global"}
    for arm in ("local", "global"):
        directory = root / "c3" / arm / "31"
        for name in ("critic-warm-start.json", "critic-warm-start-arrays.npz", "initial-policy.pt",
                     "episodes.jsonl", "trajectory-distances.npz", "calibration.json"):
            assert (directory / name).exists()
    for source in ("c1b-plan-teacher", "c1c-scripted", "c2-floor"):
        rows = [json.loads(line) for line in (root / source / "episodes.jsonl").read_text().splitlines()]
        assert [r["seed"] for r in rows] == [925000, 925001, 925002]  # inclusive bounds; blocks of 2 then 1
    audit = json.loads((root / "c1b-plan-teacher/label-audit.json").read_text())
    assert sum(audit["actionTypeCounts"].values()) > 0
    declaration = json.loads((root / "declaration.json").read_text())
    assert all(entry["match"] for entry in declaration["pinnedE3Digests"].values())
    manifest = json.loads((root / "manifest.json").read_text())
    assert "report.json" in manifest["artifacts"] and "d-research.json" in manifest["artifacts"]
    assert set(report["decisionRules"]["arms"]) == {"local", "global"}
    with pytest.raises(FileExistsError):
        d.execute(root, cfg)


def test_budget_guard_stops_a_run_that_exceeds_its_cap(tmp_path):
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match="budget"):
        d.execute(tmp_path / "capped", tiny(simulatorBudget=10))
