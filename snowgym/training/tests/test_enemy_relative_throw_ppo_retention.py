import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw_ppo as s14
from snowgym_training.options import enemy_relative_throw_ppo_retention as s15
from snowgym_training.options import roster_baseline as rb

TEST_SEED_BASE = 5980000  # off-band: no real run or other test uses 5980000-5980999
# Captured at collection time, before other modules' fixtures call torch.set_num_threads(1): S14's real run used the
# process default, and an exact replay needs the same thread count as well as the same 64-world block.
DEFAULT_THREADS = torch.get_num_threads()


def tiny(**overrides):
    return {**s15.configuration(), "cohorts": (1,), "optionHorizon": 8, "blockWorlds": 2, "fidelityWorlds": 2,
            "evaluationSeedBase": TEST_SEED_BASE, "evaluationWorlds": 2, "stochasticEvalSeedBase": TEST_SEED_BASE + 500,
            "bootstrapSamples": 50, "budgetCap": 10_000, **overrides}


def write_cell(root, cohort, arm, policy, mode, rows):
    directory = s15.cell_directory(root, cohort, arm, policy, mode)
    directory.mkdir(parents=True)
    (directory / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(seed, success, lost, timed_out=False, alive=3):
    return {"seed": seed, "success": success, "unitsLostFraction": lost, "timedOut": timed_out,
            "blueAliveCount": alive}


def synthetic_root(tmp_path, cfg, final_row, initial_row):
    for cohort in cfg["cohorts"]:
        for arm in s15.ARMS:
            for mode in s15.MODES:
                write_cell(tmp_path, cohort, arm, "final", mode, [final_row(s) for s in s15.evaluation_seeds(cfg)])
                write_cell(tmp_path, cohort, arm, "initializer", mode,
                           [initial_row(s) for s in s15.evaluation_seeds(cfg)])
    return tmp_path


def summaries_with_rejection(cfg, rate):
    return {str(c): {f"{arm}/{policy}-{mode}": {"rejectedActionRate": rate, "success": .8}
                     for arm in s15.ARMS for policy in s15.POLICIES for mode in s15.MODES} for c in cfg["cohorts"]}


# -- Offline: configuration, seeds, budget (declaration §3, §7) --------------------------------


def test_configuration_and_budget_match_the_declaration():
    cfg = s15.configuration()
    assert cfg["format"] == "snowgym.m8-s15-ppo-retention-eval-config.v0"
    assert cfg["finalRun"] == "m8_s14_ppo_continuation_v0" and cfg["finalCheckpoint"] == "update-200.pt"
    assert cfg["sourceRun"] == "m8_s12_enemy_relative_throw_v0"  # prepare_policy reads the initializer from here
    assert s15.budget_bound(cfg) == {"fidelity": 153_600, "evaluation": 1_920_000, "total": 2_073_600}
    assert s15.budget_bound(cfg)["total"] <= cfg["budgetCap"] == 2_100_000
    assert cfg["autonomousQualificationEligible"] is False


def test_retention_worlds_are_fresh_and_fidelity_reuses_exactly_s14s_first_block():
    cfg = s15.configuration()
    worlds = s15.evaluation_seeds(cfg)
    assert worlds == list(range(2700000, 2700400)) and len(set(worlds)) == 400
    s14_worlds = set(s14.evaluation_seeds(s14.configuration()))
    assert not (set(worlds) & s14_worlds)
    assert not (set(worlds) & set(range(2100000, 2100400)))  # S4-S13's paired worlds
    assert s15.fidelity_seeds(cfg) == sorted(s14_worlds)[:64]


def test_fidelity_torch_seeds_reproduce_s14s_formula_and_retention_seeds_are_distinct():
    cfg = s15.configuration()
    s14_base = s14.configuration()["stochasticEvalSeedBase"]
    for cohort in cfg["cohorts"]:
        assert s15.fidelity_torch_seed(cfg, cohort, "initializer") == s14_base + 10 * cohort
        assert s15.fidelity_torch_seed(cfg, cohort, "final") == s14_base + 10 * cohort + 1
    seeds = [s15.stochastic_torch_seed(cfg, c, a, p) for c in cfg["cohorts"] for a in s15.ARMS for p in s15.POLICIES]
    assert len(set(seeds)) == len(seeds) == 12
    assert all(985100 <= s < 985400 for s in seeds)


# -- Offline: analysis and decision rules (declaration §4, §5) ----------------------------------


def test_paired_analysis_reports_every_cell_metric_and_cohort(tmp_path):
    cfg = {**s15.configuration(), "evaluationSeedBase": TEST_SEED_BASE, "evaluationWorlds": 4, "bootstrapSamples": 50}
    root = synthetic_root(tmp_path, cfg, lambda s: row(s, True, 0.), lambda s: row(s, s % 2 == 0, 1 / 3, alive=2))
    cells = s15.paired_analysis(root, cfg)
    assert set(cells) == {f"{arm}/{mode}" for arm in s15.ARMS for mode in s15.MODES}
    for cell in cells.values():
        assert set(cell["perCohort"]) == {"1", "2", "3"}
        for differences in (*cell["perCohort"].values(), cell["cohortAveraged"]):
            assert set(differences) == set(s15.METRICS)
        assert cell["cohortAveraged"]["success"]["mean"] == pytest.approx(.5)
        assert cell["cohortAveraged"]["unitsLostFraction"]["mean"] == pytest.approx(-1 / 3)
        assert cell["cohortAveraged"]["teamWipe"]["mean"] == 0.


def test_team_wipe_is_derived_from_the_alive_count():
    assert s15.metric_value(row(1, False, 1., alive=0), "teamWipe") == 1.
    assert s15.metric_value(row(1, True, 0., alive=3), "teamWipe") == 0.


def test_paired_analysis_rejects_rows_that_miss_a_declared_world(tmp_path):
    cfg = {**s15.configuration(), "cohorts": (1,), "evaluationSeedBase": TEST_SEED_BASE, "evaluationWorlds": 3,
           "bootstrapSamples": 50}
    root = synthetic_root(tmp_path, cfg, lambda s: row(s, True, 0.), lambda s: row(s, True, 0.))
    path = s15.cell_directory(root, 1, "easy", "final", "stochastic") / "episodes.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:2]))
    with pytest.raises(RuntimeError):
        s15.paired_analysis(root, cfg)


def difference(low, high, mean=None):
    return {"mean": (low + high) / 2 if mean is None else mean, "interval95": [low, high]}


def test_cell_status_uses_the_harmful_direction_of_each_metric():
    neutral = {"unitsLostFraction": difference(-.02, .02), "timedOut": difference(-.01, .01),
               "teamWipe": difference(-.01, .01)}
    assert s15.cell_status({**neutral, "success": difference(-.04, .03)}, .05)["status"] == "non-inferior"
    assert s15.cell_status({**neutral, "success": difference(-.12, -.06)}, .05)["status"] == "regressed"
    assert s15.cell_status({**neutral, "success": difference(-.09, .02)}, .05)["status"] == "inconclusive"
    # timeouts harm by rising: a confident increase inside the margin is non-inferior but flagged
    within = s15.cell_status({**neutral, "success": difference(-.01, .01), "timedOut": difference(.01, .04)}, .05)
    assert within == {"status": "non-inferior", "confidentHarm": ["timedOut"]}
    assert s15.cell_status({**neutral, "success": difference(-.01, .01), "timedOut": difference(.02, .07)},
                           .05)["status"] == "regressed"
    # a large success GAIN is never harm
    assert s15.cell_status({**neutral, "success": difference(.3, .5)}, .05)["status"] == "non-inferior"


def test_decision_rules_follow_the_declared_precedence(tmp_path):
    cfg = {**s15.configuration(), "evaluationSeedBase": TEST_SEED_BASE, "evaluationWorlds": 6, "bootstrapSamples": 50}
    same = synthetic_root(tmp_path / "same", cfg, lambda s: row(s, True, 0.), lambda s: row(s, True, 0.))
    cells = s15.paired_analysis(same, cfg)
    assert s15.decision_rules(cells, summaries_with_rejection(cfg, 0.), cfg)["outcome"] == "retained"
    # the rejection gate alone turns an otherwise-retained result into a regression
    failing = s15.decision_rules(cells, summaries_with_rejection(cfg, .002), cfg)
    assert failing["outcome"] == "regressed" and not failing["rejectionGate"]["passed"]
    worse = synthetic_root(tmp_path / "worse", cfg, lambda s: row(s, False, 1., timed_out=True, alive=0),
                           lambda s: row(s, True, 0.))
    rules = s15.decision_rules(s15.paired_analysis(worse, cfg), summaries_with_rejection(cfg, 0.), cfg)
    assert rules["outcome"] == "regressed"
    assert {flag["cohort"] for flag in rules["cohortFlags"]} == {"1", "2", "3"}


def test_one_cohort_beyond_the_margin_is_flagged_without_failing_the_averaged_cells():
    cfg = s15.configuration()
    fine = {name: difference(-.01, .01) for name in s15.METRICS}
    bad = {**fine, "unitsLostFraction": difference(.06, .09)}
    cells = {f"{arm}/{mode}": {"cohortAveraged": fine, "perCohort": {"1": fine, "2": bad if arm == "easy" else fine,
                                                                      "3": fine}}
             for arm in s15.ARMS for mode in s15.MODES}
    rules = s15.decision_rules(cells, summaries_with_rejection(cfg, 0.), cfg)
    assert rules["outcome"] == "retained-with-cohort-regression"
    assert {(f["cell"], f["cohort"], f["metric"]) for f in rules["cohortFlags"]} == {
        ("easy/deterministic", "2", "unitsLostFraction"), ("easy/stochastic", "2", "unitsLostFraction")}


# -- Offline against the real archives: policies (declaration §1) --------------------------------


def test_load_policies_rebuilds_s14s_reference_and_the_update_200_checkpoint():
    cfg = s15.configuration()
    policies = s15.load_policies(cfg, 1)
    assert isinstance(policies["final"], s14.FullAuthorityPolicyV1EnemyThrowPPO)
    assert not policies["final"].training and not policies["initializer"].training
    assert float(policies["initializer"].throw_offset_log_std.mean()) == pytest.approx(cfg["offsetLogStdTarget"])
    _, reference = s14.prepare_policy(cfg, 1, sigma_scale=.5)
    for name, value in reference.state_dict().items():
        assert torch.equal(policies["initializer"].state_dict()[name], value)
    stored = torch.load(s15.final_checkpoint_path(cfg, 1), map_location="cpu", weights_only=True)["model"]
    for name, value in stored.items():
        assert torch.equal(policies["final"].state_dict()[name], value)
    assert dr.parameter_distance(policies["final"], policies["initializer"]) > 0


def test_archived_rows_cover_s14s_evaluation_worlds():
    cfg = s15.configuration()
    rows = s15.archived_rows(cfg, 1, "final", "stochastic")
    assert sorted(rows) == s14.evaluation_seeds(s14.configuration())


# -- Live ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


@pytest.fixture
def default_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(DEFAULT_THREADS)
    yield
    torch.set_num_threads(previous)


def test_deterministic_replay_of_s14s_first_block_matches_its_archived_rows_exactly(client, default_threads):
    """The whole 64-world block: a 4-world block reproduces outcomes but differs in the sixth decimal of distances,
    because batched float math in the policy depends on batch size."""
    cfg = s15.configuration()
    policies = s15.load_policies(cfg, 1)
    seeds = s15.fidelity_seeds(cfg)
    rows = rb.collect_cell(client, seeds, cfg, "normal", model=policies["final"], mode="deterministic",
                           account=lambda n: None)
    archived = s15.archived_rows(cfg, 1, "final", "deterministic")
    assert rows == [archived[s] for s in seeds]


def test_tiny_evaluation_writes_every_cell_and_aggregates(tmp_path, client):
    cfg = tiny()
    policies = s15.load_policies(cfg, 1)
    summaries = {"1": s15.evaluate_cohort(client, cfg, 1, policies, tmp_path, account=lambda n: None)}
    assert set(summaries["1"]) == {f"{a}/{p}-{m}" for a in s15.ARMS for p in s15.POLICIES for m in s15.MODES}
    cells = s15.paired_analysis(tmp_path, cfg)
    rules = s15.decision_rules(cells, summaries, cfg)
    assert rules["outcome"] in {"retained", "retained-with-cohort-regression", "regressed", "inconclusive"}
    assert s15.load_cell(tmp_path, 1, "random", "final", "stochastic").keys() == set(s15.evaluation_seeds(cfg))


def test_a_fidelity_mismatch_stops_before_any_new_world_and_seals(tmp_path, monkeypatch):
    cfg = tiny()
    monkeypatch.setattr(s15, "archived_rows", lambda *args: {})
    report = s15.run(tmp_path / "run", cfg)
    assert report["fidelityPassed"] is False and report["decisionRules"]["outcome"] == "fidelity-failed"
    assert report["evaluationSimulatorDecisions"] == 0 and report["summaries"] == {}
    assert not (tmp_path / "run" / "cohort-1").exists()
    assert dr.verify_sealed(tmp_path / "run")
    with pytest.raises(FileExistsError):
        s15.declare(tmp_path / "run", cfg)
