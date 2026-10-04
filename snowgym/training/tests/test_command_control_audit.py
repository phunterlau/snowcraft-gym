
import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import command_control_audit as s16
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw_ppo_retention as s15
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options.plans import teacher_option_plan

TEST_SEED_BASE = 5984000  # off-band: no real run or other test uses 5984000-5984999
DEFAULT_THREADS = torch.get_num_threads()  # see test_enemy_relative_throw_ppo_retention.py


# -- Offline: configuration, panel, seeds (declaration §2, §3, §8) -----------------------------------------


def test_configuration_and_budget_match_the_declaration():
    cfg = s16.configuration()
    assert s16.budget_bound(cfg) == {"gate": 153_600, "teacher": 307_200, "learned": 4_608_000,
                                     "total": 5_068_800}
    assert s16.budget_bound(cfg)["total"] <= cfg["budgetCap"] == 5_100_000
    assert cfg["commandHorizon"] == cfg["optionHorizon"] == 200 and cfg["blockWorlds"] == 64
    assert cfg["autonomousQualificationEligible"] is False


def test_panel_is_fresh_cycles_spreads_and_avoids_the_reserved_qualification_panel():
    cfg = s16.configuration()
    worlds = s16.panel(cfg)
    seeds = [seed for _, seed, _ in worlds]
    assert seeds == list(range(2720000, 2720192))
    assert [spread for _, _, spread in worlds][:6] == [10, 20, 30, 10, 20, 30]
    assert {d: sum(w[2] == d for w in worlds) for d in s16.SPREADS} == {10: 64, 20: 64, 30: 64}
    low, high = cfg["reservedQualificationSeeds"]
    used = set(seeds) | set(s15.fidelity_seeds(cfg))
    assert not any(low <= s <= high for s in used)
    assert not used & set(range(2700000, 2700400)) and not set(seeds) & set(range(2600000, 2600400))


def test_stochastic_torch_seeds_are_distinct_and_in_the_declared_range():
    cfg = s16.configuration()
    seeds = [s16.stochastic_torch_seed(cfg, c, p, k, t) for c in cfg["cohorts"] for p in s16.POLICIES
             for k in s16.CONDITIONS for t in s16.TASKS]
    assert len(seeds) == len(set(seeds)) == 3 * 2 * 5 * 2
    assert min(seeds) == 986100 and max(seeds) == 986383


def test_execution_selects_follow_the_declared_conditions():
    cfg = s16.configuration()
    shuffled = s16.shuffle_table(cfg)
    assert shuffled == s16.shuffle_table(cfg)
    mismatch = np.mean([shuffled[key] != key[1] for key in shuffled])
    assert .4 < mismatch < .6
    assert s16.execution_select("correct", "leftmost", 0, shuffled) == "leftmost"
    assert s16.execution_select("zero-plan", "rightmost", 0, shuffled) == "rightmost"
    assert s16.execution_select("other", "leftmost", 0, shuffled) == "rightmost"
    assert s16.execution_select("canonical", "rightmost", 0, shuffled) == "nearest"
    assert s16.execution_select("shuffled", "leftmost", 7, shuffled) == shuffled[(7, "leftmost")]
    with pytest.raises(ValueError):
        s16.execution_select("largest", "leftmost", 0, shuffled)


def test_engage_plan_changes_only_the_selector_and_nearest_is_the_training_plan():
    canonical, _ = teacher_option_plan("engage")
    assert s16.engage_plan("nearest") == canonical
    left = s16.engage_plan("leftmost")
    assert left["groups"][0]["order"]["objective"] == {"kind": "enemy_cluster", "select": "leftmost"}
    left["groups"][0]["order"]["objective"]["select"] = "nearest"
    assert left == canonical


def test_scenarios_place_three_singletons_and_the_default_layout_matches_the_generated_spawns():
    layout = s16.scenario_for(20)
    assert layout["redSpawns"] == [{"x": 30, "y": -20}, {"x": 30, "y": 0}, {"x": 30, "y": 20}]
    assert layout["redController"] == "scripted" and layout["redDifficulty"] == "normal"
    assert layout["blueUnits"] == layout["redUnits"] == 3
    assert s16.scenario_for(None)["redSpawns"] == [{"x": 30, "y": -5}, {"x": 30, "y": 0}, {"x": 30, "y": 5}]


# -- Offline: scoring (declaration §4) --------------------------------------------------------------------------


def test_flank_order_scores_ties_and_missing_crossings_as_declared():
    assert s16.flank_order(10, 20) == 1.
    assert s16.flank_order(20, 10) == 0.
    assert s16.flank_order(15, 15) == .5
    assert s16.flank_order(None, None) == .5
    assert s16.flank_order(30, None) == 1.
    assert s16.flank_order(None, 30) == 0.


def test_flank_contrast_is_signed_and_zero_without_damage():
    assert s16.flank_contrast(60., 20.) == pytest.approx(.5)
    assert s16.flank_contrast(0., 40.) == -1.
    assert s16.flank_contrast(0., 0.) == 0.


def test_bearing_closest_picks_the_enemy_nearest_the_throw_heading_and_skips_the_dead():
    enemies = [{"id": 4, "x": 30, "y": -20, "alive": True}, {"id": 5, "x": 30, "y": 0, "alive": True},
               {"id": 6, "x": 30, "y": 20, "alive": True}]
    assert s16.bearing_closest((0, 0), (10, 6), enemies) == 6
    assert s16.bearing_closest((0, 0), (10, 0.5), enemies) == 5
    enemies[1]["alive"] = False
    assert s16.bearing_closest((0, 0), (10, 0.5), enemies) in (4, 6)
    assert s16.bearing_closest((0, 0), (0, 0), enemies) is None


def delta(mean, low, high):
    return {"mean": mean, "interval95": [low, high]}


def test_classification_follows_the_declared_table():
    cfg = s16.configuration()
    assert s16.classify(delta(.5, .3, .7), .9, cfg) == "controllable"
    assert s16.classify(delta(.3, .1, .5), .9, cfg) == "sensitive"
    assert s16.classify(delta(.45, -.01, .9), .9, cfg) == "insensitive"
    assert s16.classify(delta(-.2, -.3, -.1), .9, cfg) == "counter-sensitive"
    assert s16.classify(delta(.9, .8, 1.), .4, cfg) == "uninformative"


def audit_row(seed, order, contrast, *, requested=10, mirror=20, centre_first=False, lost=0.):
    return {"seed": seed, "flankOrder": order, "flankContrast": contrast, "requestedCross": requested,
            "mirrorCross": mirror, "centreIds": [5], "firstOverallIds": [5] if centre_first else [6],
            "unitsLostFraction": lost, "blueAliveCount": 3, "rejectedActions": 0, "totalActions": 30,
            "aim": {"requested": 2, "mirror": 1, "centre": 1, "other": 0}, "lateralTowardRequested": 1.,
            "stopReason": "requested-complete"}


def synthetic_cells(worlds, orders):
    return {condition: {task: [audit_row(seed, order, order - .5) for _, seed, _ in worlds] for task in s16.TASKS}
            for condition, order in orders.items()}


def test_contrasts_are_world_paired_overall_and_per_spread():
    cfg = {**s16.configuration(), "worlds": 6, "bootstrapSamples": 50}
    worlds = s16.panel(cfg)
    cells = synthetic_cells(worlds, {"correct": 1., "other": 0., "canonical": .5, "shuffled": .5, "zero-plan": .5})
    out = s16.contrasts(cells, worlds, cfg, s16.CONDITIONS)
    assert set(out) == {"other", "canonical", "shuffled", "zero-plan"}
    assert set(out["other"]) == {"all", "d10", "d20", "d30"}
    assert out["other"]["all"]["order"]["mean"] == 1.
    assert out["shuffled"]["d20"]["contrast"]["mean"] == .5


def test_condition_summary_reports_order_by_side_aim_shares_and_stop_reasons():
    worlds = s16.panel({**s16.configuration(), "worlds": 3})
    cells = synthetic_cells(worlds, {"correct": 1.})
    cells["correct"]["rightmost"] = [audit_row(seed, 0., -.5, centre_first=True) for _, seed, _ in worlds]
    summary = s16.condition_summary(cells["correct"], worlds)
    assert summary["order"] == .5 and summary["orderBySide"] == {"leftmost": 1., "rightmost": 0.}
    assert summary["centreFirst"] == .5 and summary["informative"] == 1.
    assert summary["aimShares"]["requested"] == pytest.approx(.5)
    assert summary["stopReasons"]["requested-complete"] == 6


def test_mirror_identity_counts_pairs_where_other_does_not_replay_the_mirrored_correct_cell():
    worlds = s16.panel({**s16.configuration(), "worlds": 4})
    cells = {c: {t: [audit_row(seed, 1. if c == "correct" else 0., 0.) for _, seed, _ in worlds] for t in s16.TASKS}
             for c in ("correct", "other")}
    cells["canonical"] = {"leftmost": [audit_row(seed, 1., 0.) for _, seed, _ in worlds],
                          "rightmost": [audit_row(seed, 0., 0.) for _, seed, _ in worlds]}
    assert s16.mirror_identity(cells) == {"otherVsCorrect": 0., "canonical": 0.}
    cells["other"]["leftmost"][0]["flankOrder"] = .5
    cells["canonical"]["rightmost"][1]["flankOrder"] = 1.
    assert s16.mirror_identity(cells) == {"otherVsCorrect": 1 / 8, "canonical": 2 / 8}


def test_averaged_cells_average_each_world_over_policies():
    worlds = s16.panel({**s16.configuration(), "worlds": 3})
    first = synthetic_cells(worlds, {"correct": 1., "other": 0.})
    second = synthetic_cells(worlds, {"correct": 0., "other": 0.})
    averaged = s16.averaged_cells([first, second])
    assert {r["flankOrder"] for r in averaged["correct"]["leftmost"]} == {.5}


def test_load_cells_rejects_a_cell_missing_a_declared_world(tmp_path):
    worlds = s16.panel({**s16.configuration(), "worlds": 3})
    for condition in ("correct",):
        for task in s16.TASKS:
            rows = [audit_row(seed, 1., .5) for _, seed, _ in worlds]
            s16.write_rows(tmp_path / condition / task, rows if task == "leftmost" else rows[:2])
    with pytest.raises(RuntimeError):
        s16.load_cells(tmp_path, ("correct",), worlds)


# -- Live ----------------------------------------------------------------------------------------------------------


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


def test_preview_grounds_the_three_selectors_to_three_distinct_singletons_at_every_spread(client):
    cfg = s16.configuration()
    worlds = [(i, TEST_SEED_BASE + i, d) for i, d in enumerate(s16.SPREADS)]
    wrapper = s16.CommandAuditBatch(v1.SelectiveBatchEnv(3, client=client, observation_version=3), gamma=cfg["gamma"])
    seeds = [seed for _, seed, _ in worlds]
    wrapper.reset(seeds, [s16.scenario_for(d) for _, _, d in worlds], [f"t-{s}" for s in seeds],
                  [s16.engage_plan("leftmost")] * 3, [v1.engage_spec(cfg)] * 3)
    before = list(wrapper.environment.state_hashes)
    ids = {select: s16.preview_ids(wrapper, seeds, select, "test") for select in ("leftmost", "rightmost", "nearest")}
    assert list(wrapper.environment.state_hashes) == before
    for index in range(3):
        assert all(len(ids[select][index]) == 1 for select in ids)
        assert len({ids[select][index][0] for select in ids}) == 3
        assert wrapper.trackers[index].activated_target_ids == ids["leftmost"][index]
    enemies = {u["id"]: u["y"] for u in wrapper.environment.raw_observations[0]["enemies"]}
    assert enemies[ids["nearest"][0][0]] == 0


def test_runner_reproduces_s14s_first_block_exactly_on_the_default_layout(client, default_threads):
    cfg = s16.configuration()
    policies = s15.load_policies(cfg, 1)
    worlds = [(i, seed, None) for i, seed in enumerate(s15.fidelity_seeds(cfg))]
    rows = s16.run_cell(client, cfg, worlds, requested_select="nearest", mirror_select=None,
        execution_selects=["nearest"] * len(worlds), choose=s16.model_chooser(policies["final"], True),
        source="test", account=lambda n: None)
    assert s16.gate_compare(rows, s15.archived_rows(cfg, 1, "final", "deterministic")) == []


def test_teacher_finishes_the_requested_flank_before_its_mirror_more_often_than_under_the_other_command(client):
    cfg = s16.configuration()
    worlds = [(i, TEST_SEED_BASE + 10 + i, 20) for i in range(4)]
    orders = {}
    for condition, select in (("correct", "leftmost"), ("other", "rightmost")):
        rows = s16.run_cell(client, cfg, worlds, requested_select="leftmost", mirror_select="rightmost",
            execution_selects=[select] * 4, choose=s16.teacher_chooser, source=f"test-{condition}",
            account=lambda n: None)
        orders[condition] = np.mean([r["flankOrder"] for r in rows])
        assert all(r["requestedIds"] != r["mirrorIds"] for r in rows)
        assert all(r["stopReason"] in ("requested-complete", "blue-wiped", "environment-done", "horizon") for r in rows)
    assert orders["correct"] > orders["other"]


def test_zero_plan_removes_only_the_command_channels_and_does_not_mutate_the_observation(client):
    cfg = s16.configuration()
    policies = s15.load_policies(cfg, 1)
    wrapper = s16.CommandAuditBatch(v1.SelectiveBatchEnv(2, client=client, observation_version=3), gamma=cfg["gamma"])
    seeds = [TEST_SEED_BASE + 20, TEST_SEED_BASE + 21]
    observation, _ = wrapper.reset(seeds, [s16.scenario_for(20)] * 2, [f"z-{s}" for s in seeds],
                                   [s16.engage_plan("leftmost")] * 2, [v1.engage_spec(cfg)] * 2)
    rows = s16.tensor_dict(observation)
    original = {key: value.clone() for key, value in rows.items()}
    model = policies["final"]
    with torch.no_grad():
        stripped, *_ = s16.ZeroPlan(model).act(rows, deterministic=True)
        manual = {k: (torch.zeros_like(v) if k in s16.ZEROED_PLAN_KEYS else v) for k, v in rows.items()}
        expected, *_ = model.act(manual, deterministic=True)
        features = model.features(rows) - model.features(manual)
    assert all(torch.equal(rows[k], original[k]) for k in rows)
    for key in expected:
        assert torch.equal(stripped[key], expected[key])
    assert float(rows["plan_role_state"].abs().sum()) > 0
    own = rows["allies"].shape[-1]
    pools = len(model.encoders) * 2 * 16
    assert float(features[..., :own + pools].abs().max()) == 0.  # physical features untouched
    assert float(features[..., own + pools:].abs().max()) > 0.  # only directive/role channels change


def test_tiny_run_reaches_aggregate_and_seals_with_every_learned_cell(tmp_path, client, monkeypatch):
    cfg = {**s16.configuration(), "cohorts": (1,), "worlds": 3, "worldSeedBase": TEST_SEED_BASE + 100,
           "optionHorizon": 8, "commandHorizon": 8, "earlyWindow": 4, "lateralDecision": 2, "blockWorlds": 3,
           "bootstrapSamples": 50, "teacherOrderMin": -1., "teacherDeltaLowerMin": -1., "teacherSpreadOrderMin": -1.,
           "budgetCap": 100_000}
    monkeypatch.setattr(s16, "regression_gate", lambda *args: {"stub": {"exact": True}})
    report = s16.run(tmp_path / "run", cfg)
    assert report["outcome"] == "complete" and report["regressionGate"]["passed"]
    assert set(report["learned"]["policies"]) == {f"c1-{p}-{m}" for p in s16.POLICIES for m in s16.MODES}
    for entry in report["learned"]["policies"].values():
        assert set(entry["summaries"]) == set(s16.CONDITIONS) and entry["class"] in s16.CLASSES
    assert set(report["teacher"]["summaries"]) == set(s16.TEACHER_CONDITIONS)
    assert set(report["teacher"]["mirrorIdentityViolation"]) == {"otherVsCorrect", "canonical"}
    assert set(report["predictions"]) >= {"1-teacherGatePasses", "2-noLearnedControllable"}
    assert dr.verify_sealed(tmp_path / "run")
    with pytest.raises(FileExistsError):
        s16.declare(tmp_path / "run", cfg)
