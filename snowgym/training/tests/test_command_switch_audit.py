import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import command_control_audit as s16
from snowgym_training.options import command_switch_audit as s18
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw as ert
from snowgym_training.options import enemy_relative_throw_ppo_retention as s15
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options import roster_imitation_repair as rp
from snowgym_training.options.full_authority_diagnostics import TRAINING
from snowgym_training.options.reservoir import file_digest

TEST_SEED_BASE = 5987000  # off-band: no real run or other test uses 5987000-5987999
DEFAULT_THREADS = torch.get_num_threads()  # see test_enemy_relative_throw_ppo_retention.py


def integers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from integers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from integers(item)


# -- Offline: configuration, panel, seeds (declaration §2, §9) --------------------------------------------------------


def test_configuration_and_budget_match_the_declaration():
    cfg = s18.configuration()
    assert s18.budget_bound(cfg) == {"gate": 460_800, "teacher": 768_000, "learned": 9_216_000, "total": 10_444_800}
    assert s18.budget_bound(cfg)["total"] <= cfg["budgetCap"] == 10_450_000
    assert cfg["switchTimes"] == (30, 50, 60) and cfg["postWindow"] == 40 and cfg["spreads"] == (20, 30)
    assert cfg["autonomousQualificationEligible"] is False


def test_panel_is_fresh_with_two_spreads_and_avoids_every_earlier_and_reserved_band():
    cfg = s18.configuration()
    worlds = s16.panel(cfg)
    seeds = {seed for _, seed, _ in worlds}
    assert sorted(seeds) == list(range(2750000, 2750192))
    assert {d: sum(w[2] == d for w in worlds) for d in (20, 30)} == {20: 96, 30: 96}
    low, high = cfg["reservedQualificationSeeds"]
    assert not any(low <= s <= high for s in seeds) and (low, high) == (2760000, 2760399)
    for band in (range(2720000, 2720192), range(2730000, 2730400), range(2740000, 2740200), range(2700000, 2700400)):
        assert not seeds & set(band)


def test_stochastic_torch_seeds_are_shared_across_branches_distinct_across_cells_and_in_range():
    cfg = s18.configuration()
    seeds = [s18.stochastic_torch_seed(cfg, c, p, k, t) for c in cfg["cohorts"] for p in s18.POLICIES
             for k in (*cfg["switchTimes"], None) for t in s18.TASKS]
    assert len(seeds) == len(set(seeds)) == 3 * 2 * 4 * 2
    assert min(seeds) == 988100 and max(seeds) == 988363


def test_configuration_integers_avoid_every_repository_json_scanner_band():
    bands = [(2100000, 2100399), (630000, 630119)]
    bands += [(low, high) for _, low, high in ri.training_bands(ri.configuration())]
    bands += [(low, high) for _, low, high in rp.seed_bands(rp.configuration())]
    bands += [(low, high) for _, low, high in ert.seed_bands(ert.configuration())]
    cfg = s18.configuration()
    assert [n for n in integers({"c": cfg, "b": s18.budget_bound(cfg)}) for lo, hi in bands if lo <= n <= hi] == []


def test_cells_cover_every_switch_branch_and_one_reference_per_task():
    cells = s18.cells_for(s18.configuration())
    assert len(cells) == 3 * 2 * 3 + 2
    assert {(t, None, "keep") for t in s18.TASKS} <= set(cells)
    assert {b for _, k, b in cells if k is not None} == set(s18.BRANCHES)


def test_reused_s16_module_is_byte_identical_to_the_sealed_implementation():
    sealed = json.loads((TRAINING / "runs/m8_s16_command_control_audit_v0/declaration.json").read_text())
    assert file_digest(TRAINING / "src/snowgym_training/options/command_control_audit.py") == \
        sealed["implementationDigest"]


# -- Offline: analysis (declaration §6.3, §7) ------------------------------------------------------------------------


def prefix(eligible=True, state="h", decision=30):
    return {"reachedK": True, "stateHash": state, "decision": decision, "crossed": [], "flankHealth": [100., 100.],
            "groundings": [[4], [6]], "eligible": eligible}


def row(seed, contrast, order, *, eligible=True, lost=0., state="h"):
    return {"seed": seed, "postContrast": contrast, "flankOrder": order, "unitsLostFraction": lost, "blueAliveCount": 3,
            "prefix": prefix(eligible, state), "postAim": {"requested": 1, "mirror": 1, "centre": 0, "other": 0},
            "lateralTowardRequested": 1., "requestedLatency": 20, "requestedCross": 50, "stopReason": "horizon"}


def synthetic_cells(cfg, values, *, ineligible=()):
    worlds = s16.panel(cfg)
    cells = {}
    for task, k, branch in s18.cells_for(cfg):
        contrast, order = values.get(branch, (0., .5))
        cells[(task, k, branch)] = {seed: row(seed, contrast, order, eligible=seed not in ineligible)
                                    for _, seed, _ in worlds}
    return cells


def test_contrasts_use_only_eligible_pairs_and_switch_minus_reactivate_is_primary():
    cfg = {**s18.configuration(), "worlds": 6, "bootstrapSamples": 50, "minEligibleWorlds": 2}
    cells = synthetic_cells(cfg, {"keep": (-1., 0.), "reactivate": (-.9, 0.), "switch": (.8, 1.)},
                            ineligible={2750000})
    analysis = s18.analyse_group(cells, cfg)
    entry = analysis["k30"]
    assert entry["eligibleWorlds"] == 5 and entry["prefixMismatches"] == 0
    assert entry["contrasts"]["switch-reactivate"]["postContrast"]["mean"] == pytest.approx(1.7)
    assert entry["contrasts"]["reactivate-keep"]["postContrast"]["mean"] == pytest.approx(.1)
    assert entry["contrasts"]["switch-reactivate"]["flankOrder"]["mean"] == pytest.approx(1.)
    assert set(entry["bySide"]) == set(s18.TASKS)


def test_prefix_mismatch_is_counted_per_branch_and_world():
    cfg = {**s18.configuration(), "worlds": 4}
    cells = synthetic_cells(cfg, {})
    cells[("leftmost", 30, "switch")][2750001]["prefix"]["stateHash"] = "different"
    cells[("rightmost", 30, "reactivate")][2750002]["prefix"]["crossed"] = [[4, 29]]
    assert s18.prefix_mismatches(cells, 30) == 2 and s18.prefix_mismatches(cells, 50) == 0


def test_classification_is_relative_to_the_teacher_and_guards_small_samples():
    cfg = s18.configuration()

    def entry(mean, low, high, worlds=100):
        return {"eligibleWorlds": worlds, "contrasts": {"switch-reactivate": {"postContrast": {
            "mean": mean, "interval95": [low, high]}}}}
    assert s18.classify(entry(1.2, .9, 1.4), 1.8, cfg) == "redirects"
    assert s18.classify(entry(.5, .2, .8), 1.8, cfg) == "partial"
    assert s18.classify(entry(.1, -.2, .4), 1.8, cfg) == "ignores"
    assert s18.classify(entry(-.5, -.8, -.2), 1.8, cfg) == "counter"
    assert s18.classify(entry(1.6, 1.4, 1.8, worlds=47), 1.8, cfg) == "insufficient"


def test_teacher_validity_needs_enough_worlds_and_a_high_lower_bound():
    cfg = s18.configuration()

    def entry(low, worlds=100):
        return {"eligibleWorlds": worlds, "contrasts": {"switch-reactivate": {"postContrast": {
            "mean": low + .2, "interval95": [low, low + .4]}}}}
    teacher = {"k30": entry(1.5), "k50": entry(.79), "k60": entry(1.2, worlds=40)}
    assert s18.teacher_validity(teacher, cfg) == {30: True, 50: False, 60: False}


def test_cohort_averaging_uses_pairs_eligible_for_every_policy():
    cfg = {**s18.configuration(), "worlds": 6, "bootstrapSamples": 50, "minEligibleWorlds": 2}
    first = synthetic_cells(cfg, {"reactivate": (-1., 0.), "switch": (1., 1.)}, ineligible={2750000})
    second = synthetic_cells(cfg, {"reactivate": (-1., 0.), "switch": (0., .5)}, ineligible={2750001})
    out = s18.averaged([first, second], cfg)["k30"]
    assert out["commonEligibleWorlds"] == 4
    assert out["switch-reactivate"]["postContrast"]["mean"] == pytest.approx(1.5)


# -- Live (off-band seeds only, except the replay check on S16's archived development panel) -------------------------


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


def test_replay_mode_reproduces_s16s_archived_first_block(client, default_threads):
    cfg = s18.configuration()
    policies = s15.load_policies(cfg, 1)
    worlds = s16.panel(s16.configuration())[:64]
    rows = s18.run_cell(client, cfg, worlds, account=lambda n: None, initial_select="leftmost",
        requested_select="leftmost", mirror_select="rightmost", k=None, branch="keep",
        choose=s16.model_chooser(policies["final"], True), source="test-replay", stop_rule="s16")
    archived = {r["seed"]: r for r in s16.read_rows(
        TRAINING / "runs/m8_s16_command_control_audit_v0/learned/cohort-1/final/deterministic/correct/leftmost")}
    assert all(r[f] == archived[r["seed"]][f] for r in rows for f in s18.REPLAY_FIELDS)


def test_teacher_branches_share_their_prefix_and_the_switch_redirects_fire(client):
    cfg = s18.configuration()
    worlds = [(i, TEST_SEED_BASE + i, 30) for i in range(4)]
    out = {}
    for branch in s18.BRANCHES:
        out[branch] = s18.run_cell(client, cfg, worlds, account=lambda n: None, initial_select="leftmost",
            requested_select="rightmost", mirror_select="leftmost", k=30, branch=branch,
            choose=s16.teacher_chooser, source=f"test-{branch}", stop_rule="both-flanks")
    for branch in ("reactivate", "switch"):
        for a, b in zip(out["keep"], out[branch]):
            assert a["prefix"] == b["prefix"] and a["prefix"]["reachedK"]
            assert b["planVersionAfterK"] is not None
    assert all(r["planVersionAfterK"] is None for r in out["keep"])
    assert all(r["stopReason"] in ("both-flanks-crossed", "blue-wiped", "environment-done", "horizon")
               for rows in out.values() for r in rows)
    assert np.mean([r["postContrast"] for r in out["switch"]]) > np.mean([r["postContrast"] for r in out["reactivate"]])


def test_tiny_run_reaches_aggregate_and_seals(tmp_path, client, monkeypatch):
    cfg = {**s18.configuration(), "cohorts": (1,), "worlds": 4, "worldSeedBase": TEST_SEED_BASE + 100,
           "switchTimes": (4, 6, 8), "optionHorizon": 12, "commandHorizon": 12, "postWindow": 3, "lateralWindow": 2,
           "earlyWindow": 4, "blockWorlds": 4, "bootstrapSamples": 50, "minEligibleWorlds": 1,
           "teacherPostDeltaLowerMin": -9., "budgetCap": 200_000}
    monkeypatch.setattr(s18, "replay_gate", lambda *args: {"stub": {"exact": True}})
    report = s18.run(tmp_path / "run", cfg)
    assert report["outcome"] == "complete" and report["prefixIdentityHolds"] is True
    assert set(report["learned"]["policies"]) == {f"c1-{p}-{m}" for p in s18.POLICIES for m in s18.MODES}
    for entry in report["learned"]["policies"].values():
        assert {f"k{k}" for k in cfg["switchTimes"]} | {"reference"} == set(entry)
        assert all(entry[f"k{k}"]["class"] in (*s18.CLASSES, "teacher-invalid") for k in cfg["switchTimes"])
    assert set(report["predictions"]) >= {"1-teacherValidAllK", "2-finalsRedirectAtK30", "finalClasses"}
    seeds = {json.loads(line)["seed"] for path in (tmp_path / "run" / "learned").rglob("episodes.jsonl")
             for line in path.read_text().splitlines()}
    low, high = s18.configuration()["reservedQualificationSeeds"]
    assert seeds and not any(low <= s <= high or 2750000 <= s < 2750192 for s in seeds)
    assert dr.verify_sealed(tmp_path / "run")
