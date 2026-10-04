import json
import os

import pytest

from snowgym_training.options import command_control_audit as s16
from snowgym_training.options import command_control_qualification as s17
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw as ert
from snowgym_training.options import roster_imitation as ri
from snowgym_training.options import roster_imitation_repair as rp
from snowgym_training.options.full_authority_diagnostics import TRAINING
from snowgym_training.options.reservoir import file_digest

TEST_SEED_BASE = 5985000  # off-band: no real run or other test uses 5985000-5985999


def tiny(**overrides):
    """Every live test runs on off-band panels; the reserved qualification/held-out panels are never touched."""
    return {**s17.configuration(), "cohorts": (1,), "worldSeedBase": TEST_SEED_BASE, "worlds": 3,
            "panels": {"qualification": {"worldSeedBase": TEST_SEED_BASE, "worlds": 3, "spreads": (10, 20, 30)},
                       "heldout": {"worldSeedBase": TEST_SEED_BASE + 100, "worlds": 2, "spreads": (15, 25)}},
            "optionHorizon": 8, "commandHorizon": 8, "earlyWindow": 4, "lateralDecision": 2, "blockWorlds": 3,
            "bootstrapSamples": 50, "teacherOrderMin": -1., "teacherDeltaLowerMin": -1., "teacherSpreadOrderMin": -1.,
            "budgetCap": 100_000, **overrides}


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


# -- Offline: configuration, panels, seeds (declaration §3, §7) --------------------------------------------------


def test_configuration_and_budget_match_the_declaration():
    cfg = s17.configuration()
    assert s17.budget_bound(cfg) == {"gate": 153_600, "teacher": 480_000, "learned": 5_760_000, "total": 6_393_600}
    assert s17.budget_bound(cfg)["total"] <= cfg["budgetCap"] == 6_400_000
    assert cfg["conditions"] == ("correct", "other") and cfg["autonomousQualificationEligible"] is False
    assert (cfg["worldSeedBase"], cfg["worlds"]) == (2730000, 400)  # never S16's development panel


def test_panels_are_the_reserved_band_and_a_fresh_held_out_geometry_band():
    cfg = s17.configuration()
    qualification, heldout = s17.panel_worlds(cfg, "qualification"), s17.panel_worlds(cfg, "heldout")
    assert [seed for _, seed, _ in qualification] == list(range(2730000, 2730400))
    low, high = s16.configuration()["reservedQualificationSeeds"]
    assert (qualification[0][1], qualification[-1][1]) == (low, high)
    assert {d: sum(w[2] == d for w in qualification) for d in (10, 20, 30)} == {10: 134, 20: 133, 30: 133}
    assert [seed for _, seed, _ in heldout] == list(range(2740000, 2740200))
    assert {d: sum(w[2] == d for w in heldout) for d in (15, 25)} == {15: 100, 25: 100}
    development = {seed for _, seed, _ in s16.panel(s16.configuration())}
    for seeds in s17.reserved_seeds(cfg).values():
        assert not seeds & development and not seeds & set(range(2600000, 2600400))
        assert not seeds & set(range(2700000, 2700400))


def test_no_test_configuration_touches_a_reserved_panel():
    reserved = set().union(*s17.reserved_seeds(s17.configuration()).values())
    test_cfg = tiny()
    for panel in s17.PANELS:
        assert not {seed for _, seed, _ in s17.panel_worlds(test_cfg, panel)} & reserved
    assert not {seed for _, seed, _ in s16.panel(test_cfg)} & reserved


def test_stochastic_torch_seeds_are_distinct_and_in_the_declared_range():
    cfg = s17.configuration()
    seeds = [s17.stochastic_torch_seed(cfg, c, g, p, k, t) for c in cfg["cohorts"] for g in s17.PANELS
             for p in s16.POLICIES for k in s17.CONDITIONS for t in s16.TASKS]
    assert len(seeds) == len(set(seeds)) == 3 * 2 * 2 * 2 * 2
    assert min(seeds) == 987100 and max(seeds) == 987363


def test_configuration_integers_avoid_every_repository_json_scanner_band():
    bands = [(2100000, 2100399), (630000, 630119)]
    bands += [(low, high) for _, low, high in ri.training_bands(ri.configuration())]
    bands += [(low, high) for _, low, high in rp.seed_bands(rp.configuration())]
    bands += [(low, high) for _, low, high in ert.seed_bands(ert.configuration())]
    cfg = s17.configuration()
    hits = [n for n in integers({"config": cfg, "budget": s17.budget_bound(cfg)}) for low, high in bands
            if low <= n <= high]
    assert hits == []


# -- Offline: verdict and predictions (declaration §5, §6) --------------------------------------------------------


def classed(classes, mode="deterministic", identity_ok=True):
    return {"policies": {f"c{c}-{p}-{mode}": {"class": value if p == "final" else "sensitive",
                                              "mirrorIdentityWithinTolerance": identity_ok}
                         for c, value in classes.items() for p in s16.POLICIES}}


def test_verdict_counts_controllable_finals_and_treats_uninformative_as_not_replicated():
    cfg = s17.configuration()
    assert s17.verdict(classed({1: "controllable", 2: "controllable", 3: "controllable"}), cfg)["verdict"] == \
        "replicated"
    partial = s17.verdict(classed({1: "controllable", 2: "uninformative", 3: "controllable"}), cfg)
    assert partial["verdict"] == "partially-replicated" and partial["controllableFinals"] == 2
    assert s17.verdict(classed({1: "sensitive", 2: "insensitive", 3: "uninformative"}), cfg)["verdict"] == \
        "not-replicated"
    flagged = s17.verdict(classed({1: "controllable", 2: "controllable", 3: "controllable"}, identity_ok=False), cfg)
    assert flagged["identityWithinTolerance"] is False and flagged["note"]


def test_reused_s16_module_is_byte_identical_to_the_sealed_implementation():
    sealed = json.loads((TRAINING / "runs/m8_s16_command_control_audit_v0/declaration.json").read_text())
    assert file_digest(TRAINING / "src/snowgym_training/options/command_control_audit.py") == \
        sealed["implementationDigest"]


def test_declare_refuses_when_the_reused_module_differs_from_s16s_sealed_digest(tmp_path):
    fake = tmp_path / "fake-s16"
    fake.mkdir()
    (fake / "declaration.json").write_text(json.dumps({"implementationDigest": "sha256:not-this-one"}))
    cfg = {**s17.configuration(), "s16Run": os.path.relpath(fake, TRAINING / "runs")}
    with pytest.raises(RuntimeError, match="differs from the implementation S16 sealed"):
        s17.declare(tmp_path / "run", cfg)
    assert not (tmp_path / "run").exists()


# -- Live (off-band panels only) ------------------------------------------------------------------------------------


def test_tiny_run_reaches_aggregate_on_both_panels_and_seals(tmp_path, monkeypatch):
    cfg = tiny()
    monkeypatch.setattr(s16, "regression_gate", lambda *args: {"stub": {"exact": True}})
    report = s17.run(tmp_path / "run", cfg)
    assert report["outcome"] == "complete" and report["verdict"]["verdict"] in s17.VERDICTS
    for panel in s17.PANELS:
        policies = report["learned"][panel]["policies"]
        assert set(policies) == {f"c1-{p}-{m}" for p in s16.POLICIES for m in s16.MODES}
        assert all(set(entry["summaries"]) == set(s17.CONDITIONS) for entry in policies.values())
        assert set(report["teacher"][panel]["summaries"]) == set(s17.CONDITIONS)
    assert set(report["predictions"]) >= {"1-teacherGatePasses", "2-replicated", "5-heldoutFinalsControllable"}
    seeds = {json.loads(line)["seed"] for path in (tmp_path / "run").rglob("episodes.jsonl")
             for line in path.read_text().splitlines()}
    assert seeds and not seeds & set().union(*s17.reserved_seeds(s17.configuration()).values())
    assert dr.verify_sealed(tmp_path / "run")
