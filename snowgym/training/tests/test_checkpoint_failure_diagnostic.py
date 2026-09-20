import json

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import checkpoint_failure_diagnostic as cfd
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import mixture_imitation as mi


def tiny(**overrides):
    # optionHorizon is NOT shrunk: it must match the archive's value (200) exactly, or
    # episodes terminate/time out differently and the reproduction gate fails for reasons
    # that have nothing to do with the checkpoint (discovered live — see dev notes).
    return {**cfd.configuration(), "evaluationEpisodes": 3,
            "collectionBlockWorlds": 3, "checkpoints": (("M", 97101),), "budgetCap": 20_000, **overrides}


# -- budget ---------------------------------------------------------------------------------


def test_budget_bound_matches_declared_figures():
    cfg = cfd.configuration()
    bound = cfd.budget_bound(cfg)
    assert bound == {"perCheckpoint": 20_000, "checkpoints": 6, "total": 120_000}
    assert cfg["budgetCap"] == 200_000
    assert bound["total"] <= cfg["budgetCap"]


def test_checkpoints_cover_both_conditions_and_all_three_seeds():
    assert set(cfd.CHECKPOINTS) == {("M", 97101), ("M", 97102), ("M", 97103),
                                     ("C", 97101), ("C", 97102), ("C", 97103)}


# -- reproduction gate (pure) -----------------------------------------------------------------


def make_episode(seed, *, success, alive, timed_out=False, target_damage=None):
    damage = target_damage if target_damage is not None else ([0.0] if not success else [0.0, 80.0])
    return {"seed": seed, "source": "test", "success": success, "failed": not success, "timedOut": timed_out,
            "finalDecision": len(damage), "blueAliveAtEnd": alive, "rejectedActions": 0,
            "totalActions": len(damage), "fold": None, "rewards": [0.0] * len(damage),
            "distances": [1.0] * len(damage), "targetDamage": damage}


def test_reproduction_gate_passes_when_regenerated_matches_archive(tmp_path):
    archived = tmp_path / "episodes.jsonl"
    from snowgym_training.options.full_authority_train_v1 import episode_row
    archived_rows = [episode_row(make_episode(1, success=True, alive=True)),
                      episode_row(make_episode(2, success=False, alive=False))]
    archived.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in archived_rows), encoding="utf-8")

    regenerated_episodes = [make_episode(1, success=True, alive=True), make_episode(2, success=False, alive=False)]
    gate = cfd.reproduction_gate(regenerated_episodes, archived)
    assert gate == {"testedWorlds": 2, "matchedWorlds": 2, "mismatchedSeeds": [], "missingFromArchive": [],
                     "passed": True}


def test_reproduction_gate_can_check_a_strict_subset_of_the_archive(tmp_path):
    from snowgym_training.options.full_authority_train_v1 import episode_row
    archived = tmp_path / "episodes.jsonl"
    archived_rows = [episode_row(make_episode(s, success=(s == 1), alive=(s == 1))) for s in (1, 2, 3)]
    archived.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in archived_rows), encoding="utf-8")

    # Only world 1 regenerated (a tiny test's subset) — must not read the missing worlds 2/3
    # as mismatches.
    gate = cfd.reproduction_gate([make_episode(1, success=True, alive=True)], archived)
    assert gate["testedWorlds"] == 1
    assert gate["passed"] is True


def test_reproduction_gate_fails_on_a_genuine_mismatch(tmp_path):
    from snowgym_training.options.full_authority_train_v1 import episode_row
    archived = tmp_path / "episodes.jsonl"
    archived.write_text(json.dumps(episode_row(make_episode(1, success=True, alive=True)), sort_keys=True) + "\n",
                        encoding="utf-8")
    gate = cfd.reproduction_gate([make_episode(1, success=False, alive=False)], archived)
    assert gate["passed"] is False
    assert gate["mismatchedSeeds"] == [1]


# -- deployed_view geometry (pure, hand-computed) ----------------------------------------------


class FakeDeployedModel:
    """Stand-in exercising only `__call__`. One decision, one ally, one enemy, one red
    projectile. Ally at world origin; enemy at world (25, 0) (ARENA_HALF_EXTENT=(50,40),
    so raw 0.5 on the x-axis); the model deterministically selects THROW and aims exactly
    at world (25, 0) too (raw = atanh(0.5) so tanh(.)*50 = 25) — zero aim error expected."""

    def __call__(self, o, with_value=False):
        import math
        return {"living": torch.ones(1, 1, dtype=torch.bool),
                "action_logits": torch.tensor([[[0., 0., 10., 0.]]]),  # THROW
                "move_raw": torch.zeros(1, 1, 2),
                "throw_raw": torch.tensor([[[math.atanh(0.5), 0.]]]),
                "power_raw": torch.zeros(1, 1)}


def make_deployed_part():
    obs = {"allies": torch.zeros(1, 1, 21), "enemies": torch.zeros(1, 1, 21),
           "projectiles": torch.zeros(1, 1, 9), "projectile_mask": torch.ones(1, 1, dtype=torch.int8)}
    obs["enemies"][0, 0, 1] = 1.0  # present
    obs["enemies"][0, 0, 2] = 0.5  # raw x -> world 25
    obs["projectiles"][0, 0, 1] = 1.0  # red-owned
    labels = {"action_type": torch.tensor([[2]]), "target": torch.zeros(1, 1, 2), "power": torch.zeros(1, 1)}
    return {"observation": obs, "labels": labels, "seed": torch.tensor([42]), "decision": torch.tensor([0])}


def test_deployed_view_reproduces_a_hand_computed_case():
    cfg = {**cfd.configuration(), "engageRange": 9.0}
    part = make_deployed_part()
    view = cfd.deployed_view(FakeDeployedModel(), part, cfg)
    assert view["distance_to_enemy"][0, 0].item() == pytest.approx(25.0, abs=1e-4)
    assert bool(view["in_range"][0, 0]) is False  # 25 > engageRange 9
    assert bool(view["model_throws"][0, 0]) is True
    assert view["deployed_aim_error"][0, 0].item() == pytest.approx(0.0, abs=1e-3)
    assert bool(view["red_present"][0]) is True


def test_deployed_view_in_range_flag_when_enemy_is_close():
    cfg = {**cfd.configuration(), "engageRange": 9.0}
    part = make_deployed_part()
    part["observation"]["enemies"][0, 0, 2] = 0.1  # raw x -> world 5, inside engageRange
    view = cfd.deployed_view(FakeDeployedModel(), part, cfg)
    assert bool(view["in_range"][0, 0]) is True
    assert bool(view["close_range_throw"][0, 0]) is True


def test_contact_failure_summary_rates():
    cfg = {**cfd.configuration(), "engageRange": 9.0}
    part = make_deployed_part()
    part["observation"]["enemies"][0, 0, 2] = 0.1
    view = cfd.deployed_view(FakeDeployedModel(), part, cfg)
    summary = cfd.contact_failure_summary(view)
    assert summary["decisionsInRange"] == 1
    assert summary["closeRangeThrowRate"] == pytest.approx(1.0)
    assert summary["deployedThrowCount"] == 1
    assert summary["deployedAimErrorDegrees"] == pytest.approx(0.0, abs=1e-3)


# -- episode windows and finishing-failure summary (pure) --------------------------------------


def test_episode_windows_uses_zero_indexed_first_hit():
    episodes = [make_episode(1, success=True, alive=True, target_damage=[0.0, 0.0, 80.0])]
    windows = cfd.episode_windows(episodes)
    # first non-zero damage at index 2 (0-indexed) -> firstHitDecision = 3 -> window start 2.
    assert windows[1] == (2, 3)


def test_episode_windows_omits_episodes_with_no_contact():
    episodes = [make_episode(1, success=False, alive=False, target_damage=[0.0, 0.0])]
    assert cfd.episode_windows(episodes) == {}


def test_finishing_failure_summary_restricts_to_the_post_contact_window():
    episodes = [make_episode(7, success=True, alive=True, target_damage=[0.0, 80.0])]  # window (1, 2)
    view = {"seed": torch.tensor([7, 7, 7]), "decision": torch.tensor([0, 1, 2]),
            "live": torch.tensor([[True], [True], [True]]),
            "model_type": torch.tensor([2, 1, 1]),  # THROW, MOVE, MOVE (MOVE == full_authority_imitation ACTION_MOVE)
            "red_present": torch.tensor([False, True, True])}
    from snowgym_client.encoding import ACTION_MOVE
    assert view["model_type"][1].item() == ACTION_MOVE
    summary = cfd.finishing_failure_summary(view, episodes)
    assert summary["episodesWithContact"] == 1
    assert summary["postContactDecisions"] == 2  # decisions 1 and 2, not decision 0
    assert summary["underThreatDecisions"] == 2
    assert summary["keepsMovingUnderThreatRate"] == pytest.approx(1.0)


def test_finishing_failure_summary_handles_no_episodes_with_contact():
    view = {"seed": torch.tensor([1]), "decision": torch.tensor([0]),
            "live": torch.tensor([[True]]), "model_type": torch.tensor([1]),
            "red_present": torch.tensor([False])}
    summary = cfd.finishing_failure_summary(view, [make_episode(1, success=False, alive=False)])
    assert summary["episodesWithContact"] == 0
    assert summary["keepsMovingUnderThreatRate"] is None


# -- declare() digest cross-checks --------------------------------------------------------------


def test_declare_cross_checks_digests_against_r1n_h(tmp_path):
    cfg = tiny(budgetCap=200_000)
    root = tmp_path / "run"
    cfd.declare(root, cfg)
    declaration = json.loads((root / "declaration.json").read_text(encoding="utf-8"))
    assert declaration["imitationImplementationDigest"]
    assert declaration["trainImplementationDigest"]
    assert declaration["mixtureImplementationDigest"]
    with pytest.raises(FileExistsError):
        cfd.declare(root, cfg)


def test_declare_raises_if_source_run_declaration_is_tampered(tmp_path, monkeypatch):
    fake_source = tmp_path / "runs" / "fake-h"
    fake_source.mkdir(parents=True)
    real = json.loads((cfd.TRAINING / "runs/m7b_engage_r1n_h_v0/declaration.json").read_text(encoding="utf-8"))
    tampered = {**real, "imitationImplementationDigest": "sha256:" + "0" * 64}
    (fake_source / "declaration.json").write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(cfd, "TRAINING", tmp_path)
    cfg = tiny(sourceRun="fake-h")
    with pytest.raises(RuntimeError, match="full_authority_imitation.py digest"):
        cfd.declare(tmp_path / "run", cfg)


# -- live: collect_with_attribution matches fi.collect exactly ----------------------------------


def test_collect_with_attribution_matches_fi_collect_bit_for_bit():
    cfg = tiny()
    model = cfd.load_checkpoint(cfg, "M", 97101)
    seeds = mi.eval_seeds(cfg, "normal")[:2]
    with SnowGymBatchClient() as client:
        torch.manual_seed(0)
        with mi.scenario_override(mi.EVAL_ARMS["normal"]):
            fi_episodes, fi_part = fi.collect(client, seeds, cfg, model=model, source="fi-baseline",
                                              block_worlds=2, account=lambda n: None, deterministic=True, record=True)
        torch.manual_seed(0)
        with mi.scenario_override(mi.EVAL_ARMS["normal"]):
            cfd_episodes, cfd_part = cfd.collect_with_attribution(client, seeds, cfg, model=model,
                source="cfd-attributed", block_worlds=2, account=lambda n: None)

    assert [e["seed"] for e in fi_episodes] == [e["seed"] for e in cfd_episodes]
    assert [e["success"] for e in fi_episodes] == [e["success"] for e in cfd_episodes]
    for key in fi.ACTOR_KEYS:
        assert torch.equal(fi_part["observation"][key], cfd_part["observation"][key])
    for key in fi.LABEL_KEYS:
        assert torch.equal(fi_part["labels"][key], cfd_part["labels"][key])
    assert cfd_part["seed"].tolist() == [s for s in cfd_part["seed"].tolist()]  # attribution present
    assert set(cfd_part["seed"].tolist()) <= set(seeds)


# -- live: tiny end-to-end declare / run_checkpoint / aggregate ---------------------------------


def test_tiny_end_to_end_run_checkpoint_and_aggregate(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    cfd.declare(root, cfg)

    steps = 0

    def account(count):
        nonlocal steps
        steps += count

    with SnowGymBatchClient() as client:
        report = cfd.run_checkpoint(root, cfg, client, "M", 97101, account)
    assert report["reproductionGate"]["passed"] is True
    assert report["reproductionGate"]["testedWorlds"] == cfg["evaluationEpisodes"]
    assert report["labelErrorRecheck"]["deltas"]["typeAccuracy"] is not None
    assert steps > 0
    with pytest.raises(FileExistsError):
        cfd.run_checkpoint(root, cfg, client, "M", 97101, account)

    aggregated = cfd.aggregate(root, cfg)
    assert aggregated["allReproductionGatesPassed"] is True
    assert set(aggregated["checkpoints"]) == {"M-97101"}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "report.json" in manifest["artifacts"]

    # Tamper detection reuses death_rate_ppo's seal/verify_sealed, unchanged.
    rows = root / "condition-M-seed-97101" / "eval-normal-deterministic" / "episodes.jsonl"
    rows.write_text(rows.read_text(encoding="utf-8") + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        dr.verify_sealed(root)
