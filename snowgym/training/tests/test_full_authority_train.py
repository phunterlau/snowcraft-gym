import math
import shutil
from pathlib import Path

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.full_authority_ppo import FullAuthorityPolicy
from snowgym_training.options import full_authority_train as f


def test_configuration_budget_matches_the_declared_two_million_target():
    cfg = f.configuration()
    per_run = cfg["updates"] * cfg["worlds"] * cfg["rolloutDecisions"]
    assert 1_900_000 <= per_run <= 2_050_000
    assert cfg["worlds"] * cfg["rolloutDecisions"] >= 8192  # the review's declared minimum
    assert cfg["roster"] == 1 and set(cfg["arms"]) == {"local", "global"}
    assert cfg["autonomousQualificationEligible"] is False


@pytest.fixture(scope="module")
def tiny_cfg():
    return {**f.configuration(), "worlds": 6, "rolloutDecisions": 12, "updates": 2, "epochs": 1,
        "minibatchSize": 24, "warmStartTrainDecisions": 96, "warmStartHeldOutDecisions": 48,
        "warmStartEpochs": 2}


def test_collector_produces_correctly_shaped_rollouts_and_auto_resets(tiny_cfg):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = FullAuthorityPolicy(destination="local", local_radius=tiny_cfg["localRadius"])
    with SnowGymBatchClient() as client:
        wrapper = f.make_wrapper(client, tiny_cfg["worlds"], tiny_cfg["gamma"])
        collector = f.Collector(wrapper, model, f.SeedSchedule(920000, 920099))
        collector.start(tiny_cfg["rolloutDecisions"])
        done = collector.advance()
        assert done
        rollout = collector.rollout(gamma=tiny_cfg["gamma"], gae_lambda=tiny_cfg["gaeLambda"])
        expected = tiny_cfg["worlds"] * tiny_cfg["rolloutDecisions"]
        assert len(rollout["advantage"]) == len(rollout["returns"]) == expected
        assert rollout["moveLatent"].shape == (expected, 10, 2)
        assert rollout["action_type"].shape == (expected, 10)
        assert rollout["logp"].shape == (expected, 10)
        # More seeds than worlds were drawn once any episode inside the window auto-reset.
        assert collector.schedule.next_seed >= 920000 + tiny_cfg["worlds"]


def test_warm_start_critic_gate_stops_training_when_unmet(tiny_cfg):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = FullAuthorityPolicy(destination="global")
    strict_cfg = {**tiny_cfg, "warmStartMinRSquared": 1.1}  # unreachable: the arm must stop
    with SnowGymBatchClient() as client:
        report, _optimizer = f.warm_start_critic(model, client, strict_cfg, seed=12)
        assert report["heldOutRSquared"] is not None
        assert report["heldOutRSquared"] < strict_cfg["warmStartMinRSquared"]
        assert report["trainRows"] > 0 and report["heldOutRows"] > 0
        assert report["simulatorDecisions"] == report["trainRows"] + report["heldOutRows"]


def test_train_run_stops_at_the_critic_gate_and_records_it_without_actor_training(tmp_path, tiny_cfg):
    torch.set_num_threads(1)
    strict_cfg = {**tiny_cfg, "warmStartMinRSquared": 1.1}
    with SnowGymBatchClient() as client:
        model, report = f.train_run(client, strict_cfg, tmp_path / "gated", 13, "local")
    assert model is None
    assert report["stoppedAtCriticGate"] is True
    assert "history" not in report
    assert not (tmp_path / "gated" / "final-state.pt").exists()


def test_train_run_completes_and_moves_actor_parameters_when_the_gate_passes(tmp_path, tiny_cfg):
    torch.set_num_threads(1)
    lenient_cfg = {**tiny_cfg, "warmStartMinRSquared": -10.0}
    with SnowGymBatchClient() as client:
        torch.manual_seed(14)
        initial = FullAuthorityPolicy(destination="local", local_radius=lenient_cfg["localRadius"])
        initial_actor = {n: p.detach().clone() for n, p in initial.named_parameters()
                         if p in set(initial.actor_parameters())}
        model, report = f.train_run(client, lenient_cfg, tmp_path / "trained", 14, "local")
        assert model is not None and report["stoppedAtCriticGate"] is False
        assert len(report["history"]) == lenient_cfg["updates"]
        assert (tmp_path / "trained" / "final-state.pt").exists()
        moved = any(not torch.equal(p.detach(), initial_actor[n]) for n, p in model.named_parameters()
                    if n in initial_actor)
        assert moved
        evaluation = f.evaluate(model, client, [930000, 930001], lenient_cfg)
        assert len(evaluation) == 2 and all(e["rejectedActions"] == 0 for e in evaluation)


def test_uniform_random_floor_runs_full_episodes_without_crashing(tiny_cfg):
    with SnowGymBatchClient() as client:
        rows = f.uniform_random_floor(client, [940000], tiny_cfg, rng_seed=1)
    assert len(rows) == 1 and rows[0]["decisions"] > 0


def test_run_refuses_to_overwrite_an_existing_output(tmp_path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        f.run(tmp_path / "exists")
