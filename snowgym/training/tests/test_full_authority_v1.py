import math

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_THROW
from snowgym_training.executor.full_authority_ppo import FullAuthorityPolicy
from snowgym_training.executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from snowgym_training.options import full_authority_diagnostics as d
from snowgym_training.options import full_authority_train as e3
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.ppo import living_unit_mask, ppo_loss


def tiny_cfg(**overrides):
    return {**v1.configuration(), "optionHorizon": 10, "blockWorlds": 3, "warmStartTrainEpisodes": 3,
            "warmStartHeldOutEpisodes": 3, "warmStartEpochs": 1, "minibatchSize": 16, "epochs": 1,
            "criticEpochs": 2, "bootstrapSamples": 50, "calibrationRows": 8, "calibrationDraws": 4,
            "trainSeedBase": 921000, "heldOutSeedBase": 922000, **overrides}


def test_e3_sources_still_match_their_archived_digests():
    pinned = d.e3_digests_unchanged()
    assert set(pinned) == {"implementationDigest", "policyImplementationDigest"}
    assert all(entry["match"] for entry in pinned.values())


def test_throw_log_std_is_separate_and_globally_calibrated_in_both_arms():
    local, global_ = FullAuthorityPolicyV1(destination="local"), FullAuthorityPolicyV1(destination="global")
    torch.testing.assert_close(local.move_log_std.exp().detach(), torch.tensor([.25, .25]))
    torch.testing.assert_close(global_.move_log_std.exp().detach(), torch.tensor([2 / 50, 2 / 40]))
    for model in (local, global_):
        torch.testing.assert_close(model.throw_log_std.exp().detach(), torch.tensor([2 / 50, 2 / 40]))
        names = {n for n, p in model.named_parameters() if p in set(model.actor_parameters())}
        assert {"move_log_std", "throw_log_std", "power_log_std"} <= names
        assert not any(n.startswith("critic.") for n in names)
        assert not hasattr(model, "target_log_std")
    with pytest.raises(ValueError):
        FullAuthorityPolicyV1(destination="diagonal")


def test_features_and_move_decoders_are_the_pinned_e3_methods():
    for name in ("pair_features", "features", "decode_move"):
        assert getattr(FullAuthorityPolicyV1, name) is getattr(FullAuthorityPolicy, name)


def test_throw_likelihood_uses_the_throw_log_std_not_the_move_log_std():
    model = FullAuthorityPolicyV1(destination="local")
    prediction = {"living": torch.tensor([[True]]), "action_logits": torch.zeros(1, 1, 4),
                  "move_raw": torch.zeros(1, 1, 2), "throw_raw": torch.zeros(1, 1, 2), "power_raw": torch.zeros(1, 1)}
    latent = {"move": torch.zeros(1, 1, 2), "throw": torch.tensor([[[.1, -.2]]]), "power": torch.tensor([[.3]])}
    action = torch.tensor([[ACTION_THROW]])
    logp, _ = model.evaluate_latents(None, action, latent, prediction=prediction)
    expected = (Categorical(logits=torch.zeros(4)).log_prob(torch.tensor(ACTION_THROW))
                + Normal(torch.zeros(2), model.throw_log_std.exp()).log_prob(latent["throw"][0, 0]).sum()
                + Normal(torch.zeros(1), model.power_log_std.exp()).log_prob(latent["power"][0]).sum())
    torch.testing.assert_close(logp[0, 0], expected)
    wrong = Normal(torch.zeros(2), model.move_log_std.exp()).log_prob(latent["throw"][0, 0]).sum()
    assert not torch.allclose(wrong, Normal(torch.zeros(2), model.throw_log_std.exp()).log_prob(latent["throw"][0, 0]).sum())


def test_monte_carlo_returns_are_exact_discounted_sums():
    assert v1.monte_carlo_returns([0., 0., -1.], .5) == [-.25, -.5, -1.]
    assert v1.monte_carlo_returns([1., 2.], 1.) == [3., 2.]


def test_predictive_r2_penalizes_a_bias_that_explained_variance_ignores():
    target = torch.arange(20, dtype=torch.float32)
    decision = torch.arange(20)
    metrics = v1.critic_metrics(target + 1, target, decision, target, decision, horizon=20, bins=20)
    assert metrics["explainedVariance"] == pytest.approx(1.)
    assert metrics["meanResidual"] == pytest.approx(1.)
    variance = float(target.var(unbiased=False))
    assert metrics["predictiveR2"] == pytest.approx(metrics["explainedVariance"] - 1 / variance)


def test_time_only_baseline_uses_training_bin_means_with_a_global_fallback():
    train_decision = torch.tensor([0, 1, 10, 11])
    train_target = torch.tensor([-1., -1., -3., -3.])
    held_decision = torch.tensor([0, 10, 19])  # bin 1 of 2 covers decisions 10-19
    held_target = torch.tensor([-1., -3., -3.])
    metrics = v1.critic_metrics(held_target, held_target, held_decision, train_target, train_decision, horizon=20, bins=2)
    assert metrics["timeOnlyR2"] == pytest.approx(1.)
    sparse = v1.critic_metrics(held_target, held_target, held_decision, train_target[:2], train_decision[:2],
                               horizon=20, bins=2)
    assert sparse["timeOnlyR2"] < 1.  # the empty bin falls back to the training-fold mean
    constant = v1.critic_metrics(torch.zeros(3), torch.ones(3), held_decision, train_target, train_decision, horizon=20, bins=2)
    assert constant["predictiveR2"] is None and v1.critic_gate(constant, v1.configuration()) is False


def test_critic_gate_requires_the_absolute_and_the_clock_relative_condition():
    cfg = v1.configuration()
    assert v1.critic_gate({"predictiveR2": .3, "timeOnlyR2": .32}, cfg)
    assert not v1.critic_gate({"predictiveR2": .3, "timeOnlyR2": .4}, cfg)
    assert not v1.critic_gate({"predictiveR2": .2, "timeOnlyR2": .1}, cfg)
    # With a near-perfect clock the relative condition binds far above 0.25 (A8).
    conditions = v1.gate_conditions({"predictiveR2": .9, "timeOnlyR2": .998}, cfg)
    assert conditions["absolutePassed"] and not conditions["clockRelativePassed"]
    assert conditions["bindingThreshold"] == pytest.approx(.948)


def test_clock_skill_score_separates_matching_the_clock_from_beating_it():
    train_decision = torch.tensor([0, 0, 10, 10])
    train_target = torch.tensor([-1., -2., -3., -4.])
    held_decision = torch.tensor([0, 0, 10, 10])
    held_target = torch.tensor([-1., -2., -3., -4.])
    clock = torch.tensor([-1.5, -1.5, -3.5, -3.5])
    matched = v1.critic_metrics(clock, held_target, held_decision, train_target, train_decision, horizon=20, bins=2)
    perfect = v1.critic_metrics(held_target, held_target, held_decision, train_target, train_decision, horizon=20, bins=2)
    assert matched["clockSkillScore"] == pytest.approx(0.)
    assert perfect["clockSkillScore"] == pytest.approx(1.)


def test_bootstrap_interval_brackets_the_point_estimate():
    generator = torch.Generator().manual_seed(0)
    episode = torch.arange(40).repeat_interleave(5)
    target = torch.randn(200, generator=generator)
    prediction = target + .5 * torch.randn(200, generator=generator)
    point = v1.critic_metrics(prediction, target, torch.zeros(200, dtype=torch.long), target,
                              torch.zeros(200, dtype=torch.long), horizon=200, bins=20)["predictiveR2"]
    low, high = v1.bootstrap_predictive_r2(prediction, target, episode, samples=500, seed=1)
    assert low <= point <= high


class FakeTracker:
    def __init__(self, length):
        self.length, self.decision, self.finished = length, 0, False
        self.assigned_ids, self.activated_target_ids = (1,), (2,)


class FakeWrapper:
    """Worlds finish at different decisions, so the active set shrinks mid-block."""

    def __init__(self, lengths, horizon):
        self.lengths, self.horizon, self.batch_size = lengths, horizon, len(lengths)
        self.environment = self
        self.raw_observations = [None] * len(lengths)

    def _raw(self, index):
        tracker = self.trackers[index]
        return {"allies": [{"id": 1, "alive": True, "x": float(tracker.decision), "y": 0.}],
                "enemies": [{"id": 2, "alive": True, "x": 10., "y": 0.}]}

    def _rows(self, indices):
        return {"option_state": np.asarray([[1 - self.trackers[i].decision / self.horizon, 1, 1] for i in indices],
                                           dtype=np.float32),
                "world": np.asarray([[i] for i in indices], dtype=np.int64)}

    def reset(self, seeds, *_):
        self.trackers = [FakeTracker(length) for length in self.lengths]
        self.raw_observations = [self._raw(i) for i in range(self.batch_size)]
        return self._rows(range(self.batch_size)), [{}] * self.batch_size

    def step_indices(self, indices, actions):
        assert len(actions["world"]) == len(indices)
        infos = []
        for index in indices:
            tracker = self.trackers[index]
            if tracker.finished:
                raise RuntimeError("stepped a finished world")
            tracker.decision += 1
            tracker.finished = tracker.decision >= tracker.length
            self.raw_observations[index] = self._raw(index)
            infos.append({"actionResults": [{"accepted": True}], "option": {
                "decision": tracker.decision, "success": False, "failed": tracker.finished,
                "timedOut": tracker.finished, "metrics": {"targetDamage": float(tracker.decision >= 2)}}})
        return self._rows(indices), np.full(len(indices), -.1, dtype=np.float32), None, None, infos


def test_block_runner_steps_only_unfinished_worlds_and_keeps_rows_aligned():
    wrapper = FakeWrapper([2, 5, 3], horizon=10)
    seen = []

    def choose(_wrapper, active, rows, raws):
        # Observation rows handed to the controller belong to exactly the active worlds.
        assert rows["world"][:, 0].tolist() == active and len(raws) == len(active)
        seen.append(list(active))
        return {"world": rows["world"][:, 0].numpy()}

    episodes, stored, decisions = v1.run_block(wrapper, [7, 8, 9], {"optionHorizon": 10}, choose=choose,
                                               source="fake", keep_observations=True)
    assert seen[0] == [0, 1, 2] and seen[2] == [1, 2] and seen[-1] == [1]
    assert decisions == 2 + 5 + 3
    for episode, length in zip(episodes, [2, 5, 3]):
        assert len(episode["rewards"]) == episode["finalDecision"] == length
        assert episode["remainingFraction"] == pytest.approx([1 - t / 10 for t in range(length)])
    for step in stored:
        torch.testing.assert_close(step["observation"]["option_state"][:, 0], 1 - step["decision"].float() / 10)
        assert step["observation"]["world"][:, 0].tolist() == step["worlds"].tolist()
    flat = v1.flatten_block(episodes, stored, .5)
    assert len(flat["returns"]) == decisions
    last = flat["decision"] == torch.tensor([e["finalDecision"] for e in episodes])[flat["episode"]] - 1
    torch.testing.assert_close(flat["returns"][last], torch.full((3,), -.1))
    row = v1.episode_row(episodes[1])
    assert row["firstHitDecision"] == 2 and row["distanceAtFirstHit"] == pytest.approx(9.)


def test_selective_env_steps_scripted_worlds_and_reads_teacher_labels_by_index():
    plan, _ = v1.teacher_option_plan("engage")
    with SnowGymBatchClient() as client:
        environment = v1.SelectiveBatchEnv(3, client=client, observation_version=3)
        environment.reset([923100, 923101, 923102], [e3.scenario()] * 3)
        environment.activate_plans(["p0", "p1", "p2"], [plan] * 3)
        before = list(environment.state_hashes)
        observations, _, _, _, infos = environment.step_scripted_indices([2, 0])
        assert observations["tick"].tolist() == [[6], [6]]
        assert [info["stateHash"] for info in infos] == [environment.state_hashes[2], environment.state_hashes[0]]
        assert environment.state_hashes[1] == before[1]
        assert len(environment.plan_teacher_actions_indices([1, 2])) == 2
        assert environment.plan_teacher_tensor_actions_indices([2])["action_type"].shape == (1, 10)
        for indices in ([], [0, 0], [3], [-1], [True]):
            with pytest.raises(ValueError, match="indices"):
                environment.step_scripted_indices(indices)
            with pytest.raises(ValueError, match="indices"):
                environment.plan_teacher_actions_indices(indices)


def test_scripted_blocks_score_complete_episodes_and_refuse_finished_trackers():
    cfg = tiny_cfg(optionHorizon=3)
    with SnowGymBatchClient() as client:
        wrapper = v1.make_wrapper(client, 2, cfg["gamma"], scripted=True)
        episodes, _, decisions = v1.run_block(wrapper, [923000, 923001], cfg, choose=None, source="scripted")
        assert decisions == sum(e["finalDecision"] for e in episodes) <= 6
        assert all(e["totalActions"] > 0 and e["rejectedActions"] == 0 for e in episodes)
        with pytest.raises(RuntimeError, match="completed option"):
            wrapper.step_scripted_indices([0])


@pytest.fixture(scope="module")
def live_rollout():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = FullAuthorityPolicyV1(destination="local")
    cfg = tiny_cfg()
    with SnowGymBatchClient() as client:
        collector = e3.Collector(e3.make_wrapper(client, 4, cfg["gamma"]), model, e3.SeedSchedule(924000, 924099))
        collector.start(6)
        collector.advance()
        rollout = collector.rollout(gamma=cfg["gamma"], gae_lambda=cfg["gaeLambda"])
    return model, rollout, cfg


def test_actor_loss_keeps_minibatch_advantage_normalization_and_leaves_the_critic_untouched(live_rollout):
    model, rollout, cfg = live_rollout
    indices = torch.arange(16)
    for parameter in model.parameters():
        parameter.grad = None
    loss, losses = v1.actor_loss(model, rollout, indices, cfg)
    loss.backward()
    assert all(p.grad is None for p in model.critic.parameters())
    assert any(p.grad is not None for p in model.actor_parameters())
    # The policy term is the same minibatch-normalized clipped surrogate E3 optimized.
    obs = {k: v[indices] for k, v in rollout["observation"].items()}
    latent = {"move": rollout["moveLatent"][indices], "throw": rollout["throwLatent"][indices],
              "power": rollout["powerLatent"][indices]}
    with torch.no_grad():
        logp, extra = model.evaluate_latents(obs, rollout["action_type"][indices], latent)
        reference = ppo_loss(logp, rollout["logp"][indices], rollout["advantage"][indices], extra["value"],
                             rollout["returns"][indices], extra["entropy"], v1.ppo_config(cfg),
                             active_mask=living_unit_mask(obs))
    torch.testing.assert_close(losses["policy"].detach(), reference["policy"])


def test_critic_still_trains_when_the_actor_stops_on_kl(live_rollout):
    model, rollout, cfg = live_rollout
    stopping = {**cfg, "movementKlStop": -1.0}  # every minibatch exceeds it: the actor stops at once
    actor_before = [p.detach().clone() for p in model.actor_parameters()]
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=cfg["learningRate"])
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg["learningRate"])
    trace = v1.ppo_update(model, actor_optimizer, critic_optimizer, rollout, stopping)
    assert trace["klStopped"] and trace["actorOptimizerSteps"] == 0
    assert trace["criticOptimizerSteps"] == cfg["criticEpochs"] * math.ceil(len(rollout["advantage"]) / cfg["minibatchSize"])
    assert all(torch.equal(p.detach(), q) for p, q in zip(model.actor_parameters(), actor_before))
    assert any(not torch.equal(p.detach(), q) for p, q in zip(model.critic.parameters(), critic_before))


def test_monte_carlo_warm_start_reports_metrics_arrays_and_calibration():
    torch.set_num_threads(1)
    cfg = tiny_cfg()
    torch.manual_seed(cfg["trainingRngs"][0])
    model = FullAuthorityPolicyV1(destination="global")
    with SnowGymBatchClient() as client:
        wrapper = v1.make_wrapper(client, cfg["blockWorlds"], cfg["gamma"])
        report, arrays, episodes, held, _ = v1.warm_start_critic_mc(model, wrapper, cfg, 0, source="test")
    assert report["trainEpisodes"] == report["heldOutEpisodes"] == 3
    assert report["simulatorDecisions"] == sum(e["finalDecision"] for e in episodes)
    assert report["trainRows"] + report["heldOutRows"] == report["simulatorDecisions"]
    for key in ("predictiveR2", "explainedVariance", "meanResidual", "timeOnlyR2", "clockSkillScore",
                "untrainedPredictiveR2", "gatePassed", "gateConditions"):
        assert key in report
    assert len(arrays["heldOutReturn"]) == len(arrays["heldOutValueAfter"]) == report["heldOutRows"]
    assert set(arrays["heldOutSeed"].tolist()) == {922000, 922001, 922002}
    assert set(arrays["trainSeed"].tolist()) == {921000, 921001, 921002}
    np.testing.assert_allclose(arrays["heldOutOptionState"][:, 0], 1 - arrays["heldOutDecision"] / cfg["optionHorizon"],
                               atol=1e-6)
    held_episodes = [e for e in episodes if e["fold"] == "heldOut"]
    expected = [r for e in held_episodes for r in v1.monte_carlo_returns(e["rewards"], cfg["gamma"])]
    np.testing.assert_allclose(sorted(arrays["heldOutReturn"]), sorted(expected), rtol=1e-5)
    calibration = v1.exploration_calibration(model, held["observation"], cfg)
    assert calibration["states"] > 0 and calibration["moveDisplacementWorld"]["median"] > 0
    assert 0 <= calibration["moveClampedFraction"] <= 1
