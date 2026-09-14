import json

import numpy as np
import pytest
import torch
from torch import nn

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_MOVE
from snowgym_training.executor.full_authority_critics import EgocentricCritic
from snowgym_training.executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options import pre_ppo_diagnostics as pd


@pytest.fixture(scope="module")
def live_rows():
    """Real observation rows from a short random-policy block."""
    torch.set_num_threads(1)
    cfg = {**v1.configuration(), "optionHorizon": 6}
    torch.manual_seed(3)
    model = FullAuthorityPolicyV1(destination="global")
    with SnowGymBatchClient() as client:
        _, stored, _ = v1.run_block(v1.make_wrapper(client, 2, cfg["gamma"]), [931000, 931001], cfg,
                                    choose=pd.mode_chooser(model, pd.MODES["full1"]), source="rows",
                                    keep_observations=True)
    rows = {k: torch.cat([s["observation"][k] for s in stored]) for k in stored[0]["observation"]}
    return model, rows


def test_modes_reproduce_act_and_switch_draws_independently(live_rows):
    model, rows = live_rows
    with torch.no_grad():
        torch.manual_seed(7)
        expected, _, _, _ = model.act(rows)
        torch.manual_seed(7)
        sampled = pd.sample_actions(model, rows, sample_type=True, sample_continuous=True)
        deterministic, _, _, _ = model.act(rows, deterministic=True)
        mean = pd.sample_actions(model, rows, sample_type=False, sample_continuous=False)
    for key in expected:
        assert torch.equal(expected[key], sampled[key])
        assert torch.equal(deterministic[key], mean[key])
    with torch.no_grad():
        torch.manual_seed(9)
        _, one = pd.sample_actions(model, rows, sample_type=True, sample_continuous=True, latents=True)
        torch.manual_seed(9)
        _, half = pd.sample_actions(model, rows, sample_type=True, sample_continuous=True, sigma_scale=.5, latents=True)
        prediction = model(rows, with_value=False)
        typed = pd.sample_actions(model, rows, sample_type=True, sample_continuous=False)
        continuous = pd.sample_actions(model, rows, sample_type=False, sample_continuous=True)
    for name, raw in (("move", "move_raw"), ("throw", "throw_raw"), ("power", "power_raw")):
        torch.testing.assert_close(half[name] - prediction[raw], .5 * (one[name] - prediction[raw]))
    assert torch.equal(continuous["action_type"], prediction["action_logits"].argmax(-1))
    moves = typed["action_type"] == ACTION_MOVE
    torch.testing.assert_close(typed["target"][moves], torch.tanh(prediction["move_raw"])[moves])


def test_replayed_prefix_matches_the_source_and_a_perturbed_prefix_fails(live_rows):
    model, _ = live_rows
    cfg = {**pd.configuration(), "optionHorizon": 12, "branchDecisions": [0, 4, 9], "rolloutsPerState": 2}
    with SnowGymBatchClient() as client:
        torch.manual_seed(11)
        episodes, recorder = pd.collect_source(client, model, [931100], cfg, account=lambda _count: None)
        rows = pd.run_rollouts(client, model, episodes[0], recorder.actions[0], recorder.digests[0], cfg,
                               account=lambda _count: None, source="replay")
        assert len(rows) == 6 and all(r["replayIdentity"] for r in rows)
        assert sorted({r["branchDecision"] for r in rows}) == [0, 4, 9]
        tampered = [{**a, "action_type": np.full_like(a["action_type"], ACTION_MOVE), "target": np.full_like(a["target"], .9)}
                    for a in recorder.actions[0]]  # every prefix action becomes a far move
        with pytest.raises(RuntimeError, match="replay identity mismatch"):
            pd.run_rollouts(client, model, episodes[0], tampered, recorder.digests[0], {**cfg, "branchDecisions": [9]},
                            account=lambda _count: None, source="tampered")


def test_ceiling_estimator_corrects_the_within_state_bias_and_capture_is_calibrated():
    rng = np.random.default_rng(0)
    means = rng.normal(0, .5, size=6000)
    returns = means[:, None] + rng.normal(0, 1, size=(6000, 8))
    stats = pd.ceiling_statistics(returns)
    assert stats["stateValueVariance"] == pytest.approx(.25, abs=.02)
    assert stats["ceilingR2"] == pytest.approx(.2, abs=.015)
    naive = stats["betweenVariance"] / (stats["betweenVariance"] + stats["withinVariance"])
    assert naive > .25  # without the W/m correction the ceiling is biased upward
    exact = pd.branch_metrics(means, returns)
    constant = pd.branch_metrics(np.full_like(means, means.mean()), returns)
    assert exact["capture"] == pytest.approx(1, abs=.05) and constant["capture"] == pytest.approx(0, abs=.05)
    assert exact["rolloutR2"] == pytest.approx(stats["ceilingR2"], abs=.02)
    assert pd.ceiling_statistics(returns[:1])["ceilingR2"] is None


def test_cluster_interval_resamples_episodes_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, .5, size=(60, 1)) + rng.normal(0, 1, size=(60, 8))
    states = [(j // 3, k) for j, k in zip(range(60), [0, 25, 50] * 20)]
    report = pd.ceiling_report(states, returns, {"C0": returns.mean(1)}, {**pd.configuration(), "bootstrapSamples": 300})
    low, high = report["pooled"]["ceilingR2Interval95"]
    assert low <= report["pooled"]["ceilingR2"] <= high
    assert report["byDecision"]["25"]["states"] == 20 and report["byDecision"]["125"]["states"] == 0
    assert report["critics"]["C0"]["captureInterval95"] is not None


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, observation):
        return self.layer(observation["x"]).squeeze(-1)


def dataset(x, y, seeds):
    return {"observation": {"x": x}, "returns": y, "episode": seeds.clone(), "decision": torch.zeros_like(seeds),
            "seed": seeds}


def test_early_stopping_restores_the_best_validation_epoch_and_splits_by_episode():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 64)[:, None]
    train = dataset(x, 3 * x.squeeze(-1), torch.arange(64))
    validation = dataset(x, -3 * x.squeeze(-1), torch.arange(64, 128))  # improving on train hurts validation
    cfg = {**pd.configuration(), "learningRate": .05, "minibatchSize": 16, "earlyStoppingPatience": 3}
    critic = Linear()
    result = pd.train_critic(critic, train, cfg, epochs=50, seed=1, clip=None, validation=validation)
    best = result["history"][result["bestEpoch"]]
    assert result["epochsRun"] == result["bestEpoch"] + 4 < 50
    assert pd.mean_squared_error(critic, validation) == pytest.approx(best["validationMse"])
    split_cfg = {**pd.configuration(), "trainSeedBase": 100, "seedBandStride": 1000, "validationFromIndex": 3}
    seeds = torch.tensor([1100, 1100, 1102, 1103, 1103, 1104])
    data = dataset(torch.zeros(6, 1), torch.zeros(6), seeds)
    fit, held = pd.validation_split(data, split_cfg, 1)
    assert fit["seed"].tolist() == [1100, 1100, 1102] and held["seed"].tolist() == [1103, 1103, 1104]


def test_egocentric_critic_reads_living_units_and_option_state(live_rows):
    _, rows = live_rows
    torch.manual_seed(2)
    critic = EgocentricCritic()
    with torch.no_grad():
        value = critic(rows)
        assert value.shape == (len(rows["allies"]),)
        changed = {**rows, "option_state": rows["option_state"].clone()}
        changed["option_state"][:, 0] = 1 - changed["option_state"][:, 0]
        assert not torch.allclose(critic(changed), value)
        dead = {**rows, "allies": rows["allies"].clone(), "ally_mask": rows["ally_mask"].clone()}
        empty = ~dead["ally_mask"].bool()[0]
        assert empty.any()  # 1v1 fills one of ten roster slots; a masked-out slot must not move the value
        dead["allies"][:, empty.nonzero()[0, 0]] = 5.
        torch.testing.assert_close(critic(dead), value)


def ceilings_with(upper, value=.3):
    return {"pooled": {"ceilingR2": value, "ceilingR2Interval95": [0., upper]}}


def test_branch_window_r2_uses_rows_near_each_branch_decision():
    cfg = {**pd.configuration(), "branchDecisions": [0, 25], "branchWindowRadius": 2}
    decision = torch.arange(40)
    returns = torch.linspace(-1, 1, 40)
    values = torch.where((decision - 25).abs() <= 2, returns, torch.zeros(40))  # exact only near k = 25
    result = pd.branch_window_r2(values, {"decision": decision, "returns": returns}, cfg)
    assert result["byDecision"]["25"] == pytest.approx(1.)
    assert result["byDecision"]["0"] < 0 and result["pooled"] < 1


def test_critic_and_exploration_rules_follow_the_declaration():
    cfg = pd.configuration()
    seeds = [str(s) for s in cfg["policySeeds"]]
    repair = pd.critic_rules({s: ceilings_with(.6) for s in seeds},
                             {s: {"variant": "C2", "heldOutPredictiveR2": .3} for s in seeds}, cfg)
    assert repair["outcome"] == "repairable"
    unreachable = pd.critic_rules({s: ceilings_with(.2) for s in seeds},
                                  {s: {"variant": "C1", "heldOutPredictiveR2": .1} for s in seeds}, cfg)
    assert unreachable["outcome"] == "gate-unreachable"
    mixed = pd.critic_rules({seeds[0]: ceilings_with(.2), seeds[1]: ceilings_with(.6), seeds[2]: ceilings_with(.6)},
                            {s: {"variant": "C2", "heldOutPredictiveR2": .1 if s == seeds[0] else .3} for s in seeds},
                            cfg)
    assert mixed["outcome"] == "bracketed" and mixed["perPolicy"][seeds[0]]["row"] == "gate-unreachable"

    def d1(det, **modes):
        return {s: {"det": {"successFraction": det}, **{m: {"successFraction": v} for m, v in modes.items()}}
                for s in seeds}

    rules = pd.exploration_rules(d1(.7, type=.5, cont1=.65, full1=.45, full05=.6, full025=.68), cfg)
    assert rules["recommendedSigmaScale"] == .5 and rules["typeSamplingDominates"]
    assert pd.exploration_rules(d1(.7, type=.65, cont1=.5, full1=.3, full05=.4, full025=.55), cfg)[
        "recommendedSigmaScale"] is None


def test_configuration_budget_and_seed_bands_match_the_declaration():
    cfg = pd.configuration()
    assert pd.budget_bound(cfg) == {"d1": 360_000, "d2": 705_600, "d3": 230_400, "total": 1_296_000}
    assert cfg["simulatorBudget"] == 1_350_000
    assert v1.fold_seeds(cfg, 2, held_out=False)[-1] == 686255 and v1.fold_seeds(cfg, 2, held_out=True)[0] == 689000
    assert cfg["d1Seeds"] == [680000, 680099] and cfg["d2SourceSeedBase"] == 690000
    assert list(cfg["modes"]) == ["det", "type", "cont1", "full1", "full05", "full025"]


def tiny(**overrides):
    return {**pd.configuration(), "optionHorizon": 8, "policySeeds": [97101], "trainingRngs": [97101],
            "d1Seeds": [931200, 931201], "evaluationBlockWorlds": 2, "d2SourceSeedBase": 931300,
            "d2SourceEpisodes": 2, "branchDecisions": [0, 3], "rolloutsPerState": 2, "trainSeedBase": 931400,
            "heldOutSeedBase": 931500, "warmStartTrainEpisodes": 4, "warmStartHeldOutEpisodes": 2,
            "validationFromIndex": 3, "blockWorlds": 2, "criticBaselineEpochs": 1, "maxCriticEpochs": 3,
            "earlyStoppingPatience": 1, "egocentricHidden": 8, "minibatchSize": 16, "bootstrapSamples": 20,
            "simulatorBudget": 5000, **overrides}


def test_tiny_end_to_end_run_retains_every_declared_artifact(tmp_path):
    report = pd.execute(tmp_path / "run", tiny())
    root = tmp_path / "run"
    assert report["simulatorDecisions"] <= pd.budget_bound(tiny())["total"]
    assert set(report["d1"]["97101"]) == set(pd.MODES) and report["exploration"]["seedAveragedGaps"]["det"] == 0
    assert set(report["d3"]["97101"]) == set(pd.VARIANTS) and report["selections"]["97101"]["variant"] in ("C1", "C2", "C3")
    assert report["criticRules"]["outcome"] in ("repairable", "gate-unreachable", "bracketed")
    rollouts = [json.loads(line) for line in (root / "d2/policy-97101/rollouts.jsonl").read_text().splitlines()]
    assert rollouts and all(r["replayIdentity"] for r in rollouts)
    branch = np.load(root / "d2/policy-97101/branch-states.npz")
    assert branch["returns"].shape[1] == 2 and set(branch["decision"].tolist()) <= {0, 3}
    for name in pd.VARIANTS:
        assert (root / f"d3/policy-97101/critic-{name}.pt").exists() and f"value{name}" in branch.files
    manifest = json.loads((root / "manifest.json").read_text())
    assert "declaration.json" in manifest["artifacts"] and "report.json" in manifest["artifacts"]
    declaration = json.loads((root / "declaration.json").read_text())
    archived = json.loads((pd.TRAINING / "runs/m7b_engage_r1n_c_v0/manifest.json").read_text())
    assert declaration["sourceRun"]["artifacts"] == len(archived["artifacts"])
    assert declaration["budgetBound"] == pd.budget_bound(tiny())
    windows = report["d3"]["97101"]["C0"]["branchWindowR2"]
    assert set(windows["byDecision"]) == {"0", "3"} and "pooled" in windows
    with pytest.raises(FileExistsError):
        pd.execute(root, tiny())


def test_budget_guard_stops_the_run(tmp_path):
    with pytest.raises(ValueError, match="budget exceeded"):
        pd.execute(tmp_path / "run", tiny(simulatorBudget=10))
