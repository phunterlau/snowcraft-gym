import copy
import json
import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as pp
from snowgym_training.options import full_authority_train_v1 as v1


def tiny(**overrides):
    return {**pp.configuration(), "optionHorizon": 8, "initializerSeeds": [97101], "runSeeds": [97301],
            "trainingRngs": [97301], "updates": 2, "episodesPerUpdate": 3, "trainingSeedBase": 932000,
            "checkpointUpdates": [2], "trainSeedBase": 932100, "heldOutSeedBase": 932200,
            "warmStartTrainEpisodes": 3, "warmStartHeldOutEpisodes": 3, "warmStartEpochs": 1, "blockWorlds": 3,
            "minibatchSize": 16, "epochs": 1, "criticEpochs": 1, "evaluationSeeds": [932300, 932302],
            "evaluationBlockWorlds": 3, "bootstrapSamples": 50, "criticSanityMinR2": -1e9,
            "policySimulatorBudget": 5000, **overrides}


@pytest.fixture(scope="module")
def live_rollout():
    """A real complete-episode rollout from the archived initializer at sigma x0.5."""
    torch.set_num_threads(1)
    cfg = tiny()
    torch.manual_seed(1)
    model, reference = pp.prepare_policy(cfg, 0)
    with SnowGymBatchClient() as client:
        recorder = pp.RolloutRecorder(model)
        episodes, stored, _ = v1.run_block(v1.make_wrapper(client, 3, cfg["gamma"]), [932400, 932401, 932402], cfg,
                                           choose=recorder, source="rollout", keep_observations=True)
    return cfg, model, reference, episodes, pp.build_rollout(episodes, stored, recorder, cfg)


def test_sigma_is_halved_frozen_and_outside_the_actor_optimizer():
    cfg = tiny()
    model, reference = pp.prepare_policy(cfg, 0)
    source = torch.load(pp.TRAINING / cfg["sourceRun"] / "seed-97101/fit-4.pt", map_location="cpu")["model"]
    trainable = {id(p) for p in model.actor_parameters()}
    for name in pp.LOG_STDS:
        parameter = getattr(model, name)
        torch.testing.assert_close(parameter.detach(), source[name] + math.log(.5))
        assert not parameter.requires_grad and id(parameter) not in trainable
        torch.testing.assert_close(getattr(reference, name).detach(), parameter.detach())
    assert not any(p.requires_grad for p in reference.parameters())


def test_hybrid_kl_is_zero_for_identical_policies_and_matches_a_monte_carlo_estimate(live_rollout):
    _, model, reference, _, rollout = live_rollout
    rows = {k: v[:4] for k, v in rollout["observation"].items()}
    with torch.no_grad():
        assert float(pp.hybrid_kl(reference, reference, rows)) == pytest.approx(0, abs=1e-6)
        shifted = copy.deepcopy(reference)
        shifted.move_head[-1].bias.add_(torch.tensor([.02, -.01]))
        shifted.throw_head[-1].bias.add_(torch.tensor([-.03, .02]))
        shifted.power_head[-1].bias.add_(.2)
        shifted.action_head.bias.add_(torch.tensor([.3, -.2, .5, 0.]))
        exact = float(pp.hybrid_kl(shifted, reference, rows))
        repeated = {k: v.repeat(4000, *([1] * (v.dim() - 1))) for k, v in rows.items()}
        torch.manual_seed(2)
        action, latent, logp, _ = shifted.act(repeated)
        reference_logp, _ = reference.evaluate_latents(repeated, action["action_type"], latent, with_value=False)
        live = pp.living_unit_mask(repeated)
        estimate = float(((logp - reference_logp) * live).sum() / live.sum())
    assert exact > .01
    assert estimate == pytest.approx(exact, rel=.1)


def test_hybrid_kl_weights_continuous_terms_by_type_probability(live_rollout):
    _, _, reference, _, rollout = live_rollout
    rows = {k: v[:4] for k, v in rollout["observation"].items()}
    with torch.no_grad():
        moved = copy.deepcopy(reference)
        moved.move_head[-1].bias.add_(torch.tensor([.05, .05]))
        no_moves = copy.deepcopy(moved)
        no_moves.action_head.bias[1] = -1e4  # MOVE probability ~0: the move mean shift must not count
        reference_no_moves = copy.deepcopy(reference)
        reference_no_moves.action_head.bias[1] = -1e4
        assert float(pp.hybrid_kl(moved, reference, rows)) > .01
        assert float(pp.hybrid_kl(no_moves, reference_no_moves, rows)) == pytest.approx(0, abs=1e-5)


def test_chunked_rollout_anchor_kl_equals_the_full_batch_value(live_rollout):
    _, _, reference, _, rollout = live_rollout
    with torch.no_grad():
        shifted = copy.deepcopy(reference)
        shifted.action_head.bias.add_(torch.tensor([.3, -.2, .5, 0.]))
        shifted.move_head[-1].bias.add_(torch.tensor([.02, -.01]))
        full = float(pp.hybrid_kl(shifted, reference, rollout["observation"]))
    assert len(rollout["advantage"]) > 7 and full > 0
    assert pp.rollout_anchor_kl(shifted, reference, rollout["observation"], chunk=7) == pytest.approx(full, rel=1e-5)


def test_rollout_logs_match_evaluate_latents_and_advantages_are_monte_carlo(live_rollout):
    cfg, model, _, episodes, rollout = live_rollout
    with torch.no_grad():
        latent = {"move": rollout["moveLatent"], "throw": rollout["throwLatent"], "power": rollout["powerLatent"]}
        logp, _ = model.evaluate_latents(rollout["observation"], rollout["action_type"], latent, with_value=False)
        values = model.critic(rollout["observation"])
    torch.testing.assert_close(logp, rollout["logp"])
    torch.testing.assert_close(values, rollout["value"])
    torch.testing.assert_close(rollout["advantage"], rollout["returns"] - rollout["value"])
    first = rollout["episode"] == 0
    expected = v1.monte_carlo_returns(episodes[0]["rewards"], cfg["gamma"])
    torch.testing.assert_close(rollout["returns"][first], torch.tensor(expected, dtype=torch.float32))
    assert len(rollout["advantage"]) == sum(len(e["rewards"]) for e in episodes)


def test_anchored_update_reduces_to_v1_without_entropy_or_anchor(live_rollout):
    base, model, reference, _, rollout = live_rollout
    for stop in (base["movementKlStop"], 1e9):  # with the declared KL stop, and with enough steps to move
        cfg = {**base, "entropyWeight": 0., "anchorWeight": 0., "epochs": 2, "criticEpochs": 1, "minibatchSize": 8,
               "movementKlStop": stop}
        ours, theirs = copy.deepcopy(model), copy.deepcopy(model)
        optimizers = [(torch.optim.Adam(m.actor_parameters(), lr=1e-3), torch.optim.Adam(m.critic.parameters(), lr=1e-3))
                      for m in (ours, theirs)]
        torch.manual_seed(5)
        a = pp.anchored_ppo_update(ours, reference, *optimizers[0], rollout, cfg)
        torch.manual_seed(5)
        b = v1.ppo_update(theirs, *optimizers[1], rollout, cfg)
        assert a["actorOptimizerSteps"] == b["actorOptimizerSteps"] and a["klStopped"] == b["klStopped"]
        for (name, p), q in zip(ours.named_parameters(), theirs.parameters()):
            torch.testing.assert_close(p, q, msg=name)
    assert a["actorOptimizerSteps"] > 1 and not a["klStopped"]
    anchored = copy.deepcopy(model)
    optimizer = torch.optim.Adam(anchored.actor_parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(anchored.critic.parameters(), lr=1e-3)
    torch.manual_seed(5)
    pp.anchored_ppo_update(anchored, reference, optimizer, critic_optimizer, rollout, {**cfg, "anchorWeight": 1.})
    assert any(not torch.equal(p, q) for p, q in zip(anchored.actor_parameters(), ours.actor_parameters()))
    for p, q in zip(anchored.critic.parameters(), ours.critic.parameters()):  # the critic ignores the anchor
        torch.testing.assert_close(p, q)


def rows_for(worlds, died, success, timeout):
    return [{"seed": w, "blueAliveAtEnd": not d, "success": s, "timedOut": t} for w, d, s, t in zip(worlds, died, success, timeout)]


def write_evaluation(directory, name, rows):
    path = directory / "evaluation" / name
    path.mkdir(parents=True)
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_seed_averaged_paired_analysis_resamples_worlds(tmp_path):
    cfg = {**pp.configuration(), "bootstrapSamples": 400}
    worlds = list(range(40))
    directories = []
    for k in range(3):
        directory = tmp_path / f"policy-{k}"
        before = [w % 2 == 0 for w in worlds]  # 50% deaths
        after = [w % 4 == 0 for w in worlds]    # 25% deaths
        for mode in ("deterministic",):
            write_evaluation(directory, f"initializer-{mode}", rows_for(worlds, before, [not d for d in before], [False] * 40))
            write_evaluation(directory, f"final-{mode}", rows_for(worlds, after, [not d for d in after], [False] * 40))
        directories.append(directory)
    analysis = pp.paired_analysis(directories, cfg, "deterministic")
    death = analysis["seedAveraged"]["death"]
    assert death["mean"] == pytest.approx(-.25) and death["interval95"][1] < 0
    assert analysis["perPolicy"]["policy-0"]["success"]["mean"] == pytest.approx(.25)


def analysis_with(death, success, timeout):
    return {"seedAveraged": {"death": {"mean": death[0], "interval95": list(death[1:])},
                             "success": {"mean": 0., "interval95": [success, 1.]},
                             "timeout": {"mean": 0., "interval95": [-1., timeout]}},
            "perPolicy": {"policy-1": {"death": {"mean": death[0]}}}}


def test_decision_rules_follow_the_declared_precedence():
    cfg = pp.configuration()
    reports = {"policy-1": {"criticSanityStop": False, "parameterDistance": 1., "rejectionRate": 0.,
                            "finalAnchorKl": .05}}
    rule = lambda *args: pp.decision_rules(analysis_with(*args), reports, cfg)["outcome"]
    assert rule((-.08, -.12, -.04), -.02, .01) == "survival-improved"
    assert rule((-.03, -.05, -.01), -.02, .01) == "improved-below-threshold"
    assert rule((-.02, -.05, .01), -.02, .01) == "no-detectable-change"
    still = {"policy-1": {**reports["policy-1"], "finalAnchorKl": .001}}
    assert pp.decision_rules(analysis_with((-.02, -.05, .01), -.02, .01), still, cfg)["outcome"] == "no-effective-training"
    assert pp.decision_rules(analysis_with((-.03, -.05, -.01), -.02, .01), still, cfg)["outcome"] == "improved-below-threshold"
    assert rule((-.08, -.12, -.04), -.07, .01) == "harm-or-avoidance"  # fewer deaths bought with fewer wins
    assert rule((-.08, -.12, -.04), -.02, .06) == "harm-or-avoidance"  # or with more timeouts
    assert rule((.04, .01, .07), -.02, .01) == "harm-or-avoidance"
    stopped = {"policy-1": {**reports["policy-1"], "criticSanityStop": True}}
    assert pp.decision_rules(None, stopped, cfg)["outcome"] == "incomplete"


def test_critic_sanity_stop_needs_the_whole_interval_below_zero():
    cfg = pp.configuration()
    assert not pp.critic_sanity_stop({"predictiveR2": .059, "predictiveR2Interval95": [-.043, .113]}, cfg)  # R1n-c 97103
    assert pp.critic_sanity_stop({"predictiveR2": -.08, "predictiveR2Interval95": [-.15, -.01]}, cfg)
    assert pp.critic_sanity_stop({"predictiveR2": None, "predictiveR2Interval95": None}, cfg)


def test_configuration_budget_and_seed_bands_match_the_declaration():
    cfg = pp.configuration()
    assert pp.budget_bound(cfg) == {"warmStart": 76_800, "training": 2_560_000, "evaluation": 320_000,
                                    "perPolicy": 2_956_800, "total": 8_870_400}
    assert cfg["policySimulatorBudget"] == 3_000_000 and cfg["simulatorBudget"] == 9_000_000
    assert pp.training_seeds(cfg, 2, 199)[-1] == 1_400_000 + 100_000 + 12_799
    bands = [set().union(*(pp.training_seeds(cfg, i, u) for u in range(cfg["updates"]))) for i in range(3)]
    assert all(len(band) == 12_800 for band in bands) and not (bands[0] & bands[1] or bands[0] & bands[2] or bands[1] & bands[2])
    assert v1.fold_seeds(cfg, 1, held_out=False)[0] == 881000 and v1.fold_seeds(cfg, 2, held_out=True)[-1] == 887127
    assert cfg["evaluationSeeds"] == [870000, 870399] and cfg["gaeLambda"] == 1.
    assert cfg["actorLearningRate"] == 1e-5 and cfg["learningRate"] == 3e-4  # amendment A1; critic keeps R1n-b's


def test_tiny_end_to_end_declare_policy_aggregate_and_tamper(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    pp.declare(root, cfg)
    report = pp.run_policy(root, cfg, 0)
    directory = root / "policy-97101"
    assert not report["criticSanityStop"] and report["parameterDistance"] > 0 and report["finalAnchorKl"] >= 0
    assert (directory / "update-002.pt").exists() and len(json.loads((directory / "training-history.json").read_text())) == 2
    assert set(report["evaluation"]) == {f"{n}-{m}" for n in ("initializer", "final") for m in ("deterministic", "stochastic")}
    with pytest.raises(FileExistsError):
        pp.run_policy(root, cfg, 0)
    final = pp.aggregate(root, cfg)
    assert final["decisionRules"]["outcome"] in {"survival-improved", "improved-below-threshold",
                                                 "no-effective-training", "no-detectable-change", "harm-or-avoidance"}
    manifest = json.loads((root / "manifest.json").read_text())
    assert "policy-97101/manifest.json" in manifest["artifacts"] and "report.json" in manifest["artifacts"]
    rows = directory / "training-episodes.jsonl"
    rows.write_text(rows.read_text() + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        pp.verify_sealed(directory)
    with pytest.raises(RuntimeError, match="configuration differs"):
        pp.run_policy(root, {**cfg, "anchorWeight": .5}, 0)


def test_critic_sanity_stop_skips_training_and_the_outcome_is_incomplete(tmp_path):
    cfg = tiny(criticSanityMinR2=1e9)
    root = tmp_path / "run"
    pp.declare(root, cfg)
    report = pp.run_policy(root, cfg, 0)
    assert report["criticSanityStop"] and "evaluation" not in report
    assert pp.aggregate(root, cfg)["decisionRules"]["outcome"] == "incomplete"


def test_budget_guard_stops_a_policy_run(tmp_path):
    cfg = tiny(policySimulatorBudget=10)
    root = tmp_path / "run"
    pp.declare(root, cfg)
    with pytest.raises(ValueError, match="budget exceeded"):
        pp.run_policy(root, cfg, 0)
