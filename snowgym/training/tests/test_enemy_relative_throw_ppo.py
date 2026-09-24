import copy
import json
import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw_ppo as erp
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options.opponent_transfer import scenario_override
from snowgym_training.ppo import living_unit_mask

TEST_SEED_BASE = 5960000  # off-band: not in any real cohort's declared 5{1,2,3}xxxxx range


def tiny(**overrides):
    return {**erp.configuration(), "optionHorizon": 8, "cohorts": (1,), "runSeedBase": 5900000,
            "probeEpisodes": 3, "sigmaProbeSuccessGapMin": -1e9, "sigmaProbeEntropyMargin": -1e9,
            "sigmaProbeSeedBaseByCohort": {1: TEST_SEED_BASE + 100}, "lrProbeSeedBaseByCohort": {1: TEST_SEED_BASE + 200},
            "trainingSeedBaseByCohort": {1: TEST_SEED_BASE + 1000}, "updates": 2, "episodesPerUpdate": 3,
            "checkpointUpdates": [2], "trainSeedBase": TEST_SEED_BASE + 300, "heldOutSeedBase": TEST_SEED_BASE + 400,
            "seedBandStride": 1000, "warmStartTrainEpisodes": 3, "warmStartHeldOutEpisodes": 3, "warmStartEpochs": 1,
            "blockWorlds": 3, "minibatchSize": 16, "epochs": 1, "criticEpochs": 1, "criticSanityMinR2": -1e9,
            "evaluationSeedBase": TEST_SEED_BASE + 500, "evaluationWorlds": 3,
            "stochasticEvalSeedBase": TEST_SEED_BASE + 600, "bootstrapSamples": 50,
            "probeBudgetCap": 50_000, "trainingBudgetCap": 50_000, **overrides}


@pytest.fixture(scope="module")
def live_rollout():
    """A real complete-episode rollout from a real S12 cohort-1 checkpoint at sigma x0.5."""
    torch.set_num_threads(1)
    cfg = tiny()
    torch.manual_seed(1)
    model, reference = erp.prepare_policy(cfg, 1)
    with SnowGymBatchClient() as client:
        recorder = erp.RolloutRecorderEnemyRelative(model)
        seeds = list(range(TEST_SEED_BASE + 700, TEST_SEED_BASE + 703))
        with scenario_override(rb.arm_scenario("normal")):
            episodes, stored, _ = v1.run_block(v1.make_wrapper(client, 3, cfg["gamma"]), seeds, cfg,
                                               choose=recorder, source="rollout", keep_observations=True)
    return cfg, model, reference, episodes, erp.build_rollout_enemy_relative(episodes, stored, recorder, cfg)


# -- Policy preparation (declaration §4) --------------------------------------------------


def test_log_stds_are_frozen_and_excluded_from_the_actor_optimizer():
    cfg = tiny()
    model, reference = erp.prepare_policy(cfg, 1)
    source = torch.load(erp.TRAINING / "runs" / cfg["sourceRun"] / "cohort-1" / "new" / cfg["policyCheckpoint"],
                        map_location="cpu", weights_only=True)["model"]
    trainable = {id(p) for p in model.actor_parameters()}
    for name in erp.LOG_STDS:
        parameter = getattr(model, name)
        torch.testing.assert_close(parameter.detach(), source[name] + math.log(cfg["sigmaScale"]))
        assert not parameter.requires_grad and id(parameter) not in trainable
        torch.testing.assert_close(getattr(reference, name).detach(), parameter.detach())
    offset = model.throw_offset_log_std
    torch.testing.assert_close(offset.detach(), torch.full_like(offset, cfg["offsetLogStdTarget"]))
    assert not offset.requires_grad and id(offset) not in trainable
    assert not any(p.requires_grad for p in reference.parameters())


def test_prepare_policy_offset_log_std_compensates_for_a_non_default_sigma_scale():
    cfg = tiny()
    model, _ = erp.prepare_policy(cfg, 1, sigma_scale=0.25)
    expected = cfg["offsetLogStdTarget"] + math.log(0.25 / cfg["sigmaScale"])
    torch.testing.assert_close(model.throw_offset_log_std.detach(), torch.full_like(model.throw_offset_log_std, expected))


# -- KL anchor (declaration §7) ------------------------------------------------------------


def test_hybrid_kl_is_zero_for_identical_policies_and_matches_a_monte_carlo_estimate(live_rollout):
    _, model, reference, _, rollout = live_rollout
    rows = {k: v[:4] for k, v in rollout["observation"].items()}
    with torch.no_grad():
        assert float(erp.hybrid_kl_enemy_relative(reference, reference, rows)) == pytest.approx(0, abs=1e-6)
        shifted = copy.deepcopy(reference)
        shifted.move_head[-1].bias.add_(torch.tensor([.02, -.01]))
        torch.manual_seed(11)
        shifted.enemy_score_head[-1].weight.add_(.05 * torch.randn_like(shifted.enemy_score_head[-1].weight))
        shifted.throw_offset_head[-1].bias.add_(torch.tensor([-.03, .02]))
        shifted.power_head[-1].bias.add_(.2)
        shifted.action_head.bias.add_(torch.tensor([.3, -.2, .5, 0.]))
        exact = float(erp.hybrid_kl_enemy_relative(shifted, reference, rows))
        repeated = {k: v.repeat(4000, *([1] * (v.dim() - 1))) for k, v in rows.items()}
        torch.manual_seed(2)
        action, latent, logp, _ = shifted.act(repeated)
        reference_logp, _ = reference.evaluate_latents(repeated, action["action_type"], latent, with_value=False)
        live = living_unit_mask(repeated)
        estimate = float(((logp - reference_logp) * live).sum() / live.sum())
    assert exact > .01
    assert estimate == pytest.approx(exact, rel=.15)


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
        assert float(erp.hybrid_kl_enemy_relative(moved, reference, rows)) > .01
        assert float(erp.hybrid_kl_enemy_relative(no_moves, reference_no_moves, rows)) == pytest.approx(0, abs=1e-5)


def test_hybrid_kl_is_finite_when_a_row_has_no_living_enemy(live_rollout):
    _, _, reference, _, rollout = live_rollout
    rows = {k: v[:4].clone() for k, v in rollout["observation"].items()}
    rows["enemy_mask"][0] = 0  # force every enemy slot masked for the first row
    with torch.no_grad():
        shifted = copy.deepcopy(reference)
        shifted.throw_offset_head[-1].bias.add_(torch.tensor([-.03, .02]))
        value = erp.hybrid_kl_enemy_relative(shifted, reference, rows)
    assert torch.isfinite(value)


def test_chunked_rollout_anchor_kl_equals_the_full_batch_value(live_rollout):
    _, _, reference, _, rollout = live_rollout
    with torch.no_grad():
        shifted = copy.deepcopy(reference)
        shifted.action_head.bias.add_(torch.tensor([.3, -.2, .5, 0.]))
        shifted.move_head[-1].bias.add_(torch.tensor([.02, -.01]))
        full = float(erp.hybrid_kl_enemy_relative(shifted, reference, rollout["observation"]))
    assert len(rollout["advantage"]) > 7 and full > 0
    assert erp.rollout_anchor_kl_enemy_relative(shifted, reference, rollout["observation"], chunk=7) \
        == pytest.approx(full, rel=1e-5)


# -- logp and rollout (declaration §3) -----------------------------------------------------


def test_rollout_logs_match_evaluate_latents_and_advantages_are_monte_carlo(live_rollout):
    cfg, model, _, episodes, rollout = live_rollout
    with torch.no_grad():
        latent = {"move": rollout["moveLatent"], "enemy": rollout["enemyLatent"],
                  "offset": rollout["offsetLatent"], "power": rollout["powerLatent"]}
        logp, _ = model.evaluate_latents(rollout["observation"], rollout["action_type"], latent, with_value=False)
        values = model.critic(rollout["observation"])
    torch.testing.assert_close(logp, rollout["logp"])
    torch.testing.assert_close(values, rollout["value"])
    torch.testing.assert_close(rollout["advantage"], rollout["returns"] - rollout["value"])
    first = rollout["episode"] == 0
    expected = v1.monte_carlo_returns(episodes[0]["rewards"], cfg["gamma"])
    torch.testing.assert_close(rollout["returns"][first], torch.tensor(expected, dtype=torch.float32))
    assert len(rollout["advantage"]) == sum(len(e["rewards"]) for e in episodes)


def test_evaluate_latents_logp_matches_a_hand_computed_sum_on_a_synthetic_batch(live_rollout):
    """Built from a real observation (correct keys/shapes/fixed unit capacity for `features()`), with
    `action_type` and the enemy mask forced so both MOVE and THROW rows, and a row with no living enemy, are all
    exercised deterministically, only at slots the observation already marks living."""
    _, model, _, _, rollout = live_rollout
    full_live = living_unit_mask(rollout["observation"])
    counts = full_live.sum(-1)
    row_with_two = int(torch.nonzero(counts >= 2)[0])  # needs a MOVE and a THROW unit
    row_with_one = int(torch.nonzero(counts >= 1)[0])
    observation = {k: torch.stack([v[row_with_two], v[row_with_one]]) for k, v in rollout["observation"].items()}
    live = living_unit_mask(observation)
    alive_slots = [torch.nonzero(live[row]).flatten().tolist() for row in range(2)]
    assert len(alive_slots[0]) >= 2 and len(alive_slots[1]) >= 1, "fixture rollout needs at least the exercised units alive"
    observation["enemy_mask"][0] = 1  # row 0: living enemies present
    observation["enemies"][0, :, 1] = 1.0
    observation["enemy_mask"][1] = 0  # row 1: no living enemy at all
    action_type = torch.zeros_like(rollout["action_type"][:2])
    action_type[0, alive_slots[0][0]] = 1  # MOVE
    action_type[0, alive_slots[0][1]] = 2  # THROW
    action_type[1, alive_slots[1][0]] = 2  # THROW, but this row has no living enemy
    units = action_type.shape[-1]
    torch.manual_seed(7)
    latent = {"move": torch.randn(2, units, 2), "enemy": torch.randint(0, observation["enemies"].shape[1], (2, units)),
              "offset": torch.randn(2, units, 2), "power": torch.randn(2, units)}
    logp, extra = model.evaluate_latents(observation, action_type, latent, with_value=False)
    with torch.no_grad():
        prediction = model(observation, with_value=False)
    from torch.distributions import Categorical, Normal
    action_categorical = Categorical(logits=prediction["action_logits"])
    enemy_categorical = Categorical(logits=prediction["enemy_logits"])
    move_normal = Normal(prediction["move_raw"], model.move_log_std.exp())
    offset_normal = Normal(prediction["throw_offset_raw"], model.throw_offset_log_std.exp())
    power_normal = Normal(prediction["power_raw"], model.power_log_std.exp())
    moves, throws = live & (action_type == 1), live & (action_type == 2)
    expected = (action_categorical.log_prob(action_type) * live
                + move_normal.log_prob(latent["move"]).sum(-1) * moves
                + enemy_categorical.log_prob(latent["enemy"]) * throws
                + offset_normal.log_prob(latent["offset"]).sum(-1) * throws
                + power_normal.log_prob(latent["power"]) * throws)
    torch.testing.assert_close(logp, expected)
    torch.testing.assert_close(extra["entropy"], action_categorical.entropy() * live)


def test_act_returns_a_nonzero_logp_and_a_target_within_bounds(live_rollout):
    _, model, _, _, rollout = live_rollout
    rows = {k: v[:4] for k, v in rollout["observation"].items()}
    action, _, logp, value = model.act(rows, deterministic=True)
    assert action["target"].abs().max() <= 1.0 + 1e-5
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()
    # S12's placeholder returned exactly zero; this must not.
    assert float(logp.detach().abs().sum()) > 0


# -- ppo_loss at roster 3 (declaration §14) -------------------------------------------------


def test_ppo_loss_broadcasts_and_masks_correctly_at_roster_three():
    from snowgym_training.ppo import PPOConfig, ppo_loss
    batch = 5
    logp = torch.randn(batch, 3)
    old_logp = logp + 0.01 * torch.randn(batch, 3)
    advantage = torch.randn(batch)
    entropy = torch.rand(batch, 3)
    active = torch.tensor([[True, True, True], [True, True, False], [True, False, False],
                           [True, True, True], [False, True, True]])
    config = PPOConfig(gae_lambda=1., ratio_mode="per-unit")
    result = ppo_loss(logp, old_logp, advantage, torch.zeros(batch), torch.zeros(batch), entropy, config,
                      active_mask=active)
    assert torch.isfinite(result["total"]) and torch.isfinite(result["policy"])
    # a fully-dead row must not raise a divide-by-zero: counts are clamped to at least 1
    all_dead = torch.zeros(1, 3, dtype=torch.bool)
    result_dead = ppo_loss(logp[:1], old_logp[:1], advantage[:1], torch.zeros(1), torch.zeros(1), entropy[:1],
                           config, active_mask=all_dead)
    assert torch.isfinite(result_dead["total"])


# -- Anchored update (declaration §8) --------------------------------------------------------


def test_anchored_update_drops_entropy_and_anchor_terms_when_their_weights_are_zero(live_rollout):
    base, model, reference, _, rollout = live_rollout
    cfg = {**base, "entropyWeight": 0., "anchorWeight": 0., "epochs": 1, "criticEpochs": 1, "minibatchSize": 8}
    indices = torch.arange(min(8, len(rollout["advantage"])))
    loss, losses = erp.anchored_actor_loss_enemy_relative(model, reference, rollout, indices, cfg)
    torch.testing.assert_close(loss, losses["policy"])


def test_anchor_weight_changes_the_resulting_parameters(live_rollout):
    base, model, reference, _, rollout = live_rollout
    cfg = {**base, "entropyWeight": 0., "epochs": 2, "criticEpochs": 1, "minibatchSize": 8, "movementKlStop": 1e9}
    unanchored, anchored = copy.deepcopy(model), copy.deepcopy(model)
    for target, weight in ((unanchored, 0.), (anchored, 1.)):
        actor_optimizer = torch.optim.Adam(target.actor_parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(target.critic.parameters(), lr=1e-3)
        torch.manual_seed(5)
        step = erp.anchored_ppo_update_enemy_relative(target, reference, actor_optimizer, critic_optimizer,
            rollout, {**cfg, "anchorWeight": weight})
        assert step["actorOptimizerSteps"] > 0 and not step["klStopped"]
    assert any(not torch.equal(p, q) for p, q in zip(anchored.actor_parameters(), unanchored.actor_parameters()))
    for p, q in zip(anchored.critic.parameters(), unanchored.critic.parameters()):  # the critic ignores the anchor
        torch.testing.assert_close(p, q)


# -- Paired analysis and decision rules (declaration §9, §10) -------------------------------


def rows_for(worlds, success, lost, timeout):
    return [{"seed": w, "success": s, "unitsLostFraction": l, "timedOut": t}
            for w, s, l, t in zip(worlds, success, lost, timeout)]


def write_evaluation(directory, name, rows):
    path = directory / "evaluation" / name
    path.mkdir(parents=True)
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_cohort_averaged_paired_analysis_resamples_worlds(tmp_path):
    cfg = {**erp.configuration(), "bootstrapSamples": 400}
    worlds = list(range(40))
    directories = []
    for k in range(3):
        directory = tmp_path / f"cohort-{k}"
        before_l = [0.5 if w % 2 == 0 else 0.0 for w in worlds]
        after_l = [0.25 if w % 4 == 0 else 0.0 for w in worlds]
        write_evaluation(directory, "initializer-deterministic",
            rows_for(worlds, [l == 0 for l in before_l], before_l, [False] * 40))
        write_evaluation(directory, "final-deterministic",
            rows_for(worlds, [l == 0 for l in after_l], after_l, [False] * 40))
        directories.append(directory)
    analysis = erp.paired_analysis_enemy_relative(directories, cfg)
    lost = analysis["cohortAveraged"]["unitsLostFraction"]
    assert lost["mean"] < 0 and lost["interval95"][1] <= 0
    assert "cohort-0" in analysis["perCohort"]


def analysis_with(lost, success, timeout):
    return {"cohortAveraged": {"unitsLostFraction": {"mean": lost[0], "interval95": list(lost[1:])},
                               "success": {"mean": 0., "interval95": [success, 1.]},
                               "timedOut": {"mean": 0., "interval95": [-1., timeout]}},
            "perCohort": {"cohort-1": {"unitsLostFraction": {"mean": lost[0]}}}}


def test_decision_rules_follow_the_declared_precedence():
    cfg = erp.configuration()
    reports = {"cohort-1": {"criticSanityStop": False, "parameterDistance": 1., "finalAnchorKl": .05}}
    rule = lambda *args: erp.decision_rules(analysis_with(*args), reports, cfg)["outcome"]
    assert rule((-.08, -.12, -.04), -.02, .01) == "improved"
    assert rule((-.03, -.05, -.01), -.02, .01) == "improved-below-threshold"
    assert rule((-.02, -.05, .01), -.02, .01) == "no-detectable-change"
    still = {"cohort-1": {**reports["cohort-1"], "finalAnchorKl": .001}}
    assert erp.decision_rules(analysis_with((-.02, -.05, .01), -.02, .01), still, cfg)["outcome"] == "no-effective-training"
    assert erp.decision_rules(analysis_with((-.03, -.05, -.01), -.02, .01), still, cfg)["outcome"] == "improved-below-threshold"
    assert rule((-.08, -.12, -.04), -.07, .01) == "harm-or-avoidance"
    assert rule((-.08, -.12, -.04), -.02, .06) == "harm-or-avoidance"
    assert rule((.04, .01, .07), -.02, .01) == "harm-or-avoidance"
    stopped = {"cohort-1": {**reports["cohort-1"], "criticSanityStop": True}}
    assert erp.decision_rules(None, stopped, cfg)["outcome"] == "incomplete"
    failed = {"cohort-1": {**reports["cohort-1"], "outcome": "sigma-probe-failed"}}
    assert erp.decision_rules(None, failed, cfg)["outcome"] == "incomplete"


# -- Configuration, budgets, and seed bands (declaration §11, §13) --------------------------


def test_configuration_and_budgets_match_the_declaration():
    cfg = erp.configuration()
    assert erp.probe_budget_bound(cfg) == {"sigma1x": 25_600, "sigmaFallbackWorstCase": 51_200, "lr": 12_800,
                                           "perCohortWorstCase": 89_600, "total": 268_800}
    assert erp.training_budget_bound(cfg) == {"warmStart": 76_800, "training": 2_560_000, "evaluation": 320_000,
                                              "perCohort": 2_956_800, "total": 8_870_400}
    assert cfg["probeBudgetCap"] == 300_000 and cfg["trainingBudgetCap"] == 9_000_000
    assert cfg["gaeLambda"] == 1. and cfg["advantage"] == "monte-carlo"


def test_seed_bands_are_disjoint_across_cohorts_and_from_test_seeds():
    cfg = erp.configuration()
    for cohort in cfg["cohorts"]:
        assert erp.training_seeds(cfg, cohort, 199)[-1] == cfg["trainingSeedBaseByCohort"][cohort] + 199 * 64 + 63
    bands = []
    for c in cfg["cohorts"]:
        sigma = set(erp.sigma_probe_seeds(cfg, c))
        lr = set(erp.lr_probe_seeds(cfg, c))
        train = set().union(*(erp.training_seeds(cfg, c, u) for u in range(cfg["updates"])))
        critic_train = set(v1.fold_seeds(cfg, c, held_out=False))
        critic_held = set(v1.fold_seeds(cfg, c, held_out=True))
        assert not (sigma & lr)
        bands.append(sigma | lr | train | critic_train | critic_held)
    assert not (bands[0] & bands[1] or bands[0] & bands[2] or bands[1] & bands[2])
    evaluation = set(erp.evaluation_seeds(cfg))
    assert not any(evaluation & b for b in bands)
    assert not any(TEST_SEED_BASE <= s < TEST_SEED_BASE + 2000 for b in bands for s in b)


# -- Live: probes and a tiny end-to-end run --------------------------------------------------


def test_sigma_candidates_are_absolute_multiples_of_the_declared_sigma_scale():
    cfg = erp.configuration()
    assert erp.sigma_candidates(cfg) == pytest.approx([0.5, 0.25, 0.125])
    # the first probe candidate must reproduce declaration §4's own log-std values exactly
    model, _ = erp.prepare_policy(cfg, 1, sigma_scale=erp.sigma_candidates(cfg)[0])
    reference, _ = erp.prepare_policy(cfg, 1)  # default sigma_scale=cfg["sigmaScale"]
    for name in (*erp.LOG_STDS, "throw_offset_log_std"):
        torch.testing.assert_close(getattr(model, name).detach(), getattr(reference, name).detach())


def test_sigma_probe_selects_a_candidate_against_a_real_checkpoint():
    cfg = tiny()
    with SnowGymBatchClient() as client:
        report = erp.sigma_probe(client, cfg, 1, lambda n: None)
    assert report["selectedSigmaScale"] in erp.sigma_candidates(cfg)
    assert len(report["candidates"]) >= 1


def test_lr_probe_selects_a_candidate_by_first_step_kl_not_by_whether_a_full_update_ever_stops():
    cfg = tiny()
    report = erp.lr_probe(cfg, 1, cfg["sigmaScale"], lambda n: None)
    assert set(c["lr"] for c in report["candidates"]) == set(erp.LR_CANDIDATES)
    assert report["selectedLr"] in erp.LR_CANDIDATES or report["selectedLr"] is None
    for candidate in report["candidates"]:
        assert isinstance(candidate["firstStepKl"], float) and candidate["firstStepKl"] >= 0
    # a larger lr's one-step KL should be at least as large as a smaller lr's on the same rollout/minibatch draw
    by_lr = sorted(report["candidates"], key=lambda c: c["lr"])
    assert all(a["firstStepKl"] <= b["firstStepKl"] + 1e-6 for a, b in zip(by_lr, by_lr[1:]))


def test_chunked_approximate_kl_matches_a_hand_stepped_model(live_rollout):
    """Regression guard for the original bug: `lr_probe`'s reported KL must reflect ONE Adam step's effect,
    not whether a full multi-epoch update eventually trips the KL stop. Takes one step by hand exactly as
    `lr_probe` does, then checks `chunked_approximate_kl` against a directly hand-computed (unchunked)
    approximate KL on the same stepped model."""
    _, model, reference, _, rollout = live_rollout
    size = len(rollout["advantage"])
    all_indices = torch.arange(size)
    probe_model = copy.deepcopy(model)
    actor_optimizer = torch.optim.Adam(probe_model.actor_parameters(), lr=3e-4)
    torch.manual_seed(99)
    step_indices = torch.randperm(size)[:min(16, size)]
    loss, _ = erp.anchored_actor_loss_enemy_relative(probe_model, reference, rollout, step_indices,
        {**live_rollout[0], "anchorWeight": 0., "entropyWeight": 0.})
    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_optimizer.step()

    chunked = erp.chunked_approximate_kl(probe_model, rollout, all_indices, live_rollout[0], chunk=5)
    with torch.no_grad():
        latent = {"move": rollout["moveLatent"], "enemy": rollout["enemyLatent"],
                  "offset": rollout["offsetLatent"], "power": rollout["powerLatent"]}
        new_logp, _ = probe_model.evaluate_latents(rollout["observation"], rollout["action_type"], latent,
            with_value=False)
    live = living_unit_mask(rollout["observation"]).float()
    counts = live.sum(-1).clamp_min(1.0)
    log_ratio = (new_logp - rollout["logp"]) * live
    ratio = log_ratio.exp()
    unit_kl = (ratio - 1 - log_ratio) * live
    hand_computed = float((unit_kl.sum(-1) / counts).mean())
    assert chunked == pytest.approx(hand_computed, rel=1e-4)
    # this step actually moved the model -- a KL of ~0 here would mean the "step" never happened
    assert chunked > 1e-8


def test_select_shared_lr_takes_the_intersection_across_cohorts():
    cfg = erp.configuration()
    reports = {
        "1": {"candidates": [{"lr": 3e-4, "firstStepKl": 0.02}, {"lr": 1e-4, "firstStepKl": 0.005},
                              {"lr": 1e-5, "firstStepKl": 0.0001}]},
        "2": {"candidates": [{"lr": 3e-4, "firstStepKl": 0.05}, {"lr": 1e-4, "firstStepKl": 0.02},
                              {"lr": 1e-5, "firstStepKl": 0.0002}]},
    }
    # only 1e-5 passes cfg["movementKlStop"]=0.01 for BOTH cohorts, even though cohort 1 alone would pass at 1e-4
    assert erp.select_shared_lr(reports, cfg) == pytest.approx(1e-5)


def test_tiny_end_to_end_probe_then_cohort_then_aggregate(tmp_path):
    cfg = tiny()
    root = tmp_path / "run"
    erp.declare(root, cfg)
    probes = erp.probe_all_cohorts(root, cfg, lambda n: None)
    assert probes["selectedSigmaByCohort"]["1"] in erp.sigma_candidates(cfg)
    assert probes["selectedSharedLr"] in erp.LR_CANDIDATES or probes["selectedSharedLr"] is None
    assert (root / "probe-report.json").exists()

    report = erp.run_cohort(root, cfg, 1, probes["selectedSigmaByCohort"]["1"], probes["selectedSharedLr"],
                            lambda n: None)
    directory = root / "cohort-1"
    assert not report["criticSanityStop"]
    assert (directory / "update-002.pt").exists()
    history = json.loads((directory / "training-history.json").read_text())
    assert len(history) == 2 and "meanUnitsLostFraction" in history[0]
    assert set(report["evaluation"]) == {f"{n}-{m}" for n in ("initializer", "final") for m in ("deterministic", "stochastic")}
    assert dr.verify_sealed(directory)
    with pytest.raises(FileExistsError):
        erp.run_cohort(root, cfg, 1, probes["selectedSigmaByCohort"]["1"], probes["selectedSharedLr"], lambda n: None)

    final = erp.aggregate(root, cfg, {"1": report}, 1000, 2000)
    assert final["decisionRules"]["outcome"] in {"improved", "improved-below-threshold", "no-effective-training",
                                                 "no-detectable-change", "harm-or-avoidance"}
    manifest = json.loads((root / "manifest.json").read_text())
    assert "cohort-1/manifest.json" in manifest["artifacts"] and "report.json" in manifest["artifacts"]


def test_training_budget_guard_stops_a_cohort_run(tmp_path):
    cfg = tiny(trainingBudgetCap=10)
    root = tmp_path / "run"
    erp.declare(root, cfg)
    probes = erp.probe_all_cohorts(root, cfg, lambda n: None)

    def tiny_account(count):
        nonlocal spent
        spent += count
        if spent > cfg["trainingBudgetCap"]:
            raise ValueError("M8-S14 training budget exceeded")

    spent = 0
    with pytest.raises(ValueError, match="budget exceeded"):
        erp.run_cohort(root, cfg, 1, probes["selectedSigmaByCohort"]["1"], probes["selectedSharedLr"], tiny_account)
