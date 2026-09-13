import copy
import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.recovery_ppo import RecoveryPolicy
from snowgym_training.options import horizon_null_train as n
from snowgym_training.options import horizon_train as h


def test_permutation_is_seeded_shape_preserving_and_leaves_returns_alone():
    selected = {"advantage": torch.arange(20, dtype=torch.float32), "returns": torch.arange(20, dtype=torch.float32) * 7.,
        "observation": {"allies": torch.zeros(20, 3)}}
    a = n.permute_advantage(selected, 99301, 0)
    b = n.permute_advantage(selected, 99301, 0)
    c = n.permute_advantage(selected, 99301, 1)
    torch.testing.assert_close(a["advantage"], b["advantage"], atol=0, rtol=0)  # deterministic given (seed, update)
    assert sorted(a["advantage"].tolist()) == sorted(selected["advantage"].tolist())  # same multiset, reordered
    assert not torch.equal(a["advantage"], selected["advantage"])  # a real permutation, not the identity
    assert not torch.equal(a["advantage"], c["advantage"])  # independent stream per update
    torch.testing.assert_close(a["returns"], selected["returns"], atol=0, rtol=0)  # critic target untouched
    assert a["observation"] is selected["observation"]  # untouched keys are not copied


def test_permutation_stream_is_independent_of_row_selection_and_episode_seeds():
    # The three seeds used elsewhere in horizon_train/horizon_null_train for the same (seed, update)
    # must not collide with the permutation tag.
    seed, update, batch_size = 99301, 3, 8
    row_selection_seed = seed * 100000 + update
    episode_seeds = {seed * 100000 + update * batch_size + slot for slot in range(batch_size)}
    assert row_selection_seed not in episode_seeds
    assert [seed, update, n.PERMUTATION_TAG] != [row_selection_seed] and [seed, update, n.PERMUTATION_TAG] not in (
        [s] for s in episode_seeds)


def test_constant_vector_fit_recovers_a_planted_vector():
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(200, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    planted = np.array([0.07, 0.71])
    rows = [{"recommendationWorldGap": 5. + float(rng.uniform(0, 3)), "worldDirection": d.tolist(),
             "finalGapReduction": float(d @ planted) + float(rng.normal(scale=1e-4)),
             "finalWorldShift": float(rng.uniform(0.5, 1.0))} for d in directions]
    fit = n.constant_vector_fit(rows, gap_threshold=4.)
    assert fit["count"] == 200
    assert fit["r2"] > 0.99
    np.testing.assert_allclose(fit["vector"], planted, atol=1e-2)
    below_threshold = [{**r, "recommendationWorldGap": 1.0} for r in rows[:2]]
    assert n.constant_vector_fit(below_threshold, gap_threshold=4.)["count"] == 0
    assert n.constant_vector_fit(below_threshold, gap_threshold=4.)["r2"] is None


def test_parameter_breakdown_matches_manual_group_sums():
    initial = {"geometry.encoders.0.weight": torch.zeros(3), "geometry.move.0.weight": torch.zeros(2),
               "option_move.weight": torch.zeros(4)}
    final = {"geometry.encoders.0.weight": torch.tensor([1., 0., 0.]), "geometry.move.0.weight": torch.tensor([3., 4.]),
             "option_move.weight": torch.zeros(4)}
    result = n.parameter_breakdown(initial, final, steps=100, learning_rate=3e-4)
    assert result["l2"] == pytest.approx((1. + 25.) ** .5)
    assert result["groups"]["geometry.encoders"]["l2"] == pytest.approx(1.)
    assert result["groups"]["geometry.move.0"]["l2"] == pytest.approx(5.)
    assert result["groups"]["option_move"]["l2"] == pytest.approx(0.)
    assert result["lrSqrtSteps"] == pytest.approx(3e-4 * 10)
    with pytest.raises(StopIteration):
        n.parameter_breakdown({"unknown.head.weight": torch.zeros(1)}, {"unknown.head.weight": torch.zeros(1)}, 1, 3e-4)


def test_training_success_gain_adjusts_for_visited_frame_difficulty():
    history = ([{"update": u, "successes": 3, "seeds": [1, 2, 3, 4]} for u in range(1, 11)] +
               [{"update": u, "successes": 5, "seeds": [1, 2, 3, 4]} for u in range(21, 31)])
    deterministic = {1: True, 2: False, 3: False, 4: False}  # one of four visited frames already succeeds unaided
    gain = n.training_success_gain(history, deterministic)
    assert gain["early"]["stochasticSuccesses"] == 30 and gain["early"]["deterministicSuccesses"] == 10
    assert gain["late"]["stochasticSuccesses"] == 50 and gain["late"]["deterministicSuccesses"] == 10
    assert gain["stochasticGain"] == 20
    assert gain["adjustedGain"] == (50 - 10) - (30 - 10)


@pytest.fixture(scope="module")
def reference():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source, metadata, data, lineage = n.inputs()
    return source, metadata, data, lineage


def test_s9_and_s10_ancestry_verify_and_source_stays_frozen(reference):
    source, metadata, data, lineage = reference
    assert lineage["s9"]["manifestDigest"] and lineage["s10"]["manifestDigest"]
    assert all(not p.requires_grad for p in source.parameters())
    cfg = n.configuration()
    assert cfg["arm"] == "full" and set(cfg["nullTrainingRngs"]) == {99301, 99302, 99303}
    assert cfg["realTrainingRngs"] == [99302, 99303]


def test_live_capture_and_null_permutation_share_the_row_schedule_but_not_the_advantage_order(reference, tmp_path):
    source, metadata, data, lineage = reference
    frames = data["datasets"]["training"][:2]
    cfg = {**n.configuration(), "updates": 2, "batchSize": 2, "rowsPerUpdate": 6, "epochs": 1, "minibatchSize": 6}
    before = semantic_state_digest(source.state_dict())
    with SnowGymBatchClient() as client:
        real_model, real_report, real_capture, real_initial = n.train(
            source, metadata, client, frames, cfg, tmp_path / "real", 99301, null=False)
        null_model, null_report, null_capture, null_initial = n.train(
            source, metadata, client, frames, cfg, tmp_path / "null", 99301, null=True)
    assert semantic_state_digest(source.state_dict()) == before
    # Same schedule: identical frame draws and identical selected-row indices at every update.
    assert [x["seeds"] for x in real_report["history"]] == [x["seeds"] for x in null_report["history"]]
    assert [x["selectedRows"] for x in real_report["history"]] == [x["selectedRows"] for x in null_report["history"]]
    assert real_report["initialStateDigest"] == null_report["initialStateDigest"]
    # But the null run's parameters moved differently, because its optimizer saw permuted advantages.
    assert real_report["actorParameterL2Change"] > 0 and null_report["actorParameterL2Change"] > 0
    assert semantic_state_digest(real_model.state_dict()) != semantic_state_digest(null_model.state_dict())
    # Update-1 capture: one units-list per collected decision row, matching the update-1 rollout size.
    assert real_capture["observation"] is not None
    assert len(real_capture["units"]) == len(null_capture["units"])
    opportunities = n.finish_capture(real_capture, real_model)
    assert all({"finalWorldShift", "finalGapReduction", "worldDirection", "recommendationWorldGap"} <= r.keys()
               for r in opportunities)
    with pytest.raises(FileExistsError):
        n.train(source, metadata, client, frames, cfg, tmp_path / "real", 99302, null=False)


def test_stochastic_evaluate_is_seed_reproducible_and_draw_varying(reference):
    source, metadata, data, lineage = reference
    cfg = n.configuration()
    frame = data["datasets"]["historical"][0]
    torch.manual_seed(99301)
    model = RecoveryPolicy(copy.deepcopy(source), standard_deviation=cfg["latentStd"]).eval().requires_grad_(False)
    steps = []
    with SnowGymBatchClient() as client:
        first = n.stochastic_evaluate(model, source, client, [frame], cfg, steps.append, draws=2)
        second = n.stochastic_evaluate(model, source, client, [frame], cfg, steps.append, draws=2)
    assert [r["draw"] for r in first] == [0, 1]
    assert first == second  # deterministic given the fixed base seed and frame seed
    assert sum(steps) > 0
