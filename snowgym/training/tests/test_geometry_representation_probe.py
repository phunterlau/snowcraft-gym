import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.geometry_probe import GeometryProbe
from snowgym_training.options import geometry_representation_probe as g
from snowgym_training.ppo_checkpoint import load_ppo_checkpoint
from snowgym_training.options.identity import checkpoint_model
from snowgym_training.options.movement_train import REFERENCE


def test_attention_pool_ignores_masked_entries_and_handles_all_masked_rows():
    keys = torch.tensor([[[1., 0.], [0., 1.], [5., 5.]]])  # one row, three entities, dim 2
    query = torch.tensor([[1., 0.]])
    mask = torch.tensor([[True, True, False]])
    pooled = g.attention_pool(query, keys, mask)
    assert pooled.shape == (1, 2)
    assert not torch.isnan(pooled).any()
    # The masked-out (5,5) outlier must not leak into a fully-attended-elsewhere pool.
    assert pooled[0, 0] > pooled[0, 1]  # the query favors the first (aligned) key
    empty_mask = torch.zeros_like(mask)
    pooled_empty = g.attention_pool(query, keys, empty_mask)
    torch.testing.assert_close(pooled_empty, torch.zeros_like(pooled_empty))


@pytest.fixture(scope="module")
def source():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    metadata, state = load_ppo_checkpoint(REFERENCE)
    model = checkpoint_model(metadata)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


def test_build_arm_shapes_and_parameter_parity(source):
    import copy
    models = {arm: g.build_arm(copy.deepcopy(source), arm) for arm in g.ARMS}
    assert isinstance(models["A"], GeometryProbe) and not models["A"].relative
    assert isinstance(models["A-rel"], GeometryProbe) and models["A-rel"].relative
    assert isinstance(models["B"], g.AttentionGeometryProbe)
    counts = {arm: sum(p.numel() for p in m.parameters() if p.requires_grad) for arm, m in models.items()}
    assert counts["A"] == counts["A-rel"]  # same class, same shapes; only pair_features() differs at runtime
    assert abs(counts["B"] - counts["A"]) / counts["A"] <= 0.05  # declared +/-5% parity


def test_zero_initialized_arms_reproduce_the_frozen_source_on_a_real_observation(source):
    import copy
    from snowgym_training.ppo_collect import tensor_dict
    with SnowGymBatchClient() as client:
        wrapper = g.make_wrapper(client, 1, g.GAMMA)
        plan, spec = g.teacher_option_plan("engage")
        observation, _ = wrapper.reset([100000], [g.teacher_option_scenario("engage")], ["probe-zero-init"], [plan], [spec])
    batch = tensor_dict(observation)
    with torch.no_grad():
        reference = source(batch)
    for arm in g.ARMS:
        model = g.build_arm(copy.deepcopy(source), arm)
        with torch.no_grad():
            output = model(batch)
        torch.testing.assert_close(output["target_by_action"], reference["target_by_action"])
        torch.testing.assert_close(output["power"], reference["power"])


def test_heading_loss_picks_cosine_far_and_clipped_endpoint_near():
    prediction = {"target_by_action": torch.zeros(1, 1, 4, 2), "action_logits": torch.zeros(1, 1, 4)}
    prediction["action_logits"][0, 0, 1] = 10.  # classifier chooses MOVE
    observation = {"allies": torch.tensor([[[1., 1., 0., 0.] + [0.] * 17]]), "ally_mask": torch.ones(1, 1)}
    available = torch.ones(1, 1, dtype=torch.bool)
    # Far case: own at origin, teacher target far along +x; learned endpoint perpendicular (+y) -> cosine=0, heading=1.
    prediction["target_by_action"][0, 0, 1] = torch.tensor([0., 0.5])  # world (0, 20) at scale (50,40)
    teacher_far = torch.tensor([[[0.6, 0.]]])  # world (30, 0): distance 30 > 2.1
    far = g.heading_loss(prediction, teacher_far, available, observation)
    assert far["nearUnits"] == 0 and far["farUnits"] == 1
    assert far["heading"] == pytest.approx(1.0, abs=1e-4)
    assert far["headingErrorDegrees"] == pytest.approx(90.0, abs=1e-2)
    # Near case: teacher target within the slow radius -> clipped endpoint MSE, no heading term.
    teacher_near = torch.tensor([[[0.02, 0.]]])  # world (1, 0): distance 1 < 2.1
    near = g.heading_loss(prediction, teacher_near, available, observation)
    assert near["farUnits"] == 0 and near["nearUnits"] == 1
    assert near["headingErrorDegrees"] is None
    assert near["endpoint"] > 0


def test_heading_loss_respects_availability_and_classifier_mask():
    prediction = {"target_by_action": torch.zeros(1, 2, 4, 2), "action_logits": torch.zeros(1, 2, 4)}
    prediction["action_logits"][0, :, 1] = 10.  # both units choose MOVE
    observation = {"allies": torch.tensor([[[1., 1., 0., 0.] + [0.] * 17, [1., 0., 0., 0.] + [0.] * 17]]),
                   "ally_mask": torch.ones(1, 2)}  # second unit is dead (index 1 == alive flag == 0)
    teacher = torch.zeros(1, 2, 2)
    unavailable = torch.tensor([[True, True]])
    result = g.heading_loss(prediction, teacher, unavailable, observation)
    assert result["movingUnits"] == 1  # the dead unit is excluded by living_unit_mask regardless of availability
    all_unavailable = torch.zeros(1, 2, dtype=torch.bool)
    result_none = g.heading_loss(prediction, teacher, all_unavailable, observation)
    assert result_none["movingUnits"] == 0 and result_none["total"] == 0


def test_configuration_and_source_checkpoint_digest_agree(source):
    cfg = g.configuration()
    metadata, _ = load_ppo_checkpoint(REFERENCE)
    assert cfg["checkpointDigest"] == metadata["checkpointDigest"]
    assert set(cfg["arms"]) == set(g.ARMS) == {"A", "A-rel", "B"}


def test_dataset_batches_align_observation_target_and_availability():
    rows = [{"observation": {"x": torch.tensor([[float(i)]])}, "moveTarget": torch.tensor([float(i), 0.]),
             "moveAvailable": torch.tensor([i % 2 == 0])} for i in range(5)]
    dataset = g.Dataset(rows)
    assert len(dataset) == 5
    observation, target, available = dataset.batch(torch.tensor([2, 4]))
    torch.testing.assert_close(observation["x"], torch.tensor([[2.], [4.]]))
    torch.testing.assert_close(target, torch.tensor([[2., 0.], [4., 0.]]))
    torch.testing.assert_close(available, torch.tensor([[True], [True]]))
    with pytest.raises(ValueError):
        g.Dataset([])


def test_live_episode_and_one_fit_step_end_to_end(source):
    import copy
    cfg = {**g.configuration(), "stepsPerRound": 2, "minibatchSize": 4}
    with SnowGymBatchClient() as client:
        wrapper = g.make_wrapper(client, 1, g.GAMMA)
        rows, summary = g.episode(None, source, wrapper, 100000, teacher_move=True, collect=True)
        assert summary["seed"] == 100000 and summary["decisions"] > 0
        assert len(rows) == summary["decisions"]
        model = g.build_arm(copy.deepcopy(source), "A")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learningRate"])
        before = semantic_state_digest(source.state_dict())
        traces = g.fit_round(model, optimizer, g.Dataset(rows), cfg, seed=1)
        assert len(traces) == 2
        assert all(math.isfinite(t["total"]) for t in traces)
        assert semantic_state_digest(source.state_dict()) == before  # frozen source never moves
        held_out_rows, eval_summary = g.episode(model, source, wrapper, 100000, teacher_move=False, collect=False)
        assert held_out_rows == [] and "success" in eval_summary
