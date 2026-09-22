import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import death_rate_ppo as dr
from snowgym_training.options import enemy_relative_throw as ert
from snowgym_training.options import full_authority_imitation as fi
from snowgym_training.options import roster_baseline as rb
from snowgym_training.options.opponent_transfer import scenario_override
from snowgym_training.options.plans import teacher_option_plan

D = math.radians


# -- Geometry -----------------------------------------------------------------------------


def unit(angle):
    return torch.tensor([math.cos(angle), math.sin(angle)])


def test_rotate_composes_angles_by_addition():
    composed = ert.rotate(unit(D(30)), unit(D(20)))
    torch.testing.assert_close(composed, unit(D(50)), atol=1e-6, rtol=0)
    # rotating by zero (offset = (1,0)) leaves the base unchanged
    torch.testing.assert_close(ert.rotate(unit(D(73)), unit(0.0)), unit(D(73)), atol=1e-6, rtol=0)


def test_safe_normalize_handles_a_near_zero_vector_without_nan():
    out = ert.safe_normalize(torch.zeros(1, 2))
    assert torch.isfinite(out).all()
    out = ert.safe_normalize(torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(out, torch.tensor([[0.6, 0.8]]))


def test_enemy_relative_world_uses_world_units_not_normalized_axes():
    """The arena is 100x80 (half-extents 50, 40) -- not square. A relative position that is normalized-space
    equal on both axes must NOT be equal in world space, or bearings computed from the normalized delta would
    be systematically distorted along the shorter (height) axis."""
    observation = {
        "allies": torch.zeros(1, 1, 21),
        "enemies": torch.zeros(1, 1, 21),
        "enemy_mask": torch.ones(1, 1),
    }
    # own at normalized origin; one enemy at normalized delta (0.2, 0.2) -- equal on both axes.
    observation["enemies"][0, 0, 2] = 0.2
    observation["enemies"][0, 0, 3] = 0.2
    observation["enemies"][0, 0, 1] = 1.0  # alive
    world = ert.enemy_relative_world(observation)
    # world delta must be (0.2*50, 0.2*40) = (10, 8), NOT (0.2, 0.2) and NOT equal on both axes.
    torch.testing.assert_close(world[0, 0, 0], torch.tensor([10.0, 8.0]))
    assert world[0, 0, 0, 0] != world[0, 0, 0, 1]


# -- Loss: hand-computed on a synthetic single-decision batch --------------------------------


def make_synthetic_observation_and_labels():
    """One world, one unit, three enemy slots at world positions (10,0), (0,8), (-6,-6) relative to the
    thrower (who sits at normalized origin). Teacher throws at world point (9.5, 0.3) -- nearest enemy is
    slot 0 (10,0), with a small offset."""
    obs = {
        "allies": torch.zeros(1, 3, 21), "enemies": torch.zeros(1, 3, 21),
        "ally_mask": torch.ones(1, 3), "enemy_mask": torch.ones(1, 3),
        "unit_action_mask": torch.ones(1, 3, 4),
    }
    obs["allies"][0, 0, 1] = 1.0  # unit 0 alive; units 1/2 dead/unused for this test
    # enemy world positions -> normalized by (50, 40)
    positions_world = [(10.0, 0.0), (0.0, 8.0), (-6.0, -6.0)]
    for slot, (x, y) in enumerate(positions_world):
        obs["enemies"][0, slot, 1] = 1.0
        obs["enemies"][0, slot, 2] = x / 50.0
        obs["enemies"][0, slot, 3] = y / 40.0
    labels = {"action_type": torch.zeros(1, 3, dtype=torch.long), "target": torch.zeros(1, 3, 2),
              "power": torch.zeros(1, 3)}
    labels["action_type"][0, 0] = 2  # ACTION_THROW
    labels["target"][0, 0, 0] = 9.5 / 50.0
    labels["target"][0, 0, 1] = 0.3 / 40.0
    return obs, labels, positions_world


class StubModel(torch.nn.Module):
    """Bypasses the real network: returns fixed, hand-chosen predictions so the loss can be checked by hand."""

    def __init__(self, enemy_logits, offset_raw):
        super().__init__()
        self._enemy_logits, self._offset_raw = enemy_logits, offset_raw
        self.move_head = torch.nn.Linear(1, 1)  # unused by this test's decode_move path (no MOVE rows)

    def features(self, observation):
        return torch.zeros(observation["allies"].shape[0], observation["allies"].shape[1], 1)

    def decode_move(self, observation, latent):
        return torch.zeros_like(latent)

    def forward(self, observation, *, with_value=True):
        live = torch.tensor([[True, False, False]])
        action_logits = torch.zeros(1, 3, 4)
        enemy_mask = ert.enemy_living_mask(observation)
        enemy_logits = self._enemy_logits.masked_fill(~enemy_mask[:, None, :], -1e9)
        return {"action_logits": action_logits, "living": live, "move_raw": torch.zeros(1, 3, 2),
                "power_raw": torch.zeros(1, 3), "enemy_logits": enemy_logits, "throw_offset_raw": self._offset_raw,
                "enemy_mask": enemy_mask, "enemy_world": ert.enemy_relative_world(observation)}


def test_teacher_enemy_selection_picks_the_angularly_nearest_slot():
    obs, labels, positions = make_synthetic_observation_and_labels()
    cfg = {"slowRadius": 1.0, "endpointClip": 3.0}
    model = StubModel(enemy_logits=torch.zeros(1, 3, 3), offset_raw=torch.tensor([[[1.0, 0.0]]] * 1).expand(1, 3, 2).clone())
    losses = ert.imitation_loss_enemy_relative(model, obs, labels, cfg)
    assert losses["throwUsable"] == 1
    # slot 0 (10,0) is nearest to the teacher's (9.5,0.3) direction -- confirm via the enemy_loss being small
    # when the model's logits already favor slot 0, and large when they favor slot 2.
    favor_0 = StubModel(enemy_logits=torch.tensor([[[5.0, -5.0, -5.0]]] * 1).expand(1, 3, 3).clone(),
                        offset_raw=torch.tensor([[[1.0, 0.0]]] * 1).expand(1, 3, 2).clone())
    favor_2 = StubModel(enemy_logits=torch.tensor([[[-5.0, -5.0, 5.0]]] * 1).expand(1, 3, 3).clone(),
                        offset_raw=torch.tensor([[[1.0, 0.0]]] * 1).expand(1, 3, 2).clone())
    loss_favor_0 = ert.imitation_loss_enemy_relative(favor_0, obs, labels, cfg)["enemyLoss"]
    loss_favor_2 = ert.imitation_loss_enemy_relative(favor_2, obs, labels, cfg)["enemyLoss"]
    assert loss_favor_0 < loss_favor_2


def test_angle_loss_is_teacher_forced_on_the_teacher_enemy_and_zero_for_a_perfect_offset():
    obs, labels, positions = make_synthetic_observation_and_labels()
    cfg = {"slowRadius": 1.0, "endpointClip": 3.0}
    # teacher direction is world (9.5, 0.3); teacher's chosen enemy (slot 0) bearing is world (10, 0) -> unit (1,0).
    # an offset of exactly (1,0) (no rotation) composed with bearing (1,0) gives direction (1,0), which is NOT
    # exactly the teacher's true direction (9.5,0.3) normalized -- so pick the offset that reproduces it exactly.
    teacher_dir = torch.tensor([9.5, 0.3])
    teacher_dir_unit = teacher_dir / teacher_dir.norm()
    # bearing is (1,0); rotate(bearing, offset) = offset when bearing is the identity (1,0). So offset = teacher_dir_unit.
    perfect_offset = teacher_dir_unit.view(1, 1, 2).expand(1, 3, 2).clone()
    model = StubModel(enemy_logits=torch.zeros(1, 3, 3), offset_raw=perfect_offset)
    losses = ert.imitation_loss_enemy_relative(model, obs, labels, cfg)
    assert float(losses["angleLoss"]) == pytest.approx(0.0, abs=1e-5)
    # an offset perpendicular to the teacher's true offset should give a much larger loss (cosine 0 -> loss ~1).
    perpendicular = torch.tensor([-teacher_dir_unit[1], teacher_dir_unit[0]]).view(1, 1, 2).expand(1, 3, 2).clone()
    bad_model = StubModel(enemy_logits=torch.zeros(1, 3, 3), offset_raw=perpendicular)
    bad_loss = ert.imitation_loss_enemy_relative(bad_model, obs, labels, cfg)["angleLoss"]
    assert float(bad_loss) == pytest.approx(1.0, abs=1e-5)


def test_a_throw_row_with_no_living_enemy_is_excluded_from_both_new_losses_and_stays_finite():
    obs, labels, _ = make_synthetic_observation_and_labels()
    obs["enemy_mask"][:] = 0.0  # no living enemies at all
    cfg = {"slowRadius": 1.0, "endpointClip": 3.0}
    model = StubModel(enemy_logits=torch.zeros(1, 3, 3), offset_raw=torch.zeros(1, 3, 2))
    losses = ert.imitation_loss_enemy_relative(model, obs, labels, cfg)
    assert losses["throwUsable"] == 0
    assert float(losses["enemyLoss"]) == 0.0 and float(losses["angleLoss"]) == 0.0
    assert torch.isfinite(losses["total"])


def test_shared_loss_terms_match_fi_imitation_loss_exactly_on_the_same_synthetic_input():
    """type/moveHeading/moveEndpoint/power are copied from fi.imitation_loss unchanged; this pins that they
    still compute bit-identical values, using a MOVE-only synthetic batch so the (different) throw terms don't
    need to agree."""
    obs = {"allies": torch.zeros(1, 2, 21), "enemies": torch.zeros(1, 2, 21), "ally_mask": torch.ones(1, 2),
           "enemy_mask": torch.zeros(1, 2), "unit_action_mask": torch.ones(1, 2, 4)}
    obs["allies"][0, 0, 1] = 1.0
    obs["allies"][0, 0, 2:4] = torch.tensor([0.0, 0.0])
    labels = {"action_type": torch.zeros(1, 2, dtype=torch.long), "target": torch.zeros(1, 2, 2),
              "power": torch.zeros(1, 2)}
    labels["action_type"][0, 0] = 1  # ACTION_MOVE
    labels["target"][0, 0] = torch.tensor([0.3, 0.1])
    cfg = {"slowRadius": 1.0, "endpointClip": 3.0}

    class BothModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)

        def features(self, observation):
            return torch.zeros(observation["allies"].shape[0], observation["allies"].shape[1], 1)

        def decode_move(self, observation, latent):
            return latent  # identity decode for this test

        def forward(self, observation, *, with_value=True):
            live = torch.tensor([[True, False]])
            return {"action_logits": torch.zeros(1, 2, 4), "living": live, "move_raw": torch.full((1, 2, 2), 0.1),
                    "power_raw": torch.zeros(1, 2), "throw_raw": torch.zeros(1, 2, 2),
                    "enemy_logits": torch.zeros(1, 2, 2), "throw_offset_raw": torch.zeros(1, 2, 2),
                    "enemy_mask": ert.enemy_living_mask(observation), "enemy_world": ert.enemy_relative_world(observation)}

    model = BothModel()
    new = ert.imitation_loss_enemy_relative(model, obs, labels, cfg)
    old = fi.imitation_loss(model, obs, labels, cfg)
    for key in ("type", "moveHeading", "moveEndpoint", "power"):
        torch.testing.assert_close(new[key], old[key])


def test_configuration_and_budget():
    cfg = ert.configuration()
    bound = ert.budget_bound(cfg)
    assert bound["total"] == bound["perRun"] * 3 * 2
    assert bound["total"] <= cfg["budgetCap"]


def test_seed_bands_are_disjoint_and_do_not_collide_across_cohorts():
    cfg = ert.configuration()
    ranges = [(low, high) for _, low, high in ert.seed_bands(cfg)]
    ordered = sorted(ranges)
    for (_, high), (low, _) in zip(ordered, ordered[1:]):
        assert high < low


def test_seed_bands_are_unused_elsewhere_in_the_repository_json():
    import json
    from snowgym_training.options.full_authority_diagnostics import TRAINING
    cfg = ert.configuration()
    ranges = [(low, high) for _, low, high in ert.seed_bands(cfg)]

    def integers(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from integers(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from integers(item)

    collisions = []
    for path in TRAINING.parents[1].rglob("*.json"):
        if any(part in {".git", "node_modules", ".venv", "dist", "m8_s12_enemy_relative_throw_v0"} for part in path.parts) \
                or path.stat().st_size > 5_000_000:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if any(low <= n <= high for n in integers(data) for low, high in ranges):
            collisions.append(str(path))
    assert collisions == []


# -- live -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def tiny(**overrides):
    return {**ert.configuration(), "optionHorizon": 8, "roundEpisodes": 8, "fits": 2, "stepsPerFit": 6,
            "imitationMinibatch": 16, "blockWorlds": 4, "roundBlockWorlds": 4, "evaluationBlockWorlds": 4,
            "warmStartTrainEpisodes": 16, "warmStartHeldOutEpisodes": 8, "warmStartEpochs": 2,
            "pairedEvalWorlds": 4, "cohorts": (1,), "optimizerSeeds": (98401,), "budgetCap": 400_000, **overrides}


def collect_round_zero_for_test(client, cfg):
    """roundEpisodes bumped well above `tiny()`'s default so round-zero reliably contains some THROW-labelled
    decisions; block_worlds and optionHorizon stay small so this is still fast."""
    big_cfg = {**cfg, "roundEpisodes": 20, "optionHorizon": 70}  # long enough for scripted red to be reached and thrown at
    seeds0 = ert.round_seeds(big_cfg, 1, 0)
    found, part = ert.mi.collect_mixture(client, big_cfg, None, seeds0, source="t", block_worlds=big_cfg["roundBlockWorlds"],
                                         account=lambda n: None)
    assert int((part["labels"]["action_type"] == 2).sum()) > 0, "test fixture collected no THROW rows at all"
    return found, part


def test_new_decoder_act_and_loss_run_against_a_real_collected_batch(client):
    cfg = tiny()
    found, part = collect_round_zero_for_test(client, cfg)
    model = ert.FullAuthorityPolicyV1EnemyThrow(destination="global")
    # part["observation"] only carries fi.ACTOR_KEYS (no mission_progress); imitation_loss uses with_value=False,
    # matching this. act() needs a full live observation (mission_progress included), as DAgger rollout supplies.
    losses = ert.imitation_loss_enemy_relative(model, part["observation"], part["labels"], cfg)
    assert torch.isfinite(losses["total"]) and losses["throwUnits"] > 0
    plan, spec = teacher_option_plan("engage")
    with scenario_override(rb.arm_scenario("normal")):
        wrapper = ert.v1.make_wrapper(client, 2, cfg["gamma"])
        live_obs, _ = wrapper.reset([9801, 9802], [ert.v1.scenario()] * 2, ["p", "q"], [plan] * 2, [spec] * 2)
    action, latent, logp, value = model.act(ert.v1.tensor_dict(live_obs), deterministic=True)
    assert action["target"].abs().max() <= 1.0 + 1e-5


def test_fit_enemy_relative_reduces_loss_when_overfitting_a_tiny_batch(client):
    cfg = tiny()
    found, part = collect_round_zero_for_test(client, cfg)
    model = ert.FullAuthorityPolicyV1EnemyThrow(destination="global")
    optimizer = torch.optim.Adam(fi.imitation_parameters(model), lr=3e-4)
    data = fi.Aggregate()
    data.add(part)
    history = ert.fit_enemy_relative(model, optimizer, data, {**cfg, "stepsPerFit": 30}, seed=1)
    assert history[-1]["angleLoss"] < history[0]["angleLoss"]


def test_tiny_end_to_end_run_trains_both_decoders_for_one_cohort_and_seals(tmp_path, client):
    cfg = tiny()
    report = ert.run(tmp_path / "run", cfg)
    assert set(report["cohorts"]["1"]) == {"old", "new"}
    for decoder, entry in report["cohorts"]["1"].items():
        assert set(entry["pairedEval"]) == {"normal", "easy"}
        assert entry["pairedEval"]["normal"]["episodes"] == 4
    assert report["withinBudgetBound"]
    assert dr.verify_sealed(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ert.declare(tmp_path / "run", cfg)
