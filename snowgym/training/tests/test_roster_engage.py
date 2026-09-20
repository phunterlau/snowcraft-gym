"""M8-S3 checks (`reviews/m8_s3_declaration.md` section 3): the existing actor, critic, option
tracker and collector at roster 3 on the batch path, and Red's action source."""

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_client.encoding import ACTION_NOOP, ACTION_THROW
from snowgym_training.executor.full_authority_ppo_v1 import FullAuthorityPolicyV1
from snowgym_training.options import full_authority_train_v1 as v1
from snowgym_training.options import roster_engage as re
from snowgym_training.options.plans import teacher_option_plan
from snowgym_training.ppo import living_unit_mask
from snowgym_training.ppo_collect import numpy_actions, tensor_dict

GAMMA = 0.9976921765


@pytest.fixture(scope="module")
def client():
    with SnowGymBatchClient() as opened:
        yield opened


def reset_wrapper(client, seeds, units=3, arm="normal"):
    plan, spec = teacher_option_plan("engage")
    wrapper = v1.make_wrapper(client, len(seeds), GAMMA)
    observation, _ = wrapper.reset(seeds, [re.roster_scenario(units, arm)] * len(seeds),
                                   [f"roster-{seed}" for seed in seeds], [plan] * len(seeds), [spec] * len(seeds))
    return wrapper, tensor_dict(observation)


def model_for_test():
    torch.manual_seed(0)
    return FullAuthorityPolicyV1(destination="global")


# -- pure ---------------------------------------------------------------------------------


def test_roster_scenario_reuses_the_r1n_arena_and_selects_the_declared_red():
    normal, easy, rand = (re.roster_scenario(3, arm) for arm in ("normal", "easy", "random"))
    assert (normal["blueUnits"], normal["redUnits"]) == (3, 3)
    assert normal["arenaWidth"] == v1.scenario()["arenaWidth"] and normal["maxTicks"] == v1.scenario()["maxTicks"]
    assert (normal["redController"], normal["redDifficulty"]) == ("scripted", "normal")
    assert (easy["redController"], easy["redDifficulty"]) == ("scripted", "easy")
    assert rand["redController"] == "random"
    with pytest.raises(ValueError):
        re.roster_scenario(0)


def test_permute_ally_slots_reorders_only_slot_indexed_tensors_and_rejects_bad_permutations():
    observation = {"allies": torch.arange(2 * 10 * 2).reshape(2, 10, 2), "tick": torch.ones(2, 1),
                   "ally_mask": torch.ones(2, 10), "unit_action_mask": torch.ones(2, 10, 4),
                   "plan_unit_roles": torch.arange(2 * 10 * 3).reshape(2, 10, 3)}
    moved = re.permute_ally_slots(observation, [2, 0, 1])
    assert torch.equal(moved["allies"][:, 0], observation["allies"][:, 2])
    assert torch.equal(moved["allies"][:, 5:], observation["allies"][:, 5:])
    assert torch.equal(moved["plan_unit_roles"][:, 1], observation["plan_unit_roles"][:, 0])
    assert moved["tick"] is observation["tick"]
    for bad in ([0, 0, 1], [0, 1, 3], [1, 2]):
        with pytest.raises(ValueError):
            re.permute_ally_slots(observation, bad)


# -- live: the existing machinery at roster 3 ---------------------------------------------


def test_the_engage_path_runs_at_3v3_with_all_units_assigned_and_all_targets_activated(client):
    wrapper, observation = reset_wrapper(client, [1, 2])
    tracker = wrapper.trackers[0]
    assert tracker.assigned_ids == (1, 2, 3) and tracker.activated_target_ids == (4, 5, 6)
    assert observation["allies"].shape == (2, 10, 21) and observation["option_state"].shape == (2, 3)
    assert torch.equal(observation["option_state"], torch.ones(2, 3))
    assert living_unit_mask(observation).sum(-1).tolist() == [3, 3]
    action, latent, logp, value = model_for_test().act(observation)
    assert action["action_type"].shape == (2, 10) and logp.shape == (2, 10) and value.shape == (2,)
    live = living_unit_mask(observation)
    assert torch.all(logp[~live] == 0), "unused slots must contribute zero likelihood"


@pytest.mark.parametrize("permutation", [[1, 2, 0], [2, 1, 0]])
def test_actor_and_critic_are_equivariant_in_the_unit_slot_order(client, permutation):
    wrapper, observation = reset_wrapper(client, [3, 4])
    model = model_for_test()
    for _ in range(4):  # move off the symmetric spawn so slots genuinely differ
        action, *_ = model.act(observation)
        observation = tensor_dict(wrapper.step(numpy_actions(action))[0])
    gaps = re.equivariance_gap(model, observation, permutation)
    assert max(gaps.values()) < 1e-4, gaps


def test_features_do_not_mutate_the_shared_observation(client):
    _, observation = reset_wrapper(client, [5])
    before = {name: value.clone() for name, value in observation.items()}
    model = model_for_test()
    for _ in range(3):  # N per-unit reads of one team tensor must see identical inputs
        model.features(observation)
        model(observation)
    assert all(torch.equal(before[name], observation[name]) for name in before)


def test_act_likelihood_matches_reevaluation_of_its_own_latents_at_roster_3(client):
    _, observation = reset_wrapper(client, [6, 7])
    model = model_for_test()
    action, latent, logp, _ = model.act(observation)
    again, _ = model.evaluate_latents(observation, action["action_type"], latent)
    torch.testing.assert_close(again, logp)


def test_a_unit_death_mid_option_is_masked_ignored_and_does_not_end_the_option(client):
    seeds = list(range(100, 108))
    wrapper, observation = reset_wrapper(client, seeds)
    model = model_for_test()
    active = list(range(len(seeds)))
    death = None
    for _ in range(200):
        action, latent, logp, value = model.act(observation)
        picked = numpy_actions(action)
        stepped, _, done, _, _ = wrapper.step_indices(active, {k: v[active] for k, v in picked.items()})
        stepped = tensor_dict(stepped)
        for row, index in enumerate(active):
            for name in observation:
                observation[name][index] = stepped[name][row]
            alive = observation["allies"][index, :3, 1] > 0.5
            if death is None and not alive.all() and not done[row]:
                death = (index, int((~alive).nonzero()[0]))
        if death is not None:
            break
        active = [i for row, i in enumerate(active) if not done[row]]
    assert death is not None, "no blue unit died mid-option under the random-init policy"
    index, slot = death
    assert wrapper.trackers[index].finished is False
    assert not living_unit_mask(observation)[index, slot], "dead unit must leave the living mask"
    assert observation["ally_mask"][index, slot], "its slot stays present, never reindexed"
    _, _, logp, _ = model.act(observation)  # freshly sampled on the post-death observation
    assert logp[index, slot] == 0, "a dead unit contributes no likelihood"


def test_dead_slot_actions_do_not_change_the_simulation(client):
    """Two worlds, same seed and policy: one submits THROW for a dead unit's slot, the other NOOP."""
    model = model_for_test()
    seed = 100
    hashes = {}
    for label, forced_type in (("throw", ACTION_THROW), ("noop", ACTION_NOOP)):
        wrapper, observation = reset_wrapper(client, [seed])
        torch.manual_seed(1)
        dead = None
        trace = []
        for step in range(120):
            action, *_ = model.act(observation)
            picked = numpy_actions(action)
            if dead is not None:
                picked["action_type"][0, dead] = forced_type
                picked["target"][0, dead] = [0.5, 0.5]
                picked["power"][0, dead] = 1.0 if forced_type == ACTION_THROW else 0.0
            stepped, _, done, _, infos = wrapper.step_indices([0], picked)
            observation = tensor_dict(stepped)
            trace.append(infos[0]["stateHash"])
            alive = observation["allies"][0, :3, 1] > 0.5
            if dead is None and not alive.all():
                dead = int((~alive).nonzero()[0])
            if done[0]:
                break
        assert dead is not None
        hashes[label] = trace
    assert hashes["throw"] == hashes["noop"]
