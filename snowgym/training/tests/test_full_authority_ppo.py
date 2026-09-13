import math

import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.full_authority_ppo import FullAuthorityPolicy
from snowgym_training.options import full_authority_train as f
from snowgym_training.options.plans import teacher_option_plan
from snowgym_training.ppo_collect import numpy_actions, tensor_dict


def test_local_and_global_decode_differ_and_local_stays_near_own_position():
    torch.manual_seed(0)
    local = FullAuthorityPolicy(destination="local", local_radius=8.0)
    global_ = FullAuthorityPolicy(destination="global")
    observation = {"allies": torch.zeros(1, 1, 21)}
    observation["allies"][0, 0, 2:4] = torch.tensor([0.2, -0.3])  # own normalized position
    latent = torch.full((1, 1, 2), 3.0)  # a large latent, tanh(3) ~= 0.995
    local_target = local.decode_move(observation, latent)
    global_target = global_.decode_move(observation, latent)
    torch.testing.assert_close(global_target, torch.tanh(latent))
    scale = torch.tensor([50., 40.])
    norm = latent[0, 0].norm()
    expected_offset = 8.0 * torch.tanh(norm) * (latent[0, 0] / norm)
    expected_local = (observation["allies"][0, 0, 2:4] + expected_offset / scale).clamp(-1, 1)
    torch.testing.assert_close(local_target[0, 0], expected_local)
    assert not torch.allclose(local_target, global_target)
    # The local decode is an isotropic radius: even this diagonal (equal-component) latent,
    # which would reach R*sqrt(2)~=11.3 under a naive per-axis bound, stays within R=8.
    world_offset = (local_target[0, 0] - observation["allies"][0, 0, 2:4]) * scale
    assert float(world_offset.norm()) <= 8.0 + 1e-4


def test_invalid_destination_and_radius_are_rejected():
    with pytest.raises(ValueError):
        FullAuthorityPolicy(destination="diagonal")
    with pytest.raises(ValueError):
        FullAuthorityPolicy(destination="local", local_radius=0.)


@pytest.fixture(scope="module")
def source_free_setup():
    torch.set_num_threads(1)
    return teacher_option_plan("engage")


def test_act_and_evaluate_latents_agree_and_respect_masks(source_free_setup):
    plan, spec = source_free_setup
    torch.manual_seed(3)
    model = FullAuthorityPolicy(destination="local")
    with SnowGymBatchClient() as client:
        wrapper = f.make_wrapper(client, 2, f.configuration()["gamma"])
        obs, _ = wrapper.reset([800000, 800001], [f.scenario()] * 2, ["m0", "m1"], [plan, plan], [spec, spec])
        obs = tensor_dict(obs)
        action, latent, logp, value = model.act(obs)
        logp2, extra = model.evaluate_latents(obs, action["action_type"], latent)
        torch.testing.assert_close(logp, logp2)
        assert value.shape == (2,)
        assert action["target"].shape[-1] == 2 and action["power"].shape == action["action_type"].shape
        # Illegal action types (per unit_action_mask) must never be sampled for a living unit.
        # Padding slots (no unit present) get an all-zero mask -- no action is "legal" there --
        # and the environment ignores whatever is submitted for them (confirmed by zero rejections below).
        live = f.living_unit_mask(obs)
        legal = obs["unit_action_mask"].bool()
        chosen_legal = legal.gather(-1, action["action_type"][..., None]).squeeze(-1)
        assert bool(chosen_legal[live].all())
        # Executing the sampled action must not raise (mask makes it legal).
        executed = numpy_actions(action)
        _, _, terminated, truncated, infos = wrapper.step(executed)
        rejected = sum(r.get("accepted") is False for r in infos[0].get("actionResults", []))
        assert rejected == 0


def test_deterministic_act_is_argmax_and_reproducible(source_free_setup):
    plan, spec = source_free_setup
    torch.manual_seed(4)
    model = FullAuthorityPolicy(destination="global")
    with SnowGymBatchClient() as client:
        wrapper = f.make_wrapper(client, 1, f.configuration()["gamma"])
        obs, _ = wrapper.reset([800002], [f.scenario()], ["m2"], [plan], [spec])
        obs = tensor_dict(obs)
        with torch.no_grad():
            prediction = model(obs)
            action_a, _, _, _ = model.act(obs, deterministic=True)
            action_b, _, _, _ = model.act(obs, deterministic=True)
        torch.testing.assert_close(action_a["action_type"], prediction["action_logits"].argmax(-1))
        torch.testing.assert_close(action_a["target"], action_b["target"])


def test_parameter_groups_exclude_critic_from_actor_parameters():
    torch.manual_seed(5)
    model = FullAuthorityPolicy(destination="local")
    actor_names = {n for n, p in model.named_parameters() if p in set(model.actor_parameters())}
    assert all(not n.startswith("critic.") for n in actor_names)
    assert any(n.startswith("action_head") for n in actor_names)
    assert any(n.startswith("target_log_std") or n == "target_log_std" for n in actor_names)
