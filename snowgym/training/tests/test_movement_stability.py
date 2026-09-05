import copy

import pytest
import torch

from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options.movement_train import ppo_update
from snowgym_training.options.movement_stability import (ARMS, actor_identity,
    configuration, decision, run)
from test_scoped_movement import inputs


@pytest.mark.parametrize("stop", [1e-8, 1e9])
def test_independent_schedule_preserves_actor_adam_and_collection_rng(inputs, stop):
    reference, obs = inputs
    obs = {k: v[:16] for k, v in obs.items()}
    torch.manual_seed(9910)
    with torch.no_grad():
        action, latent, old, _ = reference.act(obs)
    rollout = {"observation": obs, "action_type": action["action_type"], "latent": latent,
        "logp": old, "advantage": torch.linspace(-1, 1, 16), "returns": torch.ones(16),
        "reward": torch.zeros(16)}
    outputs = []
    for arm in ARMS:
        model = copy.deepcopy(reference)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        torch.manual_seed(9920)
        trace = ppo_update(model, optimizer, rollout, {**configuration(arm),
                           "epochs": 3, "minibatchSize": 4, "movementKlStop": stop})
        outputs.append((actor_identity(model, optimizer), torch.get_rng_state(),
                        semantic_state_digest(model.critic.state_dict()), trace))
        assert all(p.grad is None for p in model.geometry.source.parameters())
        assert all(p.grad is None for p in model.geometry.shot.parameters())
    a, b = outputs
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1])
    assert b[3]["criticOptimizerSteps"] == 12
    assert b[3]["actorOptimizerSteps"] == a[3]["optimizerSteps"]
    if stop < 1:
        assert a[3]["klStopped"] and a[3]["optimizerSteps"] < 12
        assert a[2] != b[2]
        assert all(row["actorGradientNorm"] == 0 for row in b[3]["minibatches"] if row["criticOnly"])
    else:
        assert not a[3]["klStopped"] and a[2] == b[2]


def test_configuration_refusal_and_fixed_decision(tmp_path):
    with pytest.raises(ValueError):
        configuration("best-arm")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    row = {"effect": {"success": {"mean": .2, "ci95": [.01, .4]}}, "rejectedActionRate": 0}
    cases = [{"seed": seed, "development": copy.deepcopy(row), "replicationDevelopment": copy.deepcopy(row)}
             for seed in (94001, 94002, 94003)]
    assert decision(cases)["promising"]
    assert not decision(cases)["autonomousQualificationEligible"]
    cases[2]["replicationDevelopment"]["effect"]["success"]["mean"] = -.01
    assert not decision(cases)["promising"]
