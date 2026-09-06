import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import horizon_audit as a


def test_coverage_boundaries_duplicates_and_invalid_indices():
    rows = [{'summary': {'learnedDecisions': 32}}, {'summary': {'learnedDecisions': 1}}]
    result = a.coverage(rows, [0, 29, 30, 30, 31, 32])
    assert result['early']['collected'] == 31
    assert result['early']['selectedOccurrences'] == 3
    assert result['late'] == {'collected': 2, 'selectedOccurrences': 3, 'uniqueSelected': 2, 'selectionFraction': 1.}
    with pytest.raises(ValueError): a.coverage(rows, [-1])
    with pytest.raises(ValueError): a.coverage(rows, [33])


def test_duplicate_output_scores_match_autograd_and_zero_unused():
    torch.manual_seed(12)
    means = torch.randn(3, 4, 2, dtype=torch.float64)
    latents = means + torch.randn_like(means)*.02
    advantages = torch.tensor([1., -2., 4.], dtype=torch.float64)
    living = torch.tensor([[1,1,1,1], [1,1,0,0], [1,1,1,0]], dtype=torch.bool)
    moves = living.clone(); moves[:, 0] = False
    selected = [0, 0, 1, 2, 2, 1]
    order = torch.tensor([4,0,2,5,1,3])
    combined, _, _ = a.output_scores(means, latents, advantages, living, moves, selected, order, 3, .02)
    variable = means.clone().requires_grad_()
    total = 0
    for positions in order.split(3):
        indices = torch.tensor(selected)[positions]
        adv = advantages[indices]; adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        new = torch.distributions.Normal(variable[indices], .02).log_prob(latents[indices]).sum(-1)
        old = torch.distributions.Normal(means[indices], .02).log_prob(latents[indices]).sum(-1)
        total = total + (((new-old).exp()*adv[:,None]*moves[indices]).sum(-1)/living[indices].sum(-1)).mean()
    gradient, = torch.autograd.grad(total, variable)
    torch.testing.assert_close(combined, gradient, rtol=1e-12, atol=1e-12)
    assert (gradient[~moves] == 0).all()


def test_credit_decomposition_and_empty_stratum():
    batch = {'advantage': torch.tensor([1.,1.,3.,3.]), 'returns': torch.tensor([1.,1.,3.,3.]),
        'value': torch.zeros(4)}
    result = a.credit(batch, [0,0,1,1], [0,0,1,2,3])
    assert result['betweenEpisodeAdvantageFraction'] == pytest.approx(1.)
    assert result['criticExplainedVariance'] == 0
    assert a.credit(batch, [0,0,1,1], [])['decisions'] == 0


@pytest.mark.parametrize('arm', a.h.ARMS)
def test_live_frozen_episode_observer_parity(arm):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source, data, cfg, _ = a.inputs()
    model, sampler = a.behavior(source, arm, 1, cfg)
    frame = data['datasets']['training'][sampler.integers(len(data['datasets']['training']), size=8)[0]]
    root = a.ARCHIVE/str(a.SEED)/arm
    archived = a.read_rows(root/'events-001.jsonl.gz')[0]
    before = a.semantic_state_digest(model.state_dict())
    with SnowGymBatchClient() as client:
        wrapper = a.h.b.make_wrapper(client, 1, cfg['gamma'])
        observer = a.Observer(model, model, wrapper, frame['seed'], 0, 0, cfg['latentStd'])
        rng = torch.get_rng_state().clone()
        batch, _, trace = a.h.episode(observer, source, wrapper, frame, cfg, arm, sampling_seed=a.SEED*100000)
        assert torch.equal(rng, torch.get_rng_state())
        assert a.json_digest(trace) == a.json_digest(archived)
        assert len(observer.predictions) == len(batch['advantage'])
        assert all(x['finalWorldShift'] == 0 for x in observer.rows if x['recommendationAvailable'])
        assert all(x['offset'] < (30 if arm == 'short' else 200) for x in observer.rows)
    assert a.semantic_state_digest(model.state_dict()) == before


def test_immutable_output_and_manifest_tamper(tmp_path):
    with pytest.raises(FileExistsError): a.run(tmp_path)
    root = tmp_path/'archive'; root.mkdir()
    (root/'data.json').write_text('{}')
    manifest = {'artifacts': {'data.json': a.h.b.file_digest(root/'data.json')}}
    manifest['manifestDigest'] = a.json_digest(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest))
    a.h.b.verified_manifest(root)
    (root/'data.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError): a.h.b.verified_manifest(root)


def test_live_first_minibatch_full_reconstruction():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source, data, cfg, _ = a.inputs()
    path = a.ARCHIVE/str(a.SEED)/'full'
    final, _, _ = a.h.r.recovery_checkpoint.load(path/'final')
    final.eval().requires_grad_(False)
    history = json.loads((path/'training.json').read_text())['history']
    with SnowGymBatchClient() as client:
        result, rows = a.replay_update(source, final, a.h.b.make_wrapper(client, 1, cfg['gamma']),
            data['datasets']['training'], cfg, 'full', 1, a.read_rows(path/'events-001.jsonl.gz'), history[0])
    assert result['completeTrajectoryMatches'] == 8
    assert result['firstMinibatchLossMaxError'] < 1e-6
    assert result['simulatorDecisions'] <= 1600
    assert sum(x['firstMinibatchMultiplicity'] for x in rows) > 0
    assert result['strata']['late']['collectedCredit']['decisions'] > 0
