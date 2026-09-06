import copy
import gzip
import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.executor.recovery_ppo import RecoveryPolicy
from snowgym_training.executor.movement_ppo import movement_loss
from snowgym_training.options import horizon_train as h


def test_sample_budgets_and_paired_streams():
    for size in (5,240,800):
        a=h.select_rows(size,240,99301)
        assert len(a)==240 and min(a)>=0 and max(a)<size
        np.testing.assert_array_equal(a,h.select_rows(size,240,99301))
        assert len(set(a))==min(size,240)
    with pytest.raises(ValueError):h.select_rows(0,240,1)
    cfg=h.configuration(); frame={'trigger':{'decision':155}}
    assert h.episode_config(cfg,'short',frame)['recoveryWindow']==30
    assert h.episode_config(cfg,'full',frame)['recoveryWindow']==45
    with pytest.raises(ValueError):h.episode_config(cfg,'bad',frame)
    frame['trigger']['decision']=200
    with pytest.raises(ValueError):h.episode_config(cfg,'full',frame)


def test_paired_gate_keeps_initializer_and_control_requirements():
    def row(seed,success):
        return {'seed':seed,'success':success,'progress':float(success),'damageDealt':10.,'damageReceived':0.,
            'livingFraction':1.,'discounted':{'executor':float(success)},'rejectedActions':0,'totalActions':100}
    baseline=[row(i,i<2) for i in range(10)]
    short=[row(i,i<3) for i in range(10)];full=[row(i,i<9) for i in range(10)]
    assert h.gate({'short':short,'full':full},baseline,1)['passed']
    assert not h.gate({'short':full,'full':full},baseline,1)['passed']
    assert not h.gate({'short':short,'full':full},baseline,0)['passed']
    with pytest.raises(ValueError):h.paired(full,list(reversed(full)))


@pytest.fixture(scope='module')
def reference():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source,metadata,data,lineage=h.inputs()
    return source,metadata,data['datasets']['training'][0],lineage


def test_live_parity_sampling_likelihood_masks_and_control_time(reference,tmp_path):
    source,_,frame,_=reference;cfg=h.configuration()
    torch.manual_seed(99301);model=RecoveryPolicy(copy.deepcopy(source),standard_deviation=cfg['latentStd'])
    before=semantic_state_digest(model.state_dict());results={}
    with SnowGymBatchClient() as client:
        wrapper=h.b.make_wrapper(client,1,cfg['gamma'])
        for arm in h.ARMS:
            batch,row,trace=h.episode(model,source,wrapper,frame,cfg,arm,deterministic=True)
            assert row['stateHashes']==frame['suffixHashes'] and row['actionsDigest']==frame['suffixActionsDigest']
            assert row['tailDecisions']==(0 if arm=='full' else row['exposureDecisions']-min(30,row['exposureDecisions']))
            assert row['simulatorDecisions']<=200
            assert not row['autonomousQualificationEligible']
            h.diagnostics(model,batch)
            state=torch.get_rng_state().clone()
            sample,_,trajectory=h.episode(model,source,wrapper,frame,cfg,arm,sampling_seed=12345)
            assert torch.equal(state,torch.get_rng_state())
            repeat,_,other=h.episode(model,source,wrapper,frame,cfg,arm,sampling_seed=12345)
            assert h.json_digest(trajectory)==h.json_digest(other)
            torch.testing.assert_close(sample['logp'],repeat['logp'],atol=0,rtol=0)
            results[arm]=sample
        for key in ('latent','action_type'):
            torch.testing.assert_close(results['short'][key],results['full'][key][:len(results['short'][key])],atol=0,rtol=0)
        sample=results['full'];new,pred=model.evaluate_latents(sample['observation'],sample['action_type'],sample['latent'])
        torch.testing.assert_close(new,sample['logp'],atol=1e-3,rtol=0)
        loss=movement_loss(new,sample['logp'],sample['advantage'],pred['value'],sample['returns'],pred)
        model.zero_grad(set_to_none=True);loss['policy'].backward()
        assert all(p.grad is None for p in model.critic.parameters())
        assert all(p.grad is None for p in model.geometry.shot.parameters())
        assert all(p.grad is None for p in model.geometry.source.parameters())
        assert model.budget_move.weight.grad is not None
        assert (sample['logp'][sample['action_type']!=1]==0).all()
    assert semantic_state_digest(model.state_dict())==before
    with pytest.raises(FileExistsError):h.run(tmp_path)


@pytest.mark.parametrize('arm',h.ARMS)
def test_exact_update_resume_and_lineage_rejection(reference,tmp_path,arm):
    source,metadata,frame,lineage=reference
    cfg={**h.configuration(),'updates':2,'batchSize':1,'rowsPerUpdate':12,'epochs':1,'minibatchSize':6}
    with SnowGymBatchClient() as client:
        full,report=h.train(source,metadata,client,[frame],cfg,tmp_path/'whole',99301,arm)
        h.train(source,metadata,client,[frame],cfg,tmp_path/'paused',99301,arm,pause_after=1)
        resumed,other=h.train(source,metadata,client,[frame],cfg,tmp_path/'resumed',99301,arm,resume=tmp_path/'paused/paused')
        assert report['history']==other['history']
        assert semantic_state_digest(full.state_dict())==semantic_state_digest(resumed.state_dict())
        assert report['checkpointDigest']==other['checkpointDigest']
        assert report['actorParameterL2Change']==other['actorParameterL2Change']>0
        with pytest.raises(ValueError,match='lineage mismatch'):
            h.train(source,metadata,client,[frame],{**cfg,'rowsPerUpdate':10},tmp_path/'bad',99301,arm,resume=tmp_path/'paused/paused')
    assert h.inputs()[-1]==lineage


def test_archived_horizon_budgets_actions_rewards_and_gates():
    root=h.b.TRAINING/'runs/m7b_engage_r1m_s9_v0'
    h.b.verified_manifest(root)
    cfg=json.loads((root/'declaration.json').read_text())['config']
    baselines=json.loads((root/'initialization.json').read_text())
    report=json.loads((root/'report.json').read_text())
    steps=0
    for split in ('training','historical','replication'):
        for a,z in zip(baselines['short'][split],baselines['full'][split],strict=True):
            for key in ('stateHashes','actionsDigest','success','progress','discounted'):
                assert a[key]==z[key]
            steps+=a['simulatorDecisions']+z['simulatorDecisions']
    for seed,comparison in report['runs'].items():
        histories={};evaluations={};training={}
        for arm in h.ARMS:
            path=root/seed/arm
            training[arm]=json.loads((path/'training.json').read_text())
            history=training[arm]['history'];histories[arm]=history
            assert len(history)==cfg['updates']
            for update,entry in enumerate(history,1):
                assert entry['presentedRows']==cfg['rowsPerUpdate']
                assert entry['uniqueSelectedRows']==min(entry['collectedRows'],cfg['rowsPerUpdate'])
                assert entry['optimizerSteps']<=8
                assert entry['likelihoodMaxError']<=1e-3
                with gzip.open(path/f'events-{update:03d}.jsonl.gz','rt') as stream:
                    episodes=[json.loads(line) for line in stream]
                assert len(episodes)==cfg['batchSize']
                assert [t['summary']['seed'] for t in episodes]==entry['seeds']
                for t in episodes:
                    row=t['summary']; events=t['events'];steps+=row['simulatorDecisions']
                    assert row['simulatorDecisions']<=200
                    assert row['actionsDigest']==h.json_digest([e['action'] for e in events])
                    assert row['stateHashes'][1:]==[e['stateHash'] for e in events]
                    count=row['learnedDecisions']
                    assert len(t['learnerLatents'])==len(t['behaviorLogProbabilities'])==count
                    assert all(e['learnerControlled']==(i<count) for i,e in enumerate(events))
                    assert count==len(events) if arm=='full' else count==min(30,len(events))
                    for i,e in enumerate(events[:count]):
                        for j,kind in enumerate(e['action']['action_type'][0]):
                            if kind!=1:
                                assert t['behaviorLogProbabilities'][i][j]==0
                                assert t['learnerLatents'][i][j]==[0,0]
                    for key,value in row['discounted'].items():
                        assert value==pytest.approx(sum(cfg['gamma']**i*e['info']['option']['rewards'][key] for i,e in enumerate(events)))
                assert sum(t['summary']['learnedDecisions'] for t in episodes)==entry['collectedRows']
                assert sum(t['summary']['simulatorDecisions'] for t in episodes)==entry['simulatorDecisions']
                assert entry['simulatorDecisions']==entry['prefixDecisions']+entry['learnedDecisions']+entry['tailDecisions']
            evaluations[arm]=json.loads((path/'evaluation.json').read_text())
            steps+=sum(x['simulatorDecisions'] for rows in evaluations[arm].values() for x in rows)
        assert training['short']['initialStateDigest']==training['full']['initialStateDigest']
        assert [x['seeds'] for x in histories['short']]==[x['seeds'] for x in histories['full']]
        for split in ('historical','replication'):
            actual=h.gate({a:evaluations[a][split] for a in h.ARMS},baselines['short'][split],training['full']['actorParameterL2Change'])
            assert actual==comparison['gates'][split]
    assert steps==report['simulatorDecisions']<=cfg['simulatorBudget']
    first=report['runs'][str(cfg['trainingRngs'][0])]
    assert len(report['runs'])==(3 if all(g['passed'] for g in first['gates'].values()) else 1)
