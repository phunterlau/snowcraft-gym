import copy

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options import continuation_probe as c
from snowgym_training.trajectory import json_digest


def test_selection_uses_only_nonterminal_boundary(monkeypatch):
    selection = {'selected': [{'seed': 1}, {'seed': 2}, {'seed': 3}]}
    monkeypatch.setattr(c.h, 'load_row', lambda seed, arm: {'trace': [None]*{1:30, 2:31, 3:8}[seed]})
    eligible, excluded = c.select(selection)
    assert eligible == [{'seed': 2}]
    assert [r['seed'] for r in excluded] == [1, 3]
    with pytest.raises(ValueError, match='no nonterminal'):
        c.select({'selected': [{'seed': 1}]})


def test_sustained_substitution_only_changes_living_selected_movement():
    first={'action_type':np.array([[1,2,1,0]]), 'target':np.zeros((1,4,2)), 'power':np.ones((1,4))}
    raw={'allies':[{'id':i,'alive':i!=3} for i in (1,2,3,4)]}
    rec={'valid':np.array([[True,False,False,False]]), 'target':np.ones((1,4,2))}
    changed,dose=c.d.intervene(first,raw,rec,'squad-30',0,1)
    assert dose['overriddenIds']==[1]
    np.testing.assert_array_equal(changed['target'][0,1:],first['target'][0,1:])
    for key in ('action_type','power'):
        np.testing.assert_array_equal(changed[key],first[key])
    assert not first['target'].any()
    rec['valid'][0,0]=False
    with pytest.raises(ValueError,match='lacks a recommendation'):
        c.d.intervene(first,raw,rec,'squad-30',0,1)


def test_paired_gate_and_denominators():
    def row(seed,success):
        return {'seed':seed,'final':dict.fromkeys(('success','progress','damageDealt','damageReceived','livingFraction','exposureDecisions'),success),
            'discounted':{'executor':float(success)},'rejectedActions':0,'totalActions':100,
            'overriddenMoves':1,'changedTargets':1,'exposure':dict.fromkeys(c.h.METRICS,None)}
    groups=[{'source':row(i,False),'move-rest':row(i,True)} for i in range(4)]
    report=c.summarize(groups)
    assert report['diagnosticContinuationGate']['passed']
    assert report['recoveredSeeds']==list(range(4)) and not report['lostSeeds']
    assert report['pairedDifference']['success']['mean']==1
    assert report['arms']['source']['exposure']['rangeOccupancy']['mean'] is None
    groups[0]['move-rest']['rejectedActions']=1
    assert not c.summarize(groups)['diagnosticContinuationGate']['passed']
    with pytest.raises(ValueError,match='empty paired'):
        c.summarize([])


def test_live_handoff_source_suffix_repeats_tamper_and_source_preservation(tmp_path):
    with pytest.raises(FileExistsError):
        c.run(tmp_path)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source,_,selection,cfg,manifests=c.inputs()
    eligible,excluded=c.select(selection)
    assert len(eligible)==23 and excluded==[{'seed':100011,'reason':'terminal-at-or-before-handoff','decisions':8}]
    pick=eligible[0]; frame=next(f for f in selection['frames'] if f['seed']==pick['seed'])
    before=semantic_state_digest(source.state_dict())
    with SnowGymBatchClient() as client:
        wrapper=c.b.make_wrapper(client,1,cfg['config']['gamma'])
        rows={arm:c.branch(source,wrapper,frame,pick,arm,cfg['config']['gamma']) for arm in c.ARMS}
        assert rows['source']['startIdentity']==rows['move-rest']['startIdentity']
        assert rows['source']['prefix']==rows['move-rest']['prefix']
        for arm,row in rows.items():
            assert row['simulatorDecisions']<=200 and not row['autonomousQualificationEligible']
            assert row['final']['exposureDecisions']<=row['startSnapshot']['remainingBudget']
            assert row['actionsDigest']==json_digest([e['action'] for e in row['trace']])
            assert json_digest(c.branch(source,wrapper,frame,pick,arm,cfg['config']['gamma']))==json_digest(row)
            for e in row['trace']:
                for key in ('action_type','power'):
                    assert e['action'][key]==e['sourceAction'][key]
        archived=c.h.load_row(pick['seed'],'squad-30')
        bad=copy.deepcopy(archived);bad['trace'][1]['reward']+=.1
        with pytest.raises(ValueError,match='transition mismatch'):
            c.restore(wrapper,frame,pick,bad)
        bad=copy.deepcopy(pick);bad['identity']['physical']='tampered'
        with pytest.raises(ValueError,match='identity mismatch'):
            c.restore(wrapper,frame,bad,archived)
        bad=copy.deepcopy(archived);bad['trace'][0]['action']['target'][0][0][0]+=.1
        with pytest.raises(ValueError,match='invalid handoff archive'):
            c.restore(wrapper,frame,pick,bad)
    assert semantic_state_digest(source.state_dict())==before
    assert c.inputs()[-1]==manifests
