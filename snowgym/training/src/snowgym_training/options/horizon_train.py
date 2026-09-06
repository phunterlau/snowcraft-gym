"""R1m-S9: matched short versus full learnable movement control."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.recovery_ppo import RecoveryPolicy
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import continuation_probe as c
from .opportunity_audit import plain, write_jsonl
from .supervised_probe import write_json

b = c.b
r = b.recovery_train
ARMS = ('short', 'full')
ARCHIVE = b.TRAINING/'runs/m7b_engage_r1m_s8_v0'


def configuration():
    return {**r.configuration(), 'format':'snowgym.horizon-config.v0', 'trainingRngs':[99301,99302,99303],
        'arms':list(ARMS), 'rowsPerUpdate':240, 'bootstrapSeed':993001, 'simulatorBudget':430000,
        'collectionMode':'paired-first-hit-with-isolated-episode-rng', 'sampleMode':'uniform-decision-capped-with-deficit-resampling'}


def inputs():
    source, metadata, _, _, lineage = c.inputs()
    manifest = b.verified_manifest(ARCHIVE)
    decl = json.loads((ARCHIVE/'declaration.json').read_text())
    if decl['implementationDigest'] != b.file_digest(Path(c.__file__)):
        raise ValueError('S8 implementation drift')
    if decl['s6Manifest'] != b.file_digest(c.h.ARCHIVE/'manifest.json') or decl['s7Manifest'] != b.file_digest(c.ARCHIVE/'manifest.json'):
        raise ValueError('S8 ancestry drift')
    data = json.loads((b.ARCHIVE/'snapshots.json').read_text())
    if data['digest'] != json_digest(data['datasets']):
        raise ValueError('S3 dataset digest mismatch')
    seen=set()
    for name,count in (('training',57),('historical',38),('replication',36)):
        frames=data['datasets'][name]
        r.validate_frames(frames,training=name=='training')
        seeds={f['seed'] for f in frames}
        lower,upper={'training':(100000,100063),'historical':(200000,200039),'replication':(210000,210039)}[name]
        if len(frames)!=count or seen & seeds or any(not lower<=s<=upper for s in seeds):
            raise ValueError('horizon seed partition mismatch')
        seen|=seeds
    return source,metadata,data,{'ancestors':lineage,'s8':manifest}


def episode_config(cfg,arm,frame):
    if arm not in ARMS or not 0<frame['trigger']['decision']<200:
        raise ValueError('invalid horizon arm or first-hit time')
    return {**cfg,'recoveryWindow':30 if arm=='short' else 200-frame['trigger']['decision']}


def episode(model,source,wrapper,frame,cfg,arm,*,sampling_seed=None,deterministic=False):
    # Local streams align each episode's noise without changing optimizer RNG.
    with torch.random.fork_rng(devices=[]):
        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)
        rollout,row,trajectory=r.episode(model,source,wrapper,frame,episode_config(cfg,arm,frame),deterministic=deterministic)
    row.update({'arm':arm,'simulatorDecisions':len(frame['trigger']['prefix'])+len(trajectory['events'])})
    arena=wrapper.environment.raw_observations[0]['arena']
    if (arena['width'],arena['height'])!=(100,80):
        raise ValueError('horizon diagnostics require frozen 100 by 80 arena')
    gamma=cfg['gamma']
    row['discounted']={key:sum(gamma**i*e['info']['option']['rewards'][key] for i,e in enumerate(trajectory['events']))
        for key in ('mission','combat','shaping','canonical','executor')}
    if arm=='full' and row['tailDecisions']!=0:
        raise ValueError('full horizon unexpectedly has frozen tail')
    trajectory.update({'arm':arm,'samplingSeed':sampling_seed,'advantages':plain(rollout['advantage']),
        'returns':plain(rollout['returns']),'values':plain(rollout['value'])})
    return rollout,row,trajectory


def select_rows(size,count,seed):
    if size<1 or count<1:
        raise ValueError('invalid optimizer sample budget')
    rng=np.random.default_rng(seed)
    if size>=count:
        return rng.choice(size,count,replace=False)
    indices=np.concatenate([np.arange(size),rng.integers(size,size=count-size)])
    rng.shuffle(indices)
    return indices


def subset(rollout,indices):
    return {key:{k:v[indices] for k,v in value.items()} if key=='observation' else value[indices]
        for key,value in rollout.items()}


def diagnostics(model,rollout):
    with torch.no_grad():
        logp,pred=model.evaluate_latents(rollout['observation'],rollout['action_type'],rollout['latent'])
        error=float((logp-rollout['logp']).abs().max())
        returns=rollout['returns']; variance=float(returns.var(unbiased=False))
        ev=1-float((returns-pred['value']).var(unbiased=False))/variance if variance>1e-12 else None
    if error>1e-3:
        raise ValueError('horizon stored likelihood mismatch')
    return {'likelihoodMaxError':error,'criticExplainedVariance':ev,
        'advantageStd':float(rollout['advantage'].std(unbiased=False)),
        'livingMoveOpportunities':int(pred['move_mask'].sum())}


def train(source,metadata,client,frames,cfg,root,seed,arm,*,resume=None,pause_after=None):
    root=Path(root)
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    if arm not in ARMS:
        raise ValueError('invalid training arm')
    r.validate_frames(frames,training=True)
    cfg={**cfg,'arm':arm}; digest=json_digest(frames)
    torch.manual_seed(seed)
    model=RecoveryPolicy(copy.deepcopy(source),standard_deviation=cfg['latentStd'])
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and not n.startswith('critic.')}
    initial_digest=semantic_state_digest(model.state_dict())
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learningRate'])
    sampler=np.random.default_rng(seed); start=0; history=[]
    if resume:
        model,optimizer,saved=r.recovery_checkpoint.load(resume)
        if saved['config']!=cfg or saved['source']!=metadata or saved['datasetDigest']!=digest or saved['trainingSeed']!=seed:
            raise ValueError('horizon resume lineage mismatch')
        start=saved['update'];history=saved['history'];sampler.bit_generator.state=saved['sampler']
    root.mkdir(parents=True)
    wrapper=b.make_wrapper(client,1,cfg['gamma'])
    for update in range(start,cfg['updates']):
        chosen=sampler.integers(len(frames),size=cfg['batchSize']).tolist()
        batches=[];rows=[];trajectories=[]
        for slot,index in enumerate(chosen):
            batch,row,trajectory=episode(model,source,wrapper,frames[index],cfg,arm,
                sampling_seed=seed*100000+update*cfg['batchSize']+slot)
            batches.append(batch);rows.append(row);trajectories.append(trajectory)
        rollout=r.combine(batches)
        checks=diagnostics(model,rollout)
        indices=select_rows(len(rollout['advantage']),cfg['rowsPerUpdate'],seed*100000+update)
        selected=subset(rollout,indices)
        with torch.no_grad():
            old_mean=model(rollout['observation'])['mean'].detach().clone()
        # The shared unchanged update applies per-living-unit PPO and KL stopping.
        step=r.ppo_update(model,optimizer,selected,cfg)
        with torch.no_grad():
            _,prediction=model.evaluate_latents(rollout['observation'],rollout['action_type'],rollout['latent'])
            scale=torch.tensor([50.,40.])  # Frozen open arena checked in episode().
            shifts=(torch.tanh(prediction['mean'])-torch.tanh(old_mean))*scale
            # Common-state target change is not physical distance traveled.
            selected_shift=shifts.norm(dim=-1)[prediction['move_mask']]
        if semantic_state_digest(source.state_dict())!=semantic_state_digest(model.geometry.source.state_dict()):
            raise ValueError('horizon inherited source changed')
        step.update({'update':update+1,'seeds':[frames[i]['seed'] for i in chosen],**checks,
            'selectedRows':indices.tolist(),'presentedRows':len(indices),'uniqueSelectedRows':len(set(indices.tolist())),
            'collectedRows':len(rollout['advantage']),'selectedMoveOpportunities':int(((selected['action_type']==1)&(selected['observation']['allies'][...,1]>.5)).sum()),
            'successes':sum(x['success'] for x in rows),'simulatorDecisions':sum(x['simulatorDecisions'] for x in rows),
            'prefixDecisions':sum(x['simulatorDecisions']-x['learnedDecisions']-x['tailDecisions'] for x in rows),
            'learnedDecisions':sum(x['learnedDecisions'] for x in rows),'tailDecisions':sum(x['tailDecisions'] for x in rows),
            'meanWorldTargetUpdate':float(selected_shift.mean()) if selected_shift.numel() else None})
        history.append(step)
        write_jsonl(root/f'events-{update+1:03d}.jsonl.gz',trajectories)
        print(json.dumps({'seed':seed,'arm':arm,'update':update+1,'successes':step['successes'],
            'collectedRows':step['collectedRows'],'optimizerSteps':step['optimizerSteps']}),flush=True)
        name='paused' if pause_after==update+1 else 'final' if update+1==cfg['updates'] else f'update-{update+1:03d}'
        if name in ('paused','final') or (update+1)%cfg['checkpointEvery']==0:
            saved=r.recovery_checkpoint.save(root/name,model,optimizer,source=metadata,config=cfg,seed=seed,
                dataset_digest=digest,update=update+1,sampler=sampler.bit_generator.state,history=history)
        if name=='paused':
            return model,{'paused':True,'checkpointDigest':saved['checkpointDigest']}
    change=sum(float((p.detach()-initial[n]).square().sum()) for n,p in model.named_parameters() if n in initial)**.5
    report={'arm':arm,'history':history,'actorParameterL2Change':change,'initialStateDigest':initial_digest,
        'checkpointDigest':saved['checkpointDigest'],'datasetDigest':digest,'sourceUnchanged':True,**r.ASSIST_FIELDS}
    write_json(root/'training.json',report)
    return model,report


def evaluate(model,source,client,frames,cfg,arm,root,*,parity=False):
    root=Path(root);root.mkdir(parents=True)
    wrapper=b.make_wrapper(client,1,cfg['gamma']);rows=[]
    for frame in frames:
        _,row,trajectory=episode(model,source,wrapper,frame,cfg,arm,deterministic=True)
        if parity and (row['stateHashes']!=frame['suffixHashes'] or row['actionsDigest']!=frame['suffixActionsDigest']):
            raise ValueError('zero-residual horizon parity failed')
        write_jsonl(root/f"episode-{frame['seed']}.jsonl.gz",[trajectory]);rows.append(row)
    return rows


def paired(left,right):
    if [x['seed'] for x in left]!=[x['seed'] for x in right] or not left:
        raise ValueError('paired horizon seed mismatch')
    def interval(values):
        v=np.asarray(values,dtype=float);rng=np.random.default_rng(993001)
        means=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
        return {'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'count':len(v)}
    result={k:interval([float(a[k])-float(z[k]) for a,z in zip(left,right,strict=True)])
        for k in ('success','progress','damageDealt','damageReceived','livingFraction')}
    result['return']=interval([a['discounted']['executor']-z['discounted']['executor'] for a,z in zip(left,right,strict=True)])
    return result


def gate(rows,baseline,change):
    versus_short=paired(rows['full'],rows['short']);versus_initial=paired(rows['full'],baseline)
    success=sum(x['success'] for x in rows['full'])/len(baseline)
    criteria={'success':success>=.5,'initializerGain':versus_initial['success']['mean']>=.2-1e-12,
        'initializerPositiveInterval':versus_initial['success']['ci95'][0]>0,
        'shortGain':versus_short['success']['mean']>=.1-1e-12,
        'shortPositiveSuccessInterval':versus_short['success']['ci95'][0]>0,
        'shortPositiveReturnInterval':versus_short['return']['ci95'][0]>0,'parameterChange':change>1e-8,
        'rejectedActions':all(sum(x['rejectedActions'] for x in rows[a])/max(1,sum(x['totalActions'] for x in rows[a]))<.001 for a in ARMS)}
    return {'passed':all(criteria.values()),'criteria':criteria,'versusShort':versus_short,'versusInitializer':versus_initial,
        'successes':{a:sum(x['success'] for x in rows[a]) for a in ARMS},'initializerSuccesses':sum(x['success'] for x in baseline)}


def run(output):
    root=Path(output)
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    cfg=configuration();source,metadata,data,lineage=inputs()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    before=semantic_state_digest(source.state_dict());root.mkdir(parents=True)
    write_json(root/'declaration.json',{'config':cfg,'source':metadata,'datasetDigest':data['digest'],
        'sourceManifest':b.file_digest(b.ARCHIVE/'manifest.json'),'s8Manifest':b.file_digest(ARCHIVE/'manifest.json'),
        'implementationDigest':b.file_digest(Path(__file__)),
        'declarationDigest':b.file_digest(b.TRAINING/'reviews/m7b_r1m_s9_declaration.md'),
        'gitCommit':resolve_git_commit(),**r.ASSIST_FIELDS})
    write_json(root/'coverage.json',data['coverage'])
    steps=0;baselines={};reports={}
    def account(count):
        nonlocal steps
        steps+=count
        if steps>cfg['simulatorBudget']:
            raise ValueError('horizon simulator budget exceeded')
    with SnowGymBatchClient() as client:
        b.require_capabilities(client)
        for arm in ARMS:
            torch.manual_seed(cfg['trainingRngs'][0]);initial=RecoveryPolicy(copy.deepcopy(source),standard_deviation=cfg['latentStd'])
            baselines[arm]={}
            for split,frames in data['datasets'].items():
                rows=evaluate(initial,source,client,frames,cfg,arm,root/'initialization'/arm/split,parity=True)
                account(sum(x['simulatorDecisions'] for x in rows));baselines[arm][split]=rows
                print(json.dumps({'preflight':arm,'split':split,'episodes':len(rows),'simulatorDecisions':steps}),flush=True)
        write_json(root/'initialization.json',baselines)
        for seed in cfg['trainingRngs']:
            evaluations={};training={}
            for arm in ARMS:
                directory=root/str(seed)/arm
                _,training[arm]=train(source,metadata,client,data['datasets']['training'],cfg,directory,seed,arm)
                account(sum(x['simulatorDecisions'] for x in training[arm]['history']))
                model,_,_=r.recovery_checkpoint.load(directory/'final')
                evaluations[arm]={}
                for split in ('historical','replication'):
                    rows=evaluate(model,source,client,data['datasets'][split],cfg,arm,directory/'evaluation'/split)
                    account(sum(x['simulatorDecisions'] for x in rows));evaluations[arm][split]=rows
                write_json(directory/'evaluation.json',evaluations[arm])
            if training['short']['initialStateDigest']!=training['full']['initialStateDigest']:
                raise ValueError('horizon initial parameters differ')
            if [x['seeds'] for x in training['short']['history']]!=[x['seeds'] for x in training['full']['history']]:
                raise ValueError('horizon frame schedules differ')
            gates={split:gate({a:evaluations[a][split] for a in ARMS},baselines['short'][split],training['full']['actorParameterL2Change'])
                for split in ('historical','replication')}
            reports[str(seed)]={'gates':gates,'training':{a:{k:v for k,v in training[a].items() if k!='history'} for a in ARMS}}
            write_json(root/str(seed)/'comparison.json',reports[str(seed)])
            print(json.dumps({'seed':seed,'gates':gates,'simulatorDecisions':steps}),flush=True)
            if seed==cfg['trainingRngs'][0] and not all(x['passed'] for x in gates.values()):
                break
    if inputs()[-1]!=lineage or semantic_state_digest(source.state_dict())!=before:
        raise ValueError('horizon source or ancestry changed')
    report={'format':'snowgym.horizon-report.v0',**r.ASSIST_FIELDS,'runs':reports,'simulatorDecisions':steps,
        'replicationsExecuted':len(reports)-1,'replicated':len(reports)==3 and all(g['passed'] for x in reports.values() for g in x['gates'].values())}
    write_json(root/'report.json',report)
    manifest={'format':'snowgym.horizon-manifest.v0',**r.ASSIST_FIELDS,
        'artifacts':{str(p.relative_to(root)):b.file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    manifest['manifestDigest']=json_digest(manifest);write_json(root/'manifest.json',manifest);b.verified_manifest(root)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
