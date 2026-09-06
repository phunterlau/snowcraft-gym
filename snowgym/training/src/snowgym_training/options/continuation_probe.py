"""R1m-S8: same-state frozen-source versus sustained teacher-MOVE continuation."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..trajectory import json_digest
from ..trainer import resolve_git_commit
from . import handoff_audit as h
from .opportunity_audit import plain, write_jsonl
from .supervised_probe import write_json

b, d = h.b, h.d
ARCHIVE = b.TRAINING / 'runs/m7b_engage_r1m_s7_v0'
ARMS = ('source', 'move-rest')
LABEL = {'assistType': 'corrected-shots-with-post-correction-movement-probe',
         'assistVersion': 'snowgym.continuation-probe.v0', 'autonomousQualificationEligible': False}
BUDGET = 19200


def inputs():
    selection, cfg, s6 = h.inputs()
    s7 = b.verified_manifest(ARCHIVE)
    declaration = json.loads((ARCHIVE/'declaration.json').read_text())
    if declaration['implementationDigest'] != b.file_digest(Path(h.__file__)):
        raise ValueError('S7 implementation drift')
    if declaration['sourceManifest'] != b.file_digest(h.ARCHIVE/'manifest.json'):
        raise ValueError('S7 source manifest drift')
    source, metadata, _, _, _ = d.inputs()
    return source, metadata, selection, cfg, {'s6': s6, 's7': s7}


def select(selection):
    eligible, excluded = [], []
    for pick in selection['selected']:
        archived = h.load_row(pick['seed'], 'squad-30')
        if len(archived['trace']) <= 30:
            excluded.append({'seed': pick['seed'], 'reason': 'terminal-at-or-before-handoff',
                             'decisions': len(archived['trace'])})
        else:
            eligible.append(pick)
    if not eligible:
        raise ValueError('no nonterminal handoff states')
    return eligible, excluded


def arrays(action):
    return {k: np.asarray(v, dtype=np.int64 if k == 'action_type' else np.float32) for k, v in action.items()}


def check_transition(wrapper, reward, info, saved):
    if (wrapper.environment.state_hashes[0] != saved['stateHash'] or float(reward[0]) != saved['reward']
        or plain(info['option']) != saved['option'] or plain(info['actionResults']) != saved['actionResults']):
        raise ValueError('archived transition mismatch')


def restore(wrapper, frame, pick, archived):
    if (archived['seed'] != pick['seed'] or archived['arm'] != 'squad-30'
        or len(archived['trace']) <= 30 or len(archived['stateHashes']) != len(archived['trace'])+1
        or json_digest([e['action'] for e in archived['trace']]) != archived['actionsDigest']):
        raise ValueError('invalid handoff archive')
    obs = b.post_hit.restore(wrapper, frame['seed'], frame['trigger']['prefix'], pick['identity'])
    if wrapper.environment.state_hashes[0] != archived['stateHashes'][0]:
        raise ValueError('initial handoff prefix hash mismatch')
    for offset, entry in enumerate(archived['trace'][:30]):
        if entry['offset'] != offset or entry['stateHash'] != archived['stateHashes'][offset+1]:
            raise ValueError('invalid prefix sequence')
        obs, reward, terminated, truncated, infos = wrapper.step(arrays(entry['action']))
        check_transition(wrapper, reward, infos[0], entry)
        if terminated[0] or truncated[0]:
            raise ValueError('terminal handoff prefix')
    with gzip.open(ARCHIVE/f"inspection-{pick['seed']}-squad-30.jsonl.gz", 'rt') as stream:
        inspection = json.loads(stream.readline())
    if h.enemy_snapshot(wrapper) != inspection['snapshots']['handoff']:
        raise ValueError('S7 handoff snapshot mismatch')
    return obs


def branch(source, wrapper, frame, pick, arm, gamma):
    if arm not in ARMS:
        raise ValueError('unknown continuation arm')
    archived = h.load_row(pick['seed'], 'squad-30')
    obs = restore(wrapper, frame, pick, archived)
    identity = b.post_hit.identity(wrapper, obs)
    snapshot = h.enemy_snapshot(wrapper)
    initial_health = (b.team_health(snapshot['allies']), b.team_health(snapshot['enemies']))
    trace, hashes, records = [], [wrapper.environment.state_hashes[0]], []
    components = dict.fromkeys(('mission', 'combat', 'shaping', 'canonical', 'executor'), 0.)
    budget = snapshot['remainingBudget']
    for offset in range(budget):
        raw = wrapper.environment.raw_observations[0]
        before = b.post_hit.identity(wrapper, obs)
        first = b.action(source, obs, wrapper)
        movement = h.recommend_movement(raw, first['action_type'].shape[1])
        shots = h.recommend_shots(raw, first['action_type'].shape[1])
        teacher = wrapper.environment.plan_teacher_tensor_actions()
        h.validate_movement_agreement(teacher, movement); h.validate_teacher_agreement(teacher, shots)
        if b.post_hit.identity(wrapper, obs) != before:
            raise ValueError('continuation labeling mutated state')
        executed, dose = d.intervene(first, raw, movement, 'squad-30' if arm == 'move-rest' else 'keep', 0, pick['unitId'])
        units = h.unit_metrics(raw, executed, obs, movement, shots)
        previous = wrapper.trackers[0].target_health_fraction(raw)
        health = (b.team_health(raw['allies']), b.team_health(raw['enemies']))
        obs, reward, terminated, truncated, infos = wrapper.step(executed)
        info = infos[0]; done = bool(terminated[0] or truncated[0])
        hashes.append(wrapper.environment.state_hashes[0])
        if arm == 'source':
            if offset+30 >= len(archived['trace']):
                raise ValueError('source suffix length mismatch')
            saved = archived['trace'][offset+30]
            if plain(executed) != saved['action'] or done != (offset+31 == len(archived['trace'])):
                raise ValueError('source suffix action or terminal mismatch')
            check_transition(wrapper, reward, info, saved)
        raw = wrapper.environment.raw_observations[0]
        record = {'offset': offset, 'units': units, 'progressGain': previous-wrapper.trackers[0].target_health_fraction(raw),
            'damageDealt': health[1]-b.team_health(raw['enemies']), 'damageReceived': health[0]-b.team_health(raw['allies'])}
        records.append(record)
        trace.append({**record, 'sourceAction': plain(first), 'action': plain(executed), 'dose': dose,
            'stateHash': hashes[-1], 'reward': float(reward[0]), 'option': plain(info['option']), 'actionResults': plain(info['actionResults'])})
        for key in components:
            components[key] += gamma**offset*info['option']['rewards'][key]
        if done:
            break
    if not done:
        raise ValueError('continuation exceeded original budget')
    final = b.post_hit.outcome(wrapper, info, initial_health, len(trace))
    if arm == 'source':
        expected = {**archived['final'], 'exposureDecisions': archived['final']['exposureDecisions']-30,
            **{k: archived['final'][k]-archived['local'][k] for k in ('damageDealt', 'damageReceived')}}
        if final != expected or hashes != archived['stateHashes'][30:]:
            raise ValueError('source suffix outcome mismatch')
    if not np.isclose(components['executor'], components['mission']+.1*components['combat']+components['shaping'], atol=1e-6):
        raise ValueError('continuation reward accounting mismatch')
    prefix = copy.deepcopy(frame['trigger']['prefix'])+[e['action'] for e in archived['trace'][:30]]
    return {'seed': pick['seed'], 'arm': arm, **LABEL, 'startIdentity': identity, 'startSnapshot': snapshot,
        'prefix': prefix, 'prefixDigest': json_digest(prefix), 'final': final, 'discounted': components,
        'stateHashes': hashes, 'actionsDigest': json_digest([e['action'] for e in trace]), 'trace': trace,
        'exposure': h.window_metrics(records, 0, None),
        'overriddenMoves': sum(len(e['dose']['overriddenIds']) for e in trace),
        'changedTargets': sum(len(e['dose']['changedIds']) for e in trace),
        'simulatorDecisions': len(prefix)+len(trace),
        'rejectedActions': sum(a.get('accepted') is False for e in trace for a in e['actionResults']),
        'totalActions': sum(len(e['actionResults']) for e in trace)}


def interval(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {'count': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(992001)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {'count': len(values), 'mean': float(values.mean()), 'ci95': np.quantile(means, [.025,.975]).tolist()}


def summarize(groups):
    if not groups:
        raise ValueError('empty paired comparison')
    def metric(row, key):
        return row['discounted']['executor'] if key == 'return' else float(row['final'][key])
    effects = {k: interval([metric(g['move-rest'], k)-metric(g['source'], k) for g in groups])
        for k in ('success', 'return', 'progress', 'damageDealt', 'damageReceived', 'livingFraction', 'exposureDecisions')}
    arms = {a: {'successes': sum(g[a]['final']['success'] for g in groups),
        'rejectedActions': sum(g[a]['rejectedActions'] for g in groups), 'totalActions': sum(g[a]['totalActions'] for g in groups),
        'overriddenMoves': interval([g[a]['overriddenMoves'] for g in groups]),
        'changedTargets': interval([g[a]['changedTargets'] for g in groups]),
        'exposure': {k: interval([g[a]['exposure'][k] for g in groups]) for k in h.METRICS},
        'discounted': {k: interval([g[a]['discounted'][k] for g in groups]) for k in groups[0][a]['discounted']}}
        for a in ARMS}
    criteria = {'successGain': effects['success']['mean'] >= .1-1e-12,
        'positiveSuccessInterval': effects['success']['ci95'][0] > 0,
        'positiveReturnInterval': effects['return']['ci95'][0] > 0,
        'rejectedActions': all(r['rejectedActions']/max(r['totalActions'], 1) < .001 for r in arms.values())}
    return {'seeds': len(groups), 'arms': arms, 'pairedDifference': effects,
        'recoveredSeeds': [g['source']['seed'] for g in groups if g['move-rest']['final']['success'] and not g['source']['final']['success']],
        'lostSeeds': [g['source']['seed'] for g in groups if g['source']['final']['success'] and not g['move-rest']['final']['success']],
        'diagnosticContinuationGate': {'passed': all(criteria.values()), 'criteria': criteria}}


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    source, metadata, selection, cfg, manifests = inputs()
    eligible, excluded = select(selection)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = semantic_state_digest(source.state_dict()); gamma = cfg['config']['gamma']
    root.mkdir(parents=True)
    write_json(root/'declaration.json', {'format': 'snowgym.continuation-config.v0', **LABEL,
        'source': metadata, 's6Manifest': b.file_digest(h.ARCHIVE/'manifest.json'), 's7Manifest': b.file_digest(ARCHIVE/'manifest.json'),
        'implementationDigest': b.file_digest(Path(__file__)),
        'declarationDigest': b.file_digest(b.TRAINING/'reviews/m7b_r1m_s8_declaration.md'),
        'gitCommit': resolve_git_commit(), 'gamma': gamma, 'arms': ARMS, 'simulatorBudget': BUDGET,
        'repeats': 2, 'bootstrapSeed': 992001, 'bootstrapSamples': 10000})
    write_json(root/'selection.json', {'eligible': eligible, 'excluded': excluded})
    frames = {f['seed']: f for f in selection['frames']}; groups=[]; repeats=[]; steps=0
    with SnowGymBatchClient() as client:
        b.require_capabilities(client); wrapper=b.make_wrapper(client, 1, gamma)
        for pick in eligible:
            group={}
            for arm in ARMS:
                first=branch(source, wrapper, frames[pick['seed']], pick, arm, gamma)
                second=branch(source, wrapper, frames[pick['seed']], pick, arm, gamma)
                if json_digest(first) != json_digest(second):
                    raise ValueError('continuation duplicate mismatch')
                steps += first['simulatorDecisions']+second['simulatorDecisions']
                if steps > BUDGET:
                    raise ValueError('continuation simulator budget exceeded')
                repeats.append({'seed':pick['seed'], 'arm':arm, 'first':json_digest(first), 'second':json_digest(second)})
                write_jsonl(root/f"branch-{pick['seed']}-{arm}.jsonl.gz", [first]); group[arm]=first
            if group['source']['startIdentity'] != group['move-rest']['startIdentity'] or group['source']['prefix'] != group['move-rest']['prefix']:
                raise ValueError('continuation starts differ')
            groups.append(group)
            print(json.dumps({'seed':pick['seed'],'completedPairs':len(groups),'simulatorDecisions':steps}),flush=True)
    if semantic_state_digest(source.state_dict()) != before or inputs()[-1] != manifests:
        raise ValueError('source or archive changed')
    report={'format':'snowgym.continuation-report.v0',**LABEL,**summarize(groups),
        'excluded':excluded,'simulatorDecisions':steps,'duplicatePairs':len(repeats)}
    write_json(root/'report.json',report); write_json(root/'duplicates.json',repeats)
    manifest={'format':'snowgym.continuation-manifest.v0',**LABEL,
        'artifacts':{str(p.relative_to(root)):b.file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    manifest['manifestDigest']=json_digest(manifest);write_json(root/'manifest.json',manifest);b.verified_manifest(root)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
