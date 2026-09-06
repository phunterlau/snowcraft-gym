"""R1m-S10: immutable S9 coverage, credit and output-direction inspection."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from snowgym_client.batch import SnowGymBatchClient
from ..checkpoint import semantic_state_digest
from ..executor.recovery_ppo import RecoveryPolicy
from ..executor.movement_ppo import movement_loss
from ..trainer import resolve_git_commit
from ..trajectory import json_digest
from . import horizon_train as h, recovery_audit as a
from .control_channels import recommend_movement, validate_movement_agreement
from .opportunity_audit import plain, write_jsonl
from .supervised_probe import write_json

ARCHIVE = h.b.TRAINING/'runs/m7b_engage_r1m_s9_v0'
UPDATES = (1, 11, 21)
SEED = 99301


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def inputs():
    source, metadata, data, _ = h.inputs()
    manifest = h.b.verified_manifest(ARCHIVE)
    decl = json.loads((ARCHIVE/'declaration.json').read_text())
    expected = {'implementationDigest': h.b.file_digest(Path(h.__file__)),
        'declarationDigest': h.b.file_digest(h.b.TRAINING/'reviews/m7b_r1m_s9_declaration.md'),
        'sourceManifest': h.b.file_digest(h.b.ARCHIVE/'manifest.json'),
        's8Manifest': h.b.file_digest(h.ARCHIVE/'manifest.json'), 'datasetDigest': data['digest'],
        'source': metadata, 'config': h.configuration()}
    if any(decl.get(k) != v for k, v in expected.items()):
        raise ValueError('S9 audit lineage mismatch')
    return source, data, decl['config'], manifest


def coverage(rows, selected):
    offsets = np.concatenate([np.arange(t['summary']['learnedDecisions']) for t in rows])
    indices = np.asarray(selected, dtype=int)
    if not len(offsets) or np.any(indices < 0) or np.any(indices >= len(offsets)):
        raise ValueError('invalid selected decision index')
    result = {}
    for name, mask in (('early', offsets < 30), ('late', offsets >= 30)):
        count = int(mask.sum())
        occurrences = int(mask[indices].sum())
        unique = int(mask[np.unique(indices)].sum())
        result[name] = {'collected': count, 'selectedOccurrences': occurrences,
            'uniqueSelected': unique, 'selectionFraction': unique/count if count else None}
    return result


def output_scores(means, latents, advantages, living, moves, selected, order, minibatch, sigma):
    """Occurrence-coordinate scores avoid overwriting duplicated selected rows."""
    indices = torch.as_tensor(selected, dtype=torch.long)
    scores, normalized = a.score_directions(means[indices], latents[indices], advantages[indices],
        living[indices], moves[indices], order, minibatch, sigma)
    combined = torch.zeros_like(means).index_add_(0, indices, scores)
    return combined, scores, normalized


class Observer:
    """Read-only act delegate: extra deterministic forwards do not consume RNG."""
    def __init__(self, model, final, wrapper, seed, episode_index, start_index, sigma):
        self.model, self.final, self.wrapper = model, final, wrapper
        self.seed, self.episode_index, self.start_index = seed, episode_index, start_index
        self.sigma, self.offset, self.rows, self.predictions = sigma, 0, [], []

    def act(self, observation, *, deterministic=False):
        pred = self.model(observation)
        final = self.final(observation)
        if not torch.equal(pred['action_type'], final['action_type']):
            raise ValueError('frozen categorical choice changed')
        action, latent, logp, value = self.model.act(observation, deterministic=deterministic)
        raw = self.wrapper.environment.raw_observations[0]
        rec = recommend_movement(raw, self.wrapper.environment.max_team_units)
        validate_movement_agreement(self.wrapper.environment.plan_teacher_tensor_actions(), rec)
        scale = np.asarray([raw['arena']['width']/2, raw['arena']['height']/2])
        for i, unit in enumerate(raw['allies']):
            if not pred['move_mask'][0, i]:
                continue
            row = {'seed': self.seed, 'episodeIndex': self.episode_index, 'offset': self.offset,
                'observationIndex': self.start_index+self.offset, 'unitIndex': i, 'unitId': unit['id'],
                'stateHash': self.wrapper.environment.state_hashes[0],
                'ready': bool(rec['ready'][0, i]), 'legalMove': bool(observation['unit_action_mask'][0, i, 1]),
                'recommendationAvailable': bool(rec['valid'][0, i])}
            if row['recommendationAvailable']:
                mean = pred['mean'][0, i].numpy(); new = final['mean'][0, i].numpy()
                target = rec['target'][0, i]
                row.update(a.geometry(mean, latent[0, i].numpy(), target, scale, self.sigma))
                row.update({'distance': float(rec['distance'][0, i]),
                    'finalWorldShift': float(np.linalg.norm((np.tanh(new)-np.tanh(mean))*scale)),
                    'finalGapReduction': row['recommendationWorldGap']-float(np.linalg.norm((target-np.tanh(new))*scale))})
            self.rows.append(row)
        self.predictions.append({k: pred[k].detach() for k in ('mean', 'living', 'move_mask')})
        self.offset += 1
        return action, latent, logp, value


def behavior(source, arm, update, cfg):
    directory = ARCHIVE/str(SEED)/arm
    if update == 1:
        torch.manual_seed(SEED)
        model = RecoveryPolicy(copy.deepcopy(source), standard_deviation=cfg['latentStd'])
        sampler = np.random.default_rng(SEED)
    else:
        model, _, saved = h.r.recovery_checkpoint.load(directory/f'update-{update-1:03d}')
        if saved['config'] != {**cfg, 'arm': arm} or saved['update'] != update-1:
            raise ValueError('pre-update checkpoint mismatch')
        sampler = np.random.default_rng(); sampler.bit_generator.state = saved['sampler']
    return model.eval().requires_grad_(False), sampler


def credit(rollout, episodes, indices):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return {'decisions': 0, 'criticExplainedVariance': None, 'betweenEpisodeAdvantageFraction': None}
    adv = rollout['advantage'][indices].numpy(); ret = rollout['returns'][indices].numpy()
    val = rollout['value'][indices].numpy(); labels = np.asarray(episodes)[indices]
    variance = float(np.var(adv)); retvar = float(np.var(ret))
    between = sum(int((labels == label).sum())*(float(adv[labels == label].mean())-float(adv.mean()))**2
        for label in set(labels.tolist()))/len(indices)
    return {'decisions': len(indices), 'advantage': a.statistics(adv),
        'criticExplainedVariance': 1-float(np.var(ret-val))/retvar if retvar > 1e-12 else None,
        'betweenEpisodeAdvantageFraction': between/variance if variance > 1e-12 else None}


def summarize(rows):
    valid = [x for x in rows if x['recommendationAvailable']]
    selected = [x for x in valid if x['selectionMultiplicity']]
    pairs = [(x['sampleTowardRecommendation'], v) for x in selected for v in x['normalizedAdvantages']]
    return {'livingMoveOpportunities': len(rows), 'validRecommendations': len(valid),
        'selectedOpportunities': len(selected), 'selectedOccurrences': sum(x['selectionMultiplicity'] for x in rows),
        'ready': sum(x['ready'] for x in rows), 'legalMove': sum(x['legalMove'] for x in rows),
        **{k: a.statistics([x[k] for x in valid]) for k in ('sampleWorldDisplacement', 'recommendationWorldGap',
            'finalWorldShift', 'finalGapReduction', 'distance')},
        'withinThreeSigmaFraction': float(np.mean([x['withinThreeSigma'] for x in valid])) if valid else None,
        'saturatedCoordinateFraction': sum(x['saturatedCoordinates'] for x in valid)/(2*len(valid)) if valid else None,
        'selectedScoreProjection': a.statistics([x['worldScoreProjection'] for x in selected]),
        'firstMinibatchScoreProjection': a.statistics([x['firstMinibatchScoreProjection'] for x in valid
            if x['firstMinibatchMultiplicity']]),
        'sampleAdvantageCorrelation': a.correlation(*zip(*pairs)) if pairs else None}


def replay_update(source, final, wrapper, frames, cfg, arm, update, archived, entry):
    model, sampler = behavior(source, arm, update, cfg)
    before = semantic_state_digest(model.state_dict())
    selected_frames = [frames[i] for i in sampler.integers(len(frames), size=cfg['batchSize'])]
    if [f['seed'] for f in selected_frames] != entry['seeds']:
        raise ValueError('frame sampler mismatch')
    batches, opportunities, predictions, offsets, episode_ids = [], [], [], [], []
    steps = 0
    for slot, (frame, saved) in enumerate(zip(selected_frames, archived, strict=True)):
        observer = Observer(model, final, wrapper, frame['seed'], slot, len(offsets), cfg['latentStd'])
        batch, row, trace = h.episode(observer, source, wrapper, frame, cfg, arm,
            sampling_seed=SEED*100000+(update-1)*cfg['batchSize']+slot)
        if json_digest(trace) != json_digest(saved):
            raise ValueError('complete archived trajectory reconstruction mismatch')
        batches.append(batch); opportunities.extend(observer.rows); predictions.extend(observer.predictions)
        offsets.extend(range(len(batch['advantage']))); episode_ids.extend([slot]*len(batch['advantage']))
        steps += row['simulatorDecisions']
    rollout = h.r.combine(batches)
    selected = h.select_rows(len(offsets), cfg['rowsPerUpdate'], SEED*100000+update-1)
    if selected.tolist() != entry['selectedRows']:
        raise ValueError('selected rows mismatch')
    order = torch.randperm(len(selected))
    first = selected[order[:cfg['minibatchSize']].numpy()]
    batch = h.subset(rollout, first)
    with torch.no_grad():
        logp, pred = model.evaluate_latents(batch['observation'], batch['action_type'], batch['latent'])
        loss = movement_loss(logp, batch['logp'], batch['advantage'], pred['value'], batch['returns'], pred,
            clip_ratio=cfg['clipRatio'])
    original = entry['minibatches'][0]
    actual = {k: float(v) for k, v in loss.items()}
    if any(not np.isclose(v, original[k], rtol=1e-5, atol=1e-6) for k, v in actual.items()):
        raise ValueError('first minibatch loss mismatch')
    means, living, moves = [torch.cat([p[k] for p in predictions]) for k in ('mean', 'living', 'move_mask')]
    scores, occurrence_scores, normalized = output_scores(means, rollout['latent'], rollout['advantage'], living, moves,
        selected, order, cfg['minibatchSize'], cfg['latentStd'])
    first_scores = torch.zeros_like(means).index_add_(0, torch.as_tensor(first),
        occurrence_scores[order[:cfg['minibatchSize']]])
    for row in opportunities:
        index, unit = row['observationIndex'], row['unitIndex']
        positions = np.flatnonzero(selected == index)
        row.update({'selectionMultiplicity': len(positions), 'firstMinibatchMultiplicity': int((first == index).sum()),
            'advantage': float(rollout['advantage'][index]),
            'normalizedAdvantages': [float(normalized[p]) for p in positions]})
        if row['recommendationAvailable']:
            row['worldScoreProjection'] = float(np.asarray(row['worldDirection']) @
                (np.asarray(row['worldJacobian'])*scores[index, unit].numpy()))
            row['firstMinibatchScoreProjection'] = float(np.asarray(row['worldDirection']) @
                (np.asarray(row['worldJacobian'])*first_scores[index, unit].numpy()))
    if semantic_state_digest(model.state_dict()) != before:
        raise ValueError('audit changed behavior weights')
    result = {'arm': arm, 'update': update, 'simulatorDecisions': steps,
        'completeTrajectoryMatches': len(archived), 'firstEpochOrder': order.tolist(),
        'firstMinibatchLoss': actual, 'firstMinibatchLossMaxError': max(abs(v-original[k]) for k,v in actual.items()),
        'coverage': coverage(archived, selected), 'strata': {}}
    for name, mask in (('early', np.asarray(offsets)<30), ('late', np.asarray(offsets)>=30)):
        subset = [x for x in opportunities if bool(mask[x['observationIndex']])]
        result['strata'][name] = {'collectedCredit': credit(rollout, episode_ids, np.flatnonzero(mask)),
            'selectedCredit': credit(rollout, episode_ids, selected[mask[selected]]),
            'geometry': summarize(subset), 'outsideRangeGeometry': summarize([x for x in subset if x.get('distance', 0)>9]),
            'firstMinibatchGeometry': summarize([x for x in subset if x['firstMinibatchMultiplicity']])}
    # Do not let the shared serializer turn nonfinite diagnostic values into null.
    json.dumps({'result': result, 'opportunities': opportunities}, allow_nan=False)
    return result, opportunities


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f'refusing to overwrite {root}')
    source, data, cfg, manifest = inputs()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    before = semantic_state_digest(source.state_dict())
    root.mkdir(parents=True)
    write_json(root/'declaration.json', {'format': 'snowgym.horizon-audit.v0', 'trainingRng': SEED,
        'updates': list(UPDATES), 'sourceManifest': h.b.file_digest(ARCHIVE/'manifest.json'),
        'implementationDigest': h.b.file_digest(Path(__file__)),
        'declarationDigest': h.b.file_digest(h.b.TRAINING/'reviews/m7b_r1m_s10_declaration.md'),
        'gitCommit': resolve_git_commit(), **h.r.ASSIST_FIELDS})
    reports, selections, steps = [], {}, 0
    with SnowGymBatchClient() as client:
        h.b.require_capabilities(client)
        wrapper = h.b.make_wrapper(client, 1, cfg['gamma'])
        for arm in h.ARMS:
            path = ARCHIVE/str(SEED)/arm
            history = json.loads((path/'training.json').read_text())['history']
            selections[arm] = []
            final, _, _ = h.r.recovery_checkpoint.load(path/'final')
            final.eval().requires_grad_(False)
            final_digest = semantic_state_digest(final.state_dict())
            for update, entry in enumerate(history, 1):
                archived = read_rows(path/f'events-{update:03d}.jsonl.gz')
                selected = h.select_rows(entry['collectedRows'], cfg['rowsPerUpdate'], SEED*100000+update-1)
                if selected.tolist() != entry['selectedRows']:
                    raise ValueError('archived selection stream mismatch')
                selections[arm].append({'update': update, **coverage(archived, selected)})
                if update not in UPDATES:
                    continue
                result, opportunities = replay_update(source, final, wrapper, data['datasets']['training'],
                    cfg, arm, update, archived, entry)
                steps += result['simulatorDecisions']
                if steps > 9600:
                    raise ValueError('audit simulator budget exceeded')
                write_jsonl(root/f'opportunities-{arm}-{update:03d}.jsonl.gz', opportunities)
                write_json(root/f'reconstruction-{arm}-{update:03d}.json', result)
                reports.append(result)
                print(json.dumps({'arm': arm, 'update': update, 'simulatorDecisions': steps}), flush=True)
            if semantic_state_digest(final.state_dict()) != final_digest:
                raise ValueError('audit changed final weights')
    if semantic_state_digest(source.state_dict()) != before or inputs()[-1] != manifest:
        raise ValueError('audit changed source evidence')
    write_json(root/'selection.json', selections)
    report = {'format': 'snowgym.horizon-audit-report.v0', 'updates': reports,
        'simulatorDecisions': steps, 'completeTrajectoryMatches': sum(x['completeTrajectoryMatches'] for x in reports),
        **h.r.ASSIST_FIELDS}
    write_json(root/'report.json', report)
    inventory = {'format': 'snowgym.horizon-audit-manifest.v0', **h.r.ASSIST_FIELDS,
        'artifacts': {str(p.relative_to(root)): h.b.file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    inventory['manifestDigest'] = json_digest(inventory)
    write_json(root/'manifest.json', inventory)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
