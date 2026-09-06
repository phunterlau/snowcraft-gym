import copy
import gzip
import json

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.options import handoff_audit as h
from snowgym_training.trajectory import json_digest


def test_readiness_availability_actions_and_geometry_are_separate():
    raw = {"arena": {"width": 100, "height": 80}, "allies": [
        {"id": 1, "alive": True, "x": 0, "y": 0, "vx": -1, "vy": 0},
        {"id": 2, "alive": False}], "enemies": [
        {"id": 3, "alive": True, "x": 10, "y": 0, "vx": 0, "vy": 0, "health": 60}]}
    action = {"action_type": np.array([[1, 2]]), "target": np.array([[[-1., 0], [0, 0]]])}
    obs = {"unit_action_mask": np.array([[[1, 1, 0, 1], [1, 0, 0, 0]]])}
    movement = {"valid": np.array([[True, False]]), "threat": np.array([[False, False]])}
    shots = {"enemyIds": np.array([[3, -1]]), "valid": np.array([[True, False]])}
    before = json_digest(raw)
    rows = h.unit_metrics(raw, action, obs, movement, shots)
    assert len(rows) == 1
    row = rows[0]
    assert row['outwardMove'] and row['rangeOpening'] and row['shotAvailable']
    assert not row['throwReady'] and not row['inRange']
    assert row['target'] == [-49.5, 0]
    row['unit']['x'] = 99
    assert json_digest(raw) == before
    raw['enemies'][0]['x'] = 0  # Degenerate ray stays finite.
    row = h.unit_metrics(raw, action, obs, movement, shots)[0]
    assert row['inRange'] and not row['rangeOpening'] and not row['outwardMove']
    raw['enemies'][0]['alive'] = False
    with pytest.raises(ValueError, match='no inspectable enemy'):
        h.unit_metrics(raw, action, obs, movement, shots)


def record(gain=0.):
    return {"units": [{"type": 1, "throwReady": True, "inRange": False,
        "outwardMove": True, "rangeOpening": True, "distance": 12}],
        "progressGain": gain, "damageDealt": gain*500, "damageReceived": 0}


def test_windows_censoring_denominators_and_boundaries():
    rows = [record(.01)]*35
    late = h.window_metrics(rows, 20, 30)
    tail = h.window_metrics(rows, 30, 40)
    assert late['complete'] and late['decisions'] == 10
    assert not tail['complete'] and tail['decisions'] == 5
    assert tail['progressGain'] == pytest.approx(.05)
    assert tail['progressPerDecision'] == pytest.approx(.01)
    assert tail['throwsOutsideFraction'] is None
    assert tail['readyInRangeNotThrowingFraction'] is None
    assert tail['outwardOutsideMoveFraction'] == 1
    assert h.window_metrics(rows, 30, None)['complete']
    missing = h.window_metrics(rows, 40, None)
    assert not missing['complete'] and missing['rangeOccupancy'] is None
    for start, end in ((-1, 20), (20, 20), (20, 19)):
        with pytest.raises(ValueError):
            h.window_metrics(rows, start, end)


def group(seed, length=40):
    result = {}
    for arm in h.ARMS:
        rows = [record(.01 if i < 30 or arm == 'keep' else .03) for i in range(length)]
        result[arm] = {"seed": seed, "windows": {w: h.window_metrics(rows, *v) for w, v in h.WINDOWS.items()},
            "final": {"success": False, "progress": .4},
            "snapshots": {"start": {"remainingBudget": 80}}, "finalProgressFreeDecisions": 10}
    return result


def test_paired_cohort_temporal_arithmetic_and_empty_strata():
    report = h.summarize([group(1), group(2, 35)])
    assert report['completePairedSeeds'] == [1]
    assert report['arms']['keep']['immediateTail']['trajectories'] == 2
    assert report['arms']['keep']['immediateTail']['complete'] == 1
    assert report['pairedTemporalContrast']['progressPerDecision']['mean'] == pytest.approx(.02)
    assert report['completePairedWindows']['squad-30']['immediateTail']['progressPerDecision']['count'] == 1
    assert report['strata']['budget61-100']['count'] == 2
    assert report['strata']['success']['progress']['mean'] is None
    assert h.statistic([None]) == {'count': 0, 'mean': None, 'ci95': None}


def test_live_archived_replay_without_policy_calls_and_tamper(tmp_path, monkeypatch):
    with pytest.raises(FileExistsError):
        h.run(tmp_path)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    selection, cfg, manifest = h.inputs()
    pick = selection['selected'][0]
    frame = next(f for f in selection['frames'] if f['seed'] == pick['seed'])
    def no_policy(*args, **kwargs):
        raise AssertionError('inspector must not select new actions')
    monkeypatch.setattr(h.b, 'action', no_policy)
    with SnowGymBatchClient() as client:
        wrapper = h.b.make_wrapper(client, 1, cfg['config']['gamma'])
        for arm in h.ARMS:
            archived = h.load_row(pick['seed'], arm)
            actual = h.inspect(wrapper, frame, pick, archived)
            assert actual['final'] == archived['final']
            assert [r['stateHash'] for r in actual['records']] == archived['stateHashes'][1:]
            assert not actual['autonomousQualificationEligible']
            assert sum(r['progressGain'] for r in actual['records']) == pytest.approx(
                actual['snapshots']['start']['objectiveHealth']-actual['snapshots']['final']['objectiveHealth'])
            assert json_digest(h.inspect(wrapper, frame, pick, archived)) == json_digest(actual)
        bad = copy.deepcopy(archived); bad['trace'][0]['action']['target'][0][0][0] += .1
        with pytest.raises(ValueError, match='action digest mismatch'):
            h.inspect(wrapper, frame, pick, bad)
        bad = copy.deepcopy(archived); bad['trace'][0]['stateHash'] = 'tampered'
        with pytest.raises(ValueError, match='transition mismatch'):
            h.inspect(wrapper, frame, pick, bad)
        bad = copy.deepcopy(pick); bad['identity']['physical'] = 'tampered'
        with pytest.raises(ValueError, match='identity mismatch'):
            h.inspect(wrapper, frame, bad, archived)
        bad = copy.deepcopy(archived); bad['trace'] = []
        with pytest.raises(ValueError, match='trace length'):
            h.inspect(wrapper, frame, pick, bad)
    assert h.inputs()[-1] == manifest


def test_archived_s7_inventory_windows_and_report_recompute():
    root = h.b.TRAINING / 'runs/m7b_engage_r1m_s7_v0'
    manifest = h.b.verified_manifest(root)
    assert len(manifest['artifacts']) == 50
    groups = []
    for path in sorted(root.glob('inspection-*-keep.jsonl.gz')):
        seed = int(path.name.split('-')[1]); group = {}
        for arm in h.ARMS:
            with gzip.open(root/f'inspection-{seed}-{arm}.jsonl.gz', 'rt') as stream:
                row = json.loads(stream.readline())
            assert row['windows'] == {w: h.window_metrics(row['records'], *bounds) for w, bounds in h.WINDOWS.items()}
            group[arm] = row
        groups.append(group)
    report = json.loads((root/'report.json').read_text())
    assert len(groups) == 24 and report['simulatorDecisions'] == 8405
    assert sum(g[a]['simulatorDecisions'] for g in groups for a in h.ARMS) == report['simulatorDecisions']
    assert all(report[k] == v for k, v in h.summarize(groups).items())
