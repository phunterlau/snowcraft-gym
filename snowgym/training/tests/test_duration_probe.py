import copy

import numpy as np
import pytest
import torch

from snowgym_client.batch import SnowGymBatchClient
from snowgym_training.checkpoint import semantic_state_digest
from snowgym_training.options import duration_probe as d
from snowgym_training.trajectory import json_digest


def test_windows_have_exact_decision_boundaries():
    for arm in d.ARMS:
        expected = 0 if arm == "keep" else int(arm.split('-')[1])
        assert sum(d.active(arm, i) for i in range(200)) == expected
    for arm, offset in (("bad", 0), ("single-1", -1), ("squad-30", .5)):
        with pytest.raises(ValueError):
            d.active(arm, offset)


def fixture():
    first = {"action_type": np.array([[1, 1, 2, 1]]), "target": np.zeros((1, 4, 2)), "power": np.ones((1, 4))}
    raw = {"allies": [{"id": i, "alive": i != 4} for i in range(1, 5)]}
    rec = {"valid": np.ones((1, 4), dtype=bool), "target": np.ones((1, 4, 2))*.2}
    return first, raw, rec


def test_scope_masks_casualties_and_no_source_mutation():
    first, raw, rec = fixture(); before = copy.deepcopy(first)
    single, dose = d.intervene(first, raw, rec, "single-30", 29, 2)
    assert dose["overriddenIds"] == dose["changedIds"] == [2]
    squad, dose = d.intervene(first, raw, rec, "squad-30", 29, 2)
    assert dose["overriddenIds"] == [1, 2] and dose["livingMoveIds"] == [1, 2]
    np.testing.assert_array_equal(squad['target'][0, 2:], first['target'][0, 2:])
    for key in ('action_type', 'power'):
        np.testing.assert_array_equal(squad[key], first[key])
    for key in first:
        np.testing.assert_array_equal(before[key], first[key])
    for arm, offset in (("keep", 0), ("single-1", 1), ("squad-30", 30)):
        result, dose = d.intervene(first, raw, rec, arm, offset, 2)
        assert not dose["overriddenIds"]
        for key in first:
            np.testing.assert_array_equal(result[key], first[key])
    raw['allies'][1]['alive'] = False
    result, dose = d.intervene(first, raw, rec, "single-30", 2, 2)
    assert dose["overriddenIds"] == []  # ID 1 must not replace dead ID 2.


def test_missing_labels_are_checked_only_for_selected_overrides():
    first, raw, rec = fixture(); rec['valid'][0, 0] = False
    d.intervene(first, raw, rec, "single-30", 0, 2)
    with pytest.raises(ValueError, match="lacks a recommendation"):
        d.intervene(first, raw, rec, "squad-30", 0, 2)


def test_factorial_contrasts_and_predeclared_primary_gate():
    def row(x):
        return {"final": dict.fromkeys(('success','progress','damageDealt','damageReceived','livingFraction'), x),
            "discounted": {"executor": x}, "local": {"success": x}, "earlyTerminationBefore30": False,
            "exposure": dict.fromkeys(('windowDecisions','overriddenMoves','changedTargets','windowLivingMoves','rangeOccupancy','meanRangeError'), 1),
            "rejectedActions": 0, "totalActions": 10}
    group = {arm: row(x) for arm,x in zip(d.ARMS, [0, .1, .2, .4, .8])}
    result = d.summarize([group]*3)
    assert result['contrasts']['interaction']['return']['mean'] == pytest.approx(.3)
    assert result['contrasts']['durationSingle']['return']['mean'] == pytest.approx(.3)
    assert result['contrasts']['scopeThirty']['return']['mean'] == pytest.approx(.4)
    assert result['primaryTransferGate']['passed']
    group['squad-30'] = row(0)
    assert not d.summarize([group]*3)['primaryTransferGate']['passed']


def test_live_s5_parity_all_arms_repeats_source_and_tamper(tmp_path):
    with pytest.raises(FileExistsError):
        d.run(tmp_path)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source, _, selected, cfg, _ = d.inputs()
    pick = selected['selected'][0]
    frame = next(f for f in selected['frames'] if f['seed'] == pick['seed'])
    before = semantic_state_digest(source.state_dict()); gamma = cfg['config']['gamma']
    with SnowGymBatchClient() as client:
        wrapper = d.b.make_wrapper(client, 1, gamma)
        rows = {arm: d.branch(source, wrapper, frame, pick, arm, gamma) for arm in d.ARMS}
        again = d.branch(source, wrapper, frame, pick, 'squad-30', gamma)
        assert json_digest(again) == json_digest(rows['squad-30'])
        assert rows['keep']['stateHashes'] == d.archived_row(pick['seed'], 'keep')['stateHashes']
        assert rows['single-1']['stateHashes'] == d.archived_row(pick['seed'], 'single-1')['stateHashes']
        for arm, row in rows.items():
            assert not row['autonomousQualificationEligible']
            for entry in row['trace']:
                assert entry['exposure']['windowActive'] == d.active(arm, entry['offset'])
                if arm.startswith('single'):
                    assert set(entry['exposure']['overriddenIds']) <= {pick['unitId']}
        assert rows['single-1']['trace'][0]['action'] == rows['single-30']['trace'][0]['action']
        assert rows['squad-1']['trace'][0]['action'] == rows['squad-30']['trace'][0]['action']
        assert semantic_state_digest(source.state_dict()) == before
        bad = copy.deepcopy(pick); bad['identity']['physical'] = 'tampered'
        with pytest.raises(ValueError, match="identity mismatch"):
            d.branch(source, wrapper, frame, bad, 'keep', gamma)
