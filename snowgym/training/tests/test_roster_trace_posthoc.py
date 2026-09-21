import pytest

from snowgym_training.options import roster_trace_posthoc as ph
from test_roster_trace import CFG, synthetic
from snowgym_training.options import roster_trace as rt


def test_live_shares_exclude_dead_units_and_match_the_declared_measure_when_all_are_alive():
    trace = synthetic()
    shares = ph.live_action_shares([trace])
    # window decisions 0..2; the dead unit (blue slot 1) is dead only in the final state, so all 9 entries are live
    assert shares["unitDecisions"] == 9
    assert shares["throw"] == pytest.approx(1 / 3) and shares["move"] == pytest.approx(1 / 3)
    assert shares["throw"] == pytest.approx(rt.episode_metrics(trace, CFG)["throwActionShare"])


def test_a_dead_unit_entry_no_longer_dilutes_the_share():
    trace = synthetic()
    trace["states"][2]["blue"][1][0] = 0  # blue unit 2 (slot 1) dead at decision 2
    trace["acts"][2] = [[1, "move", True], [2, "throw", True], [3, "throw", True]]
    shares = ph.live_action_shares([trace])
    assert shares["unitDecisions"] == 8
    assert shares["throw"] == pytest.approx(3 / 8)
    assert rt.episode_metrics(trace, CFG)["throwActionShare"] == pytest.approx(4 / 9)  # the archived, diluted definition
