"""No-lookahead is the load-bearing invariant of a calibration project: a
prediction for match M must never see data dated on/after M. These tests pin it
down both on a tiny synthetic frame (exact boundary) and on the real data.
"""
from __future__ import annotations

import pandas as pd

from fifapreds.asof import MatchStore


def _frame(rows):
    cols = ["match_id", "date", "home_team", "away_team", "home_score",
            "away_score", "tournament", "neutral", "is_played", "went_to_et"]
    return pd.DataFrame(rows, columns=cols)


def _synthetic() -> MatchStore:
    rows = [
        (0, pd.Timestamp("2020-01-01"), "A", "B", 1, 0, "Friendly", True, True, False),
        (1, pd.Timestamp("2020-06-01"), "A", "C", 2, 2, "Friendly", True, True, False),
        (2, pd.Timestamp("2020-06-01"), "B", "C", 0, 1, "Friendly", True, True, False),  # same day
        (3, pd.Timestamp("2021-01-01"), "A", "B", 3, 1, "Friendly", True, True, False),
        (4, pd.Timestamp("2030-01-01"), "A", "B", None, None, "FIFA World Cup", True, False, False),  # fixture
    ]
    return MatchStore(_frame(rows))


def test_before_is_strictly_earlier():
    store = _synthetic()
    # As of the same-day matches (2020-06-01), only the 2020-01-01 match is visible.
    before = store.before("2020-06-01")
    assert list(before["match_id"]) == [0]
    # The boundary is strict: a match dated exactly == ts is excluded.
    assert (before["date"] < pd.Timestamp("2020-06-01")).all()


def test_before_excludes_fixtures_and_future():
    store = _synthetic()
    # Even with a far-future cutoff, the unplayed fixture (match 4) never appears.
    everything = store.before("2999-01-01")
    assert 4 not in set(everything["match_id"])
    assert everything["is_played"].all()


def test_before_is_monotonic():
    store = _synthetic()
    assert len(store.before("2020-01-01")) == 0      # nothing strictly before the first
    assert len(store.before("2020-06-02")) == 3      # both same-day matches now included
    assert len(store.before("2999-01-01")) == 4      # all played matches


def test_no_lookahead_on_real_data():
    """For a sample of real matches, the as-of read for that match's date must
    contain no row dated on/after it — the silent-leak guard."""
    store = MatchStore()
    played = store.played
    sample = played.iloc[:: max(1, len(played) // 200)]  # ~200 probes across history
    for _, m in sample.iterrows():
        before = store.before(m["date"])
        assert before.empty or before["date"].max() < m["date"]
