"""Live prediction job: an identical claim (same fixture, model config, and
training cutoff) is logged once; new information (a moved training cutoff)
re-predicts. Synthetic store anchored to the wall clock because
predict_upcoming's window is genuinely about *now*.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from fifapreds.asof import MatchStore
from fifapreds.loop.predict import predict_upcoming
from fifapreds.models import BaselineElo

_COLS = ["match_id", "date", "home_team", "away_team", "home_score",
         "away_score", "tournament", "neutral", "is_played", "went_to_et"]


def _store(now: pd.Timestamp) -> MatchStore:
    rows = [
        (0, now - pd.Timedelta(days=300), "A", "B", 2, 0, "Friendly", True, True, False),
        (1, now - pd.Timedelta(days=200), "B", "A", 1, 1, "Friendly", True, True, False),
        (2, now + pd.Timedelta(days=2), "A", "B", None, None, "FIFA World Cup", True, False, False),
        (3, now + pd.Timedelta(days=30), "B", "A", None, None, "FIFA World Cup", True, False, False),
        (4, now + pd.Timedelta(days=3), "A", "B", None, None, "Friendly", True, False, False),
    ]
    return MatchStore(pd.DataFrame(rows, columns=_COLS))


def test_predict_upcoming_dedupes_and_windows():
    now = pd.Timestamp.now().normalize()
    store = _store(now)
    model = BaselineElo().fit(store.played)
    conn = sqlite3.connect(":memory:")

    # Window covers fixture 2 only (fixture 3 is past the window; fixture 4 is
    # in the window but not a World Cup match).
    first = predict_upcoming(conn, [model], store, days=8)
    assert first == {"elo_baseline": 1}

    # Same model state, same fixtures: nothing new to say.
    assert predict_upcoming(conn, [model], store, days=8) == {"elo_baseline": 0}

    # days=None covers the whole remaining schedule (fixture 3 too).
    assert predict_upcoming(conn, [model], store, days=None) == {"elo_baseline": 1}

    # New information moves the training cutoff -> the claim is re-made.
    model.update({"date": now, "home_team": "A", "away_team": "B",
                  "home_score": 1, "away_score": 0, "neutral": True})
    assert predict_upcoming(conn, [model], store, days=8) == {"elo_baseline": 1}

    # Append-only: all three claims remain on the log (fixture 2 pre- and
    # post-update, fixture 3 once).
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert n == 3
