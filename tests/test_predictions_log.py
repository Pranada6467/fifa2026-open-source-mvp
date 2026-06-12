"""Provenance log E2E: a prediction is written with full provenance, stays
pending until its result lands, then gets graded exactly once. The integrity
rails are pinned: lookahead refused at write time, late live predictions and
leaked training cutoffs refused at scoring time (backtest rows exempt from the
wall-clock rule only).
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.loop.predict import log_prediction
from fifapreds.loop.score import score_pending
from fifapreds.models import BaselineElo

_COLS = ["match_id", "date", "home_team", "away_team", "home_score",
         "away_score", "tournament", "neutral", "is_played", "went_to_et"]


def _store(rows) -> MatchStore:
    df = pd.DataFrame(rows, columns=_COLS)
    df["date"] = pd.to_datetime(df["date"])
    return MatchStore(df)


_HISTORY = [
    (0, "2021-01-01", "A", "B", 2, 0, "Friendly", False, True, False),
    (1, "2021-03-01", "B", "A", 1, 1, "Friendly", False, True, False),
    (2, "2021-06-01", "A", "B", 3, 1, "Friendly", True, True, False),
]
_FIXTURE = (3, "2022-01-01", "A", "B", None, None, "FIFA World Cup", True, False, False)
_RESULT = (3, "2022-01-01", "A", "B", 2, 0, "FIFA World Cup", True, True, False)


@pytest.fixture()
def setup():
    conn = sqlite3.connect(":memory:")
    store = _store(_HISTORY + [_FIXTURE])
    model = BaselineElo().fit(store.before("2022-01-01"))
    fixture = store.upcoming().iloc[0]
    return conn, store, model, fixture


def test_prediction_row_has_full_provenance(setup):
    conn, _, model, fixture = setup
    pid = log_prediction(conn, model, fixture,
                         predicted_at="2021-12-31T12:00:00", context="live")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE prediction_id=?", (pid,)).fetchone()
    assert row["model_id"] == "elo_baseline" and row["model_version"] == "1"
    assert row["hyperparams_hash"] == model.hyperparams_hash
    assert row["training_cutoff"] == model.trained_through.isoformat()
    assert row["code_version"]  # running inside the git repo
    assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0)
    assert pd.Timestamp(row["predicted_at"]) < pd.Timestamp(row["kickoff_ts"])
    assert row["match_id"] == 3 and row["neutral"] == 1


def test_lookahead_refused_at_write_time(setup):
    conn, _, model, _ = setup
    # Kickoff on/before the model's training cutoff (2021-06-01): contaminated.
    stale = pd.Series({"date": "2021-06-01", "home_team": "A", "away_team": "B",
                       "neutral": True, "tournament": "Friendly", "match_id": None})
    with pytest.raises(ValueError, match="lookahead"):
        log_prediction(conn, model, stale)
    with pytest.raises(ValueError, match="not fitted"):
        log_prediction(conn, BaselineElo(), stale)


def test_pending_until_result_then_scored_once(setup):
    conn, store, model, fixture = setup
    log_prediction(conn, model, fixture, predicted_at="2021-12-31T12:00:00")

    # Result not in yet: stays pending, nothing written.
    report = score_pending(conn, store)
    assert report == {"scored": 0, "pending": 1, "violations": [],
                      "results": report["results"]} and report["results"].empty

    # Result lands (A wins 2-0): graded against the logged probabilities.
    resolved = _store(_HISTORY + [_RESULT])
    report = score_pending(conn, resolved)
    assert report["scored"] == 1 and report["pending"] == 0
    p_home = conn.execute("SELECT p_home FROM predictions").fetchone()[0]
    got = report["results"].iloc[0]
    assert got["outcome"] == "home"
    assert got["log_loss"] == pytest.approx(-math.log(p_home))

    # Scoring is idempotent: a second pass finds nothing to do.
    again = score_pending(conn, resolved)
    assert again["scored"] == 0 and again["pending"] == 0


def test_late_live_prediction_is_violation_backtest_exempt(setup):
    conn, _, model, fixture = setup
    resolved = _store(_HISTORY + [_RESULT])
    # Logged AFTER kickoff: a live row must never be graded…
    late_live = log_prediction(conn, model, fixture,
                               predicted_at="2022-01-02T00:00:00", context="live")
    # …but a replayed backtest row carries wall-clock predicted_at by design;
    # its out-of-sample claim rests on training_cutoff (which is honest here).
    replay = log_prediction(conn, model, fixture,
                            predicted_at="2026-06-10T00:00:00", context="backtest:test")
    report = score_pending(conn, resolved)
    assert report["violations"] == [late_live]
    assert report["scored"] == 1
    assert int(report["results"].iloc[0]["prediction_id"]) == replay


def test_leaked_training_cutoff_is_violation(setup):
    conn, _, model, fixture = setup
    pid = log_prediction(conn, model, fixture, predicted_at="2021-12-31T12:00:00")
    # Simulate a corrupted row (the write-time guard refuses to create one):
    # the scorer audits independently and refuses to grade it.
    conn.execute("UPDATE predictions SET training_cutoff = kickoff_ts WHERE prediction_id=?",
                 (pid,))
    report = score_pending(conn, _store(_HISTORY + [_RESULT]))
    assert report["violations"] == [pid] and report["scored"] == 0


# ---------------------------------------------------------------- E6a grids
def test_goals_model_claim_stores_its_grid(setup):
    """A goals-capable entrant's full scoreline grid is captured at predict
    time and round-trips exactly — live grids can never be backfilled."""
    import numpy as np

    from fifapreds.loop.predict import load_grid
    from fifapreds.models.base import GoalsModel, ScoreGrid

    conn, store, _, fixture = setup

    class TinyGoals(GoalsModel):
        model_id = "tiny_goals"
        model_version = "0"
        trained_through = pd.Timestamp("2021-12-30")

        def fit(self, matches):
            return self

        def hyperparams(self):
            return {}

        def predict_wdl(self, home, away, *, neutral=False):
            return self.predict_goals(home, away, neutral=neutral).wdl()

        def predict_goals(self, home, away, *, neutral=False):
            g = np.array([[0.2, 0.1, 0.05],
                          [0.2, 0.15, 0.05],
                          [0.1, 0.1, 0.05]])
            return ScoreGrid(grid=g)

    model = TinyGoals()
    pid = log_prediction(conn, model, fixture, predicted_at="2021-12-31T12:00:00")
    grid = load_grid(conn, pid)
    assert grid is not None and grid.shape == (3, 3)
    assert grid.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(grid, model.predict_goals("A", "B").grid)
    # The stored W/D/L row and the stored grid tell the same story.
    row = conn.execute(
        "SELECT p_home, p_draw, p_away FROM predictions WHERE prediction_id=?",
        (pid,)).fetchone()
    wdl = model.predict_goals("A", "B").wdl()
    assert row[0] == pytest.approx(wdl.home)
    assert row[1] == pytest.approx(wdl.draw)
    assert row[2] == pytest.approx(wdl.away)


def test_wdl_only_model_stores_no_grid(setup):
    from fifapreds.loop.predict import load_grid

    conn, _, model, fixture = setup   # BaselineElo: W/D/L only
    pid = log_prediction(conn, model, fixture, predicted_at="2021-12-31T12:00:00")
    assert load_grid(conn, pid) is None
    assert conn.execute("SELECT COUNT(*) FROM score_grids").fetchone()[0] == 0
