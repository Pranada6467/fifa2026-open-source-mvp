"""E6b scoreline grading invariants: scoreline-log-loss with tail bucket,
O/U 2.5 + BTTS derivation from grids, exact-score top-k, ET exclusion,
and the integrity gate (only goals-model predictions with stored grids).
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fifapreds.db import init_predictions
from fifapreds.loop.predict import load_grid, log_prediction
from fifapreds.loop.score import (
    EPS,
    btts_from_grid,
    ou25_from_grid,
    score_scoreline_pending,
    scoreline_log_loss,
    top_k_scorelines,
)


# ----------------------------------------------------------------- unit tests

def _simple_grid() -> np.ndarray:
    """A 5x5 grid where 1-0 is the mode and low scores dominate."""
    g = np.zeros((5, 5))
    g[1, 0] = 0.25  # 1-0
    g[0, 0] = 0.15  # 0-0
    g[0, 1] = 0.12  # 0-1
    g[1, 1] = 0.11  # 1-1
    g[2, 0] = 0.08  # 2-0
    g[2, 1] = 0.07  # 2-1
    g[0, 2] = 0.06  # 0-2
    g[1, 2] = 0.05  # 1-2
    g[3, 0] = 0.04  # 3-0
    g[3, 1] = 0.03  # 3-1
    g[2, 2] = 0.02  # 2-2
    g[4, 0] = 0.01  # 4-0
    g[0, 3] = 0.01  # 0-3
    # sum = 1.0
    return g


class TestScorelineLogLoss:
    def test_in_grid_score(self):
        g = _simple_grid()
        ll = scoreline_log_loss(g, 1, 0)
        assert ll == pytest.approx(-math.log(0.25))

    def test_zero_probability_in_grid_uses_eps(self):
        g = _simple_grid()
        ll = scoreline_log_loss(g, 4, 4)
        assert ll == pytest.approx(-math.log(EPS))

    def test_out_of_grid_uses_tail_bucket(self):
        g = _simple_grid()
        ll = scoreline_log_loss(g, 10, 10)
        assert ll == pytest.approx(-math.log(EPS))

    def test_golden_value(self):
        g = _simple_grid()
        assert scoreline_log_loss(g, 0, 0) == pytest.approx(-math.log(0.15))


class TestOU25:
    def test_known_grid(self):
        g = _simple_grid()
        # Goals >= 3: (2,1)=0.07, (0,3)=0.01, (3,0)=0.04, (3,1)=0.03,
        #             (1,2)=0.05, (2,2)=0.02, (4,0)=0.01, (0,2) has total=2 so excluded
        # Wait: (0,2) total = 2 < 3, excluded. (1,2) total = 3 >= 3, included.
        # Over 2.5: 2+1=3, 0+3=3, 3+0=3, 3+1=4, 1+2=3, 2+2=4, 4+0=4
        expected = 0.07 + 0.01 + 0.04 + 0.03 + 0.05 + 0.02 + 0.01
        assert ou25_from_grid(g) == pytest.approx(expected)

    def test_all_zeros_gives_zero(self):
        g = np.array([[0.5, 0.3], [0.15, 0.05]])
        # 0+0=0, 0+1=1, 1+0=1, 1+1=2 — all < 3
        assert ou25_from_grid(g) == pytest.approx(0.0)


class TestBTTS:
    def test_known_grid(self):
        g = _simple_grid()
        # Both score: i>=1 AND j>=1: (1,1)=0.11, (2,1)=0.07, (1,2)=0.05,
        #             (3,1)=0.03, (2,2)=0.02
        expected = 0.11 + 0.07 + 0.05 + 0.03 + 0.02
        assert btts_from_grid(g) == pytest.approx(expected)

    def test_only_zeros_in_both_score_region(self):
        g = np.zeros((3, 3))
        g[0, 0] = 0.5
        g[0, 1] = 0.3
        g[1, 0] = 0.2
        assert btts_from_grid(g) == pytest.approx(0.0)


class TestTopK:
    def test_top1_is_mode(self):
        g = _simple_grid()
        top = top_k_scorelines(g, k=1)
        assert top[0] == (1, 0)

    def test_top3_order(self):
        g = _simple_grid()
        top = top_k_scorelines(g, k=3)
        assert top == [(1, 0), (0, 0), (0, 1)]


# ------------------------------------------------ integration: score_scoreline_pending

def _stub_store(matches_data):
    """Minimal MatchStore-like object for testing."""
    class _Store:
        def __init__(self, played_df):
            self.played = played_df
    return _Store(matches_data)


def _make_fixture_and_predict(conn, model, fixture_row, **kwargs):
    """Log a prediction with a grid (goals-capable model)."""
    return log_prediction(conn, model, fixture_row, **kwargs)


class TestScoreScorelinePending:
    def test_grades_resolved_match(self, tmp_path):
        from fifapreds.models.dixoncoles import DixonColes

        # Build a small training set and a fixture
        cols = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "went_to_et"]
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rows = []
        day = pd.Timestamp("2024-01-01")
        for _ in range(10):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append((day, h, a, rng.poisson(1.5), rng.poisson(1.0),
                                 "League", True, False))
                    day += pd.Timedelta(days=1)
        train = pd.DataFrame(rows, columns=cols)
        model = DixonColes(min_matches=50).fit(train)

        # The "result" that happened
        played = pd.DataFrame([{
            "date": pd.Timestamp("2025-06-01"),
            "home_team": "A", "away_team": "B",
            "home_score": 2, "away_score": 1,
            "tournament": "Cup", "neutral": True,
            "went_to_et": False,
        }])
        fixture = played.iloc[0]

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        init_predictions(conn)

        pid = log_prediction(conn, model, fixture, context="backtest:test",
                             predicted_at="2025-05-30T00:00:00")
        assert load_grid(conn, pid) is not None

        store = _stub_store(played)
        result = score_scoreline_pending(conn, store)
        assert result["scored"] == 1
        assert result["skipped_et"] == 0

        row = conn.execute(
            "SELECT * FROM scores_scoreline WHERE prediction_id = ?", (pid,)
        ).fetchone()
        assert row is not None
        assert row["home_score"] == 2
        assert row["away_score"] == 1
        assert row["scoreline_log_loss"] > 0
        assert row["ou25_outcome"] == 1  # 2+1=3 >= 3
        assert row["btts_outcome"] == 1  # both scored
        conn.close()

    def test_skips_et_matches(self, tmp_path):
        from fifapreds.models.dixoncoles import DixonColes

        cols = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "went_to_et"]
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rows = []
        day = pd.Timestamp("2024-01-01")
        for _ in range(10):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append((day, h, a, rng.poisson(1.5), rng.poisson(1.0),
                                 "League", True, False))
                    day += pd.Timedelta(days=1)
        train = pd.DataFrame(rows, columns=cols)
        model = DixonColes(min_matches=50).fit(train)

        played = pd.DataFrame([{
            "date": pd.Timestamp("2025-06-01"),
            "home_team": "A", "away_team": "B",
            "home_score": 3, "away_score": 2,
            "tournament": "Cup", "neutral": True,
            "went_to_et": True,
        }])

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        init_predictions(conn)
        log_prediction(conn, model, played.iloc[0], context="backtest:test",
                       predicted_at="2025-05-30T00:00:00")
        store = _stub_store(played)
        result = score_scoreline_pending(conn, store)
        assert result["scored"] == 0
        assert result["skipped_et"] == 1
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running score_scoreline_pending twice doesn't double-score."""
        from fifapreds.models.dixoncoles import DixonColes

        cols = ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "went_to_et"]
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rows = []
        day = pd.Timestamp("2024-01-01")
        for _ in range(10):
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    rows.append((day, h, a, rng.poisson(1.5), rng.poisson(1.0),
                                 "League", True, False))
                    day += pd.Timedelta(days=1)
        train = pd.DataFrame(rows, columns=cols)
        model = DixonColes(min_matches=50).fit(train)

        played = pd.DataFrame([{
            "date": pd.Timestamp("2025-06-01"),
            "home_team": "A", "away_team": "B",
            "home_score": 1, "away_score": 0,
            "tournament": "Cup", "neutral": True,
            "went_to_et": False,
        }])

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        init_predictions(conn)
        log_prediction(conn, model, played.iloc[0], context="backtest:test",
                       predicted_at="2025-05-30T00:00:00")
        store = _stub_store(played)
        r1 = score_scoreline_pending(conn, store)
        r2 = score_scoreline_pending(conn, store)
        assert r1["scored"] == 1
        assert r2["scored"] == 0
        conn.close()
