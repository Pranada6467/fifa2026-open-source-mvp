"""Backtest integration gate (Elo-only, WC2022, so it stays fast): the replay
covers all 64 matches, every prediction's training cutoff strictly predates
its kickoff (the no-leak audit), the model state genuinely advances during the
tournament, and the calibration lands in the plausible band — better than a
know-nothing uniform forecast, but not suspiciously good (too good = leak).
The full 3-edition, both-models run is the CLI: `python -m fifapreds.backtest`.
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.backtest import (LEAK_CANARY, UNIFORM_LOG_LOSS, leaderboard,
                                run_backtest, world_cup_matches)
from fifapreds.models import BaselineElo


@pytest.fixture(scope="module")
def replayed():
    conn = sqlite3.connect(":memory:")
    store = MatchStore()
    report = run_backtest(conn, store, {"elo": BaselineElo}, years=[2022])
    return conn, store, report


def test_all_matches_predicted_and_scored(replayed):
    conn, store, report = replayed
    n_matches = len(world_cup_matches(store, 2022))
    assert n_matches == 64
    assert report["scored"] == n_matches
    assert report["pending"] == 0 and report["violations"] == []


def test_no_lookahead_audit(replayed):
    conn, _, _ = replayed
    rows = pd.read_sql_query(
        "SELECT training_cutoff, kickoff_ts FROM predictions", conn)
    cutoff = pd.to_datetime(rows["training_cutoff"])
    kickoff = pd.to_datetime(rows["kickoff_ts"])
    assert (cutoff < kickoff).all()  # CRITICAL: silently flattering otherwise


def test_model_state_advances_through_tournament(replayed):
    conn, _, _ = replayed
    rows = pd.read_sql_query(
        "SELECT kickoff_ts, training_cutoff FROM predictions ORDER BY kickoff_ts", conn)
    first = pd.Timestamp(rows["training_cutoff"].iloc[0])
    last = pd.Timestamp(rows["training_cutoff"].iloc[-1])
    # The final is predicted by a model that has ingested the group stage —
    # a frozen-ratings replay would be a different (wrong) experiment.
    assert last > first


def test_calibration_in_plausible_band(replayed):
    conn, _, _ = replayed
    board = leaderboard(conn)
    assert len(board) == 1
    ll = float(board["log_loss"].iloc[0])
    assert ll < UNIFORM_LOG_LOSS          # carries real information
    assert ll > LEAK_CANARY               # not impossibly sharp (leak canary)
    assert 0 < float(board["rps"].iloc[0]) < 0.5
    assert 0 < float(board["brier"].iloc[0]) < 2.0 / 3.0  # better than uniform's 0.667
