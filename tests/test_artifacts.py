"""Artifacts exporter: schema per file, latest-claim-per-(fixture, model)
dedupe, played fixtures dropped from `upcoming`, missing-source degradation,
and meta counts that match the files. The synthetic builder here is reused by
the app smoke test (tests/test_app.py).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.loop.predict import log_prediction
from fifapreds.loop.score import score_pending
from fifapreds.models import BaselineElo
from fifapreds.publish import artifacts

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
_FIX_PLAYED = (3, "2022-01-01", "A", "B", None, None, "FIFA World Cup", True, False, False)
_RESULT = (3, "2022-01-01", "A", "B", 2, 0, "FIFA World Cup", True, True, False)
_FIX_FUTURE = (4, "2022-06-01", "A", "B", None, None, "FIFA World Cup", True, False, False)


def build_synthetic_artifacts(tmp_path: Path) -> tuple[Path, dict]:
    """One resolved+scored fixture, one upcoming fixture logged twice (the
    second claim must win), backtest db absent. Returns (artifacts_dir, meta)."""
    store = _store(_HISTORY + [_FIX_PLAYED, _FIX_FUTURE])
    model = BaselineElo().fit(store.before("2022-01-01"))
    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(live_db)
    fix3 = store.upcoming().iloc[0]
    fix4 = store.upcoming().iloc[1]
    log_prediction(conn, model, fix3, predicted_at="2021-12-31T12:00:00")
    log_prediction(conn, model, fix4, predicted_at="2021-12-30T12:00:00")
    log_prediction(conn, model, fix4, predicted_at="2021-12-31T12:00:00")  # supersedes
    resolved = _store(_HISTORY + [_RESULT, _FIX_FUTURE])
    report = score_pending(conn, resolved)
    assert report["scored"] == 1
    conn.close()

    out = tmp_path / "artifacts"
    meta = artifacts.build(out, live_db=live_db,
                           backtest_db=tmp_path / "missing-backtest.db",
                           tournament_src=tmp_path / "missing-sim.parquet",
                           store=resolved)
    return out, meta


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    return build_synthetic_artifacts(tmp_path_factory.mktemp("artifacts"))


def test_all_artifact_files_written(built):
    out, _ = built
    for name in ("upcoming.parquet", "leaderboard.parquet", "calibration.parquet",
                 "scored.parquet", "meta.json"):
        assert (out / name).exists()


def test_upcoming_keeps_latest_claim_and_drops_played(built):
    out, _ = built
    upcoming = pd.read_parquet(out / "upcoming.parquet")
    # Fixture 3 has been played: gone. Fixture 4 was claimed twice: one row,
    # the later claim.
    assert list(upcoming["match_id"]) == [4]
    assert upcoming["predicted_at"].iloc[0] == "2021-12-31T12:00:00"


def test_leaderboard_and_scored(built):
    out, _ = built
    board = pd.read_parquet(out / "leaderboard.parquet")
    assert len(board) == 1
    assert board.iloc[0]["model_id"] == "elo_baseline" and board.iloc[0]["n"] == 1
    scored = pd.read_parquet(out / "scored.parquet")
    assert len(scored) == 1 and scored.iloc[0]["outcome"] == "home"


def test_calibration_pools_every_claim(built):
    out, _ = built
    calibration = pd.read_parquet(out / "calibration.parquet")
    # 1 graded prediction x 3 classes = 3 pooled claims across the bins.
    assert calibration["n"].sum() == 3
    assert set(calibration["model_id"]) == {"elo_baseline"}


def test_meta_counts_and_degradation_note(built):
    out, meta = built
    assert meta["counts"] == {"upcoming": 1, "leaderboard": 1, "scored": 1,
                              "calibration": 10, "tournament": 0}
    assert meta["models"] == ["elo_baseline"]
    assert any("missing-backtest.db" in note for note in meta["notes"])
    assert meta["data_through"] == "2022-01-01"
    assert json.loads((out / "meta.json").read_text()) == meta


def test_build_with_no_sources_is_empty_but_valid(tmp_path):
    store = _store(_HISTORY + [_FIX_FUTURE])
    meta = artifacts.build(tmp_path / "art", live_db=tmp_path / "no.db",
                           backtest_db=tmp_path / "no2.db",
                           tournament_src=tmp_path / "no3.parquet", store=store)
    assert meta["counts"] == {"upcoming": 0, "leaderboard": 0, "scored": 0,
                              "calibration": 0, "tournament": 0}
    assert len(meta["notes"]) == 3  # two dbs + the tournament sim source
    assert pd.read_parquet(tmp_path / "art" / "upcoming.parquet").empty
