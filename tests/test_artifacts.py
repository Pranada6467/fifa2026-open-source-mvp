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
                           qualification_src=tmp_path / "missing-qual.parquet",
                           store=resolved)
    return out, meta


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    return build_synthetic_artifacts(tmp_path_factory.mktemp("artifacts"))


def test_all_artifact_files_written(built):
    out, _ = built
    for name in ("upcoming.parquet", "leaderboard.parquet", "calibration.parquet",
                 "scored.parquet", "surprises.parquet", "disagreement.parquet",
                 "meta.json"):
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
    # E2: the board's two-track split happens at export time.
    assert set(calibration["track"]) == {"live"}


def test_upcoming_consensus_columns(built):
    out, _ = built
    upcoming = pd.read_parquet(out / "upcoming.parquet")
    row = upcoming.iloc[0]
    # Single model, no market blend: consensus == the model, labelled as such.
    assert row["consensus_source"] == "model_avg"
    assert row["cons_p_home"] == pytest.approx(row["p_home"])


def test_surprises_one_row_per_match(built):
    out, _ = built
    surprises = pd.read_parquet(out / "surprises.parquet")
    assert len(surprises) == 1
    row = surprises.iloc[0]
    assert row["outcome"] == "home" and row["n_models"] == 1
    assert row["worst_model_id"] == "elo_baseline"
    assert 0.0 < row["consensus_p"] < 1.0


def test_meta_counts_and_degradation_note(built):
    out, meta = built
    assert meta["counts"] == {"upcoming": 1, "leaderboard": 1, "scored": 1,
                              "calibration": 10, "surprises": 1,
                              "disagreement": 0, "leaderboard_bands": 1,
                              "qualification": 0, "tournament": 0,
                              "scoreline_leaderboard": 0,
                              "scoreline_calibration": 0,
                              "scoreline_topn": 0}
    assert meta["models"] == ["elo_baseline"]
    assert any("missing-backtest.db" in note for note in meta["notes"])
    assert meta["data_through"] == "2022-01-01"
    assert json.loads((out / "meta.json").read_text()) == meta


def test_build_with_no_sources_is_empty_but_valid(tmp_path):
    store = _store(_HISTORY + [_FIX_FUTURE])
    meta = artifacts.build(tmp_path / "art", live_db=tmp_path / "no.db",
                           backtest_db=tmp_path / "no2.db",
                           tournament_src=tmp_path / "no3.parquet",
                           qualification_src=tmp_path / "no4.parquet",
                           store=store)
    assert meta["counts"] == {"upcoming": 0, "leaderboard": 0, "scored": 0,
                              "calibration": 0, "surprises": 0,
                              "disagreement": 0, "leaderboard_bands": 0,
                              "qualification": 0, "tournament": 0,
                              "scoreline_leaderboard": 0,
                              "scoreline_calibration": 0,
                              "scoreline_topn": 0}
    assert len(meta["notes"]) == 4  # two dbs + sim + qualification sources
    assert pd.read_parquet(tmp_path / "art" / "upcoming.parquet").empty


def _h2h_payload(home: str, away: str, prices=(3.0, 3.0, 3.0)) -> str:
    """One The Odds API h2h event at the given decimal prices (h, a, d)."""
    return json.dumps([{
        "commence_time": "2022-06-01T00:00:00Z",
        "home_team": home, "away_team": away,
        "bookmakers": [{
            "key": "bk", "title": "BK",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": prices[0]},
                {"name": away, "price": prices[1]},
                {"name": "Draw", "price": prices[2]},
            ]}],
        }],
    }])


def test_surprises_dedupe_two_models_and_worst_named(tmp_path):
    """Two models grade the same match -> ONE surprise row (D8), consensus is
    the model average, and the most-wrong model is named."""
    from fifapreds.models.roster import EloDecay

    store = _store(_HISTORY + [_FIX_PLAYED])
    train = store.before("2022-01-01")
    base, decay = BaselineElo().fit(train), EloDecay().fit(train)
    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(live_db)
    fix = store.upcoming().iloc[0]
    log_prediction(conn, base, fix, predicted_at="2021-12-31T12:00:00")
    log_prediction(conn, decay, fix, predicted_at="2021-12-31T12:00:00")
    resolved = _store(_HISTORY + [_RESULT])
    assert score_pending(conn, resolved)["scored"] == 2
    conn.close()

    artifacts.build(tmp_path / "art", live_db=live_db,
                    backtest_db=tmp_path / "no.db",
                    tournament_src=tmp_path / "no.parquet", store=resolved)
    surprises = pd.read_parquet(tmp_path / "art" / "surprises.parquet")
    assert len(surprises) == 1
    row = surprises.iloc[0]
    assert row["n_models"] == 2 and row["consensus_source"] == "model_avg"
    # The named worst model really did assign the lowest p to the outcome.
    scored = pd.read_parquet(tmp_path / "art" / "scored.parquet")
    worst_p = scored.set_index("model_id")["p_home"].min()
    assert row["worst_model_p"] == pytest.approx(worst_p)
    assert row["consensus_p"] == pytest.approx(scored["p_home"].mean())


def test_disagreement_from_odds_snapshot(tmp_path):
    """An h2h snapshot covering the upcoming fixture yields a disagreement row
    against the PURE de-vigged market (not market_blend)."""
    from fifapreds.db import init_odds

    store = _store(_HISTORY + [_FIX_FUTURE])
    model = BaselineElo().fit(store.before("2022-06-01"))
    live_db = tmp_path / "live.db"
    conn = sqlite3.connect(live_db)
    fix = store.upcoming().iloc[0]
    log_prediction(conn, model, fix, predicted_at="2022-05-30T12:00:00")
    init_odds(conn)
    conn.execute(
        "INSERT INTO odds_snapshots (captured_at, sport_key, market, raw_json)"
        " VALUES ('2022-05-31T00:00:00', 'soccer_fifa_world_cup', 'h2h', ?)",
        (_h2h_payload("A", "B"),))
    conn.commit()
    conn.close()

    artifacts.build(tmp_path / "art", live_db=live_db,
                    backtest_db=tmp_path / "no.db",
                    tournament_src=tmp_path / "no.parquet", store=store)
    dis = pd.read_parquet(tmp_path / "art" / "disagreement.parquet")
    assert len(dis) == 1
    row = dis.iloc[0]
    # Fair 3.0/3.0/3.0 odds de-vig to uniform; Elo (A favoured) must disagree.
    assert row["market_p_home"] == pytest.approx(1 / 3, abs=1e-6)
    expected_delta = max(abs(row[f"model_p_{k}"] - row[f"market_p_{k}"])
                         for k in ("home", "draw", "away"))
    assert row["delta"] == pytest.approx(expected_delta) and row["delta"] > 0
    assert row["model_pick"] == "home"
    assert row["snapshot_id"] == 1

    # No snapshot at all -> empty frame, not a crash.
    artifacts.build(tmp_path / "art2", live_db=tmp_path / "missing.db",
                    backtest_db=tmp_path / "no.db",
                    tournament_src=tmp_path / "no.parquet", store=store)
    assert pd.read_parquet(tmp_path / "art2" / "disagreement.parquet").empty
