"""SQLite connection + schema bootstrap.

Storage split (per the plan): SQLite holds append-only logs and snapshots;
raw match data lives in parquet under data/raw/. Schema grows by block; only
the tables a caller needs are created on demand via init_*().
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fifapreds.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "fifa2026.db"


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path) if db_path is not None else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


# Capture-only odds storage (V6). We keep the RAW provider payload with a
# timestamp + quota reading; parsing/de-vig into probabilities is Block 2.
_ODDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at        TEXT    NOT NULL,           -- ISO-8601 UTC pull time
    sport_key          TEXT    NOT NULL,           -- e.g. soccer_fifa_world_cup
    market             TEXT    NOT NULL,           -- h2h | outrights
    provider           TEXT    NOT NULL DEFAULT 'the-odds-api',
    raw_json           TEXT    NOT NULL,           -- full raw payload
    requests_remaining INTEGER                     -- provider quota at capture
);
CREATE INDEX IF NOT EXISTS idx_odds_captured ON odds_snapshots(captured_at);
"""


def init_odds(conn: sqlite3.Connection) -> None:
    conn.executescript(_ODDS_SCHEMA)
    conn.commit()


# Predictions log (T5). Append-only: a prediction row is never mutated; its
# grade lands in `scores` (one row per prediction, written once the result is
# known). Provenance columns make every leaderboard entry auditable — which
# model config, code version, and training cutoff produced each probability.
_PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    context          TEXT    NOT NULL DEFAULT 'live',  -- 'live' | 'backtest:wc2014' | ...
    match_id         INTEGER,                          -- matches.parquet id (provenance only)
    home_team        TEXT    NOT NULL,
    away_team        TEXT    NOT NULL,
    kickoff_ts       TEXT    NOT NULL,                 -- ISO-8601 (date-only fixtures: midnight)
    neutral          INTEGER NOT NULL,
    tournament       TEXT,
    p_home           REAL    NOT NULL CHECK (p_home BETWEEN 0 AND 1),
    p_draw           REAL    NOT NULL CHECK (p_draw BETWEEN 0 AND 1),
    p_away           REAL    NOT NULL CHECK (p_away BETWEEN 0 AND 1),
    model_id         TEXT    NOT NULL,
    model_version    TEXT    NOT NULL,
    code_version     TEXT,                             -- git sha at prediction time
    hyperparams_hash TEXT    NOT NULL,
    training_cutoff  TEXT    NOT NULL,                 -- model.trained_through (no-leak audit)
    odds_snapshot_id INTEGER REFERENCES odds_snapshots(snapshot_id),
    seed             INTEGER,                          -- only Monte-Carlo-derived predictions
    predicted_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_model   ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_pred_kickoff ON predictions(kickoff_ts);

CREATE TABLE IF NOT EXISTS scores (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(prediction_id),
    outcome       TEXT NOT NULL CHECK (outcome IN ('home', 'draw', 'away')),
    log_loss      REAL NOT NULL,                       -- natural log
    brier         REAL NOT NULL,                       -- multiclass, sum over 3 classes
    rps           REAL NOT NULL,                       -- ranked probability score (primary)
    scored_at     TEXT NOT NULL
);

-- E6a: the full scoreline grid behind each goals-model claim, captured AT
-- PREDICT TIME (a grid can never be honestly backfilled for a live claim).
-- grid is the row-major float64 bytes of P(home=i, away=j), normalized over
-- the (n_rows x n_cols) truncated support; scoreline grading (E6b) handles
-- out-of-grid scores via an explicit tail bucket.
CREATE TABLE IF NOT EXISTS score_grids (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(prediction_id),
    n_rows        INTEGER NOT NULL CHECK (n_rows > 0),
    n_cols        INTEGER NOT NULL CHECK (n_cols > 0),
    grid          BLOB    NOT NULL
);
"""


def init_predictions(conn: sqlite3.Connection) -> None:
    conn.executescript(_PREDICTIONS_SCHEMA)
    conn.commit()
