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
