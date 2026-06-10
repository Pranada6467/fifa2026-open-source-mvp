"""Ingest martj42 → a single canonical `matches` table.

   results.csv ──► canonical names ──► split played / fixtures ──┐
   shootouts.csv ─► went_to_et flag ───────────────────────────►┤
                                                                 ▼
                                          data/processed/matches.parquet

Columns out:
  match_id, date, home_team, away_team, home_score, away_score,
  tournament, neutral (bool), is_played (bool), went_to_et (bool)

Notes
- Future World Cup fixtures live in results.csv with null scores → is_played=False.
- Scores include extra time, exclude penalties (Block V/V2). `went_to_et` marks
  matches that reached a shootout (definitely had ET) so the 90-minute goals
  model can exclude/down-weight them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fifapreds.config import PROJECT_ROOT
from fifapreds.registry import canonical

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MATCHES_PARQUET = PROCESSED / "matches.parquet"


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["neutral"] = df["neutral"].astype(bool)
    df["is_played"] = df["home_score"].notna() & df["away_score"].notna()
    return df


def _shootout_keys() -> set[tuple]:
    """(date, frozenset{teams}) for every match that went to a shootout."""
    s = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    s["home_team"] = s["home_team"].map(canonical)
    s["away_team"] = s["away_team"].map(canonical)
    return {
        (d.normalize(), frozenset((h, a)))
        for d, h, a in zip(s["date"], s["home_team"], s["away_team"])
    }


def build_matches(write: bool = True) -> pd.DataFrame:
    df = _load_results()
    shootouts = _shootout_keys()
    df["went_to_et"] = [
        (d.normalize(), frozenset((h, a))) in shootouts
        for d, h, a in zip(df["date"], df["home_team"], df["away_team"])
    ]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", df.index.astype("int64"))

    cols = [
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "tournament", "neutral",
        "is_played", "went_to_et",
    ]
    out = df[cols]
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(MATCHES_PARQUET, index=False)
    return out


def load_matches() -> pd.DataFrame:
    """Read the processed matches table, building it if missing."""
    if not MATCHES_PARQUET.exists():
        return build_matches(write=True)
    return pd.read_parquet(MATCHES_PARQUET)


if __name__ == "__main__":
    m = build_matches()
    played = int(m["is_played"].sum())
    print(f"matches.parquet: {len(m)} rows | played={played} | fixtures={len(m) - played}")
    print(f"date range: {m.date.min().date()} -> {m.date.max().date()}")
    print(f"went_to_et: {int(m.went_to_et.sum())} matches flagged")
