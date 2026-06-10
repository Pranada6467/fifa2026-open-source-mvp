"""Odds snapshotter (V6) — capture-only.

Stores the raw provider payload for the World Cup match (h2h) and winner
(outrights) markets, with a UTC timestamp and the remaining quota. Pre-match
odds cannot be backfilled, so this runs on a daily-ish cadence to build a
market-vs-model history. De-vig / blending happens in Block 2.

Run:  python -m fifapreds.loop.odds
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import requests

from fifapreds import config, db

BASE = "https://api.the-odds-api.com/v4"

# (sport_key, market). Each pull costs one quota unit per region.
PULLS = [
    ("soccer_fifa_world_cup", "h2h"),
    ("soccer_fifa_world_cup_winner", "outrights"),
]


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def snapshot(
    regions: str = "us",
    odds_format: str = "decimal",
    conn: sqlite3.Connection | None = None,
) -> list[tuple[str, str, int, str | None]]:
    """Pull each market once and store the raw payload. Returns a summary list
    of (sport_key, market, n_events, quota_remaining)."""
    key = config.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY not set — put it in .env (see .env.example)")

    own = conn is None
    conn = conn or db.connect()
    db.init_odds(conn)

    summary: list[tuple[str, str, int, str | None]] = []
    try:
        for sport_key, market in PULLS:
            resp = requests.get(
                f"{BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": key,
                    "regions": regions,
                    "markets": market,
                    "oddsFormat": odds_format,
                },
                timeout=30,
            )
            remaining = resp.headers.get("x-requests-remaining")
            resp.raise_for_status()
            payload = resp.json()
            conn.execute(
                "INSERT INTO odds_snapshots "
                "(captured_at, sport_key, market, raw_json, requests_remaining) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _utcnow(),
                    sport_key,
                    market,
                    json.dumps(payload),
                    int(remaining) if remaining and remaining.isdigit() else None,
                ),
            )
            summary.append((sport_key, market, len(payload), remaining))
        conn.commit()
    finally:
        if own:
            conn.close()
    return summary


if __name__ == "__main__":
    for sport_key, market, n_events, remaining in snapshot():
        print(f"saved {sport_key}/{market}: {n_events} events  (quota left: {remaining})")
