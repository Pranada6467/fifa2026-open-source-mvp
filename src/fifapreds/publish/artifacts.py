"""Precompute the artifacts the Streamlit viewer reads — the PUBLISH step.

The app never touches SQLite or fits a model: this job exports everything it
shows into committed files under `artifacts/`, so the deployed viewer is a
pure file reader and the repo history doubles as a public prediction record.

Sources: `data/fifa2026.db` (live predictions) + `data/backtest.db` (the
2014/18/22 proving ground). Either may be absent — whatever exists is
exported and `meta.json` says what was missing.

Outputs (all parquet unless noted):
- upcoming.parquet     latest claim per (match_id, model_id) for unplayed fixtures
- leaderboard.parquet  mean log-loss/Brier/RPS per model x context
- calibration.parquet  pooled reliability table per model (long format)
- scored.parquet       graded predictions with outcomes (the results feed)
- tournament.parquet   Monte Carlo trophy odds per (model_id, team), copied
                       from data/tournament_sim.parquet when the orchestrator
                       has produced one (T11/T14)
- meta.json            generated_at, git sha, data-through date, row counts
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds.asof import MatchStore
from fifapreds.config import PROJECT_ROOT
from fifapreds.db import DB_PATH
from fifapreds.loop.predict import code_version
from fifapreds.loop.score import CLASSES, calibration_table

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKTEST_DB = PROJECT_ROOT / "data" / "backtest.db"
TOURNAMENT_SRC = PROJECT_ROOT / "data" / "tournament_sim.parquet"

_PRED_COLS = ["prediction_id", "context", "match_id", "home_team", "away_team",
              "kickoff_ts", "neutral", "tournament", "p_home", "p_draw", "p_away",
              "model_id", "model_version", "hyperparams_hash", "training_cutoff",
              "predicted_at"]
_SCORED_SQL = """
    SELECT p.context, p.match_id, p.home_team, p.away_team, p.kickoff_ts,
           p.tournament, p.p_home, p.p_draw, p.p_away, p.model_id,
           s.outcome, s.log_loss, s.brier, s.rps, s.scored_at
    FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
"""
_LEADERBOARD_SQL = """
    SELECT p.model_id, p.context, COUNT(*) AS n,
           AVG(s.log_loss) AS log_loss, AVG(s.brier) AS brier, AVG(s.rps) AS rps
    FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
    GROUP BY p.model_id, p.context
    ORDER BY p.context, log_loss
"""


def _read(db_path: Path, sql: str) -> pd.DataFrame | None:
    """Query a db that may not exist yet (None = source missing/empty)."""
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn)
        except pd.errors.DatabaseError:   # schema not initialised yet
            return None


def _concat(frames: list[pd.DataFrame | None], columns: list[str]) -> pd.DataFrame:
    present = [f for f in frames if f is not None and not f.empty]
    return pd.concat(present, ignore_index=True) if present else pd.DataFrame(columns=columns)


def build(
    out_dir: Path | str = ARTIFACTS_DIR,
    *,
    live_db: Path | str = DB_PATH,
    backtest_db: Path | str = BACKTEST_DB,
    tournament_src: Path | str = TOURNAMENT_SRC,
    store: MatchStore | None = None,
) -> dict:
    """Export all artifacts; returns the meta dict that was written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = store or MatchStore()
    notes = [f"missing source: {p}" for p in (live_db, backtest_db) if not Path(p).exists()]

    # Upcoming: latest claim per (fixture, model), only for still-unplayed fixtures.
    live_preds = _read(live_db, f"SELECT {', '.join(_PRED_COLS)} FROM predictions")
    unplayed_ids = set(store.all.loc[~store.all["is_played"], "match_id"])
    if live_preds is not None and not live_preds.empty:
        upcoming = (
            live_preds[live_preds["match_id"].isin(unplayed_ids)]
            .sort_values("predicted_at")
            .groupby(["match_id", "model_id"], as_index=False)
            .last()
            .sort_values(["kickoff_ts", "match_id", "model_id"])
        )
    else:
        upcoming = pd.DataFrame(columns=_PRED_COLS)
    upcoming.to_parquet(out / "upcoming.parquet", index=False)

    leaderboard = _concat([_read(live_db, _LEADERBOARD_SQL), _read(backtest_db, _LEADERBOARD_SQL)],
                          ["model_id", "context", "n", "log_loss", "brier", "rps"])
    leaderboard.to_parquet(out / "leaderboard.parquet", index=False)

    scored = _concat([_read(live_db, _SCORED_SQL), _read(backtest_db, _SCORED_SQL)],
                     ["context", "match_id", "home_team", "away_team", "kickoff_ts",
                      "tournament", "p_home", "p_draw", "p_away", "model_id",
                      "outcome", "log_loss", "brier", "rps", "scored_at"])
    scored.to_parquet(out / "scored.parquet", index=False)

    # Calibration: pooled over every graded prediction, per model.
    tables = []
    for model_id, grp in scored.groupby("model_id"):
        table = calibration_table(
            grp[["p_home", "p_draw", "p_away"]].to_numpy(),
            grp["outcome"].map(CLASSES.index).to_numpy(),
        )
        table.insert(0, "model_id", model_id)
        tables.append(table)
    calibration = _concat(tables, ["model_id", "bin_lo", "bin_hi", "n", "p_mean", "freq"])
    calibration.to_parquet(out / "calibration.parquet", index=False)

    # Tournament odds: pass through whatever the orchestrator last simulated.
    if Path(tournament_src).exists():
        tournament = pd.read_parquet(tournament_src)
        tournament.to_parquet(out / "tournament.parquet", index=False)
    else:
        tournament = pd.DataFrame()
        notes.append(f"missing source: {tournament_src}")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "data_through": str(store.played["date"].max().date()),
        "counts": {
            "upcoming": int(len(upcoming)),
            "leaderboard": int(len(leaderboard)),
            "scored": int(len(scored)),
            "calibration": int(len(calibration)),
            "tournament": int(len(tournament)),
        },
        "models": sorted(set(leaderboard["model_id"]) | set(upcoming["model_id"])),
        "notes": notes,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    meta = build()
    print(f"artifacts -> {ARTIFACTS_DIR}")
    for name, n in meta["counts"].items():
        print(f"  {name:<12} {n} rows")
    for note in meta["notes"]:
        print(f"  NOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
