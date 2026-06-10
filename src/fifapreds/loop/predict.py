"""Write predictions with full provenance — the only path into the log.

Every row records model_id, model_version, code_version (git sha),
hyperparams_hash, training_cutoff, odds_snapshot_id, seed, predicted_at and
kickoff_ts, so any leaderboard number can be traced back to the exact model
state that produced it.

Lookahead guard at write time: the model's `trained_through` must be strictly
before kickoff. A model that has already seen results from kickoff day (or
later) raises here — predictions contaminated at creation never reach the log.
"""
from __future__ import annotations

import sqlite3
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd

from fifapreds.config import PROJECT_ROOT
from fifapreds.db import init_predictions
from fifapreds.models.base import Model


@lru_cache(maxsize=1)
def code_version() -> str | None:
    """Short git sha of the working tree (None outside a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def log_prediction(
    conn: sqlite3.Connection,
    model: Model,
    fixture: Mapping[str, Any] | pd.Series,
    *,
    predicted_at: pd.Timestamp | str | None = None,
    odds_snapshot_id: int | None = None,
    seed: int | None = None,
    context: str = "live",
) -> int:
    """Predict one fixture and append the row; returns prediction_id.

    `fixture` needs date (kickoff), home_team, away_team, neutral; tournament
    and match_id are carried through when present. The W/D/L probabilities are
    computed here, from the model being logged — they cannot drift apart.
    """
    kickoff = pd.Timestamp(fixture["date"])
    if model.trained_through is None:
        raise ValueError("model is not fitted — nothing to log")
    if model.trained_through >= kickoff:
        raise ValueError(
            f"lookahead: model trained through {model.trained_through} but "
            f"kickoff is {kickoff} — prediction would not be out-of-sample"
        )
    neutral = bool(fixture["neutral"])
    wdl = model.predict_wdl(fixture["home_team"], fixture["away_team"], neutral=neutral)
    predicted_at = pd.Timestamp(
        predicted_at if predicted_at is not None else datetime.now(timezone.utc)
    )

    init_predictions(conn)
    cur = conn.execute(
        """INSERT INTO predictions (
               context, match_id, home_team, away_team, kickoff_ts, neutral,
               tournament, p_home, p_draw, p_away, model_id, model_version,
               code_version, hyperparams_hash, training_cutoff,
               odds_snapshot_id, seed, predicted_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            context,
            int(fixture["match_id"]) if not pd.isna(fixture.get("match_id")) else None,
            fixture["home_team"],
            fixture["away_team"],
            kickoff.isoformat(),
            int(neutral),
            fixture.get("tournament"),
            wdl.home, wdl.draw, wdl.away,
            model.model_id,
            model.model_version,
            code_version(),
            model.hyperparams_hash,
            model.trained_through.isoformat(),
            odds_snapshot_id,
            seed,
            predicted_at.isoformat(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def predict_fixtures(
    conn: sqlite3.Connection,
    model: Model,
    fixtures: pd.DataFrame,
    **kwargs: Any,
) -> list[int]:
    """Log one prediction per fixture row (kwargs as in log_prediction)."""
    return [log_prediction(conn, model, row, **kwargs) for _, row in fixtures.iterrows()]
