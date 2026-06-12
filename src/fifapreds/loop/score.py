"""Grade resolved predictions — the calibration engine's scorekeeper.

Metrics (all per-match, lower = better, probability order home/draw/away):
- log_loss: -ln(p_outcome), clamped so a delusional p=0 costs -ln(EPS), not inf.
- brier: multiclass Σ(p_k - y_k)², range [0, 2].
- rps: ranked probability score over the ordered outcomes (the football-
  standard primary metric — punishes putting mass on the *far* wrong side).

Conventions cross-checked against penaltyblog 1.11.0 in tests: rps matches
`rps_array` exactly; brier matches `multiclass_brier_score`; penaltyblog's
`ignorance_score` is log2-based, ours is natural log (ratio ln 2).

`score_pending` grades every unscored prediction whose result is known,
enforcing the integrity rules: training_cutoff must predate kickoff (both live
and backtest — the no-leak audit), and live predictions must have been made
before kickoff (replayed backtests are exempt from the wall-clock rule: their
predicted_at is honest replay time, their out-of-sample claim rests on
training_cutoff). Violating rows are never scored, only reported.

Outcome semantics: scores include extra time (martj42), so a shootout counts
as a draw — correct for 'result after 120 minutes' W/D/L grading.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fifapreds.asof import MatchStore
from fifapreds.db import init_predictions

EPS = 1e-9
CLASSES = ("home", "draw", "away")


def _naive_utc(ts) -> pd.Timestamp:
    """Timestamps in the log mix conventions: predicted_at is tz-aware UTC,
    kickoff_ts/training_cutoff are naive (martj42 dates). Compare on a single
    footing: UTC with the tzinfo dropped."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def outcome_index(home_score: float, away_score: float) -> int:
    """0=home, 1=draw, 2=away (the order penaltyblog metrics expect)."""
    if home_score > away_score:
        return 0
    return 1 if home_score == away_score else 2


def _as_probs(probs) -> np.ndarray:
    p = np.atleast_2d(np.asarray(probs, dtype=float))
    if p.shape[1] != 3:
        raise ValueError(f"expected (n, 3) probabilities, got {p.shape}")
    return p


def log_loss(probs, outcomes) -> np.ndarray:
    """-ln(p assigned to what happened), clamped to [EPS, 1-EPS]."""
    p = _as_probs(probs)
    picked = p[np.arange(len(p)), np.asarray(outcomes, dtype=int)]
    return -np.log(np.clip(picked, EPS, 1.0 - EPS))


def brier(probs, outcomes) -> np.ndarray:
    p = _as_probs(probs)
    y = np.zeros_like(p)
    y[np.arange(len(p)), np.asarray(outcomes, dtype=int)] = 1.0
    return ((p - y) ** 2).sum(axis=1)


def rps(probs, outcomes) -> np.ndarray:
    p = _as_probs(probs)
    y = np.zeros_like(p)
    y[np.arange(len(p)), np.asarray(outcomes, dtype=int)] = 1.0
    cum_diff = np.cumsum(p, axis=1) - np.cumsum(y, axis=1)
    return (cum_diff[:, :-1] ** 2).sum(axis=1) / (p.shape[1] - 1)


def binary_calibration_table(probs, happened, n_bins: int = 10,
                             alpha: float = 0.05) -> pd.DataFrame:
    """Reliability table for ANY binary event (T3 — one implementation shared
    by W/D/L one-vs-rest, group qualification (E4), and O/U-2.5/BTTS (E6)).

    Among all claims of probability ~p, how often did the claimed thing
    happen? Honest calibration shows freq ≈ p_mean per bin (the diagonal).
    `ci_lo`/`ci_hi` are Wilson score intervals on freq (statsmodels, not
    hand-rolled) so thin bins visibly carry their uncertainty.
    """
    from statsmodels.stats.proportion import proportion_confint

    flat_p = np.asarray(probs, dtype=float).ravel()
    flat_y = np.asarray(happened, dtype=float).ravel()
    if flat_p.shape != flat_y.shape:
        raise ValueError(f"shape mismatch: {flat_p.shape} vs {flat_y.shape}")
    # right-closed bins; p=0 lands in the first bin
    idx = np.minimum((flat_p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            successes = int(flat_y[mask].sum())
            ci_lo, ci_hi = proportion_confint(successes, n, alpha=alpha,
                                              method="wilson")
        rows.append({
            "bin_lo": b / n_bins,
            "bin_hi": (b + 1) / n_bins,
            "n": n,
            "p_mean": float(flat_p[mask].mean()) if n else np.nan,
            "freq": float(flat_y[mask].mean()) if n else np.nan,
            "ci_lo": float(ci_lo) if n else np.nan,
            "ci_hi": float(ci_hi) if n else np.nan,
        })
    return pd.DataFrame(rows)


def calibration_table(probs, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Pooled one-vs-rest reliability table over (n, 3) W/D/L claims — each
    class probability becomes one binary claim, binned by the shared
    `binary_calibration_table`."""
    p = _as_probs(probs)
    y = np.zeros_like(p)
    y[np.arange(len(p)), np.asarray(outcomes, dtype=int)] = 1.0
    return binary_calibration_table(p.ravel(), y.ravel(), n_bins=n_bins)


def score_pending(conn: sqlite3.Connection, store: MatchStore) -> dict:
    """Grade unscored predictions whose results are in (the loop's SCORE step).

    Returns {"scored": int, "pending": int, "violations": [prediction_id, ...],
    "results": DataFrame of newly written score rows}.
    """
    init_predictions(conn)
    todo = pd.read_sql_query(
        """SELECT p.* FROM predictions p
           LEFT JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE s.prediction_id IS NULL""",
        conn,
    )
    if todo.empty:
        return {"scored": 0, "pending": 0, "violations": [], "results": pd.DataFrame()}

    played = store.played
    results = {
        (d.normalize(), h, a): outcome_index(hs, as_)
        for d, h, a, hs, as_ in zip(
            played["date"], played["home_team"], played["away_team"],
            played["home_score"], played["away_score"],
        )
    }

    scored_rows, violations, pending = [], [], 0
    now = datetime.now(timezone.utc).isoformat()
    for row in todo.itertuples(index=False):
        kickoff = _naive_utc(row.kickoff_ts)
        if _naive_utc(row.training_cutoff) >= kickoff:
            violations.append(int(row.prediction_id))      # leaked training data
            continue
        if row.context == "live" and _naive_utc(row.predicted_at) >= kickoff:
            violations.append(int(row.prediction_id))      # predicted after kickoff
            continue
        outcome = results.get((kickoff.normalize(), row.home_team, row.away_team))
        if outcome is None:
            pending += 1                                   # result not in yet
            continue
        probs = [[row.p_home, row.p_draw, row.p_away]]
        scored_rows.append({
            "prediction_id": int(row.prediction_id),
            "outcome": CLASSES[outcome],
            "log_loss": float(log_loss(probs, [outcome])[0]),
            "brier": float(brier(probs, [outcome])[0]),
            "rps": float(rps(probs, [outcome])[0]),
            "scored_at": now,
        })

    if scored_rows:
        conn.executemany(
            """INSERT INTO scores (prediction_id, outcome, log_loss, brier, rps, scored_at)
               VALUES (:prediction_id, :outcome, :log_loss, :brier, :rps, :scored_at)""",
            scored_rows,
        )
        conn.commit()
    return {
        "scored": len(scored_rows),
        "pending": pending,
        "violations": violations,
        "results": pd.DataFrame(scored_rows),
    }
