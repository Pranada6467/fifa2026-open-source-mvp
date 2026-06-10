"""Match-level backtest on past World Cups — the first real calibration readout.

Replays 2014/2018/2022 in kickoff order through the exact live-loop machinery:
as-of reads (`store.before(d)`), provenance-logged predictions
(`loop.predict.log_prediction`), and the scorer (`loop.score.score_pending`).
No tournament simulation — this grades per-match W/D/L probabilities only.

Replay discipline per match day d:
- incremental models (Elo) ingest all international results in [cursor, d)
  — exactly what a live model would have seen between match days;
- batch models (Dixon-Coles) refit on `store.before(d)` — window + anchor are
  handled inside the model, so nothing dated >= d can leak in.

Gates (printed and enforced by `main`):
- every model must beat the uniform baseline ln(3) ≈ 1.0986 on log-loss;
- a model that looks *too* good (mean log-loss < 0.60 on WC matches) trips the
  leak canary — World Cups are not that predictable; suspect lookahead;
- pooled calibration table should hug the diagonal (reported, not enforced).

Run: `.venv/bin/python -m fifapreds.backtest [--years ...] [--elo-only]`
(Dixon-Coles refits per match day: the full 3-edition run takes ~5-10 min.)
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from fifapreds import db
from fifapreds.asof import MatchStore
from fifapreds.config import PROJECT_ROOT
from fifapreds.loop.predict import predict_fixtures
from fifapreds.loop.score import CLASSES, calibration_table, log_loss, score_pending
from fifapreds.models import BaselineElo, DixonColes
from fifapreds.models.base import Model

UNIFORM_LOG_LOSS = math.log(3)   # know-nothing forecast (1/3, 1/3, 1/3)
LEAK_CANARY = 0.60               # WC matches are never this predictable


def world_cup_matches(store: MatchStore, year: int) -> pd.DataFrame:
    wc = store.played
    wc = wc[(wc["tournament"] == "FIFA World Cup") & (wc["date"].dt.year == year)]
    if wc.empty:
        raise ValueError(f"no played World Cup matches found for {year}")
    return wc.sort_values("date", kind="stable")


def run_backtest(
    conn: sqlite3.Connection,
    store: MatchStore,
    models: Mapping[str, Callable[[], Model]],
    years: Iterable[int],
    *,
    verbose: bool = False,
) -> dict:
    """Replay each edition for each model, then grade everything.

    Returns the score_pending report; raises if any integrity violation
    surfaces — a backtest that trips its own audit is worthless.
    """
    for year in years:
        wc = world_cup_matches(store, year)
        dates = [pd.Timestamp(d) for d in sorted(wc["date"].unique())]
        context = f"backtest:wc{year}"
        for label, factory in models.items():
            model = factory()
            incremental = type(model).update is not Model.update
            cursor: pd.Timestamp | None = None
            for d in dates:
                t0 = time.perf_counter()
                if incremental and cursor is not None:
                    # Catch up on every international result since the last
                    # match day — what a live model would have ingested.
                    played = store.played
                    gap = played[(played["date"] >= cursor) & (played["date"] < d)]
                    for _, row in gap.iterrows():
                        model.update(row)
                else:
                    model.fit(store.before(d))
                day = wc[wc["date"] == d]
                predict_fixtures(conn, model, day, context=context)
                if verbose:
                    print(f"  {label:<4} wc{year} {d.date()}: {len(day)} predictions "
                          f"(train {time.perf_counter() - t0:.1f}s)")
                cursor = d

    report = score_pending(conn, store)
    if report["violations"]:
        raise RuntimeError(
            f"backtest integrity violations on predictions {report['violations']}"
        )
    return report


def leaderboard(conn: sqlite3.Connection) -> pd.DataFrame:
    """Mean out-of-sample metrics per model per context (lower = better)."""
    return pd.read_sql_query(
        """SELECT p.model_id, p.context, COUNT(*) AS n,
                  AVG(s.log_loss) AS log_loss, AVG(s.brier) AS brier,
                  AVG(s.rps) AS rps
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           GROUP BY p.model_id, p.context
           ORDER BY p.context, log_loss""",
        conn,
    )


def _pooled(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT p.model_id, p.p_home, p.p_draw, p.p_away, s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id""",
        conn,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", type=int, nargs="+", default=[2014, 2018, 2022])
    ap.add_argument("--elo-only", action="store_true",
                    help="skip the slow per-day Dixon-Coles refits")
    ap.add_argument("--db", default=str(PROJECT_ROOT / "data" / "backtest.db"))
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()                       # each run is a fresh, full replay
    conn = db.connect(db_path)
    store = MatchStore()

    models: dict[str, Callable[[], Model]] = {"elo": BaselineElo}
    if not args.elo_only:
        models["dc"] = DixonColes

    report = run_backtest(conn, store, models, args.years, verbose=True)
    print(f"\nscored {report['scored']} predictions "
          f"({report['pending']} pending, {len(report['violations'])} violations)")

    board = leaderboard(conn)
    print("\n== leaderboard (mean per match; lower = better) ==")
    print(board.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"   uniform-baseline log_loss = {UNIFORM_LOG_LOSS:.4f}")

    pooled = _pooled(conn)
    failed = False
    for model_id, grp in pooled.groupby("model_id"):
        probs = grp[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = grp["outcome"].map(CLASSES.index).to_numpy()
        mean_ll = float(log_loss(probs, outcomes).mean())
        print(f"\n== {model_id}: pooled calibration ({len(grp)} matches, "
              f"log_loss {mean_ll:.4f}) ==")
        table = calibration_table(probs, outcomes)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        gap = (table.dropna(subset=["p_mean"])
                    .query("n >= 30")
                    .eval("abs(freq - p_mean)").max())
        print(f"   max |freq - p| in populated bins: {gap:.3f}")
        if mean_ll >= UNIFORM_LOG_LOSS:
            print(f"   GATE FAIL: {model_id} does not beat the uniform baseline")
            failed = True
        if mean_ll < LEAK_CANARY:
            print(f"   GATE FAIL: {model_id} is suspiciously good — audit for lookahead")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
