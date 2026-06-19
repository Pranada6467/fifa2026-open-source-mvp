"""LOTO decision gate for `DixonColesTournamentWeighted` (S1).

Replays WC2014/18/22 with plain DC and the tournament-weighted variant
through the existing backtest plumbing, then computes LOTO pooled log-loss
and the seeded paired-bootstrap SE of the gap. Ships the variant only when
the gap exceeds 1 SE in DC's favour — same gate as the S6 ν sweep.

Wall-time warning: each variant refits Dixon-Coles per WC match day, ~5–10
min per variant. Total runtime is ~10–20 min. The scratch DB at
/tmp/dc_tw_tune.db is deleted + rewritten each run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds import db
from fifapreds.asof import MatchStore
from fifapreds.backtest import run_backtest
from fifapreds.loop.score import CLASSES, log_loss
from fifapreds.models.dixoncoles import DixonColes
from fifapreds.models.roster import DixonColesTournamentWeighted


def _row_loss(df: pd.DataFrame) -> np.ndarray:
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = df["outcome"].map(CLASSES.index).to_numpy()
    return log_loss(probs, outcomes)


def _pooled(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT p.model_id, p.context, p.p_home, p.p_draw, p.p_away,
                  s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE p.context LIKE 'backtest:wc%'""",
        conn,
    )
    df["year"] = df["context"].str.removeprefix("backtest:wc").astype(int)
    return df


def _loto(pooled: pd.DataFrame, model_id: str, years: list[int]) -> float:
    mg = pooled[pooled["model_id"] == model_id]
    fold_losses = [
        float(_row_loss(mg[mg["year"] == h]).mean())
        for h in years if not mg[mg["year"] == h].empty
    ]
    return float(np.mean(fold_losses)) if fold_losses else float("nan")


def _paired_bootstrap(pooled: pd.DataFrame, model_a: str, model_b: str,
                      n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Seeded paired bootstrap on (mean_loss_a - mean_loss_b). Negative gap
    means model_a beats model_b. Pairing is fixture-level via sorted
    alignment on (context, p_home, p_draw)."""
    a = pooled[pooled["model_id"] == model_a].sort_values(
        ["context", "p_home", "p_draw"]).reset_index(drop=True)
    b = pooled[pooled["model_id"] == model_b].sort_values(
        ["context", "p_home", "p_draw"]).reset_index(drop=True)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan")
    la = _row_loss(a.iloc[:n])
    lb = _row_loss(b.iloc[:n])
    gap = float(la.mean() - lb.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = la[idx].mean() - lb[idx].mean()
    return gap, float(samples.std(ddof=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", type=int, nargs="+", default=[2014, 2018, 2022])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--db", default="/tmp/dc_tw_tune.db")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    store = MatchStore()

    factories = {
        "dixon_coles": DixonColes,
        "dixon_coles_tournament_weighted": DixonColesTournamentWeighted,
    }
    print(f"Replaying 2 DC variants × {len(args.years)} WCs (~10–20 min)...")
    run_backtest(conn, store, factories, args.years, verbose=False)

    pooled = _pooled(conn)
    print(f"\nPooled scored claims: {len(pooled)} "
          f"({pooled['model_id'].nunique()} variants × "
          f"{pooled['year'].nunique()} WCs)\n")

    base_loto = _loto(pooled, "dixon_coles", args.years)
    variant_loto = _loto(pooled, "dixon_coles_tournament_weighted", args.years)
    gap, se = _paired_bootstrap(
        pooled, "dixon_coles_tournament_weighted", "dixon_coles",
        args.n_bootstrap, args.seed,
    )

    print(f"{'variant':>32}  {'LOTO log-loss':>14}  {'gap':>9}  {'± 1 SE':>9}")
    print("-" * 72)
    print(f"{'dixon_coles':>32}  {base_loto:>14.4f}  {'  base':>9}  {'':>9}")
    print(f"{'dixon_coles_tournament_weighted':>32}  {variant_loto:>14.4f}  "
          f"{gap:>+9.4f}  {se:>9.4f}")

    ship = (gap < 0.0) and (abs(gap) > se)
    print()
    if ship:
        print(f"SHIP: tournament weighting beats base by {abs(gap):.4f} "
              f"(> 1 SE = {se:.4f}). Add DixonColesTournamentWeighted to default_roster().")
        return 0
    print(f"DROP: gap {gap:+.4f} does not clear the +1 SE bar ({se:.4f}). "
          "Tournament weighting does not help on n=3 backtest. "
          "Leave DixonColesTournamentWeighted out of default_roster().")
    return 1


if __name__ == "__main__":
    sys.exit(main())
