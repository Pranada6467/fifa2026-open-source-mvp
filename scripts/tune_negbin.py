"""LOTO decision gate for NegBin vs base Dixon-Coles (S2 / D11-A).

Same shape as the DC tournament-weighted gate: replays WC2014/18/22 with
both variants through the existing backtest plumbing, computes LOTO pooled
log-loss + the seeded paired-bootstrap SE, ships NegBin only when the gap
in NB's favour exceeds 1 bootstrap SE.

Wall-time: each variant refits its own MLE per WC match day. Plain DC
takes ~5-10 min; NegBin is similar (penaltyblog's NB likelihood). Total
runtime ~10-20 min. The scratch DB at /tmp/negbin_tune.db is deleted +
rewritten each run.
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
from fifapreds.models import DixonColes, NegBin


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
    ap.add_argument("--db", default="/tmp/negbin_tune.db")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    store = MatchStore()

    factories = {"dixon_coles": DixonColes, "neg_bin": NegBin}
    print(f"Replaying DC vs NegBin × {len(args.years)} WCs (~10–20 min)...", flush=True)
    run_backtest(conn, store, factories, args.years, verbose=False)

    pooled = _pooled(conn)
    print(f"\nPooled scored claims: {len(pooled)} "
          f"({pooled['model_id'].nunique()} models × "
          f"{pooled['year'].nunique()} WCs)\n", flush=True)

    base_loto = _loto(pooled, "dixon_coles", args.years)
    variant_loto = _loto(pooled, "neg_bin", args.years)
    gap, se = _paired_bootstrap(pooled, "neg_bin", "dixon_coles",
                                args.n_bootstrap, args.seed)

    print(f"{'variant':>14}  {'LOTO log-loss':>14}  {'gap':>9}  {'± 1 SE':>9}")
    print("-" * 54)
    print(f"{'dixon_coles':>14}  {base_loto:>14.4f}  {'  base':>9}  {'':>9}")
    print(f"{'neg_bin':>14}  {variant_loto:>14.4f}  {gap:>+9.4f}  {se:>9.4f}")

    ship = (gap < 0.0) and (abs(gap) > se)
    print()
    if ship:
        print(f"SHIP: NegBin beats DC by {abs(gap):.4f} (> 1 SE = {se:.4f}). "
              "Add NegBin() to default_roster().")
        return 0
    print(f"DROP: gap {gap:+.4f} does not clear the +1 SE bar ({se:.4f}). "
          "Tail-fattening alone doesn't help on the n=3 backtest. "
          "Leave NegBin out of default_roster().")
    return 1


if __name__ == "__main__":
    sys.exit(main())
