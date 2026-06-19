"""Combined Phase 2 LOTO gate: DC vs NegBin vs BivariatePoisson.

Replays all three goals-model variants in a single backtest pass over
WC2014/18/22, then evaluates each variant against base DC using the same
> 1 SE bootstrap gate as the Phase 1 scripts. Runs the three fits in
sequence per match day; total wall-time ~15-25 min.

Per D11-B the wrappers live in separate files (no `_mle.py` shared base).
This script also lets us measure the actual code overlap in practice
before any refactor — if NegBin and BivariatePoisson both ship, the
follow-up evaluation has a real artifact (their scratch fits side by side)
to ground the abstraction decision.
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
from fifapreds.models import BivariatePoisson, DixonColes, NegBin


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
    ap.add_argument("--db", default="/tmp/phase2_tune.db")
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    store = MatchStore()

    factories = {
        "dixon_coles": DixonColes,
        "neg_bin": NegBin,
        "bivariate_poisson": BivariatePoisson,
    }
    print(f"Replaying 3 goals variants × {len(args.years)} WCs "
          f"(~15-25 min)...", flush=True)
    run_backtest(conn, store, factories, args.years, verbose=False)

    pooled = _pooled(conn)
    print(f"\nPooled scored claims: {len(pooled)} "
          f"({pooled['model_id'].nunique()} models × "
          f"{pooled['year'].nunique()} WCs)\n", flush=True)

    base_loto = _loto(pooled, "dixon_coles", args.years)
    print(f"{'variant':>20}  {'LOTO log-loss':>14}  {'gap vs DC':>10}  "
          f"{'± 1 SE':>9}  verdict")
    print("-" * 75)
    print(f"{'dixon_coles':>20}  {base_loto:>14.4f}  {'  base':>10}  "
          f"{'':>9}  reference")

    results = []
    for variant in ("neg_bin", "bivariate_poisson"):
        ll = _loto(pooled, variant, args.years)
        gap, se = _paired_bootstrap(pooled, variant, "dixon_coles",
                                    args.n_bootstrap, args.seed)
        ship = (gap < 0.0) and (abs(gap) > se)
        verdict = "SHIP" if ship else "drop"
        print(f"{variant:>20}  {ll:>14.4f}  {gap:>+10.4f}  {se:>9.4f}  {verdict}")
        results.append({"variant": variant, "log_loss": ll, "gap": gap,
                        "se": se, "ship": ship})

    print()
    shippable = [r for r in results if r["ship"]]
    if shippable:
        print(f"SHIP candidates ({len(shippable)}):")
        for r in shippable:
            print(f"  - {r['variant']}: beats DC by {abs(r['gap']):.4f} "
                  f"(> 1 SE = {r['se']:.4f})")
        return 0
    print("NO variant beats DC by > 1 bootstrap SE on n=3 backtest.")
    print("→ Leave both NegBin and BivariatePoisson out of default_roster().")
    print("  Mechanism + tests still land; gate decision is documented in docs/tuning/.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
