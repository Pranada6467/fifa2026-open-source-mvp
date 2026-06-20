"""Train + ship the StackedEnsemble weights (S19 / D11-F).

Trains a multinomial L2-regularized logistic regression on the per-model
WDL probability vectors from `data/backtest.db`, with LOTO discipline:
report the held-out log-loss per fold, ship the FINAL coefficients
(fit on every available year), and dump everything to
`data/stacking_weights.json` for `StackedEnsemble` to load at runtime.

Decision gate per D11-A: ship the weights JSON only when LOTO log-loss
beats base `dixon_coles` by > 1 bootstrap SE. Otherwise the script exits
1 and the existing weights file stays whatever it was (no silent
overwrite of a passing artifact with a failing one).

Refresh cadence per D11-F: run after each completed tournament; commit
the new JSON via a versioned PR. During WC2026 the weights stay frozen
(see DD4 — divergence banner surfaces any drift instead).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fifapreds import db
from fifapreds.config import PROJECT_ROOT
from fifapreds.loop.score import CLASSES, log_loss

DEFAULT_BACKTEST = PROJECT_ROOT / "data" / "backtest.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stacking_weights.json"
_PROB_COLS = ("p_home", "p_draw", "p_away")
WEIGHTS_VERSION = "1"


def _wide(conn: sqlite3.Connection) -> pd.DataFrame:
    """Pivot scored backtest rows to (match_id × model_id) → (p_home, p_draw,
    p_away) features. One row per fixture × model combination is collapsed
    into a single fixture-row with model-suffixed feature columns."""
    raw = pd.read_sql_query(
        """SELECT p.context, p.match_id, p.model_id,
                  p.p_home, p.p_draw, p.p_away, s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE p.context LIKE 'backtest:wc%'""",
        conn,
    )
    raw["year"] = raw["context"].str.removeprefix("backtest:wc").astype(int)
    # Outcome is consistent per match_id (same fixture, same result).
    outcomes = raw.groupby("match_id", as_index=False)["outcome"].first()
    pivot = raw.pivot_table(
        index="match_id", columns="model_id",
        values=list(_PROB_COLS), aggfunc="first",
    )
    pivot.columns = [f"{model}__{col}" for col, model in pivot.columns]
    pivot = pivot.reset_index().merge(outcomes, on="match_id")
    pivot = pivot.merge(
        raw[["match_id", "year"]].drop_duplicates("match_id"),
        on="match_id",
    )
    return pivot.dropna()


def _features(wide: pd.DataFrame, model_ids: list[str]) -> np.ndarray:
    cols = [f"{mid}__{p}" for mid in model_ids for p in _PROB_COLS]
    return wide[cols].to_numpy()


def _outcomes(wide: pd.DataFrame) -> np.ndarray:
    return wide["outcome"].map(CLASSES.index).to_numpy()


def _fit_logistic(features: np.ndarray, outcomes: np.ndarray,
                  C: float) -> LogisticRegression:
    # sklearn ≥ 1.5 deprecated `multi_class`; multinomial is the default
    # for 3+ classes via LBFGS. Pinned random_state keeps the fit
    # bit-stable across nightlies.
    clf = LogisticRegression(
        solver="lbfgs", C=C, max_iter=1000, random_state=0,
    )
    clf.fit(features, outcomes)
    return clf


def _paired_bootstrap_se(losses_a: np.ndarray, losses_b: np.ndarray,
                         n_bootstrap: int, seed: int) -> tuple[float, float]:
    gap = float(losses_a.mean() - losses_b.mean())
    rng = np.random.default_rng(seed)
    n = len(losses_a)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = losses_a[idx].mean() - losses_b[idx].mean()
    return gap, float(samples.std(ddof=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loto", action="store_true", default=True,
                    help="(default) report per-fold LOTO log-loss")
    ap.add_argument("--C", type=float, default=0.5,
                    help="L2 strength: smaller = stronger regularization")
    ap.add_argument("--backtest-db", default=str(DEFAULT_BACKTEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    args = ap.parse_args(argv)

    conn = db.connect(Path(args.backtest_db))
    wide = _wide(conn)
    if wide.empty:
        print("ERROR: backtest DB has no scored predictions.", file=sys.stderr)
        return 2
    model_ids = sorted({c.split("__", 1)[0] for c in wide.columns
                        if c.endswith("__p_home")})
    print(f"Stacking over {len(model_ids)} base models: {model_ids}")
    print(f"Pooled scored fixtures: {len(wide)}")
    print(f"LOTO years: {sorted(wide['year'].unique())}")
    print()

    years = sorted(wide["year"].unique())
    if len(years) < 2:
        print(f"ERROR: LOTO needs >= 2 years, got {years}.", file=sys.stderr)
        return 2

    # LOTO: fit on every year EXCEPT the holdout, predict on the holdout.
    # Per-fold held-out log-loss for the gate; per-fold dc log-loss for the
    # paired bootstrap comparison.
    stack_losses_per_fold, dc_losses_per_fold = [], []
    for holdout in years:
        train = wide[wide["year"] != holdout]
        test = wide[wide["year"] == holdout]
        if "dixon_coles" not in model_ids:
            print(f"ERROR: dixon_coles missing from base models; can't gate.",
                  file=sys.stderr)
            return 2
        clf = _fit_logistic(_features(train, model_ids), _outcomes(train), args.C)
        probs = clf.predict_proba(_features(test, model_ids))
        outcomes = _outcomes(test)
        stack_losses_per_fold.append(log_loss(probs, outcomes))
        dc_cols = [f"dixon_coles__{p}" for p in _PROB_COLS]
        dc_probs = test[dc_cols].to_numpy()
        dc_losses_per_fold.append(log_loss(dc_probs, outcomes))
        print(f"  holdout wc{holdout}: stack {stack_losses_per_fold[-1].mean():.4f}  "
              f"dc {dc_losses_per_fold[-1].mean():.4f}  n={len(test)}")

    stack_all = np.concatenate(stack_losses_per_fold)
    dc_all = np.concatenate(dc_losses_per_fold)
    stack_mean = float(stack_all.mean())
    dc_mean = float(dc_all.mean())
    gap, se = _paired_bootstrap_se(stack_all, dc_all,
                                   args.n_bootstrap, args.seed)
    ship = (gap < 0.0) and (abs(gap) > se)
    print()
    print(f"LOTO stack log-loss: {stack_mean:.4f}")
    print(f"LOTO dc    log-loss: {dc_mean:.4f}")
    print(f"gap stack-dc:        {gap:+.4f}  ± {se:.4f} SE")
    print()

    if not ship:
        print(f"DROP: stacking gap {gap:+.4f} does not clear the +1 SE bar "
              f"({se:.4f}). Leaving {args.output} unchanged.")
        return 1

    # Final fit on ALL years — the shipped artifact.
    final = _fit_logistic(_features(wide, model_ids), _outcomes(wide), args.C)
    payload = {
        "weights_version": WEIGHTS_VERSION,
        "model_ids": list(model_ids),
        "coefficients": final.coef_.tolist(),
        "intercepts": final.intercept_.tolist(),
        "C": args.C,
        "holdout_years": years,
        "trained_on": years,
        "loto_log_loss": stack_mean,
        "dc_log_loss": dc_mean,
        "gap_vs_dc": gap,
        "gap_se": se,
        "fit_date": str(date.today()),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"SHIP: wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
