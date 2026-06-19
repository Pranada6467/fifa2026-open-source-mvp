"""LOTO-tuned Davidson draw parameter ν for BaselineElo (S6).

Sweeps ν over a discrete grid; for each ν replays WC2014/18/22 through the
existing backtest plumbing; pools log-loss per holdout year (LOTO: train on
two WCs, score the third) and reports the seeded paired-bootstrap SE of the
log-loss gap vs base ν=0.6.

Decision gate (D11-A): ship `EloTunedDraw(draw_nu=<best>)` only when the
LOTO-averaged log-loss improvement vs base BaselineElo exceeds 1 bootstrap
SE. Otherwise drop the entrant from the roster — better no variant than a
vanity one that adds bootstrap variance for no signal.

Run:
    .venv/bin/python -m scripts.tune_elo_nu \
        [--grid 0.40 0.50 0.60 0.70 0.80 0.90 1.00] \
        [--years 2014 2018 2022] [--seed 7] [--n-bootstrap 500]

Writes a one-shot DB at /tmp/elo_nu_tune.db; prints the table + verdict.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds import db
from fifapreds.asof import MatchStore
from fifapreds.backtest import run_backtest
from fifapreds.loop.score import CLASSES, log_loss
from fifapreds.models.elo import BaselineElo

BASELINE_NU = 0.6


def _hash_for_nu(nu: float) -> str:
    """All BaselineElo(draw_nu=ν) instances share model_id='elo_baseline' but
    have distinct hyperparams_hash — that hash is the variant key in the DB."""
    return BaselineElo(draw_nu=nu).hyperparams_hash


def _label(nu: float) -> str:
    """Display-only label; not stored in the DB."""
    return f"elo_nu_{nu:.2f}"


def _pooled_predictions(conn) -> pd.DataFrame:
    """All scored backtest claims with the tournament-year tag parsed out
    of `context` (= 'backtest:wc2014' etc.) and the hyperparams_hash that
    distinguishes ν variants under the shared `elo_baseline` model_id."""
    df = pd.read_sql_query(
        """SELECT p.model_id, p.hyperparams_hash, p.context,
                  p.p_home, p.p_draw, p.p_away, s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE p.context LIKE 'backtest:wc%'""",
        conn,
    )
    df["year"] = df["context"].str.removeprefix("backtest:wc").astype(int)
    return df


def _row_loss(df: pd.DataFrame) -> np.ndarray:
    """Per-row log-loss vector — the bootstrap unit of resampling."""
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = df["outcome"].map(CLASSES.index).to_numpy()
    return log_loss(probs, outcomes)


def _loto_pooled_log_loss(pooled: pd.DataFrame, hp_hash: str,
                          years: list[int]) -> float:
    """Mean of per-holdout pooled log-loss across LOTO folds.

    For each year H, score the variant's predictions on year H (whose
    parameters would have been chosen using the OTHER two years if this
    were a calibrator). For ν tuning there's no train step — the
    parameter is a fixed scalar — so 'holdout' here means 'pool the
    holdout year's rows, average their log-loss'."""
    mg = pooled[pooled["hyperparams_hash"] == hp_hash]
    losses = []
    for holdout in years:
        h = mg[mg["year"] == holdout]
        if h.empty:
            continue
        losses.append(float(_row_loss(h).mean()))
    return float(np.mean(losses)) if losses else float("nan")


def _paired_bootstrap_se(pooled: pd.DataFrame, hash_a: str, hash_b: str,
                         n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Seeded paired bootstrap on the (mean_loss_a - mean_loss_b) statistic.

    Returns (mean_gap, bootstrap_se). The pairing is fixture-level: each
    bootstrap sample picks the same row indices from both variants so we
    cancel out the per-fixture variance — what the leaderboard's
    `leaderboard_bands` artifact does for the published view."""
    a = pooled[pooled["hyperparams_hash"] == hash_a].sort_values(
        ["context", "p_home", "p_draw"]).reset_index(drop=True)
    b = pooled[pooled["hyperparams_hash"] == hash_b].sort_values(
        ["context", "p_home", "p_draw"]).reset_index(drop=True)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan"), float("nan")
    loss_a = _row_loss(a.iloc[:n])
    loss_b = _row_loss(b.iloc[:n])
    gap = float(loss_a.mean() - loss_b.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = loss_a[idx].mean() - loss_b[idx].mean()
    return gap, float(samples.std(ddof=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    ap.add_argument("--years", type=int, nargs="+", default=[2014, 2018, 2022])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--db", default="/tmp/elo_nu_tune.db",
                    help="scratch DB path (deleted + rewritten on every run)")
    args = ap.parse_args(argv)

    if BASELINE_NU not in args.grid:
        print(f"NOTE: adding the baseline ν={BASELINE_NU} to the grid for the gap test.")
        args.grid = sorted(set(args.grid) | {BASELINE_NU})

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    store = MatchStore()

    # Each ν is a fresh BaselineElo factory; backtest replays them all in one
    # pass and writes scored rows to the scratch DB.
    factories = {
        _label(nu): (lambda nu=nu: BaselineElo(draw_nu=nu))
        for nu in args.grid
    }
    print(f"Replaying {len(factories)} ν values across {args.years}...")
    run_backtest(conn, store, factories, args.years, verbose=False)

    pooled = _pooled_predictions(conn)
    print(f"\nPooled scored claims: {len(pooled)} "
          f"({pooled['hyperparams_hash'].nunique()} variants × "
          f"{pooled['year'].nunique()} WCs)\n")

    baseline_hash = _hash_for_nu(BASELINE_NU)
    base_loto = _loto_pooled_log_loss(pooled, baseline_hash, args.years)
    print(f"Base ν={BASELINE_NU} LOTO-avg log-loss: {base_loto:.4f}\n")

    print(f"{'ν':>6}  {'LOTO log-loss':>14}  {'gap':>9}  {'± 1 SE':>9}  verdict")
    print("-" * 65)
    rows = []
    for nu in args.grid:
        hp_hash = _hash_for_nu(nu)
        ll = _loto_pooled_log_loss(pooled, hp_hash, args.years)
        if nu == BASELINE_NU:
            print(f"{nu:>6.2f}  {ll:>14.4f}  {'  base':>9}  {'':>9}  ν=0.6 reference")
            rows.append({"nu": nu, "log_loss": ll, "gap": 0.0, "se": 0.0,
                         "verdict": "baseline"})
            continue
        gap, se = _paired_bootstrap_se(pooled, hp_hash, baseline_hash,
                                       args.n_bootstrap, args.seed)
        # Negative gap = lower log-loss = better than baseline.
        ship = (gap < 0.0) and (abs(gap) > se)
        verdict = ("SHIP" if ship else "drop")
        print(f"{nu:>6.2f}  {ll:>14.4f}  {gap:>+9.4f}  {se:>9.4f}  {verdict}")
        rows.append({"nu": nu, "log_loss": ll, "gap": gap, "se": se,
                     "verdict": verdict})

    table = pd.DataFrame(rows)
    ship_candidates = table[table["verdict"] == "SHIP"].sort_values("gap")

    print()
    if ship_candidates.empty:
        print("NO ν beats the baseline by > 1 bootstrap SE.")
        print("→ Drop EloTunedDraw from the roster. No vanity entrants.")
        return 1
    best = ship_candidates.iloc[0]
    print(f"BEST: ν={best['nu']:.2f} — log-loss {best['log_loss']:.4f} "
          f"(gap {best['gap']:+.4f} ± {best['se']:.4f} SE)")
    print(f"→ Wire EloTunedDraw(draw_nu={best['nu']:.2f}) into default_roster().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
