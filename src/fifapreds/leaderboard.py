"""E4 — leaderboard uncertainty: seeded paired bootstrap over graded claims.

Answers the question the point estimates can't: is dixon_coles beating
elo_baseline, or is the gap noise? Per track (backtest | live):

- each model gets a percentile CI on its mean RPS and log-loss, from
  resampling ITS OWN graded matches with replacement (seeded — reproducible);
- each model is compared to the track leader by a PAIRED bootstrap on the
  matches both graded (same resampled matches for both sides, so match
  difficulty cancels), yielding a verdict badge:
      "best"   — the lowest pooled RPS in the track;
      "tied"   — the paired CI of (model − best) RPS contains 0;
      "behind" — it doesn't.

Badges were deliberately deferred from E2 (design decision D7) until this
machinery existed — a "tied" claimed from an arbitrary threshold would be
exactly the fake certainty this project exists to avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fifapreds.publish.board import track_of

N_BOOT = 1000
ALPHA = 0.05
BOOTSTRAP_SEED = 0   # fixed: the bands move when the data moves, not daily

BAND_COLS = ["model_id", "track", "n", "rps", "rps_lo", "rps_hi",
             "log_loss", "log_loss_lo", "log_loss_hi", "badge"]


def _boot_ci(values: np.ndarray, rng: np.random.Generator,
             n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile CI on the mean of `values` under resampling."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, ALPHA / 2)),
            float(np.quantile(means, 1 - ALPHA / 2)))


def bootstrap_bands(scored: pd.DataFrame, *, n_boot: int = N_BOOT,
                    seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Per (model, track) bands + verdict badges from a graded-claims frame
    (needs context, match_id, model_id, rps, log_loss)."""
    if scored is None or scored.empty:
        return pd.DataFrame(columns=BAND_COLS)
    df = scored.dropna(subset=["match_id"]).copy()
    df["track"] = df["context"].map(track_of)
    df["key"] = list(zip(df["context"], df["match_id"]))
    rng = np.random.default_rng(seed)

    rows = []
    for track, tdf in df.groupby("track"):
        pooled = tdf.groupby("model_id")[["rps", "log_loss"]].mean()
        best_id = pooled["rps"].idxmin()
        best_claims = tdf[tdf["model_id"] == best_id].set_index("key")["rps"]
        for model_id, mdf in tdf.groupby("model_id"):
            rps_vals = mdf["rps"].to_numpy()
            ll_vals = mdf["log_loss"].to_numpy()
            rps_lo, rps_hi = _boot_ci(rps_vals, rng, n_boot)
            ll_lo, ll_hi = _boot_ci(ll_vals, rng, n_boot)
            if model_id == best_id:
                badge = "best"
            else:
                mine = mdf.set_index("key")["rps"]
                shared = mine.index.intersection(best_claims.index)
                if len(shared) < 2:
                    badge = "tied"   # nothing in common to test on — no verdict
                else:
                    delta = (mine.loc[shared] - best_claims.loc[shared]).to_numpy()
                    d_lo, d_hi = _boot_ci(delta, rng, n_boot)
                    badge = "tied" if d_lo <= 0.0 <= d_hi else "behind"
            rows.append({
                "model_id": model_id, "track": track, "n": len(mdf),
                "rps": float(rps_vals.mean()),
                "rps_lo": rps_lo, "rps_hi": rps_hi,
                "log_loss": float(ll_vals.mean()),
                "log_loss_lo": ll_lo, "log_loss_hi": ll_hi,
                "badge": badge,
            })
    return (pd.DataFrame(rows, columns=BAND_COLS)
            .sort_values(["track", "rps"], ignore_index=True))
