"""E4 — bootstrap bands + verdict badges: a clear gap reads "behind", an
identical model reads "tied", the leader reads "best"; seeded reproducibility;
empty input degrades to an empty frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fifapreds.leaderboard import BAND_COLS, bootstrap_bands


def _scored(model_id: str, rps: np.ndarray, context="backtest:wc") -> pd.DataFrame:
    n = len(rps)
    return pd.DataFrame({
        "context": [context] * n,
        "match_id": np.arange(n),
        "model_id": [model_id] * n,
        "rps": rps,
        "log_loss": rps + 0.8,
    })


def test_badges_best_behind_tied():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.25, 200)
    scored = pd.concat([
        _scored("alpha", base),                 # the leader
        _scored("worse", base + 0.15),          # uniformly worse, same matches
        _scored("clone", base.copy()),          # statistically identical
    ], ignore_index=True)
    bands = bootstrap_bands(scored, seed=0)
    badge = bands.set_index("model_id")["badge"]
    # alpha and clone have identical pooled RPS; either may take "best" —
    # the other must be "tied", never "behind".
    assert {badge["alpha"], badge["clone"]} == {"best", "tied"}
    assert badge["worse"] == "behind"
    row = bands.set_index("model_id").loc["worse"]
    assert row["rps_lo"] <= row["rps"] <= row["rps_hi"]
    assert row["n"] == 200 and row["track"] == "backtest"


def test_bands_are_seeded_and_reproducible():
    rng = np.random.default_rng(1)
    scored = pd.concat([
        _scored("a", rng.uniform(0.1, 0.3, 50)),
        _scored("b", rng.uniform(0.1, 0.3, 50)),
    ], ignore_index=True)
    pd.testing.assert_frame_equal(bootstrap_bands(scored, seed=7),
                                  bootstrap_bands(scored, seed=7))


def test_tracks_are_separated():
    rng = np.random.default_rng(2)
    scored = pd.concat([
        _scored("a", rng.uniform(0.1, 0.3, 40), context="backtest:wc"),
        _scored("a", rng.uniform(0.1, 0.3, 8), context="live"),
    ], ignore_index=True)
    bands = bootstrap_bands(scored)
    assert sorted(bands["track"]) == ["backtest", "live"]
    # Each track's leader is badged independently.
    assert (bands["badge"] == "best").sum() == 2


def test_empty_scored_gives_empty_frame():
    out = bootstrap_bands(pd.DataFrame())
    assert list(out.columns) == BAND_COLS and out.empty
