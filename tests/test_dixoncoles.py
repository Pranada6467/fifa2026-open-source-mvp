"""Dixon-Coles wrapper invariants: normalized/clamped grid, neutral-venue
handling, the thin-window widening fallback, ET-score cleaning, and loud
failures (unfitted/unknown-team/unusable-data). Synthetic league data keeps
these fast; one real-data smoke test pins the wrapper to the actual pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.models import DixonColes

_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "neutral", "went_to_et"]


def _league(seed: int = 7, n_rounds: int = 8) -> pd.DataFrame:
    """Synthetic double round-robin: T1 strongest … T6 weakest, real home edge."""
    rng = np.random.default_rng(seed)
    strength = {f"T{i}": s for i, s in enumerate([1.9, 1.5, 1.2, 1.0, 0.8, 0.6], start=1)}
    rows, day = [], pd.Timestamp("2024-01-01")
    for _ in range(n_rounds):
        for home in strength:
            for away in strength:
                if home == away:
                    continue
                lam_h = 1.35 * strength[home] / strength[away]  # 1.35 = home boost
                lam_a = 1.00 * strength[away] / strength[home]
                rows.append((day, home, away, rng.poisson(lam_h), rng.poisson(lam_a),
                             "League", False, False))
                day += pd.Timedelta(days=3)
    return pd.DataFrame(rows, columns=_COLS)


@pytest.fixture(scope="module")
def fitted() -> DixonColes:
    return DixonColes(min_matches=100).fit(_league())


def test_grid_is_normalized_and_positive(fitted):
    grid = fitted.predict_goals("T1", "T4").grid
    assert grid.shape == (15, 15)  # max_goals=15 → scorelines 0..14 per side
    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_wdl_collapses_grid_and_ranks_strength(fitted):
    wdl = fitted.predict_wdl("T1", "T6", neutral=True)
    grid_wdl = fitted.predict_goals("T1", "T6", neutral=True).wdl()
    assert wdl.home == pytest.approx(grid_wdl.home)
    # The fitted strengths recover the construction: T1 >> T6.
    assert wdl.home > 0.5 > wdl.away
    reverse = fitted.predict_wdl("T6", "T1", neutral=True)
    assert reverse.away == pytest.approx(wdl.home, abs=1e-9)


def test_neutral_venue_removes_home_advantage(fitted):
    at_home = fitted.predict_wdl("T3", "T4", neutral=False)
    neutral = fitted.predict_wdl("T3", "T4", neutral=True)
    # The league was generated with a real home boost, so the fitted home
    # advantage is positive and must vanish on neutral ground.
    assert at_home.home > neutral.home
    assert at_home.away < neutral.away


def test_thin_window_widens_until_enough_data():
    dc = DixonColes(window_years=0.05, min_matches=100).fit(_league())
    # ~18 days holds nowhere near 100 matches: the fallback must have widened.
    assert dc.fitted_window_years is None or dc.fitted_window_years > 0.05
    assert dc.n_training_matches >= 100


def test_et_contaminated_scores_excluded_by_default():
    league = _league()
    # Poison pill: T1 "loses" a string of extra-time blowouts (120-minute
    # scorelines). With the V2 cleaning these rows carry zero weight.
    et_rows = pd.DataFrame(
        [(pd.Timestamp("2025-06-01") + pd.Timedelta(days=i), "T1", f"T{2 + i % 5}",
          0, 5, "Cup", False, True) for i in range(20)],
        columns=_COLS,
    )
    poisoned = pd.concat([league, et_rows], ignore_index=True)
    clean = DixonColes(min_matches=100, et_weight=0.0).fit(poisoned)
    dirty = DixonColes(min_matches=100, et_weight=1.0).fit(poisoned)
    matchup = ("T1", "T4")
    assert clean.predict_wdl(*matchup, neutral=True).home > \
           dirty.predict_wdl(*matchup, neutral=True).home


def test_no_usable_matches_raises_loudly():
    all_et = _league().assign(went_to_et=True)
    with pytest.raises(RuntimeError, match="fall back to Elo"):
        DixonColes(min_matches=100, et_weight=0.0).fit(all_et)


def test_batch_model_refuses_incremental_update(fitted):
    with pytest.raises(NotImplementedError):
        fitted.update({"date": "2026-01-01"})


def test_unfitted_and_unknown_team_raise(fitted):
    with pytest.raises(RuntimeError):
        DixonColes().predict_wdl("T1", "T2")
    with pytest.raises(KeyError):
        fitted.predict_wdl("T1", "Atlantis")


def test_real_history_smoke():
    """2-year window over the real as-of frame: converges, stays anchored to
    the frame's max date (not wall clock), and prices a WC2026 host edge."""
    store = MatchStore()
    dc = DixonColes(window_years=2.0).fit(store.played)
    assert dc.trained_through == store.played["date"].max()
    assert dc.fitted_window_years == 2.0
    home = dc.predict_wdl("Mexico", "South Africa", neutral=False)
    neutral = dc.predict_wdl("Mexico", "South Africa", neutral=True)
    assert home.home > neutral.home > neutral.away  # favourite + venue edge
    assert dc.predict_goals("Brazil", "Malta", neutral=True).grid.sum() == pytest.approx(1.0)
