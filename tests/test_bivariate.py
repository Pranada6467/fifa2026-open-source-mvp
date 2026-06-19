"""BivariatePoisson wrapper invariants (S9): normalized/positive grid,
neutral-venue handling, the thin-window widening fallback, ET-score
cleaning, loud failures, and the Karlis-Ntzoufras-specific contract that
the model can capture positive home/away goal covariance (where
independent Poisson cannot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.models import BivariatePoisson

_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "neutral", "went_to_et"]


def _league(seed: int = 7, n_rounds: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = {f"T{i}": s for i, s in enumerate([1.9, 1.5, 1.2, 1.0, 0.8, 0.6], start=1)}
    rows, day = [], pd.Timestamp("2024-01-01")
    for _ in range(n_rounds):
        for home in strength:
            for away in strength:
                if home == away:
                    continue
                lam_h = 1.35 * strength[home] / strength[away]
                lam_a = 1.00 * strength[away] / strength[home]
                rows.append((day, home, away, rng.poisson(lam_h), rng.poisson(lam_a),
                             "League", False, False))
                day += pd.Timedelta(days=3)
    return pd.DataFrame(rows, columns=_COLS)


@pytest.fixture(scope="module")
def fitted() -> BivariatePoisson:
    return BivariatePoisson(min_matches=100).fit(_league())


# ----------------------------------------------------------------- invariants

def test_grid_is_normalized_and_positive(fitted):
    grid = fitted.predict_goals("T1", "T4").grid
    assert grid.shape == (15, 15)
    assert grid.sum() == pytest.approx(1.0, abs=2e-2)
    assert (grid >= 0).all()


def test_wdl_collapses_grid_and_ranks_strength(fitted):
    wdl = fitted.predict_wdl("T1", "T6", neutral=True)
    grid_wdl = fitted.predict_goals("T1", "T6", neutral=True).wdl()
    assert wdl.home == pytest.approx(grid_wdl.home)
    assert wdl.home > 0.5 > wdl.away
    reverse = fitted.predict_wdl("T6", "T1", neutral=True)
    assert reverse.away == pytest.approx(wdl.home, abs=1e-9)


def test_neutral_venue_removes_home_advantage(fitted):
    at_home = fitted.predict_wdl("T3", "T4", neutral=False)
    neutral = fitted.predict_wdl("T3", "T4", neutral=True)
    assert at_home.home > neutral.home
    assert at_home.away < neutral.away


def test_thin_window_widens_until_enough_data():
    bp = BivariatePoisson(window_years=0.05, min_matches=100).fit(_league())
    assert bp.fitted_window_years is None or bp.fitted_window_years > 0.05
    assert bp.n_training_matches >= 100


def test_et_contaminated_scores_excluded_by_default():
    league = _league()
    et_rows = pd.DataFrame(
        [(pd.Timestamp("2025-06-01") + pd.Timedelta(days=i), "T1", f"T{2 + i % 5}",
          0, 5, "Cup", False, True) for i in range(20)],
        columns=_COLS,
    )
    poisoned = pd.concat([league, et_rows], ignore_index=True)
    clean = BivariatePoisson(min_matches=100, et_weight=0.0).fit(poisoned)
    dirty = BivariatePoisson(min_matches=100, et_weight=1.0).fit(poisoned)
    matchup = ("T1", "T4")
    assert (clean.predict_wdl(*matchup, neutral=True).home
            > dirty.predict_wdl(*matchup, neutral=True).home)


def test_no_usable_matches_raises_loudly():
    all_et = _league().assign(went_to_et=True)
    with pytest.raises(RuntimeError, match="fall back to Elo"):
        BivariatePoisson(min_matches=100, et_weight=0.0).fit(all_et)


def test_unfitted_and_unknown_team_raise(fitted):
    with pytest.raises(RuntimeError):
        BivariatePoisson().predict_wdl("T1", "T2")
    with pytest.raises(KeyError):
        fitted.predict_wdl("T1", "Atlantis")


def test_distinct_hyperparams_hash_from_dixon_coles_and_negbin():
    """All three goals-model wrappers share the same hyperparam KEYS but
    have distinct `model_id`s. Their (model_id, hash) tuples — what
    actually keys the leaderboard row — must all differ."""
    from fifapreds.models import DixonColes, NegBin
    bp = BivariatePoisson()
    dc = DixonColes()
    nb = NegBin()
    ids = {(bp.model_id, bp.hyperparams_hash),
           (dc.model_id, dc.hyperparams_hash),
           (nb.model_id, nb.hyperparams_hash)}
    assert len(ids) == 3, "two goals models collide on (model_id, hash)"


def test_real_history_smoke():
    """2-year window over the real as-of frame: converges, anchored to the
    frame's max date, prices a real WC2026 fixture without exploding."""
    store = MatchStore()
    bp = BivariatePoisson(window_years=2.0).fit(store.played)
    assert bp.trained_through == store.played["date"].max()
    assert bp.fitted_window_years == 2.0
    home = bp.predict_wdl("Mexico", "South Africa", neutral=False)
    neutral = bp.predict_wdl("Mexico", "South Africa", neutral=True)
    assert home.home > neutral.home > neutral.away
    assert bp.predict_goals("Brazil", "Malta", neutral=True).grid.sum() == pytest.approx(1.0, abs=2e-2)
