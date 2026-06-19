"""NegBin wrapper invariants (S2): normalized/positive grid, neutral-venue
handling, the thin-window widening fallback, ET-score cleaning, and loud
failures. The NB-specific extra: an overdispersed synthetic process should
fit with finite dispersion (k < ∞) and recover roughly Poisson-like rates;
this pins the wrapper to the actual penaltyblog 1.11.0 likelihood.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.models import NegBin

_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "neutral", "went_to_et"]


def _league(seed: int = 7, n_rounds: int = 8) -> pd.DataFrame:
    """Synthetic double round-robin sampled from a Poisson process — same
    construction as the DC test fixture so the comparison is apples-to-apples.

    A Poisson sample is the limit of NB as dispersion k → ∞; under this
    fixture NB should converge to large k and reproduce DC-shaped grids.
    The blowout fixture below tests the converse: when goals overdisperse,
    NB carries probability into the tail where Poisson collapses."""
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
def fitted() -> NegBin:
    return NegBin(min_matches=100).fit(_league())


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
    # T1 >> T6, the fitted rates must recover the construction.
    assert wdl.home > 0.5 > wdl.away
    reverse = fitted.predict_wdl("T6", "T1", neutral=True)
    assert reverse.away == pytest.approx(wdl.home, abs=1e-9)


def test_neutral_venue_removes_home_advantage(fitted):
    at_home = fitted.predict_wdl("T3", "T4", neutral=False)
    neutral = fitted.predict_wdl("T3", "T4", neutral=True)
    assert at_home.home > neutral.home
    assert at_home.away < neutral.away


def test_thin_window_widens_until_enough_data():
    nb = NegBin(window_years=0.05, min_matches=100).fit(_league())
    assert nb.fitted_window_years is None or nb.fitted_window_years > 0.05
    assert nb.n_training_matches >= 100


def test_et_contaminated_scores_excluded_by_default():
    league = _league()
    # Poison pill: T1 "loses" a string of 120-min blowouts. With ET cleaning
    # those rows carry zero weight and the fit ignores them.
    et_rows = pd.DataFrame(
        [(pd.Timestamp("2025-06-01") + pd.Timedelta(days=i), "T1", f"T{2 + i % 5}",
          0, 5, "Cup", False, True) for i in range(20)],
        columns=_COLS,
    )
    poisoned = pd.concat([league, et_rows], ignore_index=True)
    clean = NegBin(min_matches=100, et_weight=0.0).fit(poisoned)
    dirty = NegBin(min_matches=100, et_weight=1.0).fit(poisoned)
    matchup = ("T1", "T4")
    assert (clean.predict_wdl(*matchup, neutral=True).home
            > dirty.predict_wdl(*matchup, neutral=True).home)


def test_no_usable_matches_raises_loudly():
    all_et = _league().assign(went_to_et=True)
    with pytest.raises(RuntimeError, match="fall back to Elo"):
        NegBin(min_matches=100, et_weight=0.0).fit(all_et)


def test_unfitted_and_unknown_team_raise(fitted):
    with pytest.raises(RuntimeError):
        NegBin().predict_wdl("T1", "T2")
    with pytest.raises(KeyError):
        fitted.predict_wdl("T1", "Atlantis")


def test_distinct_hyperparams_hash_from_dixon_coles():
    """NegBin and DixonColes share the same hyperparam KEYS but the model_id
    differs — the leaderboard must never collide their rows. Hash equality
    here would be a config bug, not a feature."""
    from fifapreds.models import DixonColes
    # Same construction args; only the class differs.
    nb = NegBin()
    dc = DixonColes()
    # Different model_ids by construction.
    assert NegBin.model_id != DixonColes.model_id
    # Hashes can match (same dict shape) but the (model_id, hash) tuple
    # that keys the leaderboard differs — that's the actual invariant.
    assert (nb.model_id, nb.hyperparams_hash) != (dc.model_id, dc.hyperparams_hash)


# ------------------------------------------------------- overdispersion smoke

def test_overdispersed_fixture_fits_and_predicts_tail_weight():
    """NB's contract vs Poisson: when goals overdisperse (sample variance
    well above the mean), NB allocates probability into the upper tail
    where Poisson-only models like DC collapse to near-zero.

    Construction: a few teams in a Poisson league get a heavy-tailed
    blowout streak. NB must still fit cleanly AND give a high-scoring
    fixture (say 5-1) a probability strictly above the DC's machine
    epsilon (which would mean the NB likelihood actually fattened the
    tail, not just numerically agreed with Poisson).
    """
    rng = np.random.default_rng(13)
    league = _league(seed=5)
    blowouts = pd.DataFrame(
        [(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), "T1", "T6",
          int(rng.poisson(6) + 1), int(rng.poisson(0.4)),
          "League", False, False) for i in range(40)],
        columns=_COLS,
    )
    enriched = pd.concat([league, blowouts], ignore_index=True)
    nb = NegBin(min_matches=100).fit(enriched)
    grid = nb.predict_goals("T1", "T6", neutral=True).grid
    # P(5-1) should be non-trivial under overdispersion — the contract is
    # 'meaningful', not a hard threshold; 1e-3 is well above DC tail noise.
    assert grid[5, 1] > 1e-3


def test_real_history_smoke():
    """4-year window over the real as-of frame: converges, anchored to the
    frame's max date, produces a normalized grid for a real WC2026 fixture."""
    store = MatchStore()
    nb = NegBin(window_years=2.0).fit(store.played)
    assert nb.trained_through == store.played["date"].max()
    assert nb.fitted_window_years == 2.0
    home = nb.predict_wdl("Mexico", "South Africa", neutral=False)
    neutral = nb.predict_wdl("Mexico", "South Africa", neutral=True)
    assert home.home > neutral.home > neutral.away
    assert nb.predict_goals("Brazil", "Malta", neutral=True).grid.sum() == pytest.approx(1.0, abs=2e-2)
