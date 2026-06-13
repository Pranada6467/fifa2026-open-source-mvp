"""Hierarchical Poisson invariants: determinism, normalized grids,
confederation pooling shrinkage, neutral-venue handling, and the standard
failure modes. Synthetic league data keeps sampling fast (~2s per fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fifapreds.models.hierarchical import HierarchicalPoisson

_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
         "tournament", "neutral", "went_to_et"]

_CONFED_MAP = {
    "T1": "CONF_A", "T2": "CONF_A", "T3": "CONF_A",
    "T4": "CONF_B", "T5": "CONF_B", "T6": "CONF_B",
}


def _league(seed: int = 7, n_rounds: int = 8) -> pd.DataFrame:
    """Synthetic double round-robin: T1 strongest, T6 weakest, real home edge."""
    rng = np.random.default_rng(seed)
    strength = {"T1": 1.9, "T2": 1.5, "T3": 1.2, "T4": 1.0, "T5": 0.8, "T6": 0.6}
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


def _make_model(**overrides) -> HierarchicalPoisson:
    defaults = dict(
        n_samples=200, n_chains=1, min_matches=50,
        random_seed=42, confed_map=_CONFED_MAP,
    )
    defaults.update(overrides)
    return HierarchicalPoisson(**defaults)


@pytest.fixture(scope="module")
def fitted() -> HierarchicalPoisson:
    return _make_model().fit(_league())


def test_determinism():
    """Same seed + data → identical grids (eng review T1 mandate)."""
    data = _league()
    grids = []
    for _ in range(2):
        hp = _make_model()
        hp.fit(data)
        grids.append(hp.predict_goals("T1", "T4", neutral=True).grid)
    np.testing.assert_array_equal(grids[0], grids[1])


def test_grid_is_normalized_and_positive(fitted):
    grid = fitted.predict_goals("T1", "T4").grid
    assert grid.shape == (11, 11)
    assert grid.sum() == pytest.approx(1.0, abs=0.02)
    assert (grid >= 0).all()


def test_wdl_collapses_grid_and_ranks_strength(fitted):
    wdl = fitted.predict_wdl("T1", "T6", neutral=True)
    grid_wdl = fitted.predict_goals("T1", "T6", neutral=True).wdl()
    assert wdl.home == pytest.approx(grid_wdl.home)
    assert wdl.home > wdl.away


def test_neutral_venue_removes_home_advantage(fitted):
    at_home = fitted.predict_wdl("T3", "T4", neutral=False)
    neutral = fitted.predict_wdl("T3", "T4", neutral=True)
    assert at_home.home > neutral.home
    assert at_home.away < neutral.away


def test_different_seed_gives_different_results():
    data = _league()
    hp1 = _make_model(random_seed=42).fit(data)
    hp2 = _make_model(random_seed=99).fit(data)
    g1 = hp1.predict_goals("T1", "T4", neutral=True).grid
    g2 = hp2.predict_goals("T1", "T4", neutral=True).grid
    assert not np.allclose(g1, g2, atol=1e-6)


def test_et_contaminated_scores_excluded():
    league = _league()
    et_rows = pd.DataFrame(
        [(pd.Timestamp("2025-06-01") + pd.Timedelta(days=i), "T1", f"T{2 + i % 5}",
          0, 5, "Cup", False, True) for i in range(20)],
        columns=_COLS,
    )
    poisoned = pd.concat([league, et_rows], ignore_index=True)
    confed = dict(_CONFED_MAP)
    clean = _make_model(et_weight=0.0, confed_map=confed).fit(poisoned)
    dirty = _make_model(et_weight=1.0, random_seed=42, confed_map=confed).fit(poisoned)
    assert clean.predict_wdl("T1", "T4", neutral=True).home > \
           dirty.predict_wdl("T1", "T4", neutral=True).home


def test_unfitted_and_unknown_team_raise(fitted):
    with pytest.raises(RuntimeError):
        _make_model().predict_wdl("T1", "T2")
    with pytest.raises(KeyError):
        fitted.predict_wdl("T1", "Atlantis")


def test_batch_model_refuses_incremental_update(fitted):
    with pytest.raises(NotImplementedError):
        fitted.update({"date": "2026-01-01"})


def test_confederation_pooling_shrinks_thin_data_team():
    """A team with very few matches should be pulled toward its confederation
    mean, producing more moderate predictions than a team-only model would."""
    rng = np.random.default_rng(42)
    teams = ["Strong1", "Strong2", "Mid1", "Mid2", "Weak1", "ThinTeam"]
    confed = {
        "Strong1": "A", "Strong2": "A", "Mid1": "A",
        "Mid2": "B", "Weak1": "B", "ThinTeam": "A",
    }
    strength = {"Strong1": 2.0, "Strong2": 1.8, "Mid1": 1.3,
                "Mid2": 1.0, "Weak1": 0.7, "ThinTeam": 1.5}
    rows, day = [], pd.Timestamp("2024-01-01")
    for _ in range(8):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                if "ThinTeam" in (h, a) and rng.random() > 0.15:
                    continue
                lam_h = 1.3 * strength[h] / strength[a]
                lam_a = strength[a] / strength[h]
                rows.append((day, h, a, rng.poisson(lam_h), rng.poisson(lam_a),
                             "League", True, False))
                day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=_COLS)
    thin_matches = df[(df["home_team"] == "ThinTeam") | (df["away_team"] == "ThinTeam")]
    assert len(thin_matches) < 30, "ThinTeam should have sparse data"

    hp = HierarchicalPoisson(
        n_samples=500, n_chains=1, min_matches=30,
        random_seed=42, confed_map=confed,
    ).fit(df)
    wdl = hp.predict_wdl("ThinTeam", "Weak1", neutral=True)
    assert wdl.home > 0.3, "ThinTeam should be pulled toward its strong confederation"


def test_provenance_fields(fitted):
    assert fitted.trained_through is not None
    assert fitted.n_training_matches > 0
    h = fitted.hyperparams()
    assert "random_seed" in h
    assert "window_years" in h
    assert fitted.hyperparams_hash
