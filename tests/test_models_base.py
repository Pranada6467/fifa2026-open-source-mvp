"""Interface-contract guards: probability containers validate their invariants
(bad probabilities raise loudly, never propagate), the grid→W/D/L collapse is
exact on a known grid, and the provenance hash is stable and config-sensitive.
"""
from __future__ import annotations

import numpy as np
import pytest

from fifapreds.models import WDL, BaselineElo, ScoreGrid


def test_wdl_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        WDL(home=0.5, draw=0.5, away=0.5)        # sums to 1.5
    with pytest.raises(ValueError):
        WDL(home=1.2, draw=-0.1, away=-0.1)      # out of [0,1]
    with pytest.raises(ValueError):
        WDL(home=float("nan"), draw=0.5, away=0.5)


def test_wdl_array_order_is_home_draw_away():
    p = WDL(home=0.5, draw=0.3, away=0.2)
    assert list(p.as_array()) == [0.5, 0.3, 0.2]


def test_score_grid_collapses_to_wdl():
    # grid[i, j] = P(home i, away j): home-win mass below the diagonal.
    grid = np.array([
        [0.10, 0.05, 0.05],   # 0-0, 0-1, 0-2
        [0.20, 0.10, 0.05],   # 1-0, 1-1, 1-2
        [0.20, 0.15, 0.10],   # 2-0, 2-1, 2-2
    ])
    wdl = ScoreGrid(grid).wdl()
    assert wdl.home == pytest.approx(0.55)
    assert wdl.draw == pytest.approx(0.30)
    assert wdl.away == pytest.approx(0.15)


def test_score_grid_renormalizes_truncation_but_rejects_garbage():
    near = np.full((3, 3), (1.0 - 0.01) / 9)          # 1% mass lost to truncation: OK
    assert ScoreGrid(near).grid.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ScoreGrid(np.full((3, 3), 0.5 / 9))           # half the mass missing: bug
    with pytest.raises(ValueError):
        ScoreGrid(np.array([[1.1, -0.1], [0.0, 0.0]]))  # negative entries


def test_hyperparams_hash_stable_and_config_sensitive():
    a1, a2 = BaselineElo(k_factor=32), BaselineElo(k_factor=32)
    b = BaselineElo(k_factor=20)
    assert a1.hyperparams_hash == a2.hyperparams_hash   # same config, same hash
    assert a1.hyperparams_hash != b.hyperparams_hash    # any knob changes it
