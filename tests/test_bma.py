"""BMA ensembles (S22) — weight derivation, member discipline, and the
two D5 failure-loud paths.

Tests use FakeWDL and FakeGoals stand-ins so the contract is exercised
without dragging in the slow real-model fits — every BMA invariant is
about composition, not training.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fifapreds.db import init_predictions
from fifapreds.ensemble import (
    BMAEnsemble,
    BMAGoalsEnsemble,
    loto_log_losses,
)
from fifapreds.ensemble.bma import _softmax_weights
from fifapreds.loop.score import CLASSES
from fifapreds.models.base import WDL, GoalsModel, Model, ScoreGrid


# ----------------------------------------------- fake members for composition

class _FakeWDL(Model):
    model_version = "1"

    def __init__(self, model_id: str, wdl: tuple[float, float, float]):
        self.model_id = model_id
        self._wdl = wdl
        self.trained_through = pd.Timestamp("2026-01-01")  # already "fitted"

    def fit(self, matches): return self
    def hyperparams(self): return {"wdl": list(self._wdl)}
    def predict_wdl(self, home, away, *, neutral=False):
        return WDL(*self._wdl)


class _FakeGoals(GoalsModel):
    model_version = "1"

    def __init__(self, model_id: str, grid: np.ndarray):
        self.model_id = model_id
        self._grid = np.asarray(grid, dtype=float)
        self.trained_through = pd.Timestamp("2026-01-01")

    def fit(self, matches): return self
    def hyperparams(self): return {"grid_shape": list(self._grid.shape)}
    def predict_wdl(self, home, away, *, neutral=False):
        return self.predict_goals(home, away, neutral=neutral).wdl()
    def predict_goals(self, home, away, *, neutral=False):
        return ScoreGrid(self._grid)


# --------------------------------------------------------- weight derivation

def test_softmax_weights_basic_shape_and_normalization():
    w = _softmax_weights({"a": 1.0, "b": 1.5, "c": 1.2})
    assert set(w) == {"a", "b", "c"}
    assert sum(w.values()) == pytest.approx(1.0)
    # Lower loss → higher weight.
    assert w["a"] > w["c"] > w["b"]


def test_softmax_weights_temperature_smooths_toward_uniform():
    losses = {"a": 0.9, "b": 1.1}
    w_low = _softmax_weights(losses, T=0.1)
    w_hi = _softmax_weights(losses, T=10.0)
    # At T→∞, weights collapse to uniform; at T→0, they concentrate.
    assert max(w_low.values()) > max(w_hi.values())


def test_softmax_weights_empty_returns_empty():
    assert _softmax_weights({}) == {}


@pytest.fixture()
def backtest_db(tmp_path: Path):
    """Synthetic backtest: two models scored across 2 WCs, with model_a
    having lower log-loss than model_b on average."""
    path = tmp_path / "backtest.db"
    conn = sqlite3.connect(path)
    init_predictions(conn)
    pid = 1
    rows_p, rows_s = [], []
    rng = np.random.default_rng(0)
    for year in (2014, 2018):
        for i in range(30):
            outcome = ["home", "draw", "away"][int(rng.integers(0, 3))]
            # model_a: nails the outcome 60% of the time → lower log-loss.
            target_idx = CLASSES.index(outcome)
            p_a = np.array([0.2, 0.2, 0.2])
            p_a[target_idx] = 0.6
            rows_p.append((pid, f"backtest:wc{year}", year * 100 + i,
                           "H", "A", "2020-01-01", 1, "WC",
                           *p_a.tolist(), "model_a", "1", "abc", "h",
                           "2010-01-01", None, None, "2010-01-02"))
            rows_s.append((pid, outcome, 0.0, 0.0, 0.0, "2020-01-02"))
            pid += 1
            # model_b: uniform → higher log-loss.
            rows_p.append((pid, f"backtest:wc{year}", year * 100 + i + 1000,
                           "H", "A", "2020-01-01", 1, "WC",
                           1/3, 1/3, 1/3, "model_b", "1", "abc", "h",
                           "2010-01-01", None, None, "2010-01-02"))
            rows_s.append((pid, outcome, 0.0, 0.0, 0.0, "2020-01-02"))
            pid += 1
    conn.executemany(
        """INSERT INTO predictions
           (prediction_id, context, match_id, home_team, away_team,
            kickoff_ts, neutral, tournament, p_home, p_draw, p_away,
            model_id, model_version, code_version, hyperparams_hash,
            training_cutoff, odds_snapshot_id, seed, predicted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows_p,
    )
    conn.executemany(
        "INSERT INTO scores (prediction_id, outcome, log_loss, brier, rps, scored_at) "
        "VALUES (?,?,?,?,?,?)",
        rows_s,
    )
    conn.commit()
    return conn


def test_loto_log_losses_returns_per_model_means(backtest_db):
    losses = loto_log_losses(backtest_db)
    assert set(losses) == {"model_a", "model_b"}
    assert losses["model_a"] < losses["model_b"], \
        "the sharper model should have lower LOTO log-loss"


# ----------------------------------------------------------- BMAEnsemble (WDL)

def test_bma_wdl_weighted_average_combines_members():
    """50/50 average of two fakes returns the elementwise mean."""
    a = _FakeWDL("a", (0.6, 0.2, 0.2))
    b = _FakeWDL("b", (0.2, 0.2, 0.6))
    bma = BMAEnsemble([a, b], weights={"a": 0.5, "b": 0.5})
    out = bma.predict_wdl("H", "A")
    assert out.home == pytest.approx(0.4)
    assert out.draw == pytest.approx(0.2)
    assert out.away == pytest.approx(0.4)


def test_bma_wdl_explicit_weights_override_backtest():
    """When `weights` is supplied directly, backtest_conn is unused."""
    a = _FakeWDL("a", (0.5, 0.3, 0.2))
    b = _FakeWDL("b", (0.2, 0.3, 0.5))
    bma = BMAEnsemble([a, b], weights={"a": 0.8, "b": 0.2})
    out = bma.predict_wdl("H", "A")
    assert out.home == pytest.approx(0.8 * 0.5 + 0.2 * 0.2)


def test_bma_wdl_loto_derived_weights_favour_lower_loss(backtest_db):
    a = _FakeWDL("model_a", (0.6, 0.2, 0.2))
    b = _FakeWDL("model_b", (1/3, 1/3, 1/3))
    bma = BMAEnsemble([a, b], backtest_conn=backtest_db)
    # model_a had lower LOTO log-loss in the fixture → bigger weight.
    hp = bma.hyperparams()
    assert hp["weights"]["model_a"] > hp["weights"]["model_b"]


def test_bma_wdl_drops_member_without_backtest_history(backtest_db):
    """A member absent from the backtest gets a structured note + drop,
    not a silent uniform fallback (the D5 contract)."""
    known = _FakeWDL("model_a", (0.6, 0.2, 0.2))
    unknown = _FakeWDL("model_z", (0.4, 0.3, 0.3))
    bma = BMAEnsemble([known, unknown], backtest_conn=backtest_db)
    assert "model_z" not in bma.hyperparams()["weights"]
    assert any("model_z" in n for n in bma.notes)


def test_bma_wdl_raises_when_every_member_absent(backtest_db):
    """No member with backtest history → loud raise per D5."""
    a = _FakeWDL("ghost_a", (0.6, 0.2, 0.2))
    b = _FakeWDL("ghost_b", (0.4, 0.3, 0.3))
    with pytest.raises(RuntimeError, match="no scored predictions"):
        BMAEnsemble([a, b], backtest_conn=backtest_db)


def test_bma_requires_weights_or_backtest():
    a = _FakeWDL("a", (0.6, 0.2, 0.2))
    with pytest.raises(ValueError, match="weights"):
        BMAEnsemble([a])


def test_bma_wdl_single_member_degenerate_is_identity():
    a = _FakeWDL("a", (0.5, 0.3, 0.2))
    bma = BMAEnsemble([a], weights={"a": 1.0})
    out = bma.predict_wdl("H", "A")
    assert (out.home, out.draw, out.away) == pytest.approx((0.5, 0.3, 0.2))


def test_bma_wdl_trained_through_is_min_across_members():
    a = _FakeWDL("a", (0.6, 0.2, 0.2))
    a.trained_through = pd.Timestamp("2026-06-15")
    b = _FakeWDL("b", (0.4, 0.3, 0.3))
    b.trained_through = pd.Timestamp("2026-06-10")
    bma = BMAEnsemble([a, b], weights={"a": 0.5, "b": 0.5})
    assert bma.trained_through == pd.Timestamp("2026-06-10")


# ---------------------------------------------------- BMAGoalsEnsemble (goals)

def _delta_grid(rows: int, cols: int, h: int, a: int, mass: float = 1.0):
    g = np.zeros((rows, cols))
    g[h, a] = mass
    return g


def test_bma_goals_raises_on_non_goals_member():
    """D5 + D7: BMAGoalsEnsemble must reject W/D/L-only members at
    construction, not at first predict_goals."""
    g = _FakeGoals("g", _delta_grid(5, 5, 1, 1))
    wdl_only = _FakeWDL("wo", (0.6, 0.2, 0.2))
    with pytest.raises(TypeError, match="GoalsModel"):
        BMAGoalsEnsemble([g, wdl_only], weights={"g": 0.5, "wo": 0.5})


def test_bma_goals_weighted_grid_sum_renormalizes():
    """Two members each putting all mass on a different cell → weighted
    sum is two-cell mass that renormalizes to 1.0 (no probability lost)."""
    g1 = _FakeGoals("g1", _delta_grid(5, 5, 2, 1))
    g2 = _FakeGoals("g2", _delta_grid(5, 5, 0, 0))
    bma = BMAGoalsEnsemble([g1, g2], weights={"g1": 0.7, "g2": 0.3})
    grid = bma.predict_goals("H", "A").grid
    assert grid.sum() == pytest.approx(1.0)
    assert grid[2, 1] == pytest.approx(0.7)
    assert grid[0, 0] == pytest.approx(0.3)


def test_bma_goals_handles_different_grid_shapes():
    """A member with a 4×4 grid and another with 6×6 should align to 6×6
    without losing mass on the bigger grid."""
    g_small = _FakeGoals("s", _delta_grid(4, 4, 0, 0))
    g_big = _FakeGoals("b", _delta_grid(6, 6, 5, 5))
    bma = BMAGoalsEnsemble([g_small, g_big], weights={"s": 0.5, "b": 0.5})
    grid = bma.predict_goals("H", "A").grid
    assert grid.shape == (6, 6)
    assert grid.sum() == pytest.approx(1.0)
    assert grid[0, 0] == pytest.approx(0.5)
    assert grid[5, 5] == pytest.approx(0.5)


def test_bma_goals_predict_wdl_collapses_grid():
    g = _FakeGoals("g", _delta_grid(5, 5, 2, 0))  # home win
    bma = BMAGoalsEnsemble([g], weights={"g": 1.0})
    wdl = bma.predict_wdl("H", "A")
    assert wdl.home == pytest.approx(1.0)
    assert wdl.draw == pytest.approx(0.0)
    assert wdl.away == pytest.approx(0.0)


def test_bma_hyperparams_hash_distinct_per_weights():
    a = _FakeWDL("a", (0.6, 0.2, 0.2))
    b = _FakeWDL("b", (0.4, 0.3, 0.3))
    bma1 = BMAEnsemble([a, b], weights={"a": 0.7, "b": 0.3})
    bma2 = BMAEnsemble([a, b], weights={"a": 0.3, "b": 0.7})
    assert bma1.hyperparams_hash != bma2.hyperparams_hash
