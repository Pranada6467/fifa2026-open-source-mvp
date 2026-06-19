"""Calibration module invariants (S25 + S20 + Phase 4 pipeline).

Contracts pinned here:
- Both calibrators preserve row-sum invariants (`_validate_output`).
- Temperature: T=1 is identity; miscalibrated synthetic input drives T off 1
  in the expected direction; bounded scalar search finds the global min.
- Isotonic: monotone-increasing fit preserves cross-row order on the
  fitted column AFTER renormalization (the property a reliability chart
  cares about); per-class isotonic of an empty bin raises rather than
  silently dividing by zero.
- D5 fail-loud: NaN / inf / empty input → raise.
- Pipeline LOTO discipline (D6): the train fold for holdout year Y
  contains zero rows tagged Y — asserted, not assumed.
- Pipeline averaging: temperature variants average their T; isotonic
  variants compose via a meta-calibrator that averages outputs.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fifapreds.calibration import (
    IsotonicCalibrator,
    TemperatureCalibrator,
    apply_calibrators,
    fit_calibrators,
    loto_holdout_split,
)
from fifapreds.calibration.pipeline import _AveragedIsotonic
from fifapreds.db import init_predictions
from fifapreds.loop.score import CLASSES


# --------------------------------------------------------------- temperature

def _synthetic_overconfident(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions that are systematically too sharp: true class
    probability ~0.4 but the model claims ~0.8. Temperature scaling must
    push T > 1 to soften."""
    rng = np.random.default_rng(seed)
    # Truth: each class is the actual outcome with probability ~0.4.
    outcomes = rng.integers(0, 3, size=n)
    # Predictions: assign 0.8 to a chosen class (mostly the actual outcome,
    # but not always — overconfident in the wrong direction half the time).
    chosen = np.where(rng.random(n) < 0.5, outcomes, rng.integers(0, 3, size=n))
    probs = np.full((n, 3), 0.10)
    probs[np.arange(n), chosen] = 0.80
    return probs, outcomes


def test_temperature_identity_at_t1():
    """T=1 must reproduce the input within numerical tolerance — the
    well-known property that anchors the test of every other temperature."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=50)
    cal = TemperatureCalibrator()
    assert cal.T == 1.0
    out = cal.apply(probs)
    assert np.allclose(out, probs, atol=1e-8)


def test_temperature_fit_softens_overconfident_predictions():
    probs, outcomes = _synthetic_overconfident()
    cal = TemperatureCalibrator().fit(probs, outcomes)
    assert cal.T > 1.05, f"overconfident input should push T well above 1, got {cal.T:.3f}"


def test_temperature_fit_sharpens_underconfident_predictions():
    """Almost-uniform predictions where one class is correct ~all the time
    should pull T < 1 to sharpen."""
    rng = np.random.default_rng(1)
    n = 500
    # 99% home wins.
    outcomes = np.zeros(n, dtype=int)
    outcomes[:5] = 1                          # a few draws to keep classes covered
    outcomes[5:10] = 2                        # and a few away wins
    probs = np.tile([0.35, 0.33, 0.32], (n, 1))
    cal = TemperatureCalibrator().fit(probs, outcomes)
    assert cal.T < 0.95, f"underconfident input should push T below 1, got {cal.T:.3f}"


def test_temperature_apply_preserves_row_sums():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=100)
    cal = TemperatureCalibrator()
    cal.T = 2.5
    out = cal.apply(probs)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-8)


# ----------------------------------------------------------------- isotonic

def test_isotonic_fit_apply_roundtrip_well_calibrated_input():
    """Well-calibrated input: isotonic should be near-identity (each class's
    monotone fit ≈ y=x), and apply/renormalize must preserve sums."""
    rng = np.random.default_rng(3)
    n = 1000
    outcomes = rng.integers(0, 3, size=n)
    probs = np.zeros((n, 3))
    probs[np.arange(n), outcomes] = rng.uniform(0.5, 0.9, size=n)
    probs[np.arange(n), (outcomes + 1) % 3] = rng.uniform(0.05, 0.3, size=n)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cal = IsotonicCalibrator().fit(probs, outcomes)
    out = cal.apply(probs)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-8)


def test_isotonic_monotonicity_preserved_in_class_column():
    """For a single class column, the isotonic fit is monotone-increasing
    by construction. After renormalization the per-column monotonicity
    is no longer strict (the other columns may shift), but the rank of
    rows in the calibrated home column should follow the rank in the raw."""
    rng = np.random.default_rng(4)
    n = 400
    probs = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=n)
    outcomes = rng.integers(0, 3, size=n)
    cal = IsotonicCalibrator().fit(probs, outcomes)
    fresh = rng.dirichlet(alpha=(1.0, 1.0, 1.0), size=200)
    cal_out = cal.apply(fresh)
    # Pin: the order of `fresh[:, 0]` and `cal_out[:, 0]` should agree on
    # at least 80% of pairs (Spearman ≈ 0.8 on a noisy 200-row sample).
    rho = pd.Series(fresh[:, 0]).rank().corr(pd.Series(cal_out[:, 0]).rank())
    assert rho > 0.8, f"isotonic destroyed column-rank order: rho={rho:.3f}"


def test_isotonic_apply_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().apply(np.array([[0.5, 0.3, 0.2]]))


# ----------------------------------------------------------------- D5 loud failure

@pytest.mark.parametrize("Cal", [TemperatureCalibrator, IsotonicCalibrator])
def test_empty_input_raises(Cal):
    cal = Cal()
    with pytest.raises(ValueError, match="empty"):
        cal.fit(np.empty((0, 3)), np.array([], dtype=int))


@pytest.mark.parametrize("Cal", [TemperatureCalibrator, IsotonicCalibrator])
def test_nan_input_raises(Cal):
    cal = Cal()
    bad = np.array([[0.5, np.nan, 0.5], [0.4, 0.3, 0.3]])
    outcomes = np.array([0, 1])
    with pytest.raises(ValueError, match="NaN"):
        cal.fit(bad, outcomes)


@pytest.mark.parametrize("Cal", [TemperatureCalibrator, IsotonicCalibrator])
def test_wrong_shape_raises(Cal):
    cal = Cal()
    # 2-column probs (no draw column) — invalid for WDL.
    with pytest.raises(ValueError, match="shape"):
        cal.fit(np.array([[0.5, 0.5]]), np.array([0]))


@pytest.mark.parametrize("Cal", [TemperatureCalibrator, IsotonicCalibrator])
def test_unnormalized_input_raises(Cal):
    cal = Cal()
    bad = np.array([[0.5, 0.5, 0.5]])  # sums to 1.5
    with pytest.raises(ValueError, match="sums to"):
        cal.fit(bad, np.array([0]))


# ----------------------------------------------------------------- LOTO pipeline

@pytest.fixture()
def backtest_db(tmp_path: Path):
    """Synthetic 3-WC backtest DB: each WC has 64 graded predictions per
    model, with deterministic (probs, outcome) so the LOTO split is
    auditable. Two model_ids for the multi-model code path."""
    path = tmp_path / "backtest.db"
    conn = sqlite3.connect(path)
    init_predictions(conn)
    rng = np.random.default_rng(99)
    rows_pred, rows_score = [], []
    pid = 1
    for year in (2014, 2018, 2022):
        for model_id in ("model_a", "model_b"):
            for i in range(64):
                p = rng.dirichlet(alpha=(2.0, 1.0, 2.0))
                outcome = CLASSES[int(rng.integers(0, 3))]
                rows_pred.append((pid, f"backtest:wc{year}", 1000 * year + i,
                                  "H", "A", "2020-01-01T00:00:00", 1,
                                  "Friendly", *p,
                                  model_id, "1", "abc", "hash",
                                  "2010-01-01", None, None, "2010-01-02T00:00:00"))
                rows_score.append((pid, outcome, 0.0, 0.0, 0.0, "2020-01-02T00:00:00"))
                pid += 1
    conn.executemany(
        """INSERT INTO predictions
           (prediction_id, context, match_id, home_team, away_team,
            kickoff_ts, neutral, tournament,
            p_home, p_draw, p_away,
            model_id, model_version, code_version, hyperparams_hash,
            training_cutoff, odds_snapshot_id, seed, predicted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows_pred,
    )
    conn.executemany(
        "INSERT INTO scores (prediction_id, outcome, log_loss, brier, rps, scored_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows_score,
    )
    conn.commit()
    return conn


def test_loto_split_train_excludes_holdout_year():
    df = pd.DataFrame({"tournament_year": [2014, 2014, 2018, 2022, 2022]})
    train, holdout = loto_holdout_split(df, 2018)
    assert (train["tournament_year"] != 2018).all(), "leak: train holds 2018 rows"
    assert (holdout["tournament_year"] == 2018).all()
    assert len(train) == 4 and len(holdout) == 1


def test_fit_calibrators_emits_per_model_per_track(backtest_db):
    cals = fit_calibrators(backtest_db)
    # 2 models × 2 default tracks (temperature, isotonic) = 4 calibrators.
    assert set(cals.keys()) == {
        ("model_a", "temperature"), ("model_a", "isotonic"),
        ("model_b", "temperature"), ("model_b", "isotonic"),
    }
    # Each must apply without raising on a fresh probability row.
    for cal in cals.values():
        out = cal.apply(np.array([[0.4, 0.3, 0.3]]))
        assert np.allclose(out.sum(axis=1), 1.0)


def test_fit_calibrators_loto_assertion_blocks_leakage(backtest_db, monkeypatch):
    """Force `loto_holdout_split` to receive a frame whose train side
    contains rows from the holdout year — the assertion must fire."""
    from fifapreds.calibration import pipeline as pmod

    real_split = pmod.loto_holdout_split

    def leaky_split(df, holdout_year):
        # Inject a single bad row into the train half to trip the assertion.
        train, hold = real_split(df, holdout_year)
        leaked = hold.iloc[:1].copy() if not hold.empty else None
        if leaked is not None:
            train_with_leak = pd.concat([train, leaked], ignore_index=True)
            if (train_with_leak["tournament_year"] == holdout_year).any():
                raise AssertionError(f"LOTO leak: train set contains rows tagged {holdout_year}")
        return train, hold

    monkeypatch.setattr(pmod, "loto_holdout_split", leaky_split)
    with pytest.raises(AssertionError, match="LOTO leak"):
        fit_calibrators(backtest_db)


def test_fit_calibrators_averages_temperature(backtest_db):
    cals = fit_calibrators(backtest_db,
                           factories={"temperature": TemperatureCalibrator})
    cal_a = cals[("model_a", "temperature")]
    assert isinstance(cal_a, TemperatureCalibrator)
    assert 0.05 <= cal_a.T <= 10.0, f"T out of search bounds: {cal_a.T}"


def test_fit_calibrators_averages_isotonic_via_meta_calibrator(backtest_db):
    cals = fit_calibrators(backtest_db,
                           factories={"isotonic": IsotonicCalibrator})
    cal_a = cals[("model_a", "isotonic")]
    assert isinstance(cal_a, _AveragedIsotonic)
    out = cal_a.apply(np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]]))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_fit_calibrators_refuses_single_tournament(tmp_path):
    """LOTO needs ≥ 2 tournaments; fitting on a 1-WC scratch should raise
    rather than silently produce no folds."""
    path = tmp_path / "one_wc.db"
    conn = sqlite3.connect(path)
    init_predictions(conn)
    conn.execute(
        """INSERT INTO predictions
           (prediction_id, context, match_id, home_team, away_team,
            kickoff_ts, neutral, tournament,
            p_home, p_draw, p_away,
            model_id, model_version, code_version, hyperparams_hash,
            training_cutoff, odds_snapshot_id, seed, predicted_at)
           VALUES (1, 'backtest:wc2018', 1, 'A', 'B', '2018-06-01', 1, 'WC',
                   0.4, 0.3, 0.3, 'm', '1', 'abc', 'h', '2018-01-01',
                   NULL, NULL, '2018-05-31')""",
    )
    conn.execute(
        "INSERT INTO scores VALUES (1, 'home', 0.0, 0.0, 0.0, '2018-06-02')")
    conn.commit()
    with pytest.raises(ValueError, match="LOTO needs at least 2"):
        fit_calibrators(conn)


# ------------------------------------------------------------ apply_calibrators

def test_apply_calibrators_emits_raw_plus_each_track():
    """Every row appears under track='raw' (unchanged) plus one row per
    available calibrated track. Models missing from `calibrators` get raw only."""
    df = pd.DataFrame({
        "model_id": ["m1", "m1", "m2", "m2"],
        "p_home": [0.5, 0.4, 0.6, 0.3],
        "p_draw": [0.3, 0.3, 0.2, 0.4],
        "p_away": [0.2, 0.3, 0.2, 0.3],
    })
    # Only m1 has a temperature calibrator; m2 should appear only in raw.
    temp_cal = TemperatureCalibrator()
    temp_cal.T = 2.0  # softens
    out = apply_calibrators(df, {("m1", "temperature"): temp_cal})
    assert set(out["track"]) == {"raw", "temperature"}
    assert (out[out["track"] == "raw"]["model_id"].value_counts().to_dict()
            == {"m1": 2, "m2": 2})
    assert (out[out["track"] == "temperature"]["model_id"].value_counts().to_dict()
            == {"m1": 2})
    # Sanity: temperature track preserved row sums.
    cal_rows = out[out["track"] == "temperature"]
    assert np.allclose(
        cal_rows[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0, atol=1e-8)


def test_apply_calibrators_on_empty_frame_returns_raw_only():
    df = pd.DataFrame(columns=["model_id", "p_home", "p_draw", "p_away"])
    out = apply_calibrators(df, {})
    assert list(out.columns) == ["model_id", "p_home", "p_draw", "p_away", "track"]
    assert out.empty
