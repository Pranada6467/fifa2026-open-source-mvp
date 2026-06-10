"""Scorer correctness: golden-value metrics (hand-computed), the log(0) clamp,
agreement with penaltyblog's reference implementations, and a calibration
table that sits on the diagonal when fed perfectly calibrated forecasts.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from penaltyblog import metrics as pb_metrics

from fifapreds.loop.score import (EPS, brier, calibration_table, log_loss,
                                  outcome_index, rps)


def test_outcome_index():
    assert outcome_index(2, 0) == 0   # home
    assert outcome_index(1, 1) == 1   # draw (incl. shootouts after 120')
    assert outcome_index(0, 3) == 2   # away


def test_log_loss_golden_values():
    assert log_loss([[0.5, 0.3, 0.2]], [0])[0] == pytest.approx(math.log(2))
    assert log_loss([[0.5, 0.3, 0.2]], [2])[0] == pytest.approx(-math.log(0.2))


def test_clamp_prevents_log_zero():
    # A delusional forecast (p=0 on what happened) costs -ln(EPS), not inf/nan.
    loss = log_loss([[1.0, 0.0, 0.0]], [2])[0]
    assert np.isfinite(loss) and loss == pytest.approx(-math.log(EPS))


def test_brier_golden_values():
    assert brier([[0.7, 0.2, 0.1]], [0])[0] == pytest.approx(0.14)
    assert brier([[1.0, 0.0, 0.0]], [0])[0] == pytest.approx(0.0)   # perfect
    assert brier([[0.0, 0.0, 1.0]], [0])[0] == pytest.approx(2.0)   # max wrong


def test_rps_golden_and_ordering_sensitivity():
    assert rps([[0.6, 0.3, 0.1]], [0])[0] == pytest.approx(0.085)
    assert rps([[1.0, 0.0, 0.0]], [0])[0] == pytest.approx(0.0)
    # RPS is ordinal: a home-heavy forecast is punished harder by an away win
    # (two steps away) than the same-sized miss one step away.
    near_miss = rps([[0.6, 0.3, 0.1]], [1])[0]
    far_miss = rps([[0.6, 0.3, 0.1]], [2])[0]
    assert far_miss > near_miss


def test_metrics_agree_with_penaltyblog():
    rng = np.random.default_rng(42)
    probs = rng.dirichlet(np.ones(3), size=50)
    outcomes = rng.integers(0, 3, size=50)
    assert rps(probs, outcomes) == pytest.approx(pb_metrics.rps_array(probs, outcomes))
    assert brier(probs, outcomes).mean() == pytest.approx(
        pb_metrics.multiclass_brier_score(probs, outcomes))
    # penaltyblog's ignorance score is log2-based; ours is natural log.
    assert log_loss(probs, outcomes).mean() == pytest.approx(
        pb_metrics.ignorance_score(probs, outcomes) * math.log(2))


def test_calibration_table_diagonal_for_calibrated_forecasts():
    # 100 identical (0.7, 0.2, 0.1) forecasts whose outcomes occur at exactly
    # the stated rates → every populated bin sits on the diagonal.
    probs = np.tile([0.7, 0.2, 0.1], (100, 1))
    outcomes = np.array([0] * 70 + [1] * 20 + [2] * 10)
    table = calibration_table(probs, outcomes, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == 300  # every (row, class) claim is pooled
    populated = table.dropna(subset=["p_mean"])
    assert np.allclose(populated["freq"], populated["p_mean"], atol=1e-12)


def test_calibration_table_flags_overconfidence():
    # Claims 90% home but it only happens half the time → freq well below p.
    probs = np.tile([0.9, 0.05, 0.05], (100, 1))
    outcomes = np.array([0, 2] * 50)
    table = calibration_table(probs, outcomes, n_bins=10)
    hot_bin = table.iloc[9]  # the 0.9-1.0 bin
    assert hot_bin["p_mean"] == pytest.approx(0.9)
    assert hot_bin["freq"] == pytest.approx(0.5)
