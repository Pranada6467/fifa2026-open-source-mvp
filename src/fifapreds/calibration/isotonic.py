"""Isotonic recalibration for WDL probabilities (S20 — Zadrozny & Elkan 2002).

For each of the three classes c ∈ {home, draw, away}, fit a separate
one-vs-rest monotonic spline on (probs[:, c], outcome == c). Apply maps
each class column through its spline, then row-renormalizes back to a
proper distribution (the per-class isotonic maps are not jointly
calibrated, so the rows almost never sum to 1 before renormalization).

`sklearn.isotonic.IsotonicRegression` does the heavy lifting:
out_of_bounds='clip' ensures any apply-time probability outside the
training range gets pinned to the nearest fitted value (the calibrator
should never extrapolate past evidence). Monotone-increasing fit + clip
+ renormalization preserves the order of any two cells AFTER scaling —
a useful property the per-test invariant pins.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from fifapreds.calibration.base import (
    EPS,
    Calibrator,
    _validate_outcomes,
    _validate_output,
    _validate_probs,
)


class IsotonicCalibrator(Calibrator):
    """One isotonic regression per WDL class, fit one-vs-rest."""

    def __init__(self):
        self._regressors: list[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        probs = _validate_probs(probs, name="fit probs")
        outcomes = _validate_outcomes(outcomes, len(probs))
        self._regressors = []
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip",
                                    y_min=0.0, y_max=1.0,
                                    increasing=True)
            y_c = (outcomes == c).astype(float)
            ir.fit(probs[:, c], y_c)
            self._regressors.append(ir)
        return self

    def apply(self, probs: np.ndarray) -> np.ndarray:
        if not self._regressors:
            raise RuntimeError("isotonic calibrator is not fitted")
        probs = _validate_probs(probs, name="apply probs")
        cols = [self._regressors[c].predict(probs[:, c]) for c in range(3)]
        calibrated = np.column_stack(cols)
        # Per-class isotonic doesn't enforce row sums; renormalize.
        row_sums = calibrated.sum(axis=1, keepdims=True)
        if (row_sums < EPS).any():
            raise ValueError("isotonic apply: row sum collapsed to ~0; "
                             "fit on a bin where the actual frequency was 0")
        return _validate_output(calibrated / row_sums)
