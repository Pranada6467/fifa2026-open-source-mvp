"""Temperature scaling for WDL probabilities (S25 — Guo et al. 2017).

A single scalar T scales raw logits before softmax:
    logits = log(probs)
    calibrated = softmax(logits / T)

T = 1 is identity. T > 1 softens (pushes mass toward uniform);
T < 1 sharpens. Standard fit: minimize NLL on the held-in fold via
`scipy.optimize.minimize_scalar` (bounded search on (0.05, 10.0); the
log-loss surface is convex in T so any bounded scalar method finds the
global min).

Only one parameter, no risk of overfitting on the tiny n=64-per-fold
backtest — the standard cheap calibrator that ships first per the plan.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from fifapreds.calibration.base import (
    EPS,
    Calibrator,
    _validate_outcomes,
    _validate_output,
    _validate_probs,
)

# scipy's `bounded` method ignores 0/inf, so pick a permissive but finite window.
_T_LO, _T_HI = 0.05, 10.0


def _softmax_logits_over_T(logits: np.ndarray, T: float) -> np.ndarray:
    """Numerically-stable softmax over scaled logits."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureCalibrator(Calibrator):
    """Single-scalar temperature. `T=1.0` after construction; `fit()` sets
    it to the NLL-minimizing value on the held-in fold."""

    def __init__(self):
        self.T: float = 1.0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "TemperatureCalibrator":
        probs = _validate_probs(probs, name="fit probs")
        outcomes = _validate_outcomes(outcomes, len(probs))
        # log of clipped probs — already finite per _validate_probs but
        # raw 0.0 cells (e.g. probability mass collapse) would blow up log.
        logits = np.log(np.clip(probs, EPS, 1.0))

        def nll(T: float) -> float:
            cal = _softmax_logits_over_T(logits, T)
            picked = cal[np.arange(len(cal)), outcomes]
            return float(-np.log(np.clip(picked, EPS, 1.0)).mean())

        result = minimize_scalar(nll, bounds=(_T_LO, _T_HI), method="bounded",
                                 options={"xatol": 1e-4})
        if not result.success:
            raise RuntimeError(f"temperature fit failed: {result.message}")
        self.T = float(result.x)
        return self

    def apply(self, probs: np.ndarray) -> np.ndarray:
        probs = _validate_probs(probs, name="apply probs")
        logits = np.log(np.clip(probs, EPS, 1.0))
        calibrated = _softmax_logits_over_T(logits, self.T)
        return _validate_output(calibrated)
