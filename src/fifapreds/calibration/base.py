"""Calibrator abstract base — the publish-time probability mapper.

A `Calibrator` is fit on (n, 3) WDL probabilities + (n,) integer outcomes
in {0=home, 1=draw, 2=away} (the same convention `loop.score.CLASSES` uses)
and then maps fresh (n, 3) probabilities to recalibrated (n, 3) ones, with
every row's sum and bounds preserved.

LOTO discipline lives in `pipeline.py`, not here: this class is the per-fit
math, called once per (model_id, holdout_year) by the pipeline. Per D5 it
must raise loudly on empty input or NaN output — silent degradation is the
exact failure mode the calibration layer cannot have.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

EPS = 1e-12


class Calibrator(ABC):
    """Maps raw WDL probabilities → calibrated WDL probabilities.

    Subclasses pick the parametric family (temperature scaling, isotonic,
    Platt, etc.). The contract here is loud failure on degenerate input.
    """

    @abstractmethod
    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        """Fit on (n, 3) probs + (n,) outcomes ∈ {0, 1, 2}. Returns self."""

    @abstractmethod
    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Map (n, 3) probs → (n, 3) calibrated probs. Rows sum to 1 ± 1e-6."""


def _validate_probs(probs: np.ndarray, *, name: str = "probs") -> np.ndarray:
    """Per D5: shape, NaN, empty, sum-to-1 ± 2% — raise on any violation
    so a bad calibrator can never silently corrupt the leaderboard."""
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError(f"{name}: empty input — calibrator cannot fit/apply on 0 rows")
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"{name}: expected shape (n, 3), got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError(f"{name}: contains NaN or inf — refuse to proceed")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=2e-2):
        bad = np.argmax(np.abs(sums - 1.0))
        raise ValueError(
            f"{name}: row {bad} sums to {sums[bad]:.4f}, not 1.0 (tol 2e-2)"
        )
    return probs


def _validate_outcomes(outcomes: np.ndarray, n: int) -> np.ndarray:
    outcomes = np.asarray(outcomes, dtype=int)
    if outcomes.shape != (n,):
        raise ValueError(f"outcomes shape {outcomes.shape} != ({n},)")
    if outcomes.min() < 0 or outcomes.max() > 2:
        raise ValueError(f"outcomes out of {{0,1,2}}: min={outcomes.min()}, max={outcomes.max()}")
    return outcomes


def _validate_output(calibrated: np.ndarray) -> np.ndarray:
    """Apply-side check — same shape, no NaN, sum-to-1 within tight bound."""
    calibrated = np.asarray(calibrated, dtype=float)
    if not np.isfinite(calibrated).all():
        raise ValueError("calibrator produced NaN/inf — refuse to publish")
    if calibrated.ndim != 2 or calibrated.shape[1] != 3:
        raise ValueError(f"calibrator output shape {calibrated.shape} != (n, 3)")
    sums = calibrated.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        bad = np.argmax(np.abs(sums - 1.0))
        raise ValueError(
            f"calibrator output row {bad} sums to {sums[bad]:.6f}, not 1.0 (tol 1e-6)"
        )
    return calibrated
