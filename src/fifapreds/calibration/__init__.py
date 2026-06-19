"""Publish-time probability recalibration (S25 + S20).

Raw model claims stay pristine in `data/fifa2026.db`. Calibrators are
fit on backtest predictions under LOTO discipline (per `/plan-eng-review`
D6) and applied at publish time to produce the `track` dimension of
`leaderboard.parquet`. The viewer renders calibrated as default; raw is
available via the track toggle (see plan DD2).

Per D5: any calibrator that emits NaN or receives empty input raises
loudly so the orchestrator's existing drop-entrant path catches it; the
viewer never sees a degraded probability.
"""
from fifapreds.calibration.base import Calibrator
from fifapreds.calibration.isotonic import IsotonicCalibrator
from fifapreds.calibration.pipeline import (
    apply_calibrators,
    fit_calibrators,
    loto_holdout_split,
)
from fifapreds.calibration.temperature import TemperatureCalibrator

__all__ = [
    "Calibrator",
    "TemperatureCalibrator",
    "IsotonicCalibrator",
    "fit_calibrators",
    "apply_calibrators",
    "loto_holdout_split",
]
