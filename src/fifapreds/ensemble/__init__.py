"""Ensemble entrants — composition of the frozen roster (Phase 5).

`BMAEnsemble` (W/D/L only) and `BMAGoalsEnsemble` (score-grid capable)
combine the fitted roster using LOTO-derived per-model weights from
`data/backtest.db`. Stacking (S19) lands as a follow-up; the package
keeps that surface explicit by holding only BMA today.
"""
from fifapreds.ensemble.bma import (
    BMAEnsemble,
    BMAGoalsEnsemble,
    loto_log_losses,
)

__all__ = ["BMAEnsemble", "BMAGoalsEnsemble", "loto_log_losses"]
