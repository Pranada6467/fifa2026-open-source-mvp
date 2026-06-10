"""Model interface — the split that keeps the leaderboard honest.

Every model implements `predict_wdl()` (win/draw/loss — the common currency the
scorer and leaderboard grade). Only goals-capable models (Dixon-Coles/Poisson)
additionally implement `predict_goals()`, returning a full scoreline grid. Elo
is W/D/L-only and therefore CANNOT drive the Monte Carlo simulator: the group
sim needs scorelines for goal-difference / goals-scored tiebreaks.

Provenance contract: every model exposes `model_id`, `model_version`,
`hyperparams()` and a stable `hyperparams_hash`, plus `trained_through` (the
date of the last result it has seen). The predictions log (T5) stores these so
each leaderboard entry is auditable.

Probability order convention is (home, draw, away) throughout — the same order
penaltyblog's rps/log-loss/Brier metrics expect.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd

# Tolerance for "these probabilities sum to 1". WDL triples are computed in
# closed form so they should be exact to float error; score grids are truncated
# at a max-goals cap and may legitimately leak a little mass.
_WDL_TOL = 1e-6
_GRID_TOL = 0.02


@dataclass(frozen=True)
class WDL:
    """A win/draw/loss probability triple (from the home side's perspective)."""

    home: float
    draw: float
    away: float

    def __post_init__(self):
        p = (self.home, self.draw, self.away)
        if not all(np.isfinite(p)) or min(p) < 0.0 or max(p) > 1.0:
            raise ValueError(f"probabilities out of [0,1]: {p}")
        if abs(sum(p) - 1.0) > _WDL_TOL:
            raise ValueError(f"probabilities sum to {sum(p)}, not 1: {p}")

    def as_array(self) -> np.ndarray:
        """[home, draw, away] — the metric/scorer order."""
        return np.array([self.home, self.draw, self.away])


@dataclass(frozen=True)
class ScoreGrid:
    """A scoreline distribution: grid[i, j] = P(home scores i, away scores j).

    Grids are truncated at a max-goals cap, so a small amount of probability
    mass is allowed to be missing; it is renormalized away on construction.
    A grid missing more than _GRID_TOL of its mass is a modelling bug — raise.
    """

    grid: np.ndarray

    def __post_init__(self):
        g = np.asarray(self.grid, dtype=float)
        if g.ndim != 2:
            raise ValueError(f"score grid must be 2-D, got shape {g.shape}")
        if not np.isfinite(g).all() or (g < 0).any():
            raise ValueError("score grid has negative or non-finite entries")
        total = g.sum()
        if abs(total - 1.0) > _GRID_TOL:
            raise ValueError(f"score grid mass {total:.4f} too far from 1")
        object.__setattr__(self, "grid", g / total)

    def wdl(self) -> WDL:
        """Collapse the grid to W/D/L: lower triangle = home win, diagonal = draw."""
        home = np.tril(self.grid, k=-1).sum()
        draw = np.trace(self.grid)
        away = np.triu(self.grid, k=1).sum()
        return WDL(home=home, draw=draw, away=away)


class Model(ABC):
    """Base class for all leaderboard entrants (W/D/L is the common currency)."""

    model_id: ClassVar[str]
    model_version: ClassVar[str]

    #: Date of the last result this model has seen (None before fit). The
    #: provenance log stores it as `training_cutoff`, and incremental models
    #: use it to refuse out-of-order updates.
    trained_through: pd.Timestamp | None = None

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "Model":
        """(Re)train from scratch on played matches (an `asof.before(ts)` frame).

        The no-lookahead guarantee comes from the caller passing an as-of
        frame; models never read match data through any other path.
        """

    def update(self, match: Mapping[str, Any] | pd.Series) -> None:
        """Incorporate one new result (incremental models override).

        Batch-fit models (Dixon-Coles) stay current by refitting via `fit()`
        on a fresh as-of window instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is batch-fit: call fit() on a new as-of window"
        )

    @abstractmethod
    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        """Win/draw/loss probabilities for a fixture, using state as of
        `trained_through`. Unknown team names must raise (silent fallbacks
        hide registry mismatches)."""

    @abstractmethod
    def hyperparams(self) -> dict[str, Any]:
        """The frozen config that defines this entrant (JSON-serializable)."""

    @property
    def hyperparams_hash(self) -> str:
        """Stable digest of the config — stored with every prediction so a
        leaderboard row can prove which exact configuration produced it."""
        blob = json.dumps(self.hyperparams(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


class GoalsModel(Model):
    """A model that can produce full scorelines — required by the simulator."""

    @abstractmethod
    def predict_goals(self, home: str, away: str, *, neutral: bool = False) -> ScoreGrid:
        """Scoreline grid for a fixture (90-minute / regulation-time goals)."""
