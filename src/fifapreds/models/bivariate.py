"""Bivariate Poisson goals model (S9) — Karlis & Ntzoufras (2003).

The third parameter λ₃ in the bivariate Poisson formulation models
positive covariance between home and away goal counts: when one side
attacks the other counter-attacks; when both sides park the bus the
total stays low. Independent Poisson (which Dixon-Coles defaults to,
softened by its tau correction at the four low-score cells) cannot
express this directly.

Same thin wrapper around penaltyblog 1.11.0 as `NegBin` and `DixonColes`:
native `neutral_venue` + per-match `weights`, identical xi-decay + ET
cleaning + thin-window fallback. Per D11-B the fallback logic is copied
inline; a shared `_mle.py` base is reconsidered AFTER both this file and
NegBin land — premature abstraction on one user was a documented
outside-voice concern, and now there are three near-clones to compare
before extracting.

λ₃-degeneracy guard: penaltyblog raises ValueError if optimization
collapses to independent Poisson (λ₃ ≈ 0); the fallback catches that and
widens the window. The model still produces valid grids in the degenerate
limit (it reduces to independent Poisson, which is a valid distribution);
the issue is purely "no information added vs the cheaper alternative."
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import BivariatePoissonGoalModel, dixon_coles_weights

from fifapreds.models.base import WDL, GoalsModel, ScoreGrid

_REQUIRED_COLS = ("date", "home_team", "away_team", "home_score", "away_score",
                  "neutral", "went_to_et")


class BivariatePoisson(GoalsModel):
    model_id = "bivariate_poisson"
    model_version = "1"

    def __init__(
        self,
        *,
        window_years: float = 4.0,
        xi: float = 0.0018,
        et_weight: float = 0.0,
        max_goals: int = 15,
        min_matches: int = 500,
    ):
        self.window_years = window_years
        self.xi = xi
        self.et_weight = et_weight
        self.max_goals = max_goals
        self.min_matches = min_matches
        self._pb: BivariatePoissonGoalModel | None = None
        self._teams: set[str] = set()
        self.trained_through: pd.Timestamp | None = None
        self.fitted_window_years: float | None = None
        self.n_training_matches: int = 0

    def hyperparams(self) -> dict[str, Any]:
        return {
            "window_years": self.window_years,
            "xi": self.xi,
            "et_weight": self.et_weight,
            "max_goals": self.max_goals,
            "min_matches": self.min_matches,
        }

    # ---------------------------------------------------------------- train

    def fit(self, matches: pd.DataFrame) -> "BivariatePoisson":
        """Batch MLE refit on a trailing window of the as-of frame.

        Same window-doubling fallback as DixonColes and NegBin: a thin
        window raises ValueError, the schedule widens, ending with the
        full frame. The only thing that changes vs DC is the optimizer's
        likelihood — Karlis-Ntzoufras bivariate Poisson instead of
        Dixon-Coles."""
        missing = [c for c in _REQUIRED_COLS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches frame missing columns: {missing}")
        if matches["home_score"].isna().any() or matches["away_score"].isna().any():
            raise ValueError("unplayed match in training data")
        if matches.empty:
            raise ValueError("empty training frame")

        anchor = matches["date"].max()
        errors: list[str] = []
        for window in self._window_schedule(matches, anchor):
            subset = (
                matches if window is None
                else matches[matches["date"] >= anchor - pd.Timedelta(days=window * 365.25)]
            )
            weights = self._weights(subset, anchor)
            keep = weights > 0.0
            subset, weights = subset[keep], weights[keep]
            if window is not None and len(subset) < self.min_matches:
                errors.append(f"{window}y: only {len(subset)} matches")
                continue
            if subset.empty:
                errors.append("full frame: 0 usable matches")
                continue
            try:
                pb = BivariatePoissonGoalModel(
                    goals_home=np.array(subset["home_score"], dtype=np.int64),
                    goals_away=np.array(subset["away_score"], dtype=np.int64),
                    teams_home=np.array(subset["home_team"], dtype=object),
                    teams_away=np.array(subset["away_team"], dtype=object),
                    weights=np.array(weights, dtype=float),
                    neutral_venue=np.array(subset["neutral"], dtype=np.int64),
                )
                pb.fit()
            except ValueError as exc:  # penaltyblog: optimization failed
                errors.append(f"{window or 'full'}: {exc}")
                continue
            self._pb = pb
            self._teams = set(subset["home_team"]) | set(subset["away_team"])
            self.trained_through = anchor
            self.fitted_window_years = window
            self.n_training_matches = len(subset)
            return self

        raise RuntimeError(
            "Bivariate Poisson fit failed at every window (caller may fall "
            f"back to Elo): {'; '.join(errors)}"
        )

    def _window_schedule(self, matches: pd.DataFrame, anchor) -> list[float | None]:
        span_years = (anchor - matches["date"].min()).days / 365.25
        schedule: list[float | None] = []
        w = self.window_years
        while w < span_years:
            schedule.append(w)
            w *= 2.0
        schedule.append(None)
        return schedule

    def _weights(self, subset: pd.DataFrame, anchor) -> np.ndarray:
        w = (
            dixon_coles_weights(subset["date"], xi=self.xi, base_date=anchor)
            if self.xi > 0.0
            else np.ones(len(subset))
        )
        return np.asarray(w, dtype=float) * np.where(subset["went_to_et"], self.et_weight, 1.0)

    # -------------------------------------------------------------- predict

    def predict_goals(self, home: str, away: str, *, neutral: bool = False) -> ScoreGrid:
        if self._pb is None:
            raise RuntimeError("model is not fitted")
        for team in (home, away):
            if team not in self._teams:
                raise KeyError(
                    f"unknown team {team!r} — not in the fitted window "
                    f"({self.fitted_window_years or 'full'}y); "
                    "check registry.canonical() spelling"
                )
        grid = self._pb.predict(
            home, away, max_goals=self.max_goals, normalize=True, neutral_venue=neutral
        )
        return ScoreGrid(grid.grid)

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        return self.predict_goals(home, away, neutral=neutral).wdl()
