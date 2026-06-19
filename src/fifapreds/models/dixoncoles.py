"""Dixon-Coles wrapper — the goals-capable model that can drive the simulator.

Thin wrapper around penaltyblog 1.11.0's `DixonColesGoalModel` (verified in
Block V: native per-match `neutral_venue`, per-match `weights`, and
`dixon_coles_weights` time decay). One upstream `predict()` yields the full
scoreline grid, so `predict_wdl()` is just the grid collapsed.

Fitting policy (all knobs hashed into provenance):
- **Trailing window** (default ~4 years) ending at the training frame's last
  date — recent form matters more than 1950s friendlies, and refit cost stays
  bounded. The window anchor is the *frame's* max date, never wall-clock time,
  so backtest replays are deterministic and leak-free.
- **Time decay** inside the window via `dixon_coles_weights(xi)` (the original
  Dixon-Coles exponential down-weighting).
- **90-minute cleaning (V2):** martj42 scores include extra time, so matches
  flagged `went_to_et` get weight `et_weight` (default 0 = excluded) to keep
  120-minute scorelines from inflating the goal rates.
- **Thin-window fallback:** if the window holds fewer than `min_matches` rows
  or the MLE fails to converge (penaltyblog raises ValueError), the window is
  doubled and the fit retried, ending with the full frame. Only if *that*
  fails does fit raise — the orchestrator may then fall back to Elo. The
  window actually used is recorded in `fitted_window_years` (None = full).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

from fifapreds.models.base import WDL, GoalsModel, ScoreGrid

_REQUIRED_COLS = ("date", "home_team", "away_team", "home_score", "away_score",
                  "neutral", "went_to_et")
# When `importance` is set the fit additionally requires the "tournament"
# column (BaselineElo's tournament-weighted update does the same). Base DC
# never reads this column, so the requirement stays opt-in.
_IMPORTANCE_REQUIRED_COLS = ("tournament",)


class DixonColes(GoalsModel):
    model_id = "dixon_coles"
    model_version = "1"

    def __init__(
        self,
        *,
        window_years: float = 4.0,
        xi: float = 0.0018,
        et_weight: float = 0.0,
        max_goals: int = 15,
        min_matches: int = 500,
        importance: dict[str, float] | None = None,
    ):
        self.window_years = window_years
        self.xi = xi
        self.et_weight = et_weight
        self.max_goals = max_goals
        self.min_matches = min_matches
        # S1: when set, multiplies xi-decay weights by tournament importance
        # (same dict shape as BaselineElo's `importance`). None == off, math
        # is byte-identical to the base — existing golden tests pin that.
        self.importance = importance
        self._pb: DixonColesGoalModel | None = None
        self._teams: set[str] = set()
        self.trained_through: pd.Timestamp | None = None
        self.fitted_window_years: float | None = None
        self.n_training_matches: int = 0

    def hyperparams(self) -> dict[str, Any]:
        hp: dict[str, Any] = {
            "window_years": self.window_years,
            "xi": self.xi,
            "et_weight": self.et_weight,
            "max_goals": self.max_goals,
            "min_matches": self.min_matches,
        }
        # S1: only include `importance` in the hash when set, so plain DC's
        # hyperparams_hash stays byte-identical to pre-S1 rows. Subclasses
        # that opt into tournament weighting get their own distinct hash.
        if self.importance is not None:
            hp["importance"] = self.importance
        return hp

    # ---------------------------------------------------------------- train

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        """Batch MLE refit on a trailing window of the as-of frame."""
        required = _REQUIRED_COLS + (_IMPORTANCE_REQUIRED_COLS if self.importance else ())
        missing = [c for c in required if c not in matches.columns]
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
                # np.array(copy=True): parquet-backed columns yield read-only
                # views, which penaltyblog's Cython internals reject.
                pb = DixonColesGoalModel(
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
            "Dixon-Coles fit failed at every window (caller may fall back to "
            f"Elo): {'; '.join(errors)}"
        )

    def _window_schedule(self, matches: pd.DataFrame, anchor) -> list[float | None]:
        """Requested window, doubling until it covers the frame, then the full
        frame (the last resort ignores min_matches)."""
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
        w = np.asarray(w, dtype=float) * np.where(subset["went_to_et"], self.et_weight, 1.0)
        if self.importance is not None:
            # Tournament-stage uplift (S1): WC matches outweigh friendlies, etc.
            # Unmapped tournaments default to 1.0 — neutral, no surprise.
            w = w * subset["tournament"].map(self.importance).fillna(1.0).to_numpy(dtype=float)
        return w

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
