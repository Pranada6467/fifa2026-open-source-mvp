"""BaselineElo — the W/D/L-only floor every other technique must beat.

Incremental Elo over the full match history, with a Davidson draw model so one
rating difference yields a proper three-way probability:

    d        = R_home - R_away + home_advantage·(0 if neutral else 1)
    γ        = 10^(d/400)
    P(home)  = γ / (γ + 1 + ν·√γ)
    P(draw)  = ν·√γ / (γ + 1 + ν·√γ)
    P(away)  = 1 / (γ + 1 + ν·√γ)

ν tunes the even-match draw rate (P(draw) = ν/(2+ν) when d=0; ν=0.6 ≈ 23%, the
international-football base rate). Draw probability peaks for even matches and
decays with mismatch, so all three probabilities stay positive at any gap.

The rating update is self-consistent with the prediction: expected score
E = P(home) + P(draw)/2 from the model's own probabilities, then the usual
    R_home += K·imp·(S − E);  R_away −= K·imp·(S − E)
which is zero-sum by construction. S counts the post-ET result (martj42 scores
include extra time, exclude penalties), so a shootout is a draw — correct for
W/D/L purposes.

Variant knobs (all hashed into provenance; defaults = plain baseline):
- `importance`: tournament-name → K multiplier (e.g. World Cup > friendly).
- `decay_rate`: per-year regression of an idle team's rating toward the
  initial rating, applied when the team next appears in an update. Predictions
  read the as-stored rating (state as of `trained_through`) — fine in a live
  loop where every team is active; documented approximation otherwise.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from fifapreds.models.base import WDL, Model

_REQUIRED_COLS = ("date", "home_team", "away_team", "home_score", "away_score", "neutral")


class BaselineElo(Model):
    model_id = "elo_baseline"
    model_version = "1"

    def __init__(
        self,
        *,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
        draw_nu: float = 0.6,
        initial_rating: float = 1500.0,
        decay_rate: float = 0.0,
        importance: dict[str, float] | None = None,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.draw_nu = draw_nu
        self.initial_rating = initial_rating
        self.decay_rate = decay_rate
        self.importance = importance
        self._ratings: dict[str, float] = {}
        self._last_played: dict[str, pd.Timestamp] = {}
        self.trained_through: pd.Timestamp | None = None

    def hyperparams(self) -> dict[str, Any]:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "draw_nu": self.draw_nu,
            "initial_rating": self.initial_rating,
            "decay_rate": self.decay_rate,
            "importance": self.importance,
        }

    # ---------------------------------------------------------------- train

    def fit(self, matches: pd.DataFrame) -> "BaselineElo":
        """Replay played matches in time order (state is reset first)."""
        missing = [c for c in _REQUIRED_COLS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches frame missing columns: {missing}")
        self._ratings = {}
        self._last_played = {}
        self.trained_through = None
        ordered = matches.sort_values("date", kind="stable")
        for m in ordered.itertuples(index=False):
            self._apply(
                pd.Timestamp(m.date), m.home_team, m.away_team,
                m.home_score, m.away_score, bool(m.neutral),
                getattr(m, "tournament", None),
            )
        return self

    def update(self, match: Mapping[str, Any] | pd.Series) -> None:
        """Incorporate one new result (the live-loop UPDATE step)."""
        date = pd.Timestamp(match["date"])
        if self.trained_through is not None and date < self.trained_through:
            raise ValueError(
                f"out-of-order update: match dated {date.date()} but model already "
                f"trained through {self.trained_through.date()}; refit instead"
            )
        self._apply(
            date, match["home_team"], match["away_team"],
            match["home_score"], match["away_score"], bool(match["neutral"]),
            match.get("tournament"),
        )

    def _apply(self, date, home, away, home_score, away_score, neutral, tournament):
        if pd.isna(home_score) or pd.isna(away_score):
            raise ValueError(f"unplayed match in training data: {home} v {away} on {date}")
        r_home = self._effective(home, date)
        r_away = self._effective(away, date)
        expected = self._probs(r_home, r_away, neutral)
        e = expected.home + 0.5 * expected.draw
        s = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
        k = self.k_factor * (self.importance or {}).get(tournament, 1.0)
        delta = k * (s - e)
        self._ratings[home] = r_home + delta
        self._ratings[away] = r_away - delta
        self._last_played[home] = date
        self._last_played[away] = date
        self.trained_through = date

    def _effective(self, team: str, date: pd.Timestamp) -> float:
        """Pre-match rating: stored rating, regressed toward initial if idle."""
        r = self._ratings.get(team, self.initial_rating)
        last = self._last_played.get(team)
        if self.decay_rate > 0.0 and last is not None and date > last:
            years_idle = (date - last).days / 365.25
            r = self.initial_rating + (r - self.initial_rating) * math.exp(
                -self.decay_rate * years_idle
            )
        return r

    # -------------------------------------------------------------- predict

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        for team in (home, away):
            if team not in self._ratings:
                raise KeyError(
                    f"unknown team {team!r} — not in training history; "
                    "check registry.canonical() spelling"
                )
        return self._probs(self._ratings[home], self._ratings[away], neutral)

    def _probs(self, r_home: float, r_away: float, neutral: bool) -> WDL:
        d = r_home - r_away + (0.0 if neutral else self.home_advantage)
        gamma = 10.0 ** (d / 400.0)
        root = math.sqrt(gamma)
        denom = gamma + 1.0 + self.draw_nu * root
        return WDL(home=gamma / denom, draw=self.draw_nu * root / denom, away=1.0 / denom)

    # ---------------------------------------------------------------- state

    def rating(self, team: str) -> float:
        """Stored rating as of `trained_through` (raises for unseen teams)."""
        return self._ratings[team]

    @property
    def ratings(self) -> dict[str, float]:
        """Copy of all ratings — feed for rating_snapshots (T5)."""
        return dict(self._ratings)
