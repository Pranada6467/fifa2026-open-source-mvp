"""Hierarchical Poisson model — Bayesian attack/defense with confederation pooling.

PyMC implementation of the Baio & Blangiardo (2010) hierarchical model:

    home_goals ~ Poisson(exp(intercept + home_adv + attack[home] + defense[away]))
    away_goals ~ Poisson(exp(intercept + attack[away] + defense[home]))

where attack/defense parameters are partially pooled by confederation:

    attack[team]  ~ Normal(mu_att[confed[team]], sigma_att)
    defense[team] ~ Normal(mu_def[confed[team]], sigma_def)

Confederation-level priors let the model borrow strength across teams with
thin schedules (the documented DC overconfidence at the extremes for cross-
confederation matchups). A team with 10 matches still inherits its
confederation's base rate, which stabilizes the tails.

Determinism contract (eng review T1): `random_seed` pins the NUTS sampler;
predictions use posterior-mean parameters → deterministic score grids given
(seed, data). The sim downstream is seeded separately.

Optional dependency: PyMC + nutpie. If not installed, importing this module
raises ImportError — the roster skips the entrant, the orchestrator's
_fit_roster drops it gracefully.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import pymc as pm

from fifapreds.models.base import WDL, GoalsModel, ScoreGrid
from fifapreds.models.confederations import team_confederation

_REQUIRED_COLS = ("date", "home_team", "away_team", "home_score", "away_score",
                  "neutral", "went_to_et")


class HierarchicalPoisson(GoalsModel):
    model_id = "hierarchical_poisson"
    model_version = "1"

    def __init__(
        self,
        *,
        window_years: float = 4.0,
        et_weight: float = 0.0,
        max_goals: int = 10,
        min_matches: int = 500,
        random_seed: int = 42,
        n_samples: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.9,
        confed_map: dict[str, str] | None = None,
    ):
        self.window_years = window_years
        self.et_weight = et_weight
        self.max_goals = max_goals
        self.min_matches = min_matches
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.target_accept = target_accept
        self._confed_map = confed_map

        self.trained_through: pd.Timestamp | None = None
        self.fitted_window_years: float | None = None
        self.n_training_matches: int = 0

        self._teams: list[str] = []
        self._team_idx: dict[str, int] = {}
        self._attack: np.ndarray | None = None
        self._defense: np.ndarray | None = None
        self._intercept: float | None = None
        self._home_adv: float | None = None

    def hyperparams(self) -> dict[str, Any]:
        return {
            "window_years": self.window_years,
            "et_weight": self.et_weight,
            "max_goals": self.max_goals,
            "min_matches": self.min_matches,
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "target_accept": self.target_accept,
        }

    # ------------------------------------------------------------------ fit

    def fit(self, matches: pd.DataFrame) -> "HierarchicalPoisson":
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
            subset = self._apply_et_weight(subset)
            if subset is None or (window is not None and len(subset) < self.min_matches):
                errors.append(f"{window or 'full'}y: only {len(subset) if subset is not None else 0} matches")
                continue
            try:
                self._fit_pymc(subset)
            except Exception as exc:
                errors.append(f"{window or 'full'}: {exc}")
                continue
            self.trained_through = anchor
            self.fitted_window_years = window
            self.n_training_matches = len(subset)
            return self

        raise RuntimeError(
            "HierarchicalPoisson fit failed at every window (caller may fall "
            f"back): {'; '.join(errors)}"
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

    def _apply_et_weight(self, subset: pd.DataFrame) -> pd.DataFrame | None:
        if self.et_weight == 0.0:
            subset = subset[~subset["went_to_et"]]
        if subset.empty:
            return None
        return subset

    def _fit_pymc(self, df: pd.DataFrame) -> None:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        if self._confed_map is not None:
            confeds = [self._confed_map[t] for t in teams]
        else:
            confeds = [team_confederation(t) for t in teams]
        confed_names = sorted(set(confeds))
        confed_to_idx = {c: i for i, c in enumerate(confed_names)}
        team_confed_idx = np.array([confed_to_idx[c] for c in confeds])

        home_idx = np.array([team_to_idx[t] for t in df["home_team"]])
        away_idx = np.array([team_to_idx[t] for t in df["away_team"]])
        home_goals = df["home_score"].values.astype(int)
        away_goals = df["away_score"].values.astype(int)
        is_neutral = df["neutral"].values.astype(int)

        with pm.Model():
            intercept = pm.Normal("intercept", mu=0.3, sigma=0.5)
            home_adv = pm.Normal("home_adv", mu=0.25, sigma=0.25)

            mu_att = pm.Normal("mu_att", mu=0, sigma=0.5, shape=len(confed_names))
            mu_def = pm.Normal("mu_def", mu=0, sigma=0.5, shape=len(confed_names))
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

            attack_raw = pm.Normal("attack_raw", mu=0, sigma=1, shape=n_teams)
            defense_raw = pm.Normal("defense_raw", mu=0, sigma=1, shape=n_teams)

            attack = pm.Deterministic(
                "attack",
                mu_att[team_confed_idx] + sigma_att * attack_raw,
            )
            defense = pm.Deterministic(
                "defense",
                mu_def[team_confed_idx] + sigma_def * defense_raw,
            )

            home_advantage = home_adv * (1 - is_neutral)

            log_lambda_home = intercept + home_advantage + attack[home_idx] + defense[away_idx]
            log_lambda_away = intercept + attack[away_idx] + defense[home_idx]

            pm.Poisson("home_obs", mu=pm.math.exp(log_lambda_home), observed=home_goals)
            pm.Poisson("away_obs", mu=pm.math.exp(log_lambda_away), observed=away_goals)

            trace = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                random_seed=self.random_seed,
                target_accept=self.target_accept,
                progressbar=False,
                nuts_sampler="nutpie",
            )

        self._teams = teams
        self._team_idx = team_to_idx
        self._attack = trace.posterior["attack"].mean(dim=["chain", "draw"]).values
        self._defense = trace.posterior["defense"].mean(dim=["chain", "draw"]).values
        self._intercept = float(trace.posterior["intercept"].mean())
        self._home_adv = float(trace.posterior["home_adv"].mean())

    # --------------------------------------------------------------- predict

    def predict_goals(self, home: str, away: str, *, neutral: bool = False) -> ScoreGrid:
        if self._attack is None:
            raise RuntimeError("model is not fitted")
        for team in (home, away):
            if team not in self._team_idx:
                raise KeyError(
                    f"unknown team {team!r} — not in the fitted window "
                    f"({self.fitted_window_years or 'full'}y); "
                    "check registry.canonical() spelling"
                )
        hi = self._team_idx[home]
        ai = self._team_idx[away]

        home_advantage = self._home_adv * (0 if neutral else 1)
        lam_home = np.exp(self._intercept + home_advantage + self._attack[hi] + self._defense[ai])
        lam_away = np.exp(self._intercept + self._attack[ai] + self._defense[hi])

        g = self.max_goals + 1
        home_probs = np.array([_poisson_pmf(k, lam_home) for k in range(g)])
        away_probs = np.array([_poisson_pmf(k, lam_away) for k in range(g)])
        grid = np.outer(home_probs, away_probs)
        return ScoreGrid(grid)

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        return self.predict_goals(home, away, neutral=neutral).wdl()


def _poisson_pmf(k: int, lam: float) -> float:
    from math import exp, factorial
    return exp(-lam) * lam**k / factorial(k)
