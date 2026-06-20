"""Bayesian model averaging over the fitted roster (S22).

Two entrants, by design (D7):

- `BMAEnsemble` — weights every member that predicts WDL (every model
  in the roster). The hero metric "log-loss" is WDL-shaped, so the most
  inclusive average competes there.
- `BMAGoalsEnsemble` — weights only goals-capable members so it can
  also expose `predict_goals()`. Constructed with a non-goals member it
  raises immediately per D5 — no runtime branching, no surprises.

Weights are derived ONCE at construction from LOTO-averaged log-loss on
`data/backtest.db` (D6): `w_i ∝ exp(-L_i / T)`, T=1, normalized. Models
without a backtest history (e.g. brand-new entrants on their first
nightly) are excluded with a structured note rather than smuggled in
under a uniform weight — over-weighting an unproven member is the
exact failure D5 forbids.

Per D5:
- All members dropped (no backtest history at all) → raise loudly so
  the orchestrator's drop-entrant path catches it.
- Any member dropped from BMAGoalsEnsemble because it isn't a
  `GoalsModel` → raise at construction, not later.
- `predict_*` against an empty member list → raise (never silently
  fall through to a uniform prior).
"""
from __future__ import annotations

import sqlite3
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from fifapreds.loop.score import CLASSES, log_loss
from fifapreds.models.base import WDL, GoalsModel, Model, ScoreGrid

_PROB_COLS = ("p_home", "p_draw", "p_away")


def loto_log_losses(conn: sqlite3.Connection,
                    model_ids: Iterable[str] | None = None,
                    ) -> dict[str, float]:
    """LOTO-averaged log-loss per model_id from a backtest DB.

    For each tournament year present in the DB, computes per-model mean
    log-loss on THAT year; returns the mean across years per model.
    Models absent from the backtest DB are absent from the result.
    """
    rows = pd.read_sql_query(
        """SELECT p.model_id, p.context, p.p_home, p.p_draw, p.p_away,
                  s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE p.context LIKE 'backtest:wc%'""",
        conn,
    )
    if rows.empty:
        return {}
    rows["year"] = rows["context"].str.removeprefix("backtest:wc").astype(int)
    if model_ids is not None:
        rows = rows[rows["model_id"].isin(set(model_ids))]
    out: dict[str, float] = {}
    for model_id, g in rows.groupby("model_id"):
        per_year = []
        for _, gy in g.groupby("year"):
            probs = gy[list(_PROB_COLS)].to_numpy()
            outcomes = gy["outcome"].map(CLASSES.index).to_numpy()
            per_year.append(float(log_loss(probs, outcomes).mean()))
        if per_year:
            out[str(model_id)] = float(np.mean(per_year))
    return out


def _softmax_weights(losses: Mapping[str, float], *, T: float = 1.0) -> dict[str, float]:
    """w_i ∝ exp(-L_i / T), normalized."""
    if not losses:
        return {}
    ids = list(losses.keys())
    vals = np.array([losses[i] for i in ids], dtype=float)
    vals = vals - vals.min()                    # numerical stability
    w = np.exp(-vals / T)
    w = w / w.sum()
    return {i: float(wi) for i, wi in zip(ids, w)}


def _common_init(members: list[Model], weights: dict[str, float] | None,
                 backtest_conn: sqlite3.Connection | None, T: float,
                 ) -> tuple[list[Model], dict[str, float], list[str]]:
    """Resolve member list + weights, return (kept_members, weights, notes).

    Either `weights` (explicit) OR `backtest_conn` (LOTO derive) must be
    supplied. Members without weight metadata get dropped with a note.
    """
    notes: list[str] = []
    if weights is None and backtest_conn is None:
        raise ValueError("BMA: provide either `weights` or `backtest_conn`")
    if weights is None:
        losses = loto_log_losses(backtest_conn, [m.model_id for m in members])
        weights = _softmax_weights(losses, T=T)
        if not losses:
            raise RuntimeError(
                "BMA: backtest DB has no scored predictions for any member "
                f"({[m.model_id for m in members]}); cannot derive weights")

    keep, kept_weights = [], {}
    for m in members:
        if m.model_id in weights and weights[m.model_id] > 0.0:
            keep.append(m)
            kept_weights[m.model_id] = weights[m.model_id]
        else:
            notes.append(f"BMA: dropped {m.model_id} (no backtest weight)")
    if not keep:
        raise RuntimeError(
            "BMA: every candidate member was dropped — backtest history empty?")
    # Renormalize after drops so the kept weights sum to 1 exactly.
    total = sum(kept_weights.values())
    kept_weights = {k: v / total for k, v in kept_weights.items()}
    return keep, kept_weights, notes


def _check_fitted(members: list[Model]) -> list[Model]:
    """Filter to members that actually fit (have a predict-ready state).

    Per D5, ALL members dead → raise. A subset alive → use them; this
    keeps the ensemble useful when one optional dep (e.g. PyMC) is
    missing on a particular run.
    """
    alive = [m for m in members if m.trained_through is not None]
    if not alive:
        raise RuntimeError("BMA: no member is fitted (every member dead)")
    return alive


class BMAEnsemble(Model):
    """Posterior-weighted average over every W/D/L member (D7)."""

    model_id: ClassVar[str] = "bma_ensemble"
    model_version: ClassVar[str] = "1"

    def __init__(self, members: list[Model], *,
                 weights: dict[str, float] | None = None,
                 backtest_conn: sqlite3.Connection | None = None,
                 T: float = 1.0):
        self._T = T
        self._members, self._weights, self.notes = _common_init(
            members, weights, backtest_conn, T,
        )
        # Adopt the most-conservative training cutoff so the publish-time
        # leak guard sees the right date (a fitted ensemble's claim only
        # holds where every member was trained-through that point).
        cutoffs = [m.trained_through for m in self._members
                   if m.trained_through is not None]
        self.trained_through = min(cutoffs) if cutoffs else None

    def fit(self, matches: pd.DataFrame) -> "BMAEnsemble":
        """Per the plan: members come in PRE-FITTED from the orchestrator.
        Calling fit again refits each in turn; the ensemble doesn't add
        its own parameters to learn."""
        for m in self._members:
            m.fit(matches)
        self._members = _check_fitted(self._members)
        cutoffs = [m.trained_through for m in self._members]
        self.trained_through = min(cutoffs) if cutoffs else None
        return self

    def hyperparams(self) -> dict[str, Any]:
        return {
            "members": [m.model_id for m in self._members],
            "weights": {k: round(v, 6) for k, v in self._weights.items()},
            "temperature": self._T,
        }

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        alive = _check_fitted(self._members)
        acc = np.zeros(3)
        for m in alive:
            w = self._weights[m.model_id]
            p = m.predict_wdl(home, away, neutral=neutral).as_array()
            acc += w * p
        # Renormalize against tiny float drift; weights already sum to 1
        # so the constant factor is ~1.0 ± float epsilon.
        acc = acc / acc.sum()
        return WDL(home=float(acc[0]), draw=float(acc[1]), away=float(acc[2]))


class BMAGoalsEnsemble(GoalsModel):
    """Posterior-weighted average over goals-capable members (D7).

    Members MUST all be GoalsModel — passing a W/D/L-only member raises
    at construction, not at first predict_goals call. This is the
    'explicit > clever' choice the eng-review locked in for this entrant.
    """

    model_id: ClassVar[str] = "bma_goals_ensemble"
    model_version: ClassVar[str] = "1"

    def __init__(self, members: list[Model], *,
                 weights: dict[str, float] | None = None,
                 backtest_conn: sqlite3.Connection | None = None,
                 T: float = 1.0):
        non_goals = [m for m in members if not isinstance(m, GoalsModel)]
        if non_goals:
            raise TypeError(
                "BMAGoalsEnsemble: every member must be a GoalsModel; "
                f"got non-goals: {[m.model_id for m in non_goals]}")
        self._T = T
        self._members, self._weights, self.notes = _common_init(
            members, weights, backtest_conn, T,
        )
        cutoffs = [m.trained_through for m in self._members
                   if m.trained_through is not None]
        self.trained_through = min(cutoffs) if cutoffs else None

    def fit(self, matches: pd.DataFrame) -> "BMAGoalsEnsemble":
        for m in self._members:
            m.fit(matches)
        self._members = _check_fitted(self._members)
        cutoffs = [m.trained_through for m in self._members]
        self.trained_through = min(cutoffs) if cutoffs else None
        return self

    def hyperparams(self) -> dict[str, Any]:
        return {
            "members": [m.model_id for m in self._members],
            "weights": {k: round(v, 6) for k, v in self._weights.items()},
            "temperature": self._T,
        }

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        return self.predict_goals(home, away, neutral=neutral).wdl()

    def predict_goals(self, home: str, away: str, *, neutral: bool = False) -> ScoreGrid:
        alive = _check_fitted(self._members)
        # Each member's grid may have a different `max_goals` shape; align
        # to the max across members so we can sum elementwise.
        grids = [
            (m.predict_goals(home, away, neutral=neutral).grid,
             self._weights[m.model_id])
            for m in alive
        ]
        rows = max(g.shape[0] for g, _ in grids)
        cols = max(g.shape[1] for g, _ in grids)
        acc = np.zeros((rows, cols))
        for g, w in grids:
            acc[:g.shape[0], :g.shape[1]] += w * g
        # Per-grid normalization to 1 is enforced by ScoreGrid; the
        # weighted sum may carry small drift, renormalize one more time.
        return ScoreGrid(acc / acc.sum())
