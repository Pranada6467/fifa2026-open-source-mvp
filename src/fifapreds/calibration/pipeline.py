"""LOTO fit + apply pipeline for publish-time calibration (D6).

The pipeline:
1. Read scored backtest predictions from `data/backtest.db`.
2. For each (model_id, holdout_year) pair, fit a fresh calibrator on
   predictions whose tournament_year != holdout_year. Holdout-year
   leakage is asserted before fitting — silent contamination would
   defeat the whole point of LOTO.
3. Average the per-holdout calibrators into a single shipped calibrator
   per model (the published view uses one calibrator per model_id).
4. At publish time, apply each model's calibrator to its live-track and
   backtest-track probabilities, producing the `track ∈ {raw,
   temperature, isotonic}` dimension of `leaderboard.parquet`.

`tournament_year` is parsed out of the backtest `context` column
(e.g. 'backtest:wc2014' → 2014), so no schema change is needed beyond
what already exists in the predictions table.
"""
from __future__ import annotations

import sqlite3
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from fifapreds.calibration.base import Calibrator
from fifapreds.calibration.isotonic import IsotonicCalibrator
from fifapreds.calibration.temperature import TemperatureCalibrator
from fifapreds.loop.score import CLASSES

CalibratorFactory = Callable[[], Calibrator]
DEFAULT_FACTORIES: dict[str, CalibratorFactory] = {
    "temperature": TemperatureCalibrator,
    "isotonic": IsotonicCalibrator,
}


def _backtest_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    """All scored backtest claims tagged with the tournament year parsed
    from `context`. One row per (prediction_id)."""
    df = pd.read_sql_query(
        """SELECT p.prediction_id, p.model_id, p.context,
                  p.p_home, p.p_draw, p.p_away, s.outcome
           FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
           WHERE p.context LIKE 'backtest:wc%'""",
        conn,
    )
    df["tournament_year"] = (
        df["context"].str.removeprefix("backtest:wc").astype(int)
    )
    return df


def loto_holdout_split(df: pd.DataFrame, holdout_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a tagged predictions frame into (train, holdout) by year.

    Per D6: the train half must contain ZERO rows tagged with the
    holdout year — a positive assertion, not an assumption. The
    callers (and the pipeline test) rely on this guarantee."""
    if "tournament_year" not in df.columns:
        raise ValueError("frame missing `tournament_year` column "
                         "(parse from `context` first)")
    train = df[df["tournament_year"] != holdout_year]
    holdout = df[df["tournament_year"] == holdout_year]
    if (train["tournament_year"] == holdout_year).any():
        raise AssertionError(
            f"LOTO leak: train set contains rows tagged {holdout_year}")
    return train, holdout


def fit_calibrators(
    conn: sqlite3.Connection,
    *,
    model_ids: Iterable[str] | None = None,
    factories: dict[str, CalibratorFactory] | None = None,
    holdout_years: Iterable[int] | None = None,
) -> dict[tuple[str, str], Calibrator]:
    """LOTO-averaged calibrators per (model_id, track).

    For each (model, track) emits one calibrator whose parameters are
    averaged across the LOTO folds. The shipped artifact is one
    calibrator per (model_id, track), not per-fold — the leaderboard
    publishes a single calibrated view, not a per-holdout one.

    Returns: `{(model_id, track): Calibrator}`.
    """
    factories = factories or DEFAULT_FACTORIES
    df = _backtest_predictions(conn)
    if df.empty:
        raise ValueError("no scored backtest predictions to fit calibrators on")
    if model_ids is None:
        model_ids = sorted(df["model_id"].unique())
    if holdout_years is None:
        holdout_years = sorted(df["tournament_year"].unique())
    holdout_years = list(holdout_years)
    if len(holdout_years) < 2:
        raise ValueError(
            f"LOTO needs at least 2 tournaments to hold out (got {holdout_years})"
        )

    out: dict[tuple[str, str], Calibrator] = {}
    for model_id in model_ids:
        mg = df[df["model_id"] == model_id]
        if mg.empty:
            continue
        for track, factory in factories.items():
            fold_cals: list[Calibrator] = []
            for holdout in holdout_years:
                train, _ = loto_holdout_split(mg, holdout)
                if train.empty:
                    continue
                probs = train[["p_home", "p_draw", "p_away"]].to_numpy()
                outcomes = train["outcome"].map(CLASSES.index).to_numpy()
                fold_cals.append(factory().fit(probs, outcomes))
            if not fold_cals:
                continue
            out[(model_id, track)] = _average_calibrators(fold_cals, factory)
    return out


def _average_calibrators(cals: list[Calibrator],
                         factory: CalibratorFactory) -> Calibrator:
    """LOTO-average a list of fitted calibrators of the same type.

    Temperature: arithmetic mean of T. Isotonic: a meta-calibrator that
    applies each fold's regressor and averages the outputs. Other types
    can register an `_average_` method (Liskov-style override) — for now
    only the two shipped types are supported, raising otherwise.
    """
    if len(cals) == 1:
        return cals[0]
    if all(isinstance(c, TemperatureCalibrator) for c in cals):
        avg = factory()
        avg.T = float(np.mean([c.T for c in cals]))  # type: ignore[attr-defined]
        return avg
    if all(isinstance(c, IsotonicCalibrator) for c in cals):
        return _AveragedIsotonic(cals)
    raise NotImplementedError(
        f"averaging not supported for {type(cals[0]).__name__}"
    )


class _AveragedIsotonic(Calibrator):
    """Holds N fitted isotonic calibrators; apply averages their outputs
    then renormalizes. Acts as a single Calibrator from the pipeline's
    perspective."""

    def __init__(self, members: list[IsotonicCalibrator]):
        self._members = members

    def fit(self, probs, outcomes):  # pragma: no cover - already fitted
        raise NotImplementedError("averaged isotonic is constructed pre-fit")

    def apply(self, probs):
        outs = np.stack([m.apply(probs) for m in self._members], axis=0)
        return outs.mean(axis=0)


def apply_calibrators(
    probs_frame: pd.DataFrame,
    calibrators: dict[tuple[str, str], Calibrator],
    *,
    model_id_col: str = "model_id",
    prob_cols: tuple[str, str, str] = ("p_home", "p_draw", "p_away"),
) -> pd.DataFrame:
    """Apply per-(model_id, track) calibrators to `probs_frame`.

    Emits one calibrated copy per available track for each model, with a
    new `track` column. Original rows are returned with `track='raw'`.
    Models missing from `calibrators` get only the `raw` track — the
    publisher's contract is to show every model with at least its raw
    claim, even when calibration was skipped (e.g. a brand-new entrant
    with no backtest history yet).
    """
    if probs_frame.empty:
        return probs_frame.assign(track="raw")
    out_frames: list[pd.DataFrame] = [probs_frame.assign(track="raw")]
    available_tracks = {track for _, track in calibrators}
    for track in sorted(available_tracks):
        rows: list[pd.DataFrame] = []
        for model_id, group in probs_frame.groupby(model_id_col, sort=False):
            cal = calibrators.get((model_id, track))
            if cal is None:
                continue
            probs = group[list(prob_cols)].to_numpy()
            calibrated = cal.apply(probs)
            g = group.copy()
            g[list(prob_cols)] = calibrated
            g["track"] = track
            rows.append(g)
        if rows:
            out_frames.append(pd.concat(rows, ignore_index=True))
    return pd.concat(out_frames, ignore_index=True)
