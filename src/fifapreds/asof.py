"""Point-in-time data access — the no-lookahead spine.

Every model reads match history through `MatchStore.before(ts)`, which returns
only *played* matches strictly before `ts`. The historical backtest and the live
loop share this one path, so a prediction can never see its own result or any
future result. The guarantee is enforced by tests/test_asof.py.

Same-day note: international teams play at most once per day, so a strict
`date < ts` cutoff is leak-free for the two teams involved in any fixture. When
an explicit kickoff timestamp is available (e.g. from the odds feed) it can be
passed as `ts` for finer resolution.
"""
from __future__ import annotations

import pandas as pd

from fifapreds.ingest import load_matches


class MatchStore:
    def __init__(self, matches: pd.DataFrame | None = None):
        m = load_matches() if matches is None else matches
        self._all = m
        # Only played matches are usable as history, ordered in time.
        self._played = (
            m[m["is_played"]]
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

    @property
    def played(self) -> pd.DataFrame:
        return self._played

    @property
    def all(self) -> pd.DataFrame:
        return self._all

    def before(self, ts) -> pd.DataFrame:
        """Played matches strictly before `ts` (the no-lookahead read)."""
        ts = pd.Timestamp(ts)
        return self._played[self._played["date"] < ts]

    def upcoming(self, start=None, end=None) -> pd.DataFrame:
        """Not-yet-played fixtures, optionally within [start, end)."""
        fx = self._all[~self._all["is_played"]]
        if start is not None:
            fx = fx[fx["date"] >= pd.Timestamp(start)]
        if end is not None:
            fx = fx[fx["date"] < pd.Timestamp(end)]
        return fx.sort_values("date", kind="stable").reset_index(drop=True)
