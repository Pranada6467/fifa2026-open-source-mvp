"""MarketBlend (T12) — de-vigged bookmaker consensus blended with a model.

Why the market is fair game: bookmaker odds are *pre-match public information*,
snapshotted before kickoff (loop/odds.py stores the raw payload with a capture
timestamp). Using them is not result leakage — the no-lookahead invariant
guards against models seeing *results* they shouldn't, and an odds quote
contains no result. The efficient-market literature says closing odds are the
strongest single forecast available, so a heavy market weight with a model
correction is the natural Block 2 entrant: if our models carry signal the
market misses, the blend beats both parents on out-of-sample log-loss; if not,
the leaderboard says so honestly.

De-vig: bookmaker prices overstate probabilities (the overround/vig). We use
penaltyblog's **power** method — raise the naive inverse odds to a common
exponent chosen so they sum to 1. Unlike proportional (multiplicative)
scaling, power removes margin disproportionately from longshots, correcting
the documented favourite–longshot bias bookmakers price in. Shin's method is
similarly principled (models insider trading) and gave near-identical numbers
on the real snapshot, but penaltyblog's power output sums to 1.0 exactly while
Shin leaves ~1e-13 residual; power is the cleaner primitive. Per fixture we
de-vig each bookmaker separately, take the **median** across bookmakers (robust
to one stale/outlier book), and renormalize to sum exactly 1.0.

Provenance: the snapshot used is recorded per prediction row via the
`odds_snapshot_id` column (pass `MarketBlend.snapshot_id` to `log_prediction`).
It is deliberately NOT part of `hyperparams()` — the snapshot changes every
run (row provenance), while the hash pins the frozen *config* (blend weight +
base model identity).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping

import numpy as np
import pandas as pd
from penaltyblog.implied import calculate_implied

from fifapreds.models.base import WDL, Model
from fifapreds.registry import canonical

DEVIG_METHOD = "power"

_PARSE_COLS = ["home_team", "away_team", "kickoff",
               "p_home", "p_draw", "p_away", "n_bookmakers"]


def parse_h2h_snapshot(raw_json: str) -> pd.DataFrame:
    """Parse one The Odds API h2h payload into de-vigged consensus probabilities.

    Returns one row per fixture: home_team, away_team (registry-canonical),
    kickoff (Timestamp from commence_time), p_home/p_draw/p_away (power-de-vigged
    per bookmaker, median across bookmakers, renormalized to sum exactly 1),
    n_bookmakers (how many complete books fed the median). Bookmakers missing
    any of the three prices are skipped; fixtures with no complete book are
    dropped — partial coverage is normal, not an error.
    """
    events = json.loads(raw_json)
    if not isinstance(events, list):
        raise ValueError(f"unexpected h2h payload: {type(events).__name__}, not a list")
    rows: list[dict[str, Any]] = []
    for ev in events:
        raw_home, raw_away = ev["home_team"], ev["away_team"]
        per_book: list[np.ndarray] = []
        for bm in ev.get("bookmakers", []):
            prices = _h2h_prices(bm, raw_home, raw_away)
            if prices is None:
                continue
            devig = calculate_implied(list(prices), method=DEVIG_METHOD).probabilities
            per_book.append(np.asarray(devig, dtype=float))
        if not per_book:
            continue
        consensus = np.median(np.vstack(per_book), axis=0)
        consensus = consensus / consensus.sum()
        rows.append({
            "home_team": canonical(raw_home),
            "away_team": canonical(raw_away),
            "kickoff": pd.Timestamp(ev["commence_time"]),
            "p_home": consensus[0],
            "p_draw": consensus[1],
            "p_away": consensus[2],
            "n_bookmakers": len(per_book),
        })
    return pd.DataFrame(rows, columns=_PARSE_COLS)


def _h2h_prices(bookmaker: Mapping[str, Any], raw_home: str, raw_away: str
                ) -> tuple[float, float, float] | None:
    """(home, draw, away) decimal prices from one bookmaker's h2h market, or
    None when the 3-way market is missing, incomplete, or malformed."""
    market = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
    if market is None:
        return None
    prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
    triple = tuple(prices.get(name) for name in (raw_home, "Draw", raw_away))
    if any(not isinstance(p, (int, float)) or not np.isfinite(p) or p <= 1.0
           for p in triple):
        return None
    return triple  # type: ignore[return-value]


def latest_h2h_probs(conn: sqlite3.Connection) -> tuple[int, pd.DataFrame]:
    """Most recent h2h snapshot, parsed: (snapshot_id, consensus frame).

    The id is what callers pass to `log_prediction(odds_snapshot_id=...)` so
    every market-informed row points at the exact payload behind it.
    """
    try:
        row = conn.execute(
            "SELECT snapshot_id, raw_json FROM odds_snapshots "
            "WHERE market = 'h2h' ORDER BY captured_at DESC, snapshot_id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is None:
        raise LookupError(
            "no h2h snapshot in odds_snapshots — run `python -m fifapreds.loop.odds` first"
        )
    return int(row[0]), parse_h2h_snapshot(row[1])


def _market_lookup(
    market: pd.DataFrame | Mapping[tuple[str, str], Any],
) -> dict[tuple[str, str], np.ndarray]:
    """Normalize parse output (or a hand mapping) into {(home, away): [h, d, a]}.

    Keys pass through registry.canonical() so a mapping built from raw feed
    names ("USA") still matches canonical fixture names at predict time.
    """
    if isinstance(market, pd.DataFrame):
        items: Any = (((r.home_team, r.away_team), (r.p_home, r.p_draw, r.p_away))
                      for r in market.itertuples(index=False))
    else:
        items = market.items()
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for (home, away), probs in items:
        p = probs.as_array() if isinstance(probs, WDL) else np.asarray(probs, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f"bad market probabilities for {(home, away)}: {probs}")
        lookup[(canonical(home), canonical(away))] = p / p.sum()
    return lookup


class MarketBlend(Model):
    """blend_weight·market + (1 − blend_weight)·base, per fixture.

    Fixtures the market doesn't quote fall back to pure base probabilities —
    the feed only covers near-term matches, so missing coverage is normal
    operation, never an error. The base model also supplies `trained_through`
    and is the only fitted component; the market frame is a frozen snapshot.
    """

    model_id = "market_blend"
    model_version = "1"

    def __init__(
        self,
        base: Model,
        market: pd.DataFrame | Mapping[tuple[str, str], Any],
        blend_weight: float = 0.75,
        snapshot_id: int | None = None,
    ):
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}")
        self.base = base
        self.blend_weight = float(blend_weight)
        #: provenance for the caller: pass as log_prediction(odds_snapshot_id=...)
        self.snapshot_id = snapshot_id
        self._market = _market_lookup(market)

    @property
    def trained_through(self) -> pd.Timestamp | None:  # type: ignore[override]
        """The base model's match-history cutoff.

        The write-time lookahead guard exists to prove predictions never saw
        *results* at/after kickoff — that claim rests entirely on the base
        model's training data, so we delegate. Market odds are pre-match
        public information with no result content; their capture time is
        audited separately through the odds_snapshot_id provenance column.
        """
        return self.base.trained_through

    def hyperparams(self) -> dict[str, Any]:
        """Frozen config: blend weight + the full nested base config, so the
        hash uniquely pins which (weight, base model) produced a row.
        snapshot_id is excluded — it is per-run row provenance, not config."""
        return {
            "blend_weight": self.blend_weight,
            "base": {
                "model_id": self.base.model_id,
                "model_version": self.base.model_version,
                "hyperparams": self.base.hyperparams(),
            },
        }

    def fit(self, matches: pd.DataFrame) -> "MarketBlend":
        """Delegate to the base model (the market frame needs no fitting) so
        the orchestrator can treat every leaderboard entrant uniformly."""
        self.base.fit(matches)
        return self

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        base = self.base.predict_wdl(home, away, neutral=neutral)
        market = self._market.get((canonical(home), canonical(away)))
        if market is None:
            return base
        p = self.blend_weight * market + (1.0 - self.blend_weight) * base.as_array()
        p = p / p.sum()
        return WDL(home=float(p[0]), draw=float(p[1]), away=float(p[2]))
