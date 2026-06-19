"""Pure helpers behind the public board (E2) — kept out of the Streamlit
script so the verdict copy and freshness rules are unit-testable.

The viewer must never hardcode a verdict: every sentence here is computed
from the published artifacts, and degrades to an honest "too early to judge"
at small n rather than overclaiming (the D3 copy rule).
"""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

UNIFORM_LOG_LOSS = math.log(3)
# The nightly Action runs at 06:30 UTC (.github/workflows/nightly.yml);
# grading/refresh promises in the UI are derived from this, not hardcoded ad hoc.
NIGHTLY_UTC = time(6, 30)
STALE_AFTER_HOURS = 36
MIN_VERDICT_N = 20   # below this, claimed-vs-happened is noise, say so
DIM_BIN_N = 5        # calibration bins thinner than this render dimmed
DEFAULT_CALIBRATION = "raw"  # D2/DD2: default for legacy callers; viewer overrides


def pooled_confidence_bin(
    calibration: pd.DataFrame, track: str, lo: float = 0.55, hi: float = 0.85,
    *, calibration_track: str = DEFAULT_CALIBRATION,
) -> dict | None:
    """Claimed-vs-happened pooled over the mid-to-high-confidence bins of one
    track, weighted by bin population. None when the track has no such claims.

    `calibration_track` (Phase 4 / DD2) filters by the recalibration variant
    ∈ {raw, temperature, isotonic}; the column is added by the publisher.
    When the column is missing (old artifact predating Phase 4), the filter
    is skipped — keeps the helper backward compatible during the transition."""
    if calibration is None or calibration.empty or "track" not in calibration:
        return None
    sub = calibration[(calibration["track"] == track) & calibration["p_mean"].notna()]
    if "calibration" in sub.columns:
        sub = sub[sub["calibration"] == calibration_track]
    sub = sub[(sub["p_mean"] >= lo) & (sub["p_mean"] <= hi) & (sub["n"] > 0)]
    if sub.empty:
        return None
    n = int(sub["n"].sum())
    return {
        "claimed": float(np.average(sub["p_mean"], weights=sub["n"])),
        "freq": float(np.average(sub["freq"], weights=sub["n"])),
        "n": n,
    }


def verdict_sentence(calibration: pd.DataFrame, track: str = "backtest",
                     *, calibration_track: str = DEFAULT_CALIBRATION) -> str | None:
    """The hero's one-line verdict, or None when the track has no claims at all.
    Small samples get an explicitly hedged sentence instead of a confident one.

    Passes `calibration_track` through so the viewer toggle (DD2) can swap
    the underlying numbers without re-emitting the artifact."""
    pooled = pooled_confidence_bin(calibration, track,
                                   calibration_track=calibration_track)
    if pooled is None:
        return None
    label = "backtest 2014/18/22" if track == "backtest" else "live 2026"
    if pooled["n"] < MIN_VERDICT_N:
        return (f"Too early to judge the {label} record — only {pooled['n']} "
                f"confident claims graded so far.")
    return (f"When this system says ~{pooled['claimed']:.0%}, it has happened "
            f"{pooled['freq']:.0%} of the time ({label}, n={pooled['n']} claims).")


def next_nightly_utc(now: datetime | None = None) -> datetime:
    """The next 06:30 UTC after `now` — when the board's numbers move next."""
    now = now or datetime.now(timezone.utc)
    candidate = now.replace(hour=NIGHTLY_UTC.hour, minute=NIGHTLY_UTC.minute,
                            second=0, microsecond=0)
    return candidate if candidate > now else candidate + timedelta(days=1)


def is_stale(generated_at_iso: str, now: datetime | None = None,
             hours: float = STALE_AFTER_HOURS) -> bool:
    """True when the artifacts predate the last expected nightly refresh —
    the viewer shows a trust banner instead of silently serving old numbers."""
    now = now or datetime.now(timezone.utc)
    generated = datetime.fromisoformat(generated_at_iso)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return (now - generated) > timedelta(hours=hours)


def track_of(context: str) -> str:
    """Collapse prediction contexts to the board's two tracks."""
    return "live" if context == "live" else "backtest"
