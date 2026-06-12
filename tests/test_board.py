"""Board helpers (E2): the verdict copy is computed, honest at small n, and
the freshness/next-update rules match the nightly cadence. These are golden
tests — the public board's headline sentence must never drift silently.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from fifapreds.publish.board import (
    is_stale,
    next_nightly_utc,
    pooled_confidence_bin,
    track_of,
    verdict_sentence,
)


def _calibration(track: str, rows: list[tuple[float, float, int]]) -> pd.DataFrame:
    """rows = [(p_mean, freq, n), ...] all landing in one track."""
    return pd.DataFrame([
        {"model_id": "m", "track": track, "bin_lo": 0.0, "bin_hi": 1.0,
         "p_mean": p, "freq": f, "n": n}
        for p, f, n in rows
    ])


def test_verdict_golden_sentence():
    cal = _calibration("backtest", [(0.70, 0.68, 100), (0.30, 0.35, 50)])
    # Only the 0.70 bin is in the confidence window; n=100 >= threshold.
    assert verdict_sentence(cal, "backtest") == (
        "When this system says ~70%, it has happened 68% of the time "
        "(backtest 2014/18/22, n=100 claims)."
    )


def test_verdict_pools_bins_weighted():
    cal = _calibration("live", [(0.60, 0.60, 30), (0.80, 0.70, 10)])
    pooled = pooled_confidence_bin(cal, "live")
    assert pooled["n"] == 40
    assert pooled["claimed"] == pytest.approx((0.60 * 30 + 0.80 * 10) / 40)


def test_verdict_hedges_at_small_n():
    cal = _calibration("live", [(0.70, 1.0, 4)])
    sentence = verdict_sentence(cal, "live")
    assert sentence.startswith("Too early to judge the live 2026 record")
    assert "4 confident claims" in sentence


def test_verdict_none_when_track_absent():
    cal = _calibration("backtest", [(0.70, 0.68, 100)])
    assert verdict_sentence(cal, "live") is None
    assert verdict_sentence(pd.DataFrame(), "backtest") is None
    assert verdict_sentence(None, "backtest") is None


def test_next_nightly_utc_wraps_correctly():
    before = datetime(2026, 6, 12, 5, 0, tzinfo=timezone.utc)
    after = datetime(2026, 6, 12, 7, 0, tzinfo=timezone.utc)
    assert next_nightly_utc(before) == datetime(2026, 6, 12, 6, 30, tzinfo=timezone.utc)
    assert next_nightly_utc(after) == datetime(2026, 6, 13, 6, 30, tzinfo=timezone.utc)


def test_is_stale_threshold():
    now = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)
    assert not is_stale("2026-06-12T06:35:00+00:00", now)         # this morning
    assert is_stale("2026-06-10T06:35:00+00:00", now)             # missed a run
    assert not is_stale("2026-06-11T06:35:00", now)               # naive, 29.4h


def test_track_of():
    assert track_of("live") == "live"
    assert track_of("backtest:wc2014") == "backtest"
