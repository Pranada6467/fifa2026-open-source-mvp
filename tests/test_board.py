"""Board helpers (E2): the verdict copy is computed, honest at small n, and
the freshness/next-update rules match the nightly cadence. These are golden
tests — the public board's headline sentence must never drift silently.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fifapreds.publish.board import (
    is_stale,
    modal_scoreline_from_grid,
    modal_scoreline_label,
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


# ----------------------------------------------------- DD2: calibration filter

def _multi_calibration(rows: list[tuple[str, str, float, float, int]]) -> pd.DataFrame:
    """rows = [(track, calibration, p_mean, freq, n), ...] — Phase 4 shape."""
    return pd.DataFrame([
        {"model_id": "m", "track": t, "calibration": c, "bin_lo": 0.0,
         "bin_hi": 1.0, "p_mean": p, "freq": f, "n": n}
        for t, c, p, f, n in rows
    ])


def test_verdict_filters_by_calibration_track_when_column_present():
    """Same track, two calibration variants — verdict must reflect the
    selected one, not pool across them."""
    cal = _multi_calibration([
        ("backtest", "raw", 0.70, 0.55, 100),
        ("backtest", "isotonic", 0.70, 0.68, 100),
    ])
    raw = verdict_sentence(cal, "backtest", calibration_track="raw")
    iso = verdict_sentence(cal, "backtest", calibration_track="isotonic")
    assert "55%" in raw and "68%" in iso
    assert raw != iso  # the toggle must actually change the headline


def test_verdict_calibration_filter_misses_emit_none():
    cal = _multi_calibration([("backtest", "raw", 0.70, 0.68, 100)])
    # No isotonic row exists; the toggle should yield None (no confident
    # claims) rather than silently fall through to a different track.
    assert verdict_sentence(cal, "backtest", calibration_track="isotonic") is None


def test_verdict_calibration_filter_skipped_when_column_absent():
    """Old artifact (pre-Phase 4, no `calibration` column): the filter is
    a no-op so legacy data still renders a verdict."""
    cal = _calibration("backtest", [(0.70, 0.68, 100)])
    assert "calibration" not in cal.columns
    sentence = verdict_sentence(cal, "backtest", calibration_track="isotonic")
    assert sentence is not None and "68%" in sentence


# ----------------------------------------------- Item 11: modal scoreline

def _delta_grid(rows: int, cols: int, h: int, a: int) -> np.ndarray:
    grid = np.zeros((rows, cols))
    grid[h, a] = 1.0
    return grid


def test_modal_scoreline_on_point_mass():
    """Pure point mass at (2, 1): E[h]=2, E[a]=1 → modal = (2, 1)."""
    grid = _delta_grid(5, 5, 2, 1)
    assert modal_scoreline_from_grid(grid) == (2, 1)


def test_modal_scoreline_rounds_mean_not_argmax():
    """Half mass on (0, 0), half on (4, 2): argmax is tied at first cell,
    but the modal-by-mean is (2, 1) — the rounded posterior mean. This
    is the property that makes modal useful where argmax is invariant."""
    grid = np.zeros((5, 5))
    grid[0, 0] = 0.5
    grid[4, 2] = 0.5
    assert modal_scoreline_from_grid(grid) == (2, 1)


def test_modal_scoreline_on_dixoncoles_shaped_grid():
    """Synthetic DC-shaped grid centred on 1.3 home goals / 0.9 away
    goals; rounded mean should land on (1, 1)."""
    rng = np.random.default_rng(0)
    grid = np.zeros((6, 6))
    for h in range(6):
        for a in range(6):
            # Tail toward (1.3, 0.9).
            grid[h, a] = np.exp(-((h - 1.3) ** 2 + (a - 0.9) ** 2))
    grid = grid / grid.sum()
    assert modal_scoreline_from_grid(grid) == (1, 1)


def test_modal_scoreline_label_two_lines_with_audit_reference():
    bold, explainer = modal_scoreline_label(
        modal_h=1, modal_a=1, home="Argentina", away="Algeria",
        top1_h=1, top1_a=0, top1_p=0.18,
    )
    assert bold == "Argentina **1–1** Algeria"
    assert "(1–0, 18%)" in explainer
    assert "audit" in explainer


def test_modal_scoreline_label_drops_audit_clause_when_no_top1():
    bold, explainer = modal_scoreline_label(0, 0, "A", "B")
    assert bold == "A **0–0** B"
    assert "audit" not in explainer
    assert "E[goals]" in explainer


# ----------------------------------------------- DD4: divergence banner

def _cal_bin(track: str, calibration: str, bin_lo: float, bin_hi: float,
             p_mean: float, freq: float, n: int, ci_lo: float, ci_hi: float):
    return {"model_id": "m", "track": track, "calibration": calibration,
            "bin_lo": bin_lo, "bin_hi": bin_hi, "n": n,
            "p_mean": p_mean, "freq": freq, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _banner(**kw):
    """Convenience: invoke divergence_banner with the standard DD4 args."""
    from fifapreds.publish.board import divergence_banner
    return divergence_banner(
        kw.pop("cal"),
        calibration_track=kw.pop("calibration_track", "isotonic"),
        weights_refit_date="2026-06-19",
        **kw,
    )


def test_divergence_banner_silent_when_no_divergence():
    """Claimed probability sits inside the Wilson interval — no banner."""
    cal = _df([_cal_bin("live", "isotonic", 0.2, 0.3, p_mean=0.25,
                        freq=0.27, n=40, ci_lo=0.18, ci_hi=0.40)])
    assert _banner(cal=cal) is None


def test_divergence_banner_fires_when_claim_below_wilson_lower():
    """Claim of 23% but live observed 41% with ci_lo 0.30 — claim sits
    below the lower bound, the bin is statistically miscalibrated."""
    cal = _df([_cal_bin("live", "isotonic", 0.2, 0.3, p_mean=0.23,
                        freq=0.41, n=40, ci_lo=0.30, ci_hi=0.55)])
    msg = _banner(cal=cal)
    assert msg is not None
    assert "20%–30%" in msg
    assert "claimed 23%" in msg and "observed 41%" in msg
    assert "2026-06-19" in msg


def test_divergence_banner_fires_when_claim_above_wilson_upper():
    cal = _df([_cal_bin("live", "isotonic", 0.7, 0.8, p_mean=0.78,
                        freq=0.40, n=35, ci_lo=0.25, ci_hi=0.55)])
    msg = _banner(cal=cal)
    assert msg is not None
    assert "70%–80%" in msg


def test_divergence_banner_picks_worst_bin_when_multiple_diverge():
    """Two divergent bins: surface the one with the larger |freq − p_mean|."""
    cal = _df([
        _cal_bin("live", "isotonic", 0.2, 0.3, p_mean=0.25, freq=0.40,
                 n=40, ci_lo=0.30, ci_hi=0.55),  # gap 0.15
        _cal_bin("live", "isotonic", 0.5, 0.6, p_mean=0.55, freq=0.85,
                 n=30, ci_lo=0.70, ci_hi=0.95),  # gap 0.30 — worst
    ])
    msg = _banner(cal=cal)
    assert msg is not None
    assert "50%–60%" in msg, msg


def test_divergence_banner_ignores_other_calibration_track():
    """A diverging RAW bin must NOT fire when the user is on isotonic."""
    cal = _df([_cal_bin("live", "raw", 0.2, 0.3, p_mean=0.23,
                        freq=0.50, n=40, ci_lo=0.35, ci_hi=0.65)])
    assert _banner(cal=cal, calibration_track="isotonic") is None
    assert _banner(cal=cal, calibration_track="raw") is not None


def test_divergence_banner_ignores_thin_bins():
    """Below MIN_VERDICT_N=20 claims, Wilson interval is too wide to be
    meaningful — no banner even if claim is outside it."""
    cal = _df([_cal_bin("live", "isotonic", 0.2, 0.3, p_mean=0.23,
                        freq=0.80, n=5, ci_lo=0.40, ci_hi=0.97)])
    assert _banner(cal=cal) is None


def test_divergence_banner_handles_missing_calibration_column():
    """Pre-Phase-4 artifact — column absent → banner is a silent no-op."""
    cal = _df([{"model_id": "m", "track": "live", "bin_lo": 0.2,
                "bin_hi": 0.3, "p_mean": 0.23, "freq": 0.80, "n": 40,
                "ci_lo": 0.50, "ci_hi": 0.95}])
    assert _banner(cal=cal) is None
