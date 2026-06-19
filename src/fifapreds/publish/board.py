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


# ----------------------------------------------------- Item 11: modal scoreline

def modal_scoreline_from_grid(grid: np.ndarray) -> tuple[int, int]:
    """The headline scoreline as `(round(E[home]), round(E[away]))`.

    Calibration / ensemble work pushes probability mass around but rarely
    changes `argmax(grid)` — the per-cell maximum is invariant under
    monotone-per-class transforms. The modal (rounded-mean) scoreline
    DOES move when the underlying posterior fattens its tail (NegBin /
    BivariatePoisson) or sharpens (temperature scaling), so this is the
    user-visible metric that actually tracks the model work.

    Returns 0-indexed (home_goals, away_goals); the grid's shape sets the
    max representable score.
    """
    rows, cols = grid.shape
    home_axis = np.arange(rows)
    away_axis = np.arange(cols)
    e_home = float((grid.sum(axis=1) * home_axis).sum())
    e_away = float((grid.sum(axis=0) * away_axis).sum())
    return int(round(e_home)), int(round(e_away))


def modal_scoreline_label(modal_h: int, modal_a: int, home: str, away: str,
                          *, top1_h: int | None = None,
                          top1_a: int | None = None,
                          top1_p: float | None = None) -> tuple[str, str]:
    """Two-line label for the viewer headline (DD3).

    Returns (bold_line, explainer_line). The explainer references the
    top-1 argmax + its probability when supplied — the audit expander
    keeps the same top-5 list, but the headline now uses the modal
    posterior mean instead.
    """
    bold = f"{home} **{modal_h}–{modal_a}** {away}"
    if top1_h is None or top1_a is None or top1_p is None:
        explainer = "Average outcome by goals model (E[goals] rounded)."
    else:
        explainer = (
            f"Average outcome by goals model. The most-likely single score "
            f"({top1_h}–{top1_a}, {top1_p:.0%}) is in the audit."
        )
    return bold, explainer


# ------------------------------------------------- DD4: mid-tournament divergence

def divergence_banner(
    calibration: pd.DataFrame,
    *,
    calibration_track: str,
    weights_refit_date: str,
    track: str = "live",
    min_bin_n: int = MIN_VERDICT_N,
) -> str | None:
    """Amber-banner copy when live-track frequency escapes the Wilson
    interval around the claimed probability (D10 mid-tournament policy).

    The check: for each populated bin of (track, calibration_track),
    test whether `p_mean` (the claimed probability) lies outside
    `[ci_lo, ci_hi]` (the Wilson interval on the observed `freq`).
    When it does, the bin is statistically distinguishable from honest
    calibration at the observed sample size — the frozen calibrator
    has drifted from live reality and the user should know.

    Returns None when there's no divergence or insufficient data; an
    amber-friendly string otherwise. The string names the worst bin so
    the banner stays concrete: "in the 0.2–0.3 bin (claimed 23%,
    observed 41%)" rather than a vague "calibration looks off"."""
    if (calibration is None or calibration.empty
            or "calibration" not in calibration.columns):
        return None
    sub = calibration[
        (calibration["track"] == track)
        & (calibration["calibration"] == calibration_track)
        & calibration["p_mean"].notna()
        & (calibration["n"] >= min_bin_n)
    ]
    if sub.empty:
        return None
    # Divergence per bin: claimed sits outside the Wilson interval on freq.
    outside_lo = sub["p_mean"] < sub["ci_lo"]
    outside_hi = sub["p_mean"] > sub["ci_hi"]
    divergent = sub[outside_lo | outside_hi]
    if divergent.empty:
        return None
    # Worst = largest absolute gap between claim and observation.
    worst = divergent.assign(
        gap=lambda d: (d["freq"] - d["p_mean"]).abs(),
    ).sort_values("gap", ascending=False).iloc[0]
    return (
        f"Live observation diverges from frozen weights in the "
        f"{worst['bin_lo']:.0%}–{worst['bin_hi']:.0%} bin "
        f"(claimed {worst['p_mean']:.0%}, observed {worst['freq']:.0%}, "
        f"Wilson interval excluded). Weights last refit {weights_refit_date} — "
        f"refresh after WC2026 ends."
    )
