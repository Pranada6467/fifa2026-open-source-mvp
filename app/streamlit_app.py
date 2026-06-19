"""Read-only PUBLIC board over committed artifacts/ (E2) — never fits models,
never simulates on page load. Content is rebuilt by the nightly Action; force
locally with:

    .venv/bin/python -m fifapreds.publish.artifacts

Page structure is the design-review consensus (plan D1): calibration hero ->
leaderboard -> market disagreement -> surprises -> utility (match odds,
tournament) -> audit expanders. Copy rule (D3): every section leads with a
computed takeaway; mechanics live in expanders. All verdict sentences come
from fifapreds.publish.board so they are unit-tested, never hardcoded.
"""
from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fifapreds.publish.board import (
    DEFAULT_CALIBRATION,
    DIM_BIN_N,
    UNIFORM_LOG_LOSS,
    is_stale,
    next_nightly_utc,
    track_of,
    verdict_sentence,
)

# DD2: track toggle is anchored to "isotonic" by default — the canonical
# calibrated view per D2. Falls back to "raw" if the artifact predates
# Phase 4 (no `calibration` column).
DEFAULT_TRACK_PILL = "isotonic"
TRACK_PILL_ORDER = ("raw", "temperature", "isotonic")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("FIFAPREDS_ARTIFACTS", ROOT / "artifacts"))
REBUILD_HINT = "No data yet — run `python -m fifapreds.publish.artifacts`."

# ----------------------------------------------------------- visual tokens (D4)
# Outcome colours (unchanged) + semantic accents. Surprise/agreement are
# orange/teal — a colourblind-safe pair, deliberately not red/green. Track
# encoding: backtest = solid filled marks, live = hollow outlined marks.
HOME_C, DRAW_C, AWAY_C = "#ff4b4b", "#9aa0a6", "#3b82f6"
SURPRISE_C, GOOD_C = "#f59e0b", "#14b8a6"
REF_C = "gray"                       # reference lines: dashed gray


@st.cache_data
def _load_parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def load(name: str) -> pd.DataFrame | dict | None:
    path = ARTIFACTS / name
    if not path.exists():
        return None
    if name.endswith(".json"):
        return json.loads(path.read_text())
    df = _load_parquet(str(path), path.stat().st_mtime)
    return None if df.empty else df


def _filter_calibration(df: pd.DataFrame, cal_track: str) -> pd.DataFrame:
    """DD2: filter by calibration column if present; pass through otherwise.
    Lets the same render code serve pre- and post-Phase-4 artifacts."""
    if df is None or "calibration" not in df.columns:
        return df
    return df[df["calibration"] == cal_track]


def odds(p: float) -> float:
    """Probability -> decimal (European) odds. 0.64 -> 1.56."""
    return round(1.0 / max(float(p), 1e-9), 2)


def next_update_label() -> str:
    nxt = next_nightly_utc()
    return nxt.astimezone(timezone.utc).strftime("%a %d %b, 06:30 UTC")


st.set_page_config(page_title="FIFA 2026 — Live Calibration Engine", layout="wide")
st.title("FIFA 2026 — Live Calibration Engine")
st.caption("A World Cup prediction system that grades its own predictions in "
           "public. Every claim is logged before kickoff, scored after the "
           "result, and never edited.")

meta = load("meta.json")
if meta:
    sha = meta.get("code_version") or "unknown"
    st.caption(
        f"Data through {meta['data_through']} · generated {meta['generated_at'][:16]}Z "
        f"· code {sha} · models: {', '.join(meta['models']) or 'none'}"
    )
    # Stale-artifact trust banner (D2): old numbers must announce themselves.
    if is_stale(meta["generated_at"]):
        st.warning(
            f"The nightly update appears to have failed — these numbers are "
            f"as of {meta['generated_at'][:16]}Z. Next scheduled refresh: "
            f"{next_update_label()}."
        )
else:
    st.info(REBUILD_HINT)

prob_col = lambda label: st.column_config.ProgressColumn(  # noqa: E731
    label, min_value=0.0, max_value=1.0, format="%.3f")
odds_col = lambda label: st.column_config.NumberColumn(label, format="%.2f")  # noqa: E731


# ===================================================== HERO — calibration (D1.1)
st.header("Does 70% mean 70%?")

calibration = load("calibration.parquet")
leaderboard = load("leaderboard.parquet")

# DD2: calibration-track toggle anchored above the verdict. The pill set is
# constrained to the tracks actually present in the artifact, so a Phase-4
# rollback (or a pre-Phase-4 artifact) gracefully degrades to whatever's
# there instead of offering dead options.
cal_track = DEFAULT_CALIBRATION
if calibration is not None and "calibration" in calibration.columns:
    present = [t for t in TRACK_PILL_ORDER if t in set(calibration["calibration"])]
    if present:
        default_idx = (present.index(DEFAULT_TRACK_PILL)
                       if DEFAULT_TRACK_PILL in present else 0)
        cal_track = st.radio(
            "Calibration track",
            present,
            index=default_idx,
            horizontal=True,
            help="Raw = model claims pre-recalibration (the audit log). "
                 "Temperature/isotonic = LOTO-tuned recalibration on the "
                 "backtest; better numbers, smaller validation surface.",
            key="cal_track_pill",
        )
        if cal_track != "raw":
            st.caption(
                f"_Experimental — `{cal_track}` calibrator fit via LOTO on "
                "n≈64 holdout per fold (small validation surface; numbers "
                "shift as more matches grade)._"
            )

if calibration is None:
    st.info("The calibration record appears once predictions have been graded. "
            + REBUILD_HINT)
else:
    # Verdict sentences (computed, never hardcoded — D3); now filtered by
    # the selected calibration track so swapping the toggle moves the verdict
    # without re-emitting any artifact.
    backtest_verdict = verdict_sentence(calibration, "backtest",
                                        calibration_track=cal_track)
    live_verdict = verdict_sentence(calibration, "live",
                                    calibration_track=cal_track)
    if backtest_verdict:
        st.markdown(f"**{backtest_verdict}**")
    if live_verdict:
        st.markdown(live_verdict)
    else:
        st.caption("Live 2026 points appear as matches are graded — first "
                   f"grading expected after {next_update_label()}.")

    # Proof strip (D1): one wrapping line, stacks naturally on phones (D5).
    if leaderboard is not None:
        lb_for_strip = _filter_calibration(leaderboard, cal_track)
        lb = lb_for_strip.assign(track=lb_for_strip["context"].map(track_of))
        n_back = int(lb.loc[lb["track"] == "backtest", "n"].sum())
        n_live = int(lb.loc[lb["track"] == "live", "n"].sum())
        best_back = lb[lb["track"] == "backtest"]["log_loss"].min()
        market_live = lb[(lb["track"] == "live")
                         & (lb["model_id"] == "market_blend")]["log_loss"]
        strip = [f"**Backtest** {n_back} claims graded",
                 f"**Live 2026** {n_live} claims graded",
                 f"coin-flip log-loss {UNIFORM_LOG_LOSS:.3f}"]
        if pd.notna(best_back):
            strip.insert(2, f"best backtest log-loss {best_back:.3f}")
        if not market_live.empty:
            strip.append(f"market log-loss {market_live.iloc[0]:.3f}")
        st.markdown(" · ".join(strip))

    # The reliability chart: solid = backtest, hollow = live (D4); thin bins
    # (n < DIM_BIN_N) render dimmed — too few claims to judge (D2). DD2:
    # filter to the selected calibration track so the toggle moves the points.
    calibration_for_chart = _filter_calibration(calibration, cal_track)
    populated = calibration_for_chart.dropna(subset=["p_mean"])
    populated = populated[populated["n"] > 0]
    diagonal = alt.Chart(pd.DataFrame({"p": [0.0, 1.0]})).mark_line(
        strokeDash=[4, 4], color=REF_C).encode(x="p", y="p")
    dim = alt.condition(alt.datum.n >= DIM_BIN_N, alt.value(0.85), alt.value(0.25))
    base = alt.Chart(populated).encode(
        x=alt.X("p_mean", title="Claimed probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("freq", title="Observed frequency", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("n", title="Claims in bin"),
        color=alt.Color("model_id", title="Model"),
        opacity=dim,
        tooltip=["model_id", "track", "bin_lo", "bin_hi", "n",
                 alt.Tooltip("p_mean", format=".3f"),
                 alt.Tooltip("freq", format=".3f")],
    )
    layers = [diagonal]
    backtest_pts = populated[populated["track"] == "backtest"]
    live_pts = populated[populated["track"] == "live"]
    if not backtest_pts.empty:
        layers.append(base.transform_filter(alt.datum.track == "backtest")
                      .mark_circle())
    if not live_pts.empty:
        layers.append(base.transform_filter(alt.datum.track == "live")
                      .mark_point(filled=False, strokeWidth=2))
    st.altair_chart(alt.layer(*layers), width="stretch")
    st.caption("Honest calibration sits on the dashed diagonal. Solid points: "
               "backtest replay of WC 2014/18/22. Hollow points: live 2026 "
               "claims. Faded points sit in bins with fewer than "
               f"{DIM_BIN_N} claims — too few to judge.")
    with st.expander("How this works"):
        st.markdown(
            "- Every prediction is logged **before kickoff** with its model id, "
            "training cutoff, and code version; rows are append-only.\n"
            "- After each result the claim is graded (log-loss, Brier, RPS) and "
            "lands in a one-vs-rest reliability bin.\n"
            "- The **backtest** track replays the 2014/18/22 World Cups through "
            "the identical predict-and-score path — that is the proof. The "
            "**live** track is this tournament, accumulating nightly — that is "
            "the demonstration.\n"
            "- A know-nothing uniform forecast scores log-loss "
            f"{UNIFORM_LOG_LOSS:.4f}; anything above carries no information."
        )

# ===================================================== leaderboard (D1.2, D7)
st.header("Which technique is winning?")
if leaderboard is None:
    st.info("The leaderboard appears once predictions have been graded. "
            + REBUILD_HINT)
else:
    # DD2: leaderboard rows are also filtered by the hero's track toggle.
    leaderboard_for_lb = _filter_calibration(leaderboard, cal_track)
    lb = leaderboard_for_lb.assign(track=leaderboard_for_lb["context"].map(track_of))
    bands = load("leaderboard_bands.parquet")
    # Verdict caption: best by RPS per track, computed not asserted (D3).
    lines = []
    for track_name, track_lb in lb.groupby("track"):
        pooled = (track_lb.assign(w_rps=track_lb["rps"] * track_lb["n"])
                  .groupby("model_id")[["n", "w_rps"]].sum())
        pooled["rps"] = pooled["w_rps"] / pooled["n"]
        best_id = pooled["rps"].idxmin()
        best = pooled.loc[best_id]
        suffix = ""
        if bands is not None:
            n_tied = int(((bands["track"] == track_name)
                          & (bands["badge"] == "tied")).sum())
            suffix = (f", with {n_tied} model{'s' if n_tied != 1 else ''} "
                      f"tied within noise" if n_tied else ", clear of the field")
        lines.append(f"**{track_name}**: {best_id} leads on RPS "
                     f"({best['rps']:.4f} over {int(best['n'])} claims){suffix}")
    st.markdown(" · ".join(lines) + ("." if bands is not None else
                " — gaps within a few thousandths are noise until "
                "uncertainty bands land."))

    # Default view: <=5 columns (D5), ranked by RPS (D7). E4's bootstrap
    # bands supply the verdict badges + CIs; until that artifact exists the
    # view falls back to pooled point estimates.
    if bands is not None:
        BADGES = {"best": "🥇 best", "tied": "≈ tied with best",
                  "behind": "behind"}
        view = bands.copy()
        view["verdict"] = view["badge"].map(BADGES)
        view["rps_ci"] = view.apply(
            lambda r: f"{r['rps']:.4f}  [{r['rps_lo']:.4f}–{r['rps_hi']:.4f}]",
            axis=1)
        view = pd.concat([
            view[["model_id", "track", "n", "verdict", "rps_ci"]],
            pd.DataFrame([{"model_id": "— coin-flip (uniform) —",
                           "track": "reference", "n": None,
                           "verdict": "reference",
                           "rps_ci": f"log-loss {UNIFORM_LOG_LOSS:.4f}"}]),
        ], ignore_index=True)
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "model_id": st.column_config.TextColumn("Model"),
                "track": st.column_config.TextColumn("Track"),
                "n": st.column_config.NumberColumn("Claims"),
                "verdict": st.column_config.TextColumn("Verdict"),
                "rps_ci": st.column_config.TextColumn("RPS (95% CI)"),
            },
        )
        st.caption("Lower RPS is better; the interval is a seeded bootstrap "
                   "over graded matches. “Tied” means the paired difference "
                   "to the leader includes zero — calling that a win would "
                   "be noise-laundering. `market_blend` is the de-vigged "
                   "bookmaker consensus blended with the best model. Configs "
                   "were frozen before kickoff; only ratings update.")
    else:
        pooled_view = (lb.assign(w_rps=lb["rps"] * lb["n"],
                                 w_ll=lb["log_loss"] * lb["n"])
                       .groupby(["model_id", "track"], as_index=False)
                       [["n", "w_rps", "w_ll"]].sum())
        pooled_view["rps"] = pooled_view["w_rps"] / pooled_view["n"]
        pooled_view["log_loss"] = pooled_view["w_ll"] / pooled_view["n"]
        view = pooled_view[["model_id", "track", "n", "rps", "log_loss"]].copy()
        uniform_row = pd.DataFrame([{
            "model_id": "— coin-flip (uniform) —", "track": "reference",
            "n": None, "rps": None, "log_loss": UNIFORM_LOG_LOSS,
        }])
        view = pd.concat([view.sort_values(["track", "rps"]), uniform_row],
                         ignore_index=True)
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "model_id": st.column_config.TextColumn("Model"),
                "track": st.column_config.TextColumn("Track"),
                "n": st.column_config.NumberColumn("Claims"),
                "rps": st.column_config.NumberColumn("RPS (primary)", format="%.4f"),
                "log_loss": st.column_config.NumberColumn("Log-loss", format="%.4f"),
            },
        )
        st.caption("Lower is better. Models above the coin-flip row carry real "
                   "information; `market_blend` is the de-vigged bookmaker "
                   "consensus blended with the best model — the bar to beat. "
                   "Configs were frozen before kickoff; only ratings update.")

# ============================================ scoreline accuracy (E6b)
st.header("And at picking the score?")
scoreline_lb = load("scoreline_leaderboard.parquet")
scoreline_cal = load("scoreline_calibration.parquet")
if scoreline_lb is None:
    st.info(
        "Scoreline grading kicks in once goals-model claims are graded against "
        f"real results — first grading expected after {next_update_label()}. "
        "Until then this section stays empty on purpose: exact-score accuracy "
        "is not estimable from zero matches.")
else:
    sl = scoreline_lb.assign(track=scoreline_lb["context"].map(track_of))
    # Verdict caption (computed, never hardcoded): best by scoreline log-loss
    # per track, plus exact-score hit rate for context.
    lines = []
    for track_name, track_lb in sl.groupby("track"):
        best = track_lb.sort_values("scoreline_log_loss").iloc[0]
        lines.append(
            f"**{track_name}**: `{best['model_id']}` leads on scoreline log-loss "
            f"({best['scoreline_log_loss']:.3f}) and called the exact score "
            f"in {best['exact_score_pct']:.0%} of {int(best['n'])} matches"
        )
    st.markdown(" · ".join(lines) + ".")

    view = sl[["model_id", "track", "n", "scoreline_log_loss",
               "exact_score_pct", "top3_pct", "ou25_brier", "btts_brier"]]
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "model_id": st.column_config.TextColumn("Model"),
            "track": st.column_config.TextColumn("Track"),
            "n": st.column_config.NumberColumn("Claims"),
            "scoreline_log_loss":
                st.column_config.NumberColumn("Scoreline log-loss", format="%.3f"),
            "exact_score_pct": prob_col("Exact-score hit"),
            "top3_pct": prob_col("Top-3 hit"),
            "ou25_brier":
                st.column_config.NumberColumn("O/U 2.5 Brier", format="%.3f"),
            "btts_brier":
                st.column_config.NumberColumn("BTTS Brier", format="%.3f"),
        },
    )
    st.caption("Scoreline log-loss is the natural-log score on the full grid "
               "(tail bucket for out-of-grid). Exact-score hit / top-3 hit are "
               "the share of matches whose actual score was the model's most "
               "likely / top-3. Lower is better for log-loss and Brier; higher "
               "for hit rates. Extra-time matches are excluded — the 90-min "
               "model can't be graded on 120-min scores.")

    # Reliability — Over 2.5 goals + BTTS, the two binary side-markets users
    # actually parse. Wilson bands per bin (T3).
    if scoreline_cal is not None:
        for event, event_label in [("ou25", "Over 2.5 goals"),
                                   ("btts", "Both teams score")]:
            cal = scoreline_cal[scoreline_cal["event"] == event].dropna(
                subset=["p_mean"])
            cal = cal[cal["n"] > 0]
            if cal.empty:
                continue
            st.markdown(f"**{event_label} — reliability**")
            diag = alt.Chart(pd.DataFrame({"p": [0.0, 1.0]})).mark_line(
                strokeDash=[4, 4], color=REF_C).encode(x="p", y="p")
            rule = alt.Chart(cal).mark_rule(strokeWidth=2, opacity=0.5).encode(
                x=alt.X("p_mean", title=f"Claimed P({event_label})",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("ci_lo", title="Observed frequency",
                        scale=alt.Scale(domain=[0, 1])),
                y2="ci_hi",
                color=alt.Color("model_id", title="Model"),
            )
            pts = alt.Chart(cal).mark_circle().encode(
                x="p_mean", y="freq",
                size=alt.Size("n", title="Matches in bin"),
                color=alt.Color("model_id", title="Model"),
                tooltip=["model_id", "track", "n",
                         alt.Tooltip("p_mean", format=".3f"),
                         alt.Tooltip("freq", format=".3f"),
                         alt.Tooltip("ci_lo", format=".3f"),
                         alt.Tooltip("ci_hi", format=".3f")],
            )
            st.altair_chart(diag + rule + pts, width="stretch")
        st.caption("Bars are 95% Wilson intervals per bin — honest uncertainty "
                   "at this sample size.")

# ============================================ qualification foresight (E4)
st.header("Could it pick the group-stage survivors?")
qual_cal = load("qualification_calibration.parquet")
qualification = load("qualification.parquet")
if qual_cal is None:
    st.info("The tournament-level backtest hasn't been published yet — "
            "qualification foresight appears once it runs.")
else:
    n_events = int(qualification["wc"].nunique() * 32) if qualification is not None else 0
    st.markdown(
        f"Each 2014/18/22 World Cup was re-simulated **before its opening "
        f"match**; every team's claimed chance of surviving the group is "
        f"graded against what happened — {n_events} yes/no events, the "
        f"tournament-level question with enough sample to actually judge.")
    qpop = qual_cal.dropna(subset=["p_mean"])
    qpop = qpop[qpop["n"] > 0]
    diag = alt.Chart(pd.DataFrame({"p": [0.0, 1.0]})).mark_line(
        strokeDash=[4, 4], color=REF_C).encode(x="p", y="p")
    bars = alt.Chart(qpop).mark_rule(strokeWidth=2, opacity=0.5).encode(
        x=alt.X("p_mean", title="Claimed P(advance)",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("ci_lo", title="Observed frequency",
                scale=alt.Scale(domain=[0, 1])),
        y2="ci_hi",
        color=alt.Color("model_id", title="Model"),
    )
    pts = alt.Chart(qpop).mark_circle().encode(
        x="p_mean", y="freq",
        size=alt.Size("n", title="Teams in bin"),
        color=alt.Color("model_id", title="Model"),
        tooltip=["model_id", "n", alt.Tooltip("p_mean", format=".3f"),
                 alt.Tooltip("freq", format=".3f"),
                 alt.Tooltip("ci_lo", format=".3f"),
                 alt.Tooltip("ci_hi", format=".3f")],
    )
    st.altair_chart(diag + bars + pts, width="stretch")
    st.caption("Vertical lines are 95% Wilson intervals per bin — honest "
               "uncertainty at this sample size. Deep-run and champion "
               "calibration are deliberately NOT claimed: three tournaments "
               "is n=3, and no forecaster can be judged on that.")

# ============================================ market disagreement (D1.3, D6)
st.header("Where we differ from the market")
disagreement = load("disagreement.parquet")
if disagreement is None:
    st.info("No bookmaker odds snapshot covers the current fixtures — "
            "disagreements appear after the next odds capture.")
else:
    top = disagreement.head(8).copy()
    top["fixture"] = top["home_team"] + " v " + top["away_team"]
    top["when"] = pd.to_datetime(top["kickoff_ts"]).dt.strftime("%a %d %b")
    rows = []
    for r in top.itertuples(index=False):
        model_p = {"home": r.model_p_home, "draw": r.model_p_draw,
                   "away": r.model_p_away}
        market_p = {"home": r.market_p_home, "draw": r.market_p_draw,
                    "away": r.market_p_away}
        # Describe the outcome where the gap is widest, in plain words.
        gaps = {k: model_p[k] - market_p[k] for k in model_p}
        key = max(gaps, key=lambda k: abs(gaps[k]))
        side = {"home": r.home_team, "draw": "the draw", "away": r.away_team}[key]
        direction = "likelier" if gaps[key] > 0 else "less likely"
        rows.append({
            "Kickoff": r.when, "Fixture": r.fixture,
            "Models say": f"{side} {direction}: {model_p[key]:.0%} vs market {market_p[key]:.0%}",
            "Gap": r.delta,
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"Gap": prob_col("Gap")},
    )
    st.caption("Model average vs the de-vigged market consensus for upcoming "
               "fixtures, biggest gaps first. The market is the calibration "
               "benchmark — being close is the win condition; differing is a "
               "claim that gets graded.")

# ===================================================== surprises (D1.4, D8)
st.header("The system didn't see these coming")
surprises = load("surprises.parquet")
if surprises is None:
    st.info("No live matches graded yet — first grading expected after "
            f"{next_update_label()}.")
else:
    for r in surprises.head(5).itertuples(index=False):
        outcome_label = {
            "home": f"{r.home_team} won", "away": f"{r.away_team} won",
            "draw": "a draw",
        }[r.outcome]
        source = "Market consensus" if r.consensus_source == "market" else "Model consensus"
        with st.container(border=True):
            st.markdown(
                f"**{r.home_team} v {r.away_team}** — {source} gave "
                f"**{r.consensus_p:.0%}** to {outcome_label}. It happened."
            )
            st.caption(
                f"{pd.to_datetime(r.kickoff_ts).strftime('%a %d %b')} · most "
                f"wrong: {r.worst_model_id} ({r.worst_model_p:.1%}) · "
                f"{r.n_models} models graded"
            )
    st.caption("Largest surprises of the tournament so far: graded matches "
               "where the consensus gave the actual outcome the least "
               "probability. Honest forecasters get surprised — at a "
               "calibrated rate.")

# ===================================================== utility — match odds
st.header("Match odds")
upcoming = load("upcoming.parquet")
scoreline_topn = load("scoreline_topn.parquet")
if upcoming is None:
    if meta:
        st.info("No upcoming fixtures to price — either the tournament is "
                f"complete, or new claims land after {next_update_label()}.")
    else:
        st.info(REBUILD_HINT)
else:
    fixtures = (
        upcoming.drop_duplicates("match_id")
        .sort_values("kickoff_ts")[
            ["match_id", "kickoff_ts", "home_team", "away_team", "neutral",
             "cons_p_home", "cons_p_draw", "cons_p_away", "consensus_source"]
        ]
    )
    fixtures["when"] = pd.to_datetime(fixtures["kickoff_ts"]).dt.strftime("%a %d %b")
    fixtures["label"] = (
        fixtures["when"] + " — " + fixtures["home_team"] + " v " + fixtures["away_team"]
    )

    choice = st.selectbox(
        "Select a match", fixtures["label"].tolist(),
        help="Pick a fixture to see every model's call plus the consensus.",
    )
    row = fixtures[fixtures["label"] == choice].iloc[0]
    home, away = row["home_team"], row["away_team"]
    neutral = bool(row["neutral"]) if row["neutral"] in (0, 1) else row["neutral"]
    venue = "neutral venue" if neutral else f"{home} at home"
    # Consensus comes labelled from the artifact (D4) — never inferred here.
    src_label = ("market consensus" if row["consensus_source"] == "market"
                 else "model average (no market coverage)")

    sel = upcoming[upcoming["match_id"] == row["match_id"]].copy()

    st.markdown(f"### {home} v {away}")
    st.caption(f"{row['when']} · {venue} · headline odds: {src_label}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win · {row['cons_p_home']:.0%}", odds(row["cons_p_home"]))
    c2.metric(f"Draw · {row['cons_p_draw']:.0%}", odds(row["cons_p_draw"]))
    c3.metric(f"{away} win · {row['cons_p_away']:.0%}", odds(row["cons_p_away"]))

    bar_df = pd.DataFrame({
        "match": [choice] * 3,
        "outcome": [f"{home} win", "Draw", f"{away} win"],
        "p": [row["cons_p_home"], row["cons_p_draw"], row["cons_p_away"]],
        "ord": [0, 1, 2],
    })
    bar = (
        alt.Chart(bar_df).mark_bar().encode(
            x=alt.X("p:Q", stack="normalize", title=None,
                    axis=alt.Axis(format="%")),
            y=alt.Y("match:N", title=None, axis=None),
            order=alt.Order("ord:Q"),
            color=alt.Color(
                "outcome:N", title=None,
                scale=alt.Scale(domain=[f"{home} win", "Draw", f"{away} win"],
                                range=[HOME_C, DRAW_C, AWAY_C]),
                legend=alt.Legend(orient="bottom")),
            tooltip=["outcome", alt.Tooltip("p:Q", title="prob", format=".1%")],
        ).properties(height=60)
    )
    st.altair_chart(bar, width="stretch")

    # ----- predicted scoreline (E6a grids surfaced via scoreline_topn) -----
    sl = (scoreline_topn[scoreline_topn["match_id"] == row["match_id"]]
          if scoreline_topn is not None else None)
    st.subheader("Predicted scoreline")
    if sl is None or sl.empty:
        st.caption("No stored score grid for this fixture — the claim was "
                   "logged before scoreline capture was deployed. Future "
                   "predictions will include scorelines.")
    else:
        sl_models = sorted(sl["model_id"].unique())
        # Single goals model → render the model name as caption, no selector.
        if len(sl_models) > 1:
            sl_pick = st.selectbox(
                "Goals model", sl_models,
                key=f"sl_model_{row['match_id']}",
                help="Only goals-capable models (Dixon-Coles family, hierarchical) "
                     "produce a full score grid. Other models predict W/D/L only.",
            )
        else:
            sl_pick = sl_models[0]
            st.caption(f"From `{sl_pick}` (the only goals-capable model with a stored grid).")
        sl_row = sl[sl["model_id"] == sl_pick].iloc[0]

        # Hero — most likely scoreline as a single readable line.
        s1_h, s1_a = int(sl_row["s1_h"]), int(sl_row["s1_a"])
        winner = (f"{home} win" if s1_h > s1_a
                  else "draw" if s1_h == s1_a
                  else f"{away} win")
        st.markdown(
            f"**Most likely:** {home} **{s1_h}–{s1_a}** {away} "
            f"— {sl_row['top1_p']:.1%} ({winner})"
        )

        # Quick stats — the two bets a casual fan actually parses.
        q1, q2 = st.columns(2)
        q1.metric("Over 2.5 goals", f"{sl_row['ou25_prob']:.0%}",
                  help="Probability the match ends with 3+ goals (regular time).")
        q2.metric("Both teams score", f"{sl_row['btts_prob']:.0%}",
                  help="Probability both sides score at least once (regular time).")

        # Top-5 list — clear hierarchy, biggest chance first.
        top_rows = []
        for i in range(1, 6):
            h_i, a_i, p_i = (int(sl_row[f"s{i}_h"]), int(sl_row[f"s{i}_a"]),
                             float(sl_row[f"s{i}_p"]))
            top_rows.append({
                "Score": f"{home} {h_i}–{a_i} {away}",
                "Probability": p_i,
            })
        st.dataframe(
            pd.DataFrame(top_rows), hide_index=True, width="stretch",
            column_config={
                "Score": st.column_config.TextColumn("Most likely scorelines"),
                "Probability": prob_col("Probability"),
            },
        )
        st.caption("Top-5 full-time scorelines (regular 90 minutes — extra time "
                   "and penalties not modelled here). The remaining ~"
                   f"{max(0.0, 1.0 - sum(r['Probability'] for r in top_rows)):.0%} "
                   "lives in the long tail of higher-scoring or unusual results.")

    # Every model's call for this match, market blend pinned to the top.
    sel["_consensus"] = (sel["model_id"] == "market_blend").astype(int)
    sel = sel.sort_values(["_consensus", "model_id"], ascending=[False, True])
    for side in ("home", "draw", "away"):
        sel[f"odds_{side}"] = sel[f"p_{side}"].map(odds)
    per_match = sel[["model_id", "p_home", "odds_home", "p_draw", "odds_draw",
                     "p_away", "odds_away"]]
    st.dataframe(
        per_match,
        hide_index=True,
        width="stretch",
        column_config={
            "model_id": st.column_config.TextColumn("Model"),
            "p_home": prob_col("P(home)"),
            "odds_home": odds_col("Odds H"),
            "p_draw": prob_col("P(draw)"),
            "odds_draw": odds_col("Odds D"),
            "p_away": prob_col("P(away)"),
            "odds_away": odds_col("Odds A"),
        },
    )

    # Scan the whole slate: one consensus row per match, straight from the
    # artifact's cons_* columns.
    st.subheader("All upcoming matches")
    scan = fixtures.copy()
    scan["odds_hda"] = scan.apply(
        lambda r: f"{odds(r['cons_p_home'])} / {odds(r['cons_p_draw'])} / "
                  f"{odds(r['cons_p_away'])}",
        axis=1)
    st.dataframe(
        scan[["when", "label", "cons_p_home", "cons_p_draw", "cons_p_away",
              "odds_hda"]].rename(columns={"when": "kickoff", "label": "fixture"}),
        hide_index=True,
        width="stretch",
        column_config={
            "kickoff": st.column_config.TextColumn("Kickoff"),
            "fixture": st.column_config.TextColumn("Fixture"),
            "cons_p_home": prob_col("P(home)"),
            "cons_p_draw": prob_col("P(draw)"),
            "cons_p_away": prob_col("P(away)"),
            "odds_hda": st.column_config.TextColumn("Odds H / D / A"),
        },
    )
    st.caption("Consensus odds per fixture (market where covered, else the "
               "model average — labelled per match). Decimal odds = 1 ÷ probability.")

# ----------------------------------------------------------- tournament
st.header("Who wins the World Cup?")
tournament = load("tournament.parquet")
if tournament is None:
    st.info("The tournament simulation updates nightly — check back after "
            f"{next_update_label()}.")
else:
    sim_models = sorted(tournament["model_id"].unique())
    pick = (
        st.selectbox("Simulating model", sim_models)
        if len(sim_models) > 1 else sim_models[0]
    )
    view = (
        tournament[tournament["model_id"] == pick]
        .sort_values("p_champion", ascending=False)
        .head(16)[["team", "group", "p_advance", "p_qf", "p_sf",
                   "p_final", "p_champion"]]
    )
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "team": st.column_config.TextColumn("Team"),
            "group": st.column_config.TextColumn("Group"),
            "p_advance": prob_col("P(advance)"),
            "p_qf": prob_col("P(quarter-final)"),
            "p_sf": prob_col("P(semi-final)"),
            "p_final": prob_col("P(final)"),
            "p_champion": prob_col("P(champion)"),
        },
    )
    st.caption("Top 16 by title chance, conditioned on group results so far. "
               "10,000 simulated tournaments per model, refreshed nightly.")
    with st.expander("How the simulation works"):
        st.markdown(
            "Seeded Monte Carlo over the full bracket: group tiebreakers per "
            "FIFA rules, the verified 495-combination third-place routing, "
            "knockout draws resolved by extra time then penalties "
            "(approximated). The simulating model supplies every score grid; "
            "the seed is the UTC date, so a rerun reproduces the same odds."
        )

# ----------------------------------------------------------- audit trail
st.header("Audit trail")
scored = load("scored.parquet")
with st.expander("Recently scored claims (full provenance)"):
    if scored is None:
        st.info("Nothing graded yet — claims are scored as results land, "
                f"next pass after {next_update_label()}.")
    else:
        recent = scored.sort_values("kickoff_ts", ascending=False).head(30)
        view = recent.assign(
            kickoff=pd.to_datetime(recent["kickoff_ts"]).dt.strftime("%Y-%m-%d"),
            fixture=recent["home_team"] + " v " + recent["away_team"],
        )[["kickoff", "fixture", "model_id", "context", "outcome",
           "p_home", "p_draw", "p_away", "log_loss", "rps"]]
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            column_config={
                "kickoff": st.column_config.TextColumn("Kickoff"),
                "fixture": st.column_config.TextColumn("Fixture"),
                "model_id": st.column_config.TextColumn("Model"),
                "context": st.column_config.TextColumn("Context"),
                "outcome": st.column_config.TextColumn("Outcome"),
                "p_home": prob_col("P(home)"),
                "p_draw": prob_col("P(draw)"),
                "p_away": prob_col("P(away)"),
                "log_loss": st.column_config.NumberColumn("Log-loss", format="%.3f"),
                "rps": st.column_config.NumberColumn("RPS", format="%.3f"),
            },
        )
with st.expander("Full leaderboard (all metrics)"):
    if leaderboard is None:
        st.info(REBUILD_HINT)
    else:
        st.dataframe(
            leaderboard.sort_values(["context", "log_loss"]),
            hide_index=True,
            column_config={
                "model_id": st.column_config.TextColumn("Model"),
                "context": st.column_config.TextColumn("Context"),
                "n": st.column_config.NumberColumn("Claims"),
                "log_loss": st.column_config.NumberColumn("Log-loss", format="%.4f"),
                "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
                "rps": st.column_config.NumberColumn("RPS", format="%.4f"),
            },
        )
st.caption("Source: committed artifacts in the public repo — the git history "
           "is the tamper-evident prediction record.")
