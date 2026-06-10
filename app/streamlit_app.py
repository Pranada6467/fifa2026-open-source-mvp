"""Read-only viewer over committed artifacts/ — never fits models, never
simulates on page load. Rebuild content with:

    .venv/bin/python -m fifapreds.loop.predict       # log fresh predictions
    .venv/bin/python -m fifapreds.publish.artifacts  # export artifacts/
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("FIFAPREDS_ARTIFACTS", ROOT / "artifacts"))
UNIFORM_LOG_LOSS = math.log(3)
REBUILD_HINT = "No data yet — run `python -m fifapreds.publish.artifacts`."


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


st.set_page_config(page_title="FIFA 2026 — Live Calibration Engine", layout="wide")
st.title("FIFA 2026 — Live Calibration Engine")

meta = load("meta.json")
if meta:
    sha = meta.get("code_version") or "unknown"
    st.caption(
        f"Data through {meta['data_through']} · generated {meta['generated_at'][:16]}Z "
        f"· code {sha} · models: {', '.join(meta['models']) or 'none'}"
    )
else:
    st.info(REBUILD_HINT)

prob_col = lambda label: st.column_config.ProgressColumn(  # noqa: E731
    label, min_value=0.0, max_value=1.0, format="%.3f")

# ----------------------------------------------------------- tournament
st.header("Tournament odds")
tournament = load("tournament.parquet")
if tournament is None:
    st.info("No simulation yet — run `python -m fifapreds.orchestrate`.")
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
    st.caption("Seeded Monte Carlo over the full bracket (group tiebreakers, "
               "verified third-place routing, ET/penalty approximation), "
               "conditioned on group results so far. Top 16 shown.")

# ------------------------------------------------------------- upcoming
st.header("Upcoming matches")
upcoming = load("upcoming.parquet")
if upcoming is None:
    st.info(REBUILD_HINT)
else:
    view = upcoming.assign(
        kickoff=pd.to_datetime(upcoming["kickoff_ts"]).dt.strftime("%a %d %b"),
        fixture=upcoming["home_team"] + " v " + upcoming["away_team"],
        venue=upcoming["neutral"].map({1: "neutral", 0: "home", True: "neutral", False: "home"}),
    )[["kickoff", "fixture", "venue", "model_id", "p_home", "p_draw", "p_away"]]
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "kickoff": st.column_config.TextColumn("Kickoff"),
            "fixture": st.column_config.TextColumn("Fixture"),
            "venue": st.column_config.TextColumn("Venue"),
            "model_id": st.column_config.TextColumn("Model"),
            "p_home": prob_col("P(home win)"),
            "p_draw": prob_col("P(draw)"),
            "p_away": prob_col("P(away win)"),
        },
    )
    st.caption("Probabilities are pre-kickoff claims with full provenance; "
               "they are graded after each result and never edited.")

# ----------------------------------------------------------- leaderboard
st.header("Technique leaderboard")
leaderboard = load("leaderboard.parquet")
if leaderboard is None:
    st.info(REBUILD_HINT)
else:
    st.dataframe(
        leaderboard.sort_values(["context", "log_loss"]),
        hide_index=True,
        column_config={
            "model_id": st.column_config.TextColumn("Model"),
            "context": st.column_config.TextColumn("Context"),
            "n": st.column_config.NumberColumn("Matches"),
            "log_loss": st.column_config.NumberColumn("Log-loss", format="%.4f"),
            "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
            "rps": st.column_config.NumberColumn("RPS", format="%.4f"),
        },
    )
    st.caption(f"Lower is better. Know-nothing uniform forecast scores "
               f"log-loss {UNIFORM_LOG_LOSS:.4f}; anything above that carries "
               f"no information. backtest:wc20XX = out-of-sample replay.")

# ----------------------------------------------------------- calibration
st.header("Calibration")
calibration = load("calibration.parquet")
if calibration is None:
    st.info(REBUILD_HINT)
else:
    populated = calibration.dropna(subset=["p_mean"])
    diagonal = alt.Chart(pd.DataFrame({"p": [0.0, 1.0]})).mark_line(
        strokeDash=[4, 4], color="gray").encode(x="p", y="p")
    points = alt.Chart(populated).mark_circle().encode(
        x=alt.X("p_mean", title="Claimed probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("freq", title="Observed frequency", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("n", title="Claims in bin"),
        color=alt.Color("model_id", title="Model",
                        scale=alt.Scale(range=["#ff4b4b", "#9aa0a6"])),
        tooltip=["model_id", "bin_lo", "bin_hi", "n",
                 alt.Tooltip("p_mean", format=".3f"), alt.Tooltip("freq", format=".3f")],
    )
    st.altair_chart(diagonal + points, width="stretch")
    st.caption("Honest calibration sits on the diagonal: when a model says 70%, "
               "it should happen about 70% of the time.")

# ---------------------------------------------------------------- scored
st.header("Recently scored")
scored = load("scored.parquet")
if scored is None:
    st.info(REBUILD_HINT)
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
