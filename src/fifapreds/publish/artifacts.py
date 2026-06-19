"""Precompute the artifacts the Streamlit viewer reads — the PUBLISH step.

The app never touches SQLite or fits a model: this job exports everything it
shows into committed files under `artifacts/`, so the deployed viewer is a
pure file reader and the repo history doubles as a public prediction record.

Sources: `data/fifa2026.db` (live predictions) + `data/backtest.db` (the
2014/18/22 proving ground). Either may be absent — whatever exists is
exported and `meta.json` says what was missing.

Outputs (all parquet unless noted):
- upcoming.parquet     latest claim per (match_id, model_id) for unplayed
                       fixtures, plus per-match consensus columns
                       (cons_p_* / consensus_source) so the viewer labels the
                       headline odds from data, never inference (E2/D4)
- leaderboard.parquet  mean log-loss/Brier/RPS per model x context
- calibration.parquet  pooled reliability table per model x track
                       (track = backtest|live; the board's two-track story)
- scored.parquet       graded predictions with outcomes (the results feed)
- surprises.parquet    one row per graded LIVE match: consensus probability
                       assigned to what actually happened + the most-wrong
                       model (E2/D8 — the "didn't see it coming" panel)
- disagreement.parquet model-average vs de-vigged market consensus per
                       upcoming fixture with odds coverage (E2/D6)
- tournament.parquet   Monte Carlo trophy odds per (model_id, team), copied
                       from data/tournament_sim.parquet when the orchestrator
                       has produced one (T11/T14)
- scoreline_topn.parquet  per (upcoming match, goals-capable model) the top-5
                       most likely scorelines + O/U 2.5 + BTTS probabilities,
                       derived from the grid stored at predict time (E6a)
- meta.json            generated_at, git sha, data-through date, row counts
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds.asof import MatchStore
from fifapreds.calibration import apply_calibrators, fit_calibrators
from fifapreds.config import PROJECT_ROOT
from fifapreds.db import DB_PATH
from fifapreds.loop.predict import code_version, load_grid
from fifapreds.loop.score import (
    CLASSES,
    binary_calibration_table,
    brier,
    btts_from_grid,
    calibration_table,
    log_loss,
    ou25_from_grid,
    rps,
    top_k_scorelines,
)
from fifapreds.publish.board import modal_scoreline_from_grid, track_of

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKTEST_DB = PROJECT_ROOT / "data" / "backtest.db"
TOURNAMENT_SRC = PROJECT_ROOT / "data" / "tournament_sim.parquet"
QUALIFICATION_SRC = PROJECT_ROOT / "data" / "qualification_backtest.parquet"

# D2 + DD2: the recalibration dimension on the published leaderboard.
# Named distinctly from the existing E2 `track ∈ {backtest, live}` column
# on calibration.parquet so the two dimensions never collide.
CALIBRATION_COL = "calibration"
RAW_CALIBRATION = "raw"

_PRED_COLS = ["prediction_id", "context", "match_id", "home_team", "away_team",
              "kickoff_ts", "neutral", "tournament", "p_home", "p_draw", "p_away",
              "model_id", "model_version", "hyperparams_hash", "training_cutoff",
              "predicted_at"]
_SCORED_SQL = """
    SELECT p.context, p.match_id, p.home_team, p.away_team, p.kickoff_ts,
           p.tournament, p.p_home, p.p_draw, p.p_away, p.model_id,
           s.outcome, s.log_loss, s.brier, s.rps, s.scored_at
    FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
"""
_LEADERBOARD_SQL = """
    SELECT p.model_id, p.context, COUNT(*) AS n,
           AVG(s.log_loss) AS log_loss, AVG(s.brier) AS brier, AVG(s.rps) AS rps
    FROM predictions p JOIN scores s ON s.prediction_id = p.prediction_id
    GROUP BY p.model_id, p.context
    ORDER BY p.context, log_loss
"""


def _read(db_path: Path, sql: str) -> pd.DataFrame | None:
    """Query a db that may not exist yet (None = source missing/empty)."""
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn)
        except pd.errors.DatabaseError:   # schema not initialised yet
            return None


def _concat(frames: list[pd.DataFrame | None], columns: list[str]) -> pd.DataFrame:
    present = [f for f in frames if f is not None and not f.empty]
    return pd.concat(present, ignore_index=True) if present else pd.DataFrame(columns=columns)


_PROB_COLS = ["p_home", "p_draw", "p_away"]


def _build_leaderboard_with_calibration(
    scored: pd.DataFrame,
    calibrators: dict,
) -> pd.DataFrame:
    """Aggregate log_loss/brier/rps per (model_id, context, calibration).

    Always emits the `raw` calibration track using the score-time metrics
    already in `scored` (no recomputation needed — those numbers ARE the
    audit log). For each available calibrator, applies it to (p_home,
    p_draw, p_away), recomputes metrics on the calibrated probabilities,
    and emits a row under that calibration track.

    Models without a calibrator (e.g. brand-new live entrants with no
    backtest history yet) appear only in the raw track. The viewer's
    track toggle (DD2) shows raw as fallback for those.
    """
    if scored.empty:
        return pd.DataFrame(columns=[
            "model_id", "context", CALIBRATION_COL, "n",
            "log_loss", "brier", "rps",
        ])

    rows: list[pd.DataFrame] = []
    raw = (
        scored.groupby(["model_id", "context"], as_index=False)
        .agg(n=("log_loss", "size"),
             log_loss=("log_loss", "mean"),
             brier=("brier", "mean"),
             rps=("rps", "mean"))
    )
    raw[CALIBRATION_COL] = RAW_CALIBRATION
    rows.append(raw)

    tracks = sorted({track for _, track in calibrators})
    for track in tracks:
        per_track: list[pd.DataFrame] = []
        for model_id, group in scored.groupby("model_id", sort=False):
            cal = calibrators.get((model_id, track))
            if cal is None:
                continue
            probs = group[_PROB_COLS].to_numpy()
            calibrated = cal.apply(probs)
            outcomes = group["outcome"].map(CLASSES.index).to_numpy()
            losses = log_loss(calibrated, outcomes)
            briers = brier(calibrated, outcomes)
            rpss = rps(calibrated, outcomes)
            g = group[["model_id", "context"]].copy()
            g["_log_loss"] = losses
            g["_brier"] = briers
            g["_rps"] = rpss
            per_track.append(g)
        if not per_track:
            continue
        cal_rows = (
            pd.concat(per_track, ignore_index=True)
            .groupby(["model_id", "context"], as_index=False)
            .agg(n=("_log_loss", "size"),
                 log_loss=("_log_loss", "mean"),
                 brier=("_brier", "mean"),
                 rps=("_rps", "mean"))
        )
        cal_rows[CALIBRATION_COL] = track
        rows.append(cal_rows)

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["context", CALIBRATION_COL, "log_loss"])
        .reset_index(drop=True)
    )


def _build_calibration_table(
    scored: pd.DataFrame,
    calibrators: dict,
) -> pd.DataFrame:
    """Reliability table per (model_id, track, calibration).

    The existing `track ∈ {backtest, live}` split stays — that's the
    board's two-track story (proof vs demonstration). The new
    `calibration ∈ {raw, temperature, isotonic}` dimension lets the
    hero toggle (DD2) switch the underlying reliability points without
    re-fitting at view time.
    """
    if scored.empty:
        return pd.DataFrame(columns=[
            "model_id", "track", CALIBRATION_COL,
            "bin_lo", "bin_hi", "n", "p_mean", "freq", "ci_lo", "ci_hi",
        ])

    by_track = scored.assign(track=scored["context"].map(track_of))
    rows: list[pd.DataFrame] = []
    cal_tracks = sorted({track for _, track in calibrators}) + [RAW_CALIBRATION]

    for (model_id, track), grp in by_track.groupby(["model_id", "track"]):
        raw_probs = grp[_PROB_COLS].to_numpy()
        outcomes = grp["outcome"].map(CLASSES.index).to_numpy()
        for cal_track in cal_tracks:
            if cal_track == RAW_CALIBRATION:
                probs = raw_probs
            else:
                cal = calibrators.get((model_id, cal_track))
                if cal is None:
                    continue
                probs = cal.apply(raw_probs)
            table = calibration_table(probs, outcomes)
            table.insert(0, "model_id", model_id)
            table.insert(1, "track", track)
            table.insert(2, CALIBRATION_COL, cal_track)
            rows.append(table)

    return pd.concat(rows, ignore_index=True)


def _consensus(group: pd.DataFrame) -> tuple[pd.Series, str]:
    """Headline probabilities for one match's claims: the market blend where
    present, else the model average — with the source labelled so the viewer
    never has to infer it (D4)."""
    mb = group[group["model_id"] == "market_blend"]
    src = mb if not mb.empty else group
    return src[_PROB_COLS].mean(), ("market" if not mb.empty else "model_avg")


_CONS_COLS = ["cons_p_home", "cons_p_draw", "cons_p_away", "consensus_source"]


def _with_consensus(upcoming: pd.DataFrame) -> pd.DataFrame:
    """Annotate every claim row with its match's consensus probabilities."""
    if upcoming.empty:
        return upcoming.reindex(columns=list(upcoming.columns) + _CONS_COLS)
    parts = []
    for _mid, group in upcoming.groupby("match_id"):
        probs, source = _consensus(group)
        group = group.copy()
        group["cons_p_home"], group["cons_p_draw"], group["cons_p_away"] = probs
        group["consensus_source"] = source
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["kickoff_ts", "match_id", "model_id"]
    )


_SURPRISE_COLS = ["match_id", "kickoff_ts", "home_team", "away_team", "tournament",
                  "outcome", "consensus_p", "consensus_source",
                  "worst_model_id", "worst_model_p", "n_models"]


def _surprises(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per graded LIVE match (D8): how much probability the consensus
    gave the outcome that actually happened, plus the most-wrong model.
    Per-match dedupe is the point — a top-5 must be five different stories."""
    live = scored[scored["context"] == "live"]
    if live.empty:
        return pd.DataFrame(columns=_SURPRISE_COLS)
    rows = []
    for mid, group in live.groupby("match_id"):
        outcome = group["outcome"].iloc[0]
        p_col = f"p_{outcome}"
        probs, source = _consensus(group)
        worst = group.loc[group[p_col].idxmin()]
        first = group.iloc[0]
        rows.append({
            "match_id": mid,
            "kickoff_ts": first["kickoff_ts"],
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "tournament": first["tournament"],
            "outcome": outcome,
            "consensus_p": float(probs[p_col]),
            "consensus_source": source,
            "worst_model_id": worst["model_id"],
            "worst_model_p": float(worst[p_col]),
            "n_models": int(group["model_id"].nunique()),
        })
    return (pd.DataFrame(rows, columns=_SURPRISE_COLS)
            .sort_values("consensus_p", ignore_index=True))


_DISAGREE_COLS = ["match_id", "kickoff_ts", "home_team", "away_team",
                  "model_p_home", "model_p_draw", "model_p_away",
                  "market_p_home", "market_p_draw", "market_p_away",
                  "delta", "model_pick", "market_pick", "snapshot_id"]


def _disagreement(upcoming: pd.DataFrame, live_db: Path | str) -> pd.DataFrame:
    """Model-average vs de-vigged market per upcoming fixture (D6). Compares
    against the PURE market consensus from the latest odds snapshot — not
    market_blend, which already contains the model. Fixtures without odds
    coverage are simply absent; no snapshot at all yields an empty frame."""
    from fifapreds.models.market import latest_h2h_probs
    from fifapreds.registry import canonical

    empty = pd.DataFrame(columns=_DISAGREE_COLS)
    models_only = upcoming[upcoming["model_id"] != "market_blend"]
    if models_only.empty or not Path(live_db).exists():
        return empty
    try:
        with sqlite3.connect(live_db) as conn:
            snapshot_id, market = latest_h2h_probs(conn)
    except (LookupError, sqlite3.Error):
        return empty
    market_lookup = {
        (canonical(r.home_team), canonical(r.away_team)): pd.Series(
            [r.p_home, r.p_draw, r.p_away], index=_PROB_COLS)
        for r in market.itertuples(index=False)
    }

    picks = {0: "home", 1: "draw", 2: "away"}
    rows = []
    for mid, group in models_only.groupby("match_id"):
        first = group.iloc[0]
        mkt = market_lookup.get((first["home_team"], first["away_team"]))
        if mkt is None:
            continue
        model = group[_PROB_COLS].mean()
        diff = (model.to_numpy() - mkt.to_numpy())
        rows.append({
            "match_id": mid,
            "kickoff_ts": first["kickoff_ts"],
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "model_p_home": model["p_home"], "model_p_draw": model["p_draw"],
            "model_p_away": model["p_away"],
            "market_p_home": mkt["p_home"], "market_p_draw": mkt["p_draw"],
            "market_p_away": mkt["p_away"],
            "delta": float(np.abs(diff).max()),
            "model_pick": picks[int(model.to_numpy().argmax())],
            "market_pick": picks[int(mkt.to_numpy().argmax())],
            "snapshot_id": snapshot_id,
        })
    return (pd.DataFrame(rows, columns=_DISAGREE_COLS)
            .sort_values("delta", ascending=False, ignore_index=True))


_TOPN_K = 5
# Item 11: modal_h / modal_a expose the new headline scoreline (E[goals]
# rounded) alongside the existing argmax top-5. The viewer chooses which
# to render where (modal in headline, top-5 in audit expander per DD3).
_SCORELINE_COLS = (
    ["match_id", "model_id", "home_team", "away_team", "kickoff_ts",
     "ou25_prob", "btts_prob", "top1_p", "modal_h", "modal_a"]
    + [f"s{i}_{f}" for i in range(1, _TOPN_K + 1) for f in ("h", "a", "p")]
)


def _scoreline_topn(upcoming: pd.DataFrame, live_db: Path | str) -> pd.DataFrame:
    """Per (upcoming match, goals-capable model): top-K most likely scorelines
    plus O/U 2.5 and BTTS, derived from the grid stored at predict time. Rows
    without a stored grid (W/D/L-only models like Elo, market_blend) are
    omitted. The viewer needs this because score_grids lives in SQLite and
    the public board is artifact-only."""
    empty = pd.DataFrame(columns=_SCORELINE_COLS)
    if upcoming.empty or not Path(live_db).exists():
        return empty
    rows = []
    with sqlite3.connect(live_db) as conn:
        for r in upcoming.itertuples(index=False):
            grid = load_grid(conn, int(r.prediction_id))
            if grid is None:
                continue
            top = top_k_scorelines(grid, k=_TOPN_K)
            modal_h, modal_a = modal_scoreline_from_grid(grid)
            row = {
                "match_id": r.match_id,
                "model_id": r.model_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "kickoff_ts": r.kickoff_ts,
                "ou25_prob": float(ou25_from_grid(grid)),
                "btts_prob": float(btts_from_grid(grid)),
                "top1_p": float(grid[top[0][0], top[0][1]]),
                "modal_h": modal_h,
                "modal_a": modal_a,
            }
            for i, (h, a) in enumerate(top, start=1):
                row[f"s{i}_h"] = int(h)
                row[f"s{i}_a"] = int(a)
                row[f"s{i}_p"] = float(grid[h, a])
            rows.append(row)
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=_SCORELINE_COLS).sort_values(
        ["kickoff_ts", "match_id", "model_id"], ignore_index=True
    )


def build(
    out_dir: Path | str = ARTIFACTS_DIR,
    *,
    live_db: Path | str = DB_PATH,
    backtest_db: Path | str = BACKTEST_DB,
    tournament_src: Path | str = TOURNAMENT_SRC,
    qualification_src: Path | str = QUALIFICATION_SRC,
    store: MatchStore | None = None,
) -> dict:
    """Export all artifacts; returns the meta dict that was written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = store or MatchStore()
    notes = [f"missing source: {p}" for p in (live_db, backtest_db) if not Path(p).exists()]

    # Upcoming: latest claim per (fixture, model), only for still-unplayed fixtures.
    live_preds = _read(live_db, f"SELECT {', '.join(_PRED_COLS)} FROM predictions")
    unplayed_ids = set(store.all.loc[~store.all["is_played"], "match_id"])
    if live_preds is not None and not live_preds.empty:
        upcoming = (
            live_preds[live_preds["match_id"].isin(unplayed_ids)]
            .sort_values("predicted_at")
            .groupby(["match_id", "model_id"], as_index=False)
            .last()
            .sort_values(["kickoff_ts", "match_id", "model_id"])
        )
    else:
        upcoming = pd.DataFrame(columns=_PRED_COLS)
    upcoming = _with_consensus(upcoming)
    upcoming.to_parquet(out / "upcoming.parquet", index=False)

    # Predicted scorelines (top-K + O/U 2.5 + BTTS per goals-model claim).
    scoreline_topn = _scoreline_topn(upcoming, live_db)
    scoreline_topn.to_parquet(out / "scoreline_topn.parquet", index=False)

    scored = _concat([_read(live_db, _SCORED_SQL), _read(backtest_db, _SCORED_SQL)],
                     ["context", "match_id", "home_team", "away_team", "kickoff_ts",
                      "tournament", "p_home", "p_draw", "p_away", "model_id",
                      "outcome", "log_loss", "brier", "rps", "scored_at"])
    scored.to_parquet(out / "scored.parquet", index=False)

    # Phase 4: fit LOTO calibrators on backtest predictions (D6). The pipeline
    # raises ValueError when fewer than 2 tournaments are available — caught
    # here so the publisher degrades to raw-only rather than failing the
    # whole nightly. Per D5, any per-calibrator failure during fit propagates
    # up and surfaces as a publish-time error (loud, not silent).
    calibrators: dict = {}
    if Path(backtest_db).exists():
        with sqlite3.connect(backtest_db) as bt_conn:
            try:
                calibrators = fit_calibrators(bt_conn)
            except ValueError as exc:
                notes.append(f"calibrators: skipped ({exc})")

    leaderboard = _build_leaderboard_with_calibration(scored, calibrators)
    leaderboard.to_parquet(out / "leaderboard.parquet", index=False)

    # Calibration reliability table: per (model_id, track, calibration), the
    # board's narrative needs both the existing two-track (backtest|live) split
    # AND the new recalibration dimension so the hero's track toggle (DD2)
    # has data to switch between.
    calibration = _build_calibration_table(scored, calibrators)
    calibration.to_parquet(out / "calibration.parquet", index=False)

    # E2 narrative panels: surprises (graded live matches) + market disagreement.
    surprises = _surprises(scored)
    surprises.to_parquet(out / "surprises.parquet", index=False)
    disagreement = _disagreement(upcoming, live_db)
    disagreement.to_parquet(out / "disagreement.parquet", index=False)

    # E6b: scoreline leaderboard + O/U 2.5 and BTTS calibration.
    scoreline_sql = """
        SELECT p.model_id, p.context, ss.*
        FROM scores_scoreline ss
        JOIN predictions p ON p.prediction_id = ss.prediction_id
    """
    scoreline_scored = _concat(
        [_read(live_db, scoreline_sql), _read(backtest_db, scoreline_sql)],
        ["model_id", "context", "prediction_id", "home_score", "away_score",
         "scoreline_log_loss", "exact_score_hit", "top3_hit",
         "ou25_prob", "ou25_outcome", "btts_prob", "btts_outcome", "scored_at"],
    )
    if not scoreline_scored.empty:
        sl_lb = (
            scoreline_scored.groupby(["model_id", "context"])
            .agg(
                n=("prediction_id", "count"),
                scoreline_log_loss=("scoreline_log_loss", "mean"),
                exact_score_pct=("exact_score_hit", "mean"),
                top3_pct=("top3_hit", "mean"),
                ou25_brier=("ou25_prob", lambda x: float(
                    ((x.values - scoreline_scored.loc[x.index, "ou25_outcome"].values) ** 2).mean()
                )),
                btts_brier=("btts_prob", lambda x: float(
                    ((x.values - scoreline_scored.loc[x.index, "btts_outcome"].values) ** 2).mean()
                )),
            )
            .reset_index()
            .sort_values(["context", "scoreline_log_loss"])
        )
        sl_lb.to_parquet(out / "scoreline_leaderboard.parquet", index=False)

        sl_cal_tables = []
        by_track = scoreline_scored.assign(track=scoreline_scored["context"].map(track_of))
        for (model_id, track), grp in by_track.groupby(["model_id", "track"]):
            for event, prob_col, outcome_col in [
                ("ou25", "ou25_prob", "ou25_outcome"),
                ("btts", "btts_prob", "btts_outcome"),
            ]:
                ct = binary_calibration_table(
                    grp[prob_col].to_numpy(), grp[outcome_col].to_numpy()
                )
                ct.insert(0, "model_id", model_id)
                ct.insert(1, "track", track)
                ct.insert(2, "event", event)
                sl_cal_tables.append(ct)
        sl_calibration = pd.concat(sl_cal_tables, ignore_index=True)
        sl_calibration.to_parquet(out / "scoreline_calibration.parquet", index=False)
    else:
        sl_lb = pd.DataFrame()
        sl_calibration = pd.DataFrame()
        pd.DataFrame().to_parquet(out / "scoreline_leaderboard.parquet", index=False)
        pd.DataFrame().to_parquet(out / "scoreline_calibration.parquet", index=False)

    # E4: leaderboard uncertainty bands + verdict badges (seeded bootstrap).
    from fifapreds.leaderboard import BAND_COLS, bootstrap_bands

    bands = bootstrap_bands(scored) if not scored.empty else pd.DataFrame(
        columns=BAND_COLS)
    bands.to_parquet(out / "leaderboard_bands.parquet", index=False)

    # E4: group-qualification backtest passthrough + its reliability table
    # (the shared T3 binning — Wilson bands included).
    qual_src = Path(qualification_src)
    if qual_src.exists():
        qualification = pd.read_parquet(qual_src)
        qualification.to_parquet(out / "qualification.parquet", index=False)
        qtables = []
        for model_id, grp in qualification.groupby("model_id"):
            qt = binary_calibration_table(grp["p_advance"].to_numpy(),
                                          grp["advanced"].to_numpy())
            qt.insert(0, "model_id", model_id)
            qtables.append(qt)
        qual_calibration = pd.concat(qtables, ignore_index=True)
        qual_calibration.to_parquet(out / "qualification_calibration.parquet",
                                    index=False)
    else:
        qualification = pd.DataFrame()
        qual_calibration = pd.DataFrame()
        notes.append(f"missing source: {qual_src}")

    # Tournament odds: pass through whatever the orchestrator last simulated.
    if Path(tournament_src).exists():
        tournament = pd.read_parquet(tournament_src)
        tournament.to_parquet(out / "tournament.parquet", index=False)
    else:
        tournament = pd.DataFrame()
        notes.append(f"missing source: {tournament_src}")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "data_through": str(store.played["date"].max().date()),
        "counts": {
            "upcoming": int(len(upcoming)),
            "leaderboard": int(len(leaderboard)),
            "scored": int(len(scored)),
            "calibration": int(len(calibration)),
            "surprises": int(len(surprises)),
            "disagreement": int(len(disagreement)),
            "leaderboard_bands": int(len(bands)),
            "qualification": int(len(qualification)),
            "tournament": int(len(tournament)),
            "scoreline_leaderboard": int(len(sl_lb)),
            "scoreline_calibration": int(len(sl_calibration)),
            "scoreline_topn": int(len(scoreline_topn)),
        },
        "models": sorted(set(leaderboard["model_id"]) | set(upcoming["model_id"])),
        "notes": notes,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    meta = build()
    print(f"artifacts -> {ARTIFACTS_DIR}")
    for name, n in meta["counts"].items():
        print(f"  {name:<12} {n} rows")
    for note in meta["notes"]:
        print(f"  NOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
