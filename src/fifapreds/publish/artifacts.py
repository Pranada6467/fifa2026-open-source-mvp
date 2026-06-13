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
from fifapreds.config import PROJECT_ROOT
from fifapreds.db import DB_PATH
from fifapreds.loop.predict import code_version
from fifapreds.loop.score import CLASSES, binary_calibration_table, calibration_table
from fifapreds.publish.board import track_of

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BACKTEST_DB = PROJECT_ROOT / "data" / "backtest.db"
TOURNAMENT_SRC = PROJECT_ROOT / "data" / "tournament_sim.parquet"
QUALIFICATION_SRC = PROJECT_ROOT / "data" / "qualification_backtest.parquet"

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

    leaderboard = _concat([_read(live_db, _LEADERBOARD_SQL), _read(backtest_db, _LEADERBOARD_SQL)],
                          ["model_id", "context", "n", "log_loss", "brier", "rps"])
    leaderboard.to_parquet(out / "leaderboard.parquet", index=False)

    scored = _concat([_read(live_db, _SCORED_SQL), _read(backtest_db, _SCORED_SQL)],
                     ["context", "match_id", "home_team", "away_team", "kickoff_ts",
                      "tournament", "p_home", "p_draw", "p_away", "model_id",
                      "outcome", "log_loss", "brier", "rps", "scored_at"])
    scored.to_parquet(out / "scored.parquet", index=False)

    # Calibration: pooled over every graded prediction, per model x track —
    # the board's two-track story (backtest = proof, live = demo) needs the
    # split at export time, not viewer-side guessing.
    tables = []
    if not scored.empty:
        by_track = scored.assign(track=scored["context"].map(track_of))
        for (model_id, track), grp in by_track.groupby(["model_id", "track"]):
            table = calibration_table(
                grp[["p_home", "p_draw", "p_away"]].to_numpy(),
                grp["outcome"].map(CLASSES.index).to_numpy(),
            )
            table.insert(0, "model_id", model_id)
            table.insert(1, "track", track)
            tables.append(table)
    calibration = _concat(
        tables, ["model_id", "track", "bin_lo", "bin_hi", "n", "p_mean", "freq",
                 "ci_lo", "ci_hi"])
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
