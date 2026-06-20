"""T14 — end-to-end orchestrator test.

One synthetic world, the real pipeline: a pre-logged prediction gets scored,
the injected roster is fitted and logs claims for all 72 fixtures, MarketBlend
joins when an odds snapshot exists (and carries its snapshot id into the
predictions log), the Monte Carlo writes tournament odds, and the publisher
exports every artifact. No network, no real DBs.

Plus the unattended-operation guarantees (E1/T2/D4): a failing fit drops that
entrant instead of crashing the loop, a fit exceeding the per-model timeout
gets dropped the same way, and integrity violations turn into a non-zero exit
code so the nightly Action goes red.
"""
import json
import signal
import sqlite3
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

from fifapreds.db import init_odds, init_predictions
from fifapreds.loop.predict import log_prediction
from fifapreds.models.elo import BaselineElo
from fifapreds.orchestrate import (
    _fit_failure_exceptions,
    _fit_roster,
    _fit_timeout,
    exit_code,
    run,
)
from tests.test_sim_montecarlo import GROUPS, TEAMS, FakeGoals, hierarchy, synthetic_fixtures


class Store:
    """Minimal MatchStore stand-in over a synthetic frame."""

    def __init__(self, matches: pd.DataFrame):
        self.all = matches
        self.played = (
            matches[matches["is_played"]]
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

    def upcoming(self, start=None, end=None):
        fx = self.all[~self.all["is_played"]]
        if start is not None:
            fx = fx[fx["date"] >= pd.Timestamp(start)]
        if end is not None:
            fx = fx[fx["date"] < pd.Timestamp(end)]
        return fx.sort_values("date", kind="stable").reset_index(drop=True)


def synthetic_world() -> pd.DataFrame:
    """2025 friendlies (every team plays its group rivals once, decided by
    strength) + the 72 unplayed 2026 group fixtures, dated in the future."""
    rows = []
    strengths = hierarchy()
    date = pd.Timestamp("2025-03-01")
    for _, sub in GROUPS.groupby("group"):
        teams = list(sub["team"])
        for i in range(4):
            for j in range(i + 1, 4):
                h, a = teams[i], teams[j]
                hs, as_ = (2, 0) if strengths[h] > strengths[a] else (0, 2)
                rows.append(
                    {"date": date + pd.Timedelta(days=len(rows) % 60),
                     "home_team": h, "away_team": a,
                     "home_score": float(hs), "away_score": float(as_),
                     "tournament": "Friendly", "neutral": True,
                     "is_played": True, "went_to_et": False}
                )
    history = pd.DataFrame(rows)
    now = pd.Timestamp.now().normalize()
    fixtures = synthetic_fixtures()
    fixtures["date"] = fixtures["date"] - fixtures["date"].min() + now + pd.Timedelta(days=1)
    world = pd.concat([history, fixtures], ignore_index=True)
    world["match_id"] = np.arange(len(world), dtype="int64")
    return world


def odds_payload(fixture: pd.Series) -> str:
    """One The Odds API h2h event covering `fixture` at fair 3.0 odds."""
    return json.dumps([{
        "commence_time": fixture["date"].isoformat(),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "bookmakers": [{
            "key": "bk", "title": "BK",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": fixture["home_team"], "price": 3.0},
                {"name": fixture["away_team"], "price": 3.0},
                {"name": "Draw", "price": 3.0},
            ]}],
        }],
    }])


@pytest.fixture()
def world(tmp_path):
    matches = synthetic_world()
    store = Store(matches)
    conn = sqlite3.connect(tmp_path / "live.db")
    conn.row_factory = sqlite3.Row
    init_predictions(conn)
    init_odds(conn)
    return conn, store, tmp_path


def test_full_pipeline(world):
    conn, store, tmp_path = world

    # A claim made long ago for an already-played friendly: the SCORE step
    # must grade it on this run. The model only knows matches strictly before
    # that fixture (every team has earlier appearances by construction).
    old_fixture = store.played.sort_values("date").iloc[-1]
    early = BaselineElo().fit(
        store.played[store.played["date"] < old_fixture["date"]]
    )
    log_prediction(conn, early, old_fixture,
                   predicted_at=pd.Timestamp(old_fixture["date"]) - pd.Timedelta(days=1))

    # An odds snapshot covering the first upcoming fixture -> MarketBlend joins.
    first_up = store.upcoming().iloc[0]
    conn.execute(
        "INSERT INTO odds_snapshots (captured_at, sport_key, market, raw_json)"
        " VALUES (?, 'soccer_fifa_world_cup', 'h2h', ?)",
        (pd.Timestamp.now().isoformat(), odds_payload(first_up)),
    )
    conn.commit()

    roster = [BaselineElo(), FakeGoals(hierarchy(), decisiveness=0.7)]
    report = run(
        conn, store, roster,
        n_sims=64, seed=7,
        sim_path=tmp_path / "tournament_sim.parquet",
        artifacts_dir=tmp_path / "artifacts",
        live_db=tmp_path / "live.db",
        backtest_db=tmp_path / "no_backtest.db",
    )

    # SCORE: the old claim graded, nothing pending, no integrity violations.
    assert report["scored"] == 1
    assert report["violations"] == []

    # UPDATE: full roster + market blend entered.
    assert report["models"] == ["elo_baseline", "fake_goals", "market_blend"]

    # PREDICT: every model claimed all 72 upcoming fixtures.
    assert report["predicted"] == {
        "elo_baseline": 72, "fake_goals": 72, "market_blend": 72,
    }
    # The market entrant's rows carry the snapshot id; others stay NULL.
    snap_ids = dict(conn.execute(
        """SELECT model_id, COUNT(odds_snapshot_id) FROM predictions
           WHERE context='live' GROUP BY model_id"""
    ).fetchall())
    assert snap_ids["market_blend"] == 72 and snap_ids["fake_goals"] == 0

    # E6a: every goals-model claim captured its scoreline grid at predict
    # time; W/D/L-only and blend entrants stored none.
    grids_by_model = dict(conn.execute(
        """SELECT p.model_id, COUNT(*) FROM score_grids g
           JOIN predictions p ON p.prediction_id = g.prediction_id
           GROUP BY p.model_id"""
    ).fetchall())
    assert grids_by_model == {"fake_goals": 72}

    # SIMULATE: only the goals model simulates; probabilities conserved.
    assert report["simulated"] == {"fake_goals": 64}
    tournament = pd.read_parquet(tmp_path / "artifacts" / "tournament.parquet")
    assert tournament["p_champion"].sum() == pytest.approx(1.0)
    assert set(tournament["team"]) == set(TEAMS)
    assert report["sim_metas"][0]["seed"] == 7

    # PUBLISH: all artifacts exist and meta agrees.
    meta = json.loads((tmp_path / "artifacts" / "meta.json").read_text())
    assert meta["counts"]["tournament"] == 48
    assert meta["counts"]["upcoming"] == 216
    for name in ("upcoming.parquet", "leaderboard.parquet", "scored.parquet",
                 "calibration.parquet", "tournament.parquet"):
        assert (tmp_path / "artifacts" / name).exists(), name

    # Idempotence: an immediate rerun adds no new claims.
    again = run(
        conn, store, [BaselineElo(), FakeGoals(hierarchy(), decisiveness=0.7)],
        n_sims=16, seed=7,
        sim_path=tmp_path / "tournament_sim.parquet",
        artifacts_dir=tmp_path / "artifacts",
        live_db=tmp_path / "live.db",
        backtest_db=tmp_path / "no_backtest.db",
    )
    assert all(n == 0 for n in again["predicted"].values())


class FakeSamplingError(RuntimeError):
    """Stands in for pymc.exceptions.SamplingError (a RuntimeError subclass)."""


class ExplodingModel:
    """Roster entrant whose fit always fails — the loop must drop it, not die."""

    model_id = "exploding"

    def __init__(self, exc: Exception):
        self._exc = exc

    def fit(self, matches):
        raise self._exc


def test_fit_roster_drops_failing_entrants(world):
    _, store, _ = world
    roster = [
        BaselineElo(),
        ExplodingModel(FakeSamplingError("chain failed to converge")),
        ExplodingModel(np.linalg.LinAlgError("singular matrix")),
    ]
    fitted, notes = _fit_roster(roster, store.played)
    assert [m.model_id for m in fitted] == ["elo_baseline"]
    assert len(notes) == 2
    assert all("exploding: fit failed, dropped" in n for n in notes)


class SlowModel:
    """Roster entrant whose fit sleeps past the timeout — must be dropped."""

    model_id = "slow"

    def __init__(self, sleep_s: float):
        self._sleep_s = sleep_s

    def fit(self, matches):
        time.sleep(self._sleep_s)
        return self


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"),
                    reason="signal.SIGALRM is Unix-only; orchestrator runs on Linux CI / macOS dev.")
def test_fit_roster_drops_on_timeout(world):
    """D4: a fit exceeding the per-model timeout is dropped like any other
    failed entrant, with a distinct 'timed out' note so the cause is visible."""
    _, store, _ = world
    roster = [BaselineElo(), SlowModel(sleep_s=3)]
    fitted, notes = _fit_roster(roster, store.played, timeout_s=1)
    assert [m.model_id for m in fitted] == ["elo_baseline"]
    assert any("slow: fit timed out, dropped" in n for n in notes)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Unix-only")
def test_fit_timeout_restores_prior_handler():
    """The context manager must not leak its SIGALRM handler — a follow-on
    fit (or any other code) gets the handler that was installed before."""

    def sentinel(signum, frame):
        pass

    prev = signal.signal(signal.SIGALRM, sentinel)
    try:
        with _fit_timeout(60):
            pass
        assert signal.getsignal(signal.SIGALRM) is sentinel
    finally:
        signal.signal(signal.SIGALRM, prev)


def test_fit_failures_pick_up_pymc_when_installed(monkeypatch):
    class BoomError(Exception):  # NOT a RuntimeError — only reachable via pymc
        pass

    exceptions_mod = types.ModuleType("pymc.exceptions")
    exceptions_mod.SamplingError = BoomError
    pymc_mod = types.ModuleType("pymc")
    pymc_mod.exceptions = exceptions_mod
    monkeypatch.setitem(sys.modules, "pymc", pymc_mod)
    monkeypatch.setitem(sys.modules, "pymc.exceptions", exceptions_mod)
    assert BoomError in _fit_failure_exceptions()


def test_loop_survives_sampler_failure(world):
    conn, store, tmp_path = world
    report = run(
        conn, store,
        [ExplodingModel(FakeSamplingError("divergences")), FakeGoals(hierarchy())],
        n_sims=16, seed=3,
        sim_path=tmp_path / "tournament_sim.parquet",
        artifacts_dir=tmp_path / "artifacts",
        live_db=tmp_path / "live.db",
        backtest_db=tmp_path / "no_backtest.db",
    )
    assert report["models"] == ["fake_goals"]
    assert any("exploding: fit failed, dropped" in n for n in report["notes"])
    assert report["predicted"] == {"fake_goals": 72}
    assert exit_code(report) == 0


def test_exit_code_gates_on_violations():
    assert exit_code({"violations": []}) == 0
    assert exit_code({"violations": [7, 9]}) == 2


def test_bma_entrants_join_when_backtest_history_exists(world):
    """When a backtest DB exists with scored predictions for at least one
    fitted roster member, BMAEnsemble (and BMAGoalsEnsemble where every
    member is goals-capable) join the orchestrator's `entrants` list."""
    conn, store, tmp_path = world

    # Build a tiny backtest DB with two scored rows for the roster's models.
    bt_path = tmp_path / "synthetic_backtest.db"
    bt_conn = sqlite3.connect(bt_path)
    init_predictions(bt_conn)
    rows_p, rows_s = [], []
    for pid, (model_id, year) in enumerate(
        [("elo_baseline", 2018), ("elo_baseline", 2022),
         ("fake_goals", 2018), ("fake_goals", 2022)], start=1):
        rows_p.append((pid, f"backtest:wc{year}", year * 10 + pid,
                       "H", "A", "2020-01-01", 1, "WC",
                       0.45, 0.30, 0.25, model_id, "1", "abc", "h",
                       "2010-01-01", None, None, "2010-01-02"))
        rows_s.append((pid, "home", 0.0, 0.0, 0.0, "2020-01-02"))
    bt_conn.executemany(
        """INSERT INTO predictions
           (prediction_id, context, match_id, home_team, away_team,
            kickoff_ts, neutral, tournament, p_home, p_draw, p_away,
            model_id, model_version, code_version, hyperparams_hash,
            training_cutoff, odds_snapshot_id, seed, predicted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows_p)
    bt_conn.executemany(
        "INSERT INTO scores (prediction_id, outcome, log_loss, brier, rps, scored_at)"
        " VALUES (?,?,?,?,?,?)", rows_s)
    bt_conn.commit()
    bt_conn.close()

    report = run(
        conn, store,
        [BaselineElo(), FakeGoals(hierarchy())],
        n_sims=16, seed=2,
        sim_path=tmp_path / "tournament_sim.parquet",
        artifacts_dir=tmp_path / "artifacts",
        live_db=tmp_path / "live.db",
        backtest_db=bt_path,
    )
    assert "bma_ensemble" in report["models"]
    assert "bma_goals_ensemble" in report["models"]
    # Notes should describe the member list each BMA used.
    assert any("bma_ensemble: over" in n for n in report["notes"])
    assert any("bma_goals_ensemble: over" in n for n in report["notes"])


def test_runs_without_odds_snapshot(world):
    conn, store, tmp_path = world
    report = run(
        conn, store, [FakeGoals(hierarchy())],
        n_sims=16, seed=1,
        sim_path=tmp_path / "tournament_sim.parquet",
        artifacts_dir=tmp_path / "artifacts",
        live_db=tmp_path / "live.db",
        backtest_db=tmp_path / "no_backtest.db",
    )
    assert report["models"] == ["fake_goals"]
    assert any("market_blend: skipped" in n for n in report["notes"])
    assert report["predicted"] == {"fake_goals": 72}
