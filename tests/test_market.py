"""MarketBlend correctness: de-vig round-trips and overround removal on a
synthetic The Odds API payload (field names mirror the real snapshot),
incomplete books skipped, alias names canonicalized, blend math golden-valued,
graceful fallback off-coverage, and config-hash provenance that excludes the
per-run snapshot id. Ends with the log_prediction E2E: a market_blend row
lands with full provenance in a tmp sqlite log.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from fifapreds.db import init_odds
from fifapreds.loop.predict import log_prediction
from fifapreds.models.base import WDL, Model
from fifapreds.models.market import (MarketBlend, latest_h2h_probs,
                                     parse_h2h_snapshot)


class StubBase(Model):
    """Fixed-output base model — keeps the blend tests fast and exact."""

    model_id = "stub_base"
    model_version = "1"

    def __init__(self, wdl: WDL = WDL(0.5, 0.3, 0.2), trained_through="2026-06-01"):
        self._wdl = wdl
        self.trained_through = pd.Timestamp(trained_through)
        self.fit_count = 0

    def fit(self, matches):
        self.fit_count += 1
        return self

    def predict_wdl(self, home, away, *, neutral=False):
        return self._wdl

    def hyperparams(self):
        return {"wdl": [self._wdl.home, self._wdl.draw, self._wdl.away]}


def _book(key: str, outcomes: list[tuple[str, float]]) -> dict:
    return {
        "key": key,
        "title": key.title(),
        "last_update": "2026-06-10T10:30:45Z",
        "markets": [{
            "key": "h2h",
            "last_update": "2026-06-10T10:30:45Z",
            "outcomes": [{"name": n, "price": p} for n, p in outcomes],
        }],
    }


# Mirrors the real soccer_fifa_world_cup h2h payload shape (verified against
# the captured snapshot): a list of events with commence_time, home_team,
# away_team and nested bookmakers[].markets[].outcomes[] name/price pairs.
_PAYLOAD = json.dumps([
    {
        "id": "evt-usa-par",
        "sport_key": "soccer_fifa_world_cup",
        "sport_title": "FIFA World Cup",
        "commence_time": "2026-06-12T16:00:00Z",
        "home_team": "USA",                       # odds-feed alias
        "away_team": "Paraguay",
        "bookmakers": [
            _book("booka", [("USA", 2.0), ("Paraguay", 4.0), ("Draw", 3.2)]),
            _book("bookb", [("USA", 2.1), ("Paraguay", 3.9), ("Draw", 3.1)]),
            # No draw price: this book must not feed the consensus.
            _book("bookc", [("USA", 1.05), ("Paraguay", 30.0)]),
        ],
    },
    {
        "id": "evt-fair",
        "sport_key": "soccer_fifa_world_cup",
        "sport_title": "FIFA World Cup",
        "commence_time": "2026-06-11T19:00:00Z",
        "home_team": "Mexico",
        "away_team": "South Africa",
        "bookmakers": [
            _book("booka", [("Mexico", 3.0), ("South Africa", 3.0), ("Draw", 3.0)]),
        ],
    },
    {
        "id": "evt-no-complete-book",
        "sport_key": "soccer_fifa_world_cup",
        "sport_title": "FIFA World Cup",
        "commence_time": "2026-06-13T19:00:00Z",
        "home_team": "Haiti",
        "away_team": "Scotland",
        "bookmakers": [
            _book("booka", [("Haiti", 2.5), ("Scotland", 2.5)]),  # no Draw
        ],
    },
])


# ------------------------------------------------------------------ parsing

def test_parse_devig_valid_probabilities():
    df = parse_h2h_snapshot(_PAYLOAD)
    assert len(df) == 2  # the fixture with zero complete books is dropped
    for _, r in df.iterrows():
        assert r["p_home"] + r["p_draw"] + r["p_away"] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < min(r["p_home"], r["p_draw"], r["p_away"])
        assert max(r["p_home"], r["p_draw"], r["p_away"]) < 1.0
    assert df.iloc[0]["kickoff"] == pd.Timestamp("2026-06-12T16:00:00Z")
    assert "Haiti" not in set(df["home_team"])


def test_fair_odds_round_trip():
    # 3.0/3.0/3.0 carries no vig: de-vig must return exactly uniform.
    row = parse_h2h_snapshot(_PAYLOAD).set_index("home_team").loc["Mexico"]
    for p in (row["p_home"], row["p_draw"], row["p_away"]):
        assert p == pytest.approx(1 / 3, abs=1e-9)


def test_overround_removed_and_ordering_preserved():
    # Naive implied probabilities of 2.0/3.2/4.0 sum past 1 (the vig)…
    assert 1 / 2.0 + 1 / 3.2 + 1 / 4.0 > 1.0
    # …the de-vigged consensus sums to exactly 1 with ordering intact
    # (shorter price → higher probability).
    row = parse_h2h_snapshot(_PAYLOAD).iloc[0]
    assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0, abs=1e-9)
    assert row["p_home"] > row["p_draw"] > row["p_away"]


def test_incomplete_bookmaker_skipped():
    # bookc quotes no draw (and a wild 1.05 home price); only 2 books count.
    row = parse_h2h_snapshot(_PAYLOAD).iloc[0]
    assert row["n_bookmakers"] == 2
    assert row["p_home"] < 0.6  # bookc's ~95% home quote left no trace


def test_alias_names_canonicalized():
    df = parse_h2h_snapshot(_PAYLOAD)
    assert df.iloc[0]["home_team"] == "United States"
    assert "USA" not in set(df["home_team"]) | set(df["away_team"])


# ----------------------------------------------------------- latest_h2h_probs

def test_latest_h2h_probs_returns_newest_snapshot():
    conn = sqlite3.connect(":memory:")
    init_odds(conn)
    older = json.dumps([])  # an earlier, empty pull
    for captured_at, raw in [("2026-06-09T10:00:00+00:00", older),
                             ("2026-06-10T10:00:00+00:00", _PAYLOAD)]:
        conn.execute(
            "INSERT INTO odds_snapshots (captured_at, sport_key, market, raw_json) "
            "VALUES (?, 'soccer_fifa_world_cup', 'h2h', ?)", (captured_at, raw))
    conn.commit()
    snapshot_id, df = latest_h2h_probs(conn)
    assert snapshot_id == 2
    assert "United States" in set(df["home_team"])


def test_latest_h2h_probs_clear_error_when_missing():
    conn = sqlite3.connect(":memory:")
    with pytest.raises(LookupError, match="no h2h snapshot"):
        latest_h2h_probs(conn)          # table absent entirely
    init_odds(conn)
    with pytest.raises(LookupError, match="no h2h snapshot"):
        latest_h2h_probs(conn)          # table present but empty


# ------------------------------------------------------------------ blending

_MARKET = {("United States", "Paraguay"): (0.6, 0.25, 0.15)}


def test_blend_matches_hand_computation():
    blend = MarketBlend(StubBase(), _MARKET, blend_weight=0.75)
    wdl = blend.predict_wdl("United States", "Paraguay", neutral=True)
    assert wdl.home == pytest.approx(0.75 * 0.6 + 0.25 * 0.5, abs=1e-9)
    assert wdl.draw == pytest.approx(0.75 * 0.25 + 0.25 * 0.3, abs=1e-9)
    assert wdl.away == pytest.approx(0.75 * 0.15 + 0.25 * 0.2, abs=1e-9)


def test_blend_weight_extremes():
    base = StubBase()
    pure_market = MarketBlend(base, _MARKET, blend_weight=1.0)
    wdl = pure_market.predict_wdl("United States", "Paraguay", neutral=True)
    assert (wdl.home, wdl.draw, wdl.away) == pytest.approx((0.6, 0.25, 0.15), abs=1e-9)
    pure_base = MarketBlend(base, _MARKET, blend_weight=0.0)
    wdl = pure_base.predict_wdl("United States", "Paraguay", neutral=True)
    assert (wdl.home, wdl.draw, wdl.away) == pytest.approx((0.5, 0.3, 0.2), abs=1e-9)


def test_blend_weight_out_of_range_rejected():
    with pytest.raises(ValueError, match="blend_weight"):
        MarketBlend(StubBase(), _MARKET, blend_weight=1.5)


def test_blend_accepts_parse_output_frame():
    blend = MarketBlend(StubBase(), parse_h2h_snapshot(_PAYLOAD), blend_weight=1.0)
    wdl = blend.predict_wdl("Mexico", "South Africa", neutral=True)
    assert wdl.home == pytest.approx(1 / 3, abs=1e-9)


def test_uncovered_fixture_falls_back_to_base():
    base = StubBase()
    blend = MarketBlend(base, _MARKET, blend_weight=0.75)
    assert blend.predict_wdl("Japan", "Tunisia", neutral=True) == base._wdl


# ---------------------------------------------------------------- provenance

def test_hyperparams_hash_pins_config_not_snapshot():
    a = MarketBlend(StubBase(), _MARKET, blend_weight=0.75, snapshot_id=1)
    b = MarketBlend(StubBase(), _MARKET, blend_weight=0.5, snapshot_id=1)
    c = MarketBlend(StubBase(), _MARKET, blend_weight=0.75, snapshot_id=99)
    assert a.hyperparams_hash != b.hyperparams_hash   # weight is config
    assert a.hyperparams_hash == c.hyperparams_hash   # snapshot id is not
    assert a.hyperparams()["base"]["model_id"] == "stub_base"
    assert c.snapshot_id == 99                        # exposed for the caller


def test_fit_and_trained_through_delegate_to_base():
    base = StubBase(trained_through="2026-06-01")
    blend = MarketBlend(base, _MARKET)
    assert blend.fit(pd.DataFrame()) is blend
    assert base.fit_count == 1
    assert blend.trained_through == base.trained_through


# ---------------------------------------------------------------- E2E logging

def test_log_prediction_carries_market_blend_provenance():
    conn = sqlite3.connect(":memory:")
    base = StubBase(trained_through="2026-06-01")
    model = MarketBlend(base, parse_h2h_snapshot(_PAYLOAD),
                        blend_weight=0.75, snapshot_id=7)
    fixture = pd.Series({"date": "2026-06-12", "home_team": "United States",
                         "away_team": "Paraguay", "neutral": True,
                         "tournament": "FIFA World Cup", "match_id": 101})
    pid = log_prediction(conn, model, fixture, predicted_at="2026-06-11T00:00:00",
                         odds_snapshot_id=model.snapshot_id)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM predictions WHERE prediction_id=?", (pid,)).fetchone()
    assert row["model_id"] == "market_blend" and row["model_version"] == "1"
    assert row["hyperparams_hash"] == model.hyperparams_hash
    assert row["odds_snapshot_id"] == 7
    assert row["training_cutoff"] == base.trained_through.isoformat()
    assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0)
