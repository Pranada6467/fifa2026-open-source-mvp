"""BaselineElo invariants: zero-sum updates, home-advantage off when neutral,
idle-rating decay, importance weighting, and loud failures on unknown teams /
out-of-order updates (the silent-corruption paths). One real-data sanity check
pins the whole pipeline to reality.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.models import BaselineElo


def _matches(rows) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


_HISTORY = _matches([
    ("2010-01-01", "A", "B", 2, 0, "Friendly", False),
    ("2010-06-01", "C", "A", 1, 1, "Friendly", True),
    ("2011-01-01", "B", "C", 0, 3, "FIFA World Cup", True),
    ("2011-06-01", "A", "C", 1, 0, "Friendly", False),
])


def test_updates_are_zero_sum():
    model = BaselineElo().fit(_HISTORY)
    # Every update moves the two teams by ±delta, so total rating is conserved.
    total = sum(model.ratings.values())
    assert total == pytest.approx(model.initial_rating * 3)
    # And a single update transfers exactly what it takes.
    before = model.ratings
    model.update({"date": "2012-01-01", "home_team": "A", "away_team": "B",
                  "home_score": 0, "away_score": 1, "neutral": True})
    gain_b = model.rating("B") - before["B"]
    loss_a = model.rating("A") - before["A"]
    assert gain_b > 0 and gain_b == pytest.approx(-loss_a)


def test_neutral_prediction_is_symmetric():
    model = BaselineElo().fit(_HISTORY)
    ab = model.predict_wdl("A", "B", neutral=True)
    ba = model.predict_wdl("B", "A", neutral=True)
    # Swapping sides on neutral ground mirrors the probabilities exactly.
    assert ab.home == pytest.approx(ba.away, rel=1e-12)
    assert ab.away == pytest.approx(ba.home, rel=1e-12)
    assert ab.draw == pytest.approx(ba.draw, rel=1e-12)


def test_home_advantage_applies_only_when_not_neutral():
    # Two teams with identical ratings (never trained, set directly to isolate
    # the venue term from history effects).
    model = BaselineElo()
    model._ratings = {"X": 1500.0, "Y": 1500.0}
    neutral = model.predict_wdl("X", "Y", neutral=True)
    at_home = model.predict_wdl("X", "Y", neutral=False)
    assert neutral.home == pytest.approx(neutral.away)   # no edge on neutral
    assert at_home.home > neutral.home                   # venue edge only at home
    assert at_home.away < neutral.away


def test_home_win_earns_less_than_neutral_win():
    # The expected score is higher at home, so beating the same opponent at
    # home moves the rating less than beating them on neutral ground.
    rows = [("2020-01-01", "X", "Y", 1, 0, "Friendly", False)]
    home_model = BaselineElo().fit(_matches(rows))
    rows[0] = ("2020-01-01", "X", "Y", 1, 0, "Friendly", True)
    neutral_model = BaselineElo().fit(_matches(rows))
    assert home_model.rating("X") < neutral_model.rating("X")
    assert home_model.rating("X") > home_model.initial_rating  # still a gain


def test_draw_probability_peaks_when_even_and_decays_with_gap():
    model = BaselineElo()
    model._ratings = {"E1": 1500.0, "E2": 1500.0, "S": 1700.0, "G": 2500.0}
    even = model.predict_wdl("E1", "E2", neutral=True)
    tilted = model.predict_wdl("S", "E1", neutral=True)
    lopsided = model.predict_wdl("G", "E1", neutral=True)
    # Even match: draw rate is the Davidson base rate nu/(2+nu).
    assert even.draw == pytest.approx(model.draw_nu / (2 + model.draw_nu))
    # Draw probability shrinks monotonically as the mismatch grows…
    assert even.draw > tilted.draw > lopsided.draw
    # …and even a 1000-point gap yields a valid, all-positive triple.
    assert lopsided.home > 0.95 and lopsided.draw > 0 and lopsided.away > 0


def test_decay_regresses_idle_rating_toward_initial():
    history = _matches([
        ("2010-01-01", "A", "B", 3, 0, "Friendly", True),
        ("2010-02-01", "A", "B", 2, 0, "Friendly", True),
        ("2010-03-01", "A", "B", 1, 0, "Friendly", True),
    ])
    gap_match = {"date": "2020-01-01", "home_team": "A", "away_team": "C",
                 "home_score": 0, "away_score": 1, "neutral": True}

    fresh = BaselineElo(decay_rate=0.0).fit(history)
    assert fresh.rating("A") > fresh.initial_rating  # built up a real lead

    # With aggressive decay, 10 idle years erase the lead: the gap match plays
    # out as 1500 v 1500 neutral (E=0.5, S=0 → delta = -k/2).
    decayed = BaselineElo(decay_rate=5.0).fit(history)
    decayed.update(gap_match)
    assert decayed.rating("A") == pytest.approx(1500.0 - decayed.k_factor / 2, abs=1e-6)

    # With decay off, the same match is played from the undecayed lead.
    fresh.update(gap_match)
    assert fresh.rating("A") > decayed.rating("A")


def test_importance_scales_the_update():
    rows = _matches([("2020-01-01", "A", "B", 1, 0, "FIFA World Cup", True)])
    base = BaselineElo().fit(rows)
    weighted = BaselineElo(importance={"FIFA World Cup": 2.0}).fit(rows)
    base_gain = base.rating("A") - base.initial_rating
    weighted_gain = weighted.rating("A") - weighted.initial_rating
    assert weighted_gain == pytest.approx(2.0 * base_gain)
    # Tournaments missing from the map fall back to multiplier 1.
    friendly = _matches([("2020-01-01", "A", "B", 1, 0, "Friendly", True)])
    unweighted = BaselineElo(importance={"FIFA World Cup": 2.0}).fit(friendly)
    assert unweighted.rating("A") == pytest.approx(base.rating("A"))


def test_unknown_team_raises():
    model = BaselineElo().fit(_HISTORY)
    with pytest.raises(KeyError):
        model.predict_wdl("A", "Naples")  # not a national team / bad spelling


def test_out_of_order_update_raises_same_day_allowed():
    model = BaselineElo().fit(_HISTORY)  # trained through 2011-06-01
    stale = {"date": "2011-01-01", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 0, "neutral": True}
    with pytest.raises(ValueError):
        model.update(stale)
    # Same-day is fine (international teams play at most once per day).
    same_day = dict(stale, date="2011-06-01", home_team="B", away_team="C")
    model.update(same_day)
    assert model.trained_through == pd.Timestamp("2011-06-01")


def test_unplayed_match_in_training_raises():
    fixture = _matches([("2026-06-11", "Mexico", "South Africa", None, None,
                         "FIFA World Cup", False)])
    with pytest.raises(ValueError):
        BaselineElo().fit(fixture)


def test_fit_is_deterministic():
    a = BaselineElo().fit(_HISTORY)
    b = BaselineElo().fit(_HISTORY)
    assert a.ratings == b.ratings and a.trained_through == b.trained_through


def test_real_history_sanity():
    """Fit on the full real history: strong national teams surface on top and
    a giant-vs-minnow prediction is confidently (not absurdly) one-sided."""
    store = MatchStore()
    model = BaselineElo().fit(store.played)
    assert model.trained_through == store.played["date"].max()

    top20 = sorted(model.ratings, key=model.ratings.get, reverse=True)[:20]
    giants = {"Brazil", "Argentina", "France", "Spain", "England",
              "Germany", "Netherlands", "Portugal"}
    assert len(giants & set(top20)) >= 3

    p = model.predict_wdl("Brazil", "Malta", neutral=True)
    assert p.home > 0.6 and p.away < p.draw < p.home
