"""Roster invariants: every entrant has a distinct, auditable identity
(model_id + hyperparams_hash), the frozen defaults actually change behaviour
in the hypothesized direction, the importance dict's keys match the real
martj42 tournament strings verbatim (a typo there would silently weight
nothing), and default_roster() hands out fresh state every call.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fifapreds.ingest import MATCHES_PARQUET
from fifapreds.models import BaselineElo, DixonColes, GoalsModel
from fifapreds.models.roster import (
    TOURNAMENT_IMPORTANCE,
    DixonColesSlowXi,
    DixonColesTournamentWeighted,
    EloDecay,
    EloImportance,
    default_roster,
    goals_models,
)


def _matches(rows) -> pd.DataFrame:
    cols = ["date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ----------------------------------------------------------------- identity

def test_roster_identities_are_unique_and_unfitted():
    roster = default_roster()
    assert len(roster) >= 5
    ids = [m.model_id for m in roster]
    assert len(set(ids)) == len(ids), f"duplicate model_id: {ids}"
    hashes = [m.hyperparams_hash for m in roster]
    assert len(set(hashes)) == len(hashes), "two entrants share a config hash"
    assert all(m.trained_through is None for m in roster)  # fresh = unfitted


def test_variant_hashes_differ_from_their_baselines():
    assert EloDecay().hyperparams_hash != BaselineElo().hyperparams_hash
    assert EloImportance().hyperparams_hash != BaselineElo().hyperparams_hash
    assert DixonColesSlowXi().hyperparams_hash != DixonColes().hyperparams_hash
    assert DixonColesTournamentWeighted().hyperparams_hash != DixonColes().hyperparams_hash
    # S1: the DC tournament-weighted variant is distinct from DC slow-xi too —
    # they vary different knobs, must not collide.
    assert (DixonColesTournamentWeighted().hyperparams_hash
            != DixonColesSlowXi().hyperparams_hash)


def test_frozen_defaults_flow_into_hyperparams_and_allow_override():
    # The subclass only moves the default; hyperparams() (inherited) must
    # already reflect it — that is what makes the provenance hash honest.
    assert EloDecay().hyperparams()["decay_rate"] == 0.1
    assert EloImportance().hyperparams()["importance"] == TOURNAMENT_IMPORTANCE
    assert DixonColesSlowXi().hyperparams()["xi"] == 0.0005
    assert DixonColesTournamentWeighted().hyperparams()["importance"] == TOURNAMENT_IMPORTANCE
    # Every other knob stays at the baseline default.
    assert EloDecay().k_factor == BaselineElo().k_factor
    assert EloDecay().importance is None
    assert EloImportance().decay_rate == BaselineElo().decay_rate
    assert DixonColesSlowXi().window_years == DixonColes().window_years
    assert DixonColesSlowXi().et_weight == DixonColes().et_weight
    assert DixonColesTournamentWeighted().xi == DixonColes().xi  # only importance moves
    # Explicit overrides still pass through to the parent constructor.
    assert EloDecay(decay_rate=0.25).decay_rate == 0.25
    assert DixonColesSlowXi(xi=0.001).xi == 0.001
    assert DixonColesTournamentWeighted(xi=0.001).xi == 0.001


# ------------------------------------------------- importance dict vs reality

def test_importance_keys_exist_verbatim_in_martj42():
    """A misspelled key (e.g. 'CONCACAF Gold Cup') would weight nothing and
    fail silently — pin every key to the actual parquet vocabulary."""
    if not MATCHES_PARQUET.exists():
        pytest.skip("matches.parquet not built (run fifapreds.ingest)")
    tournaments = set(
        pd.read_parquet(MATCHES_PARQUET, columns=["tournament"])["tournament"]
        .dropna()
        .unique()
    )
    missing = set(TOURNAMENT_IMPORTANCE) - tournaments
    assert not missing, f"importance keys absent from martj42 data: {sorted(missing)}"


# ----------------------------------------------------------------- behaviour

def test_elo_decay_weakens_an_idle_strong_team():
    history = _matches([
        ("2010-01-01", "A", "B", 3, 0, "Friendly", True),
        ("2010-02-01", "A", "B", 2, 0, "Friendly", True),
        ("2010-03-01", "A", "B", 1, 0, "Friendly", True),
        # A returns after a decade idle — decay applies to its pre-match
        # rating here (regression materializes when the team next plays).
        ("2020-03-01", "A", "C", 1, 1, "Friendly", True),
    ])
    base = BaselineElo().fit(history)
    decay = EloDecay().fit(history)
    # A built a real lead by 2010, then sat idle: the variant regresses it.
    assert base.rating("A") > base.initial_rating
    assert decay.rating("A") < base.rating("A")
    # Direction: the idle strong team's prediction edge shrinks.
    assert (
        decay.predict_wdl("A", "B", neutral=True).home
        < base.predict_wdl("A", "B", neutral=True).home
    )


def test_elo_importance_scales_updates_by_tournament():
    wc = _matches([("2020-01-01", "A", "B", 1, 0, "FIFA World Cup", True)])
    friendly = _matches([("2020-01-01", "A", "B", 1, 0, "Friendly", True)])
    gain_wc = EloImportance().fit(wc).rating("A") - 1500.0
    gain_fr = EloImportance().fit(friendly).rating("A") - 1500.0
    gain_base = BaselineElo().fit(wc).rating("A") - 1500.0
    # Bigger K -> bigger rating delta; friendlies move less than baseline.
    assert gain_wc > gain_base > gain_fr > 0
    # First match from equal ratings has identical expected score, so the
    # gains scale exactly with the frozen multipliers.
    assert gain_wc == pytest.approx(1.75 * gain_base)
    assert gain_fr == pytest.approx(0.5 * gain_base)


# -------------------------------------------------------------- roster hygiene

def test_default_roster_returns_fresh_instances_each_call():
    r1, r2 = default_roster(), default_roster()
    assert all(a is not b for a, b in zip(r1, r2))
    # Fitting one roster's model leaves the other roster untouched.
    r1[0].fit(_matches([("2020-01-01", "A", "B", 1, 0, "Friendly", True)]))
    assert r1[0].trained_through is not None and r2[0].trained_through is None
    # The importance dict is a per-instance copy, not the shared constant.
    imp1 = next(m for m in r1 if isinstance(m, EloImportance)).importance
    imp2 = next(m for m in r2 if isinstance(m, EloImportance)).importance
    assert imp1 == TOURNAMENT_IMPORTANCE
    assert imp1 is not imp2 and imp1 is not TOURNAMENT_IMPORTANCE


def test_goals_models_returns_exactly_the_goals_entries():
    roster = default_roster()
    goals = goals_models(roster)
    ids = [m.model_id for m in goals]
    assert "dixon_coles" in ids
    assert "dixon_coles_slow_xi" in ids
    assert all(isinstance(m, GoalsModel) for m in goals)
    # Elo entries (W/D/L-only) must never reach the simulator.
    assert not any(isinstance(m, BaselineElo) for m in goals)
