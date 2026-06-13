"""Confederation map invariants: every team in the match data resolves, and
the map covers all six FIFA confederations."""
from __future__ import annotations

import pandas as pd
import pytest

from fifapreds.models.confederations import team_confederation, all_confederations


def test_every_team_in_data_resolves():
    """Mirrors the registry pattern: unknown names must never silently pass."""
    matches = pd.read_parquet("data/processed/matches.parquet")
    all_teams = set(matches["home_team"]) | set(matches["away_team"])
    missing = [t for t in sorted(all_teams) if t not in all_confederations()]
    assert missing == [], f"teams missing from confederations.csv: {missing}"


def test_six_confederations():
    confeds = set(all_confederations().values())
    assert confeds == {"AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"}


def test_unknown_team_raises():
    with pytest.raises(KeyError, match="confederations.csv"):
        team_confederation("Atlantis United")


def test_known_teams():
    assert team_confederation("Brazil") == "CONMEBOL"
    assert team_confederation("Japan") == "AFC"
    assert team_confederation("France") == "UEFA"
    assert team_confederation("Nigeria") == "CAF"
    assert team_confederation("Mexico") == "CONCACAF"
    assert team_confederation("New Zealand") == "OFC"
