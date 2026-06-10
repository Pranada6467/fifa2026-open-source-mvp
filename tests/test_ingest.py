"""Ingest correctness: canonical names, played/fixture split, ET flagging."""
from __future__ import annotations

import pandas as pd

from fifapreds.ingest import build_matches


def test_played_split_matches_scores():
    m = build_matches(write=False)
    # is_played is exactly "both scores present".
    assert (m["is_played"] == (m["home_score"].notna() & m["away_score"].notna())).all()
    # The 2026 World Cup group fixtures are present but unplayed.
    wc = m[(m["tournament"] == "FIFA World Cup") & (m["date"].dt.year == 2026)]
    assert len(wc) == 72 and not wc["is_played"].any()


def test_team_names_canonical():
    m = build_matches(write=False)
    teams = set(m["home_team"]) | set(m["away_team"])
    # Aliases are normalised away.
    assert "USA" not in teams and "United States" in teams
    assert "Bosnia & Herzegovina" not in teams


def test_went_to_et_flags_shootouts():
    m = build_matches(write=False)
    # 2022 World Cup final (Argentina 3-3 France, won on penalties) reached ET.
    final = m[(m["date"] == pd.Timestamp("2022-12-18"))
              & (m[["home_team", "away_team"]].isin(["Argentina", "France"]).all(axis=1))]
    assert len(final) == 1 and bool(final["went_to_et"].iloc[0]) is True
    # A run-of-the-mill group game did not.
    assert m["went_to_et"].sum() < len(m) * 0.1  # ET is rare overall


def test_all_wc2026_teams_have_history():
    m = build_matches(write=False)
    wc = m[(m["tournament"] == "FIFA World Cup") & (m["date"].dt.year == 2026)]
    wc_teams = set(wc["home_team"]) | set(wc["away_team"])
    assert len(wc_teams) == 48
    played = m[m["is_played"]]
    hist = set(played["home_team"]) | set(played["away_team"])
    assert wc_teams <= hist  # every WC team appears in played history
