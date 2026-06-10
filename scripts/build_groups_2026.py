"""Persist the official 2026 World Cup group assignments (A-L) and validate them
against the fixture-derived clusters in martj42 — so any transcription error is
caught automatically.

Source: 2026 FIFA World Cup final draw (Dec 5, 2025). Names use martj42 canonical
spelling (via registry.canonical) so they join the history cleanly.

Output: data/raw/groups_2026.csv  (columns: group, team)
Guard: each group's 4 teams must equal one connected component of the 72
group-stage fixtures (teams that play each other share a group); host anchors
A=Mexico, B=Canada, D=United States must hold.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fifapreds.registry import canonical

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Official final-draw result (authoritative).
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
HOST_ANCHORS = {"A": "Mexico", "B": "Canada", "D": "United States"}


def fixture_clusters() -> list[set[str]]:
    """Derive group membership from the 72 group-stage fixtures: teams that face
    each other are in the same group (connected components)."""
    import networkx as nx

    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
    g = nx.Graph()
    for _, r in wc.iterrows():
        g.add_edge(r.home_team, r.away_team)
    return list(nx.connected_components(g))


def main() -> None:
    groups = {g: [canonical(t) for t in teams] for g, teams in GROUPS.items()}

    # Guard 1: host anchors.
    for g, host in HOST_ANCHORS.items():
        assert host in groups[g], f"host anchor failed: {host} not in Group {g}"

    # Guard 2: each official group equals one fixture-derived cluster.
    clusters = [frozenset(c) for c in fixture_clusters()]
    for g, teams in groups.items():
        s = frozenset(teams)
        assert len(teams) == 4, (g, teams)
        assert s in clusters, f"Group {g} {sorted(teams)} has no matching fixture cluster"

    # Guard 3: 48 distinct teams total.
    allteams = [t for ts in groups.values() for t in ts]
    assert len(allteams) == 48 and len(set(allteams)) == 48

    rows = [{"group": g, "team": t} for g, ts in groups.items() for t in ts]
    out = pd.DataFrame(rows)
    out.to_csv(RAW / "groups_2026.csv", index=False)
    print(f"groups_2026.csv: {len(out)} teams across {len(groups)} groups — all guards passed")


if __name__ == "__main__":
    main()
