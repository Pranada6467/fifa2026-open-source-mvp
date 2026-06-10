"""Group-stage standings — FIFA 2026 tiebreakers (T9).

Orders a four-team group from its six results, per the FIFA World Cup 2026
regulations (Art. 13):

 1. points, 2. goal difference, 3. goals scored — across all group matches;
 4-6. the same three criteria restricted to matches *among the still-tied
      teams* (head-to-head), reapplied recursively to any subset that the
      restriction separates;
 7. fair-play points — NOT modellable here (martj42 has no card data); skipped
    and documented, which slightly over-uses the final criterion;
 8. drawing of lots — a seeded `numpy.random.Generator`, so simulations are
    reproducible and lots-decided orderings vary across Monte Carlo draws
    instead of hiding behind a fixed arbitrary order.

The third-place ranking across groups (best 8 of 12 advance in the 48-team
format) uses criteria 1-3 then lots — head-to-head is undefined across groups.

Everything here is plain Python over small tuples: this code sits inside the
Monte Carlo hot loop (12 groups x N sims), so no pandas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Result = tuple[str, str, int, int]  # (home, away, home_goals, away_goals)


@dataclass(frozen=True)
class TeamRecord:
    """One team's line in a group table."""

    team: str
    played: int
    points: int
    gf: int
    ga: int

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _records(teams: Sequence[str], results: Iterable[Result]) -> dict[str, TeamRecord]:
    played = {t: 0 for t in teams}
    points = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for home, away, hg, ag in results:
        for side in (home, away):
            if side not in played:
                raise KeyError(f"result team {side!r} not in group {list(teams)}")
        played[home] += 1
        played[away] += 1
        gf[home] += hg
        ga[home] += ag
        gf[away] += ag
        ga[away] += hg
        if hg > ag:
            points[home] += 3
        elif hg < ag:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1
    return {
        t: TeamRecord(team=t, played=played[t], points=points[t], gf=gf[t], ga=ga[t])
        for t in teams
    }


def _key(rec: TeamRecord) -> tuple[int, int, int]:
    """Criteria 1-3 as a descending sort key."""
    return (rec.points, rec.gd, rec.gf)


def _tied_clusters(records: list[TeamRecord]) -> list[list[TeamRecord]]:
    """Partition a key-sorted list into runs with identical (pts, gd, gf)."""
    clusters: list[list[TeamRecord]] = []
    for rec in records:
        if clusters and _key(clusters[-1][0]) == _key(rec):
            clusters[-1].append(rec)
        else:
            clusters.append([rec])
    return clusters


def _break_tie(
    tied: list[TeamRecord],
    results: list[Result],
    rng: np.random.Generator,
) -> list[str]:
    """Order a fully-tied cluster: head-to-head among the tied teams (criteria
    4-6), recursing into any sub-cluster the restriction separates; teams a
    head-to-head table cannot separate go to lots (criterion 8)."""
    if len(tied) == 1:
        return [tied[0].team]
    names = {r.team for r in tied}
    h2h_results = [r for r in results if r[0] in names and r[1] in names]
    h2h = _records(sorted(names), h2h_results)
    ordered = sorted((h2h[r.team] for r in tied), key=_key, reverse=True)
    clusters = _tied_clusters(ordered)
    if len(clusters) == 1:
        # Head-to-head separated nothing new — fair play unavailable, so lots.
        teams = [r.team for r in clusters[0]]
        return list(rng.permutation(teams))
    out: list[str] = []
    for cluster in clusters:
        out.extend(_break_tie(cluster, results, rng))
    return out


def standings(
    teams: Sequence[str],
    results: Iterable[Result],
    rng: np.random.Generator,
) -> list[TeamRecord]:
    """Final group order, best first. `results` are the group's matches;
    `rng` only matters when the full tiebreaker ladder is exhausted (lots)."""
    results = list(results)
    recs = _records(teams, results)
    ordered = sorted(recs.values(), key=_key, reverse=True)
    final: list[str] = []
    for cluster in _tied_clusters(ordered):
        final.extend(_break_tie(cluster, results, rng))
    return [recs[t] for t in final]


def rank_thirds(
    thirds: Iterable[tuple[str, TeamRecord]],
    rng: np.random.Generator,
) -> list[tuple[str, TeamRecord]]:
    """Order the twelve third-placed teams, best first (top 8 advance).

    Cross-group, so only criteria 1-3 apply, then lots. Returns
    (group_letter, record) pairs.
    """
    items = list(thirds)
    # Shuffle first so equal keys land in rng order (sort is stable).
    items = [items[i] for i in rng.permutation(len(items))]
    return sorted(items, key=lambda gr: _key(gr[1]), reverse=True)
