"""Knockout-bracket routing for the 48-team 2026 format (T10).

Round of 32 = matches 73-88. Sixteen slots are fixed (group winners and
runners-up per `bracket_2026.csv`); the other eight are the best third-placed
teams, whose placement depends on WHICH eight of the twelve groups they come
from. FIFA's official allocation table covers all C(12,8) = 495 combinations;
it was reproduced, permutation-checked and user-verified in Block V as
`routing_r32.parquet`: one row per combination, mapping each of the eight
third-hosting group winners (1A,1B,1D,1E,1G,1I,1K,1L) to the group letter of
the third it plays.

This module is read-only over those two verified artifacts. Getting a third
into the wrong bracket half corrupts every downstream probability, so
`r32_pairings` re-checks each lookup against the bracket's allowed pools at
runtime, and tests/test_sim_routing.py audits all 495 rows (the CRITICAL gate
from the plan).

Rounds beyond the R32 are pure recursion on match numbers
("Winner Match N" / "Loser Match N" for the third-place play-off) and are
exposed as `knockout_template()` for the Monte Carlo to walk.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping

import pandas as pd

from fifapreds.config import PROJECT_ROOT

BRACKET_CSV = PROJECT_ROOT / "data" / "raw" / "bracket_2026.csv"
ROUTING_PARQUET = PROJECT_ROOT / "data" / "raw" / "routing_r32.parquet"

#: The eight group winners that host third-placed teams, and where (verified
#: in Block V against the official allocation table).
THIRD_HOST_MATCHES: dict[str, int] = {
    "A": 79, "B": 85, "D": 81, "E": 74, "G": 82, "I": 77, "K": 87, "L": 80,
}

R32_MATCHES = range(73, 89)

_WINNER_RE = re.compile(r"^Winner Group ([A-L])$")
_RUNNER_RE = re.compile(r"^Runner-up Group ([A-L])$")
_THIRD_RE = re.compile(r"^3rd Group ([A-L](?:/[A-L])+)$")
_FROM_MATCH_RE = re.compile(r"^(Winner|Loser) Match (\d+)$")


@lru_cache(maxsize=1)
def bracket() -> pd.DataFrame:
    """The match-73..104 skeleton, indexed by match number."""
    df = pd.read_csv(BRACKET_CSV)
    return df.set_index("match")


@lru_cache(maxsize=1)
def routing_table() -> pd.DataFrame:
    """The 495-combination third-place allocation table, keyed by the sorted
    string of the eight qualified groups (e.g. 'ABDEGIKL')."""
    df = pd.read_parquet(ROUTING_PARQUET)
    return df.set_index("groups")


@lru_cache(maxsize=1)
def third_pools() -> dict[int, frozenset[str]]:
    """Allowed third-place source groups per R32 match, parsed straight from
    the bracket CSV's '3rd Group C/E/F/H/I' slots."""
    pools: dict[int, frozenset[str]] = {}
    for match in R32_MATCHES:
        away = bracket().loc[match, "away"]
        m = _THIRD_RE.match(away)
        if m:
            pools[match] = frozenset(m.group(1).split("/"))
    return pools


def thirds_assignment(qualified_groups: Mapping[str, str] | list[str]) -> dict[int, str]:
    """{R32 match number -> group letter of the third playing there} for one
    set of eight qualified third-place groups."""
    letters = sorted(qualified_groups)
    if len(letters) != 8 or len(set(letters)) != 8:
        raise ValueError(f"need exactly 8 distinct qualified groups, got {letters}")
    key = "".join(letters)
    try:
        row = routing_table().loc[key]
    except KeyError:
        raise KeyError(f"combination {key!r} not in routing table") from None
    out: dict[int, str] = {}
    for host, match in THIRD_HOST_MATCHES.items():
        third_group = row[f"1{host}"]
        if int(row[f"1{host}_match"]) != match:
            raise AssertionError(
                f"routing artifact disagrees with bracket: 1{host} at match "
                f"{row[f'1{host}_match']}, expected {match}"
            )
        if third_group not in third_pools()[match]:
            raise AssertionError(
                f"routing artifact sends 3{third_group} to match {match}, "
                f"outside its allowed pool {sorted(third_pools()[match])}"
            )
        out[match] = third_group
    return out


@lru_cache(maxsize=1)
def r32_slots() -> dict[int, tuple[tuple[str, str | None], tuple[str, str | None]]]:
    """Matches 73-88 parsed once into {match -> (home_spec, away_spec)} with
    spec = ('winner', group) | ('runner', group) | ('third', None); the third's
    source group is combination-dependent (see `thirds_assignment`). Parsed
    eagerly so the Monte Carlo hot loop never touches pandas or regexes."""
    slots: dict[int, tuple[tuple[str, str | None], tuple[str, str | None]]] = {}
    for match in R32_MATCHES:
        sides = []
        for slot in (bracket().loc[match, "home"], bracket().loc[match, "away"]):
            if m := _WINNER_RE.match(slot):
                sides.append(("winner", m.group(1)))
            elif m := _RUNNER_RE.match(slot):
                sides.append(("runner", m.group(1)))
            elif _THIRD_RE.match(slot):
                sides.append(("third", None))
            else:  # bracket artifact corrupted — refuse to guess
                raise ValueError(f"unrecognised R32 slot {slot!r} in match {match}")
        slots[match] = (sides[0], sides[1])
    return slots


def r32_pairings(
    winners: Mapping[str, str],
    runners: Mapping[str, str],
    best_thirds: Mapping[str, str],
) -> dict[int, tuple[str, str]]:
    """Resolve matches 73-88 to actual (home, away) team names.

    `winners`/`runners`: group letter -> team, all twelve groups.
    `best_thirds`: group letter -> team for exactly the eight qualifiers.
    """
    assignment = thirds_assignment(list(best_thirds))
    pairings: dict[int, tuple[str, str]] = {}
    for match, (home_spec, away_spec) in r32_slots().items():
        sides = []
        for kind, group in (home_spec, away_spec):
            if kind == "winner":
                sides.append(winners[group])
            elif kind == "runner":
                sides.append(runners[group])
            else:
                sides.append(best_thirds[assignment[match]])
        pairings[match] = (sides[0], sides[1])
    return pairings


@lru_cache(maxsize=1)
def knockout_template() -> dict[int, tuple[tuple[str, int], tuple[str, int]]]:
    """Matches 89-104 as {match -> ((take, from_match), (take, from_match))}
    where `take` is 'winner' or 'loser'. Match 103 is the third-place
    play-off, 104 the final."""
    template: dict[int, tuple[tuple[str, int], tuple[str, int]]] = {}
    for match, row in bracket().iterrows():
        if match in R32_MATCHES:
            continue
        sides = []
        for slot in (row["home"], row["away"]):
            m = _FROM_MATCH_RE.match(slot)
            if not m:
                raise ValueError(f"unrecognised knockout slot {slot!r} in match {match}")
            sides.append((m.group(1).lower(), int(m.group(2))))
        template[match] = (sides[0], sides[1])
    return template
