"""T10 — R32 routing. CRITICAL gate (plan): the routing artifact must agree
with the official bracket on every one of the 495 third-place combinations.

The parquet itself was permutation-checked and user-verified against FIFA's
table in Block V; these tests re-derive its internal consistency from scratch
on every run so a regressed or swapped file can never slip through silently.
"""
from itertools import combinations

import pytest

from fifapreds.sim.routing import (
    R32_MATCHES,
    THIRD_HOST_MATCHES,
    bracket,
    knockout_template,
    r32_pairings,
    routing_table,
    third_pools,
    thirds_assignment,
)

GROUPS = "ABCDEFGHIJKL"


def test_artifact_covers_all_495_combinations():
    table = routing_table()
    assert len(table) == 495
    expected = {"".join(c) for c in combinations(GROUPS, 8)}
    assert set(table.index) == expected


def test_every_combination_routes_consistently():
    """The full audit: for each combo, the eight assigned thirds are exactly
    the eight qualified groups (a permutation — no team used twice, none
    dropped) and each lands inside its match's allowed pool."""
    pools = third_pools()
    for combo in routing_table().index:
        assignment = thirds_assignment(list(combo))
        assert sorted(assignment.values()) == sorted(combo), combo
        assert set(assignment) == set(THIRD_HOST_MATCHES.values()), combo
        for match, group in assignment.items():
            assert group in pools[match], (combo, match, group)


def test_host_matches_match_plan():
    # Pinned in CLAUDE.md from the user-verified official table.
    assert THIRD_HOST_MATCHES == {
        "A": 79, "B": 85, "D": 81, "E": 74, "G": 82, "I": 77, "K": 87, "L": 80,
    }
    # And the bracket CSV agrees: those matches host '3rd Group ...' slots
    # whose home side is the matching group winner.
    for host, match in THIRD_HOST_MATCHES.items():
        assert bracket().loc[match, "home"] == f"Winner Group {host}"
        assert bracket().loc[match, "away"].startswith("3rd Group ")
    assert set(third_pools()) == set(THIRD_HOST_MATCHES.values())


def test_r32_pairings_end_to_end():
    winners = {g: f"W{g}" for g in GROUPS}
    runners = {g: f"R{g}" for g in GROUPS}
    qualified = "ABCDEFGH"
    best_thirds = {g: f"T{g}" for g in qualified}
    pairings = r32_pairings(winners, runners, best_thirds)

    assert set(pairings) == set(R32_MATCHES)
    # Fixed slots straight from the bracket skeleton.
    assert pairings[73] == ("RA", "RB")
    assert pairings[76] == ("WC", "RF")
    assert pairings[88] == ("RD", "RG")
    # Third-hosting matches: home is the group winner, away is a qualified third.
    for host, match in THIRD_HOST_MATCHES.items():
        home, away = pairings[match]
        assert home == f"W{host}"
        assert away.startswith("T") and away[1:] in qualified
    # All 32 slots filled by 32 distinct teams: 12 winners + 12 runners-up +
    # 8 thirds, nobody twice.
    teams = [t for pair in pairings.values() for t in pair]
    assert len(teams) == 32 and len(set(teams)) == 32
    assert sum(t.startswith("T") for t in teams) == 8


def test_wrong_number_of_thirds_raises():
    with pytest.raises(ValueError):
        thirds_assignment(list("ABC"))
    with pytest.raises(ValueError):
        thirds_assignment(list("AABCDEFG"))


def test_knockout_template_structure():
    t = knockout_template()
    assert set(t) == set(range(89, 105))
    # R16 consumes each R32 match exactly once.
    r16_sources = [m for match in range(89, 97) for (_, m) in t[match]]
    assert sorted(r16_sources) == list(R32_MATCHES)
    # Semis feed both the final (winners) and the play-off (losers).
    assert t[103] == (("loser", 101), ("loser", 102))
    assert t[104] == (("winner", 101), ("winner", 102))
    # Every non-final match's winner is consumed exactly once downstream.
    consumed = [src for match in t for (take, src) in t[match] if take == "winner"]
    assert sorted(consumed) == sorted(set(consumed))
    assert set(consumed) == set(range(73, 103))
