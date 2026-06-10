"""T9 — group standings and FIFA tiebreakers.

Invariant/golden-first: hand-computed tables, head-to-head and recursive
tiebreak cases, lots determinism under a fixed seed, and permutation
invariance of the inputs.
"""
import numpy as np
import pytest

from fifapreds.sim.groups import TeamRecord, rank_thirds, standings


def rng(seed=0):
    return np.random.default_rng(seed)


# A clean group: W beats everyone, X beats Y and Z, Y beats Z.
CLEAN = [
    ("W", "X", 2, 0), ("W", "Y", 3, 1), ("W", "Z", 1, 0),
    ("X", "Y", 2, 1), ("X", "Z", 1, 0), ("Y", "Z", 2, 0),
]


def test_points_order_golden():
    table = standings(["W", "X", "Y", "Z"], CLEAN, rng())
    assert [r.team for r in table] == ["W", "X", "Y", "Z"]
    assert [r.points for r in table] == [9, 6, 3, 0]
    assert table[0].gf == 6 and table[0].ga == 1 and table[0].gd == 5
    assert all(r.played == 3 for r in table)


def test_goal_difference_breaks_points_tie():
    # A and B both finish on 6 points; A's GD is better.
    results = [
        ("A", "B", 0, 1),  # head-to-head favors B — must NOT matter before GD
        ("A", "C", 4, 0), ("A", "D", 3, 0),
        ("B", "C", 1, 0), ("B", "D", 0, 1),
        ("C", "D", 1, 1),
    ]
    table = standings(["A", "B", "C", "D"], results, rng())
    a, b = table[0], table[1]
    assert (a.team, b.team) == ("A", "B")  # GD +6 vs +2 outranks h2h loss
    assert a.points == b.points == 6


def test_head_to_head_after_full_tie():
    # A and B finish identical on all of points/GD/GF (criteria 1-3):
    #   A: 2-1 W (vs B), 0-1 L (vs C), 1-1 D (vs D) -> 4 pts, gf 3, ga 3
    #   B: 1-2 L (vs A), 1-0 W (vs C), 1-1 D (vs D) -> 4 pts, gf 3, ga 3
    # Only the head-to-head (criterion 4) separates them: A won it.
    results = [
        ("A", "B", 2, 1),
        ("A", "C", 0, 1),
        ("B", "C", 1, 0),
        ("A", "D", 1, 1),
        ("B", "D", 1, 1),
        ("C", "D", 9, 0),
    ]
    table = standings(["A", "B", "C", "D"], results, rng())
    pos = {r.team: i for i, r in enumerate(table)}
    assert pos["A"] < pos["B"], "head-to-head winner must rank above"


def test_three_way_tie_exhausted_goes_to_lots():
    # A perfect 1-0 cycle (A>B>C>A) plus identical 2-0 wins over D: A, B, C
    # are tied on every criterion, overall AND head-to-head -> lots.
    results = [
        ("A", "B", 1, 0),
        ("B", "C", 1, 0),
        ("C", "A", 1, 0),
        ("A", "D", 2, 0), ("B", "D", 2, 0), ("C", "D", 2, 0),
    ]
    table = standings(["A", "B", "C", "D"], results, rng(42))
    assert table[3].team == "D"
    top3 = {r.team for r in table[:3]}
    assert top3 == {"A", "B", "C"}
    # Deterministic under the same seed.
    again = standings(["A", "B", "C", "D"], results, rng(42))
    assert [r.team for r in again] == [r.team for r in table]
    # Different seeds eventually produce a different lots order.
    orders = {
        tuple(r.team for r in standings(["A", "B", "C", "D"], results, rng(s)))
        for s in range(20)
    }
    assert len(orders) > 1, "lots should vary by seed"


def test_h2h_goal_difference_separates_cycle():
    # B, C, D identical overall (6 pts, +4, 8 goals — checked by hand): the
    # trio's mutual games form a cycle, all 3 h2h points, but with different
    # scorelines, so h2h goal difference (criterion 5) orders them D, C, B.
    results = [
        ("D", "B", 3, 0),   # trio h2h gd: D +2
        ("B", "C", 1, 0),   # trio h2h gd: B -2
        ("C", "D", 1, 0),   # trio h2h gd: C  0
        ("D", "A", 5, 3), ("B", "A", 7, 1), ("C", "A", 7, 3),
    ]
    table = standings(["A", "B", "C", "D"], results, rng())
    assert [r.team for r in table] == ["D", "C", "B", "A"]
    assert [(r.points, r.gd, r.gf) for r in table[:3]] == [(6, 4, 8)] * 3


def test_permutation_invariance():
    # Shuffling team order and result order never changes a lots-free table.
    base = standings(["W", "X", "Y", "Z"], CLEAN, rng())
    for seed in range(5):
        r = np.random.default_rng(seed)
        teams = list(r.permutation(["W", "X", "Y", "Z"]))
        shuffled = [CLEAN[i] for i in r.permutation(len(CLEAN))]
        table = standings(teams, shuffled, rng(seed))
        assert [t.team for t in table] == [t.team for t in base]


def test_unknown_team_raises():
    with pytest.raises(KeyError):
        standings(["A", "B", "C", "D"], [("A", "Q", 1, 0)], rng())


def test_rank_thirds_orders_and_breaks_by_lots():
    recs = {
        "A": TeamRecord("a3", 3, 4, 5, 3),   # 4 pts, +2
        "B": TeamRecord("b3", 3, 4, 4, 2),   # 4 pts, +2, fewer scored
        "C": TeamRecord("c3", 3, 6, 4, 1),   # 6 pts — best
        "D": TeamRecord("d3", 3, 1, 2, 5),   # worst
        "E": TeamRecord("e3", 3, 4, 5, 3),   # exact tie with A -> lots
    }
    ranked = rank_thirds(recs.items(), rng(7))
    letters = [g for g, _ in ranked]
    assert letters[0] == "C" and letters[-1] == "D"
    assert set(letters[1:4]) == {"A", "B", "E"}
    # B (gf 4) must rank below both gf-5 records.
    assert letters.index("B") > max(letters.index("A"), letters.index("E"))
    # Same seed -> same order; the A/E coin flip is reproducible.
    again = rank_thirds(recs.items(), rng(7))
    assert [g for g, _ in again] == letters
