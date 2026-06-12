"""E4 — group-qualification backtest: groups recovered from fixtures alone
(self-validating), deterministic qualification under a deterministic model,
seeded reproducibility, and the pre-tournament no-leak guard.
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from fifapreds.asof import MatchStore
from fifapreds.tournament_backtest import (
    derive_groups,
    run,
    simulate_qualification,
    wc_matches,
)
from tests.test_sim_montecarlo import FakeGoals

_COLS = ["match_id", "date", "home_team", "away_team", "home_score",
         "away_score", "tournament", "neutral", "is_played", "went_to_et"]

TEAMS = [f"T{i:02d}" for i in range(1, 33)]            # T01 strongest
STRENGTHS = {t: float(32 - i) for i, t in enumerate(TEAMS)}
GROUPS = {f"G{g + 1}": TEAMS[g * 4:(g + 1) * 4] for g in range(8)}


def synthetic_wc2014() -> pd.DataFrame:
    """48 group matches (decided by strength) + 8 R16 matches for the two
    strongest of each group — a complete miniature 2014."""
    rows, mid = [], 0
    date = pd.Timestamp("2014-06-12")
    for teams in GROUPS.values():
        for h, a in itertools.combinations(teams, 2):
            hs, as_ = (2, 0) if STRENGTHS[h] > STRENGTHS[a] else (0, 2)
            rows.append((mid, date + pd.Timedelta(days=mid % 14), h, a,
                         hs, as_, "FIFA World Cup", True, True, False))
            mid += 1
    # R16 pairs winner of group g with runner-up of group g+1 (cross-group,
    # like the real bracket — a knockout match must bridge two groups).
    group_list = list(GROUPS.values())
    for g in range(8):
        winner = group_list[g][0]
        runner_up = group_list[(g + 1) % 8][1]
        rows.append((mid, pd.Timestamp("2014-06-28") + pd.Timedelta(days=g),
                     winner, runner_up, 1, 0,
                     "FIFA World Cup", True, True, False))
        mid += 1
    df = pd.DataFrame(rows, columns=_COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def store() -> MatchStore:
    return MatchStore(synthetic_wc2014())


def test_derive_groups_recovers_the_draw(store):
    edition = wc_matches(store, 2014)
    groups = derive_groups(edition.head(48))
    assert sorted(map(tuple, groups.values())) == sorted(
        tuple(sorted(ts)) for ts in GROUPS.values())


def test_derive_groups_rejects_a_bad_slice(store):
    edition = wc_matches(store, 2014)
    with pytest.raises(ValueError, match="groups of 4"):
        # one knockout match leaks in -> two groups merge into one component
        derive_groups(edition.head(49))


def test_deterministic_model_gives_certain_qualification(store):
    model = FakeGoals(STRENGTHS, decisiveness=1.0)
    edition = wc_matches(store, 2014)
    groups = derive_groups(edition.head(48))
    p = simulate_qualification(model, groups, edition.head(48),
                               n_sims=50, seed=1)
    for teams in GROUPS.values():
        assert p[teams[0]] == 1.0 and p[teams[1]] == 1.0
        assert p[teams[2]] == 0.0 and p[teams[3]] == 0.0
    # Probability is conserved: exactly 2 advance per group.
    assert sum(p.values()) == pytest.approx(16.0)


def test_seeded_reproducibility(store):
    model = FakeGoals(STRENGTHS, decisiveness=0.6)
    edition = wc_matches(store, 2014)
    groups = derive_groups(edition.head(48))
    kw = dict(n_sims=200, seed=42)
    a = simulate_qualification(model, groups, edition.head(48), **kw)
    b = simulate_qualification(model, groups, edition.head(48), **kw)
    assert a == b


def test_run_grades_against_reality_and_refuses_leaks(store, tmp_path):
    model = FakeGoals(STRENGTHS, decisiveness=1.0)
    model.trained_through = pd.Timestamp("2014-01-01")   # honest pre-tournament
    out = run(models=[model], years=(2014,), n_sims=20, seed=3,
              store=store, out_path=tmp_path / "qual.parquet")
    assert len(out) == 32
    # The two strongest per group really advanced in the synthetic edition.
    assert out[out["p_advance"] == 1.0]["advanced"].all()
    assert not out[out["p_advance"] == 0.0]["advanced"].any()
    assert (out["training_cutoff"] == "2014-01-01T00:00:00").all()
    assert (tmp_path / "qual.parquet").exists()

    leaky = FakeGoals(STRENGTHS, decisiveness=1.0)
    leaky.trained_through = pd.Timestamp("2014-06-12")   # opener day = leak
    with pytest.raises(ValueError, match="leak"):
        run(models=[leaky], years=(2014,), n_sims=5, store=store,
            out_path=None)
