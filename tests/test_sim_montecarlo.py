"""T11 — tournament Monte Carlo.

A hand-rolled FakeGoals model makes outcomes fully controllable: a strict
strength hierarchy must crown the strongest team every time; all-equal
strengths force the draw -> extra-time -> penalties path and a diffuse
champion distribution. Structural invariants (stage sizes, monotonicity,
seed determinism, conditioning on played results, the lookahead guard) hold
for any model.
"""
import numpy as np
import pandas as pd
import pytest

from fifapreds.models.base import GoalsModel, ScoreGrid, WDL
from fifapreds.sim.montecarlo import (
    STAGE_SIZES,
    group_stage_fixtures,
    load_groups,
    simulate_tournament,
)

GROUPS = load_groups()
TEAMS = list(GROUPS["team"])


class FakeGoals(GoalsModel):
    """Stronger team wins 2-0 with probability `decisiveness`, else 1-1."""

    model_id = "fake_goals"
    model_version = "0"

    def __init__(self, strengths: dict[str, float], decisiveness: float = 1.0):
        self.strengths = strengths
        self.decisiveness = decisiveness
        self.trained_through = pd.Timestamp("2026-06-01")

    def fit(self, matches):
        return self

    def hyperparams(self):
        return {"decisiveness": self.decisiveness}

    def predict_goals(self, home, away, *, neutral=False):
        grid = np.zeros((4, 4))
        sh, sa = self.strengths[home], self.strengths[away]
        if sh == sa:
            grid[1, 1] = 1.0
        elif sh > sa:
            grid[2, 0] = self.decisiveness
            grid[1, 1] = 1.0 - self.decisiveness
        else:
            grid[0, 2] = self.decisiveness
            grid[1, 1] = 1.0 - self.decisiveness
        return ScoreGrid(grid)

    def predict_wdl(self, home, away, *, neutral=False) -> WDL:
        return self.predict_goals(home, away, neutral=neutral).wdl()


def synthetic_fixtures() -> pd.DataFrame:
    """72 unplayed group fixtures built straight from the groups artifact."""
    rows = []
    date = pd.Timestamp("2026-06-11")
    for g, sub in GROUPS.groupby("group"):
        teams = list(sub["team"])
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(
                    {
                        "match_id": len(rows),
                        "date": date + pd.Timedelta(days=len(rows) % 16),
                        "home_team": teams[i],
                        "away_team": teams[j],
                        "home_score": np.nan,
                        "away_score": np.nan,
                        "tournament": "FIFA World Cup",
                        "neutral": True,
                        "is_played": False,
                        "went_to_et": False,
                    }
                )
    return pd.DataFrame(rows)


def hierarchy() -> dict[str, float]:
    """Globally unique strengths: TEAMS[0] strongest, last weakest."""
    return {t: float(len(TEAMS) - i) for i, t in enumerate(TEAMS)}


def run(model, fixtures, seed=1, n_sims=64):
    return simulate_tournament(
        model, n_sims=n_sims, seed=seed, matches=fixtures, groups=GROUPS
    )


def test_strict_hierarchy_crowns_strongest_always():
    summary, meta = run(FakeGoals(hierarchy()), synthetic_fixtures())
    best = summary.iloc[0]
    assert best["team"] == TEAMS[0]
    assert best["p_champion"] == 1.0
    assert best["p_advance"] == best["p_r16"] == best["p_sf"] == 1.0
    # The strongest team in every group always wins it.
    for g, sub in GROUPS.groupby("group"):
        strongest = max(sub["team"], key=hierarchy().get)
        row = summary[summary["team"] == strongest].iloc[0]
        assert row["p_group_win"] == 1.0
    assert meta["model_id"] == "fake_goals" and meta["n_sims"] == 64


def test_stage_sizes_and_monotonicity():
    # Mild decisiveness keeps outcomes random enough to exercise every path.
    model = FakeGoals(hierarchy(), decisiveness=0.6)
    summary, _ = run(model, synthetic_fixtures(), n_sims=128)
    for stage, size in STAGE_SIZES.items():
        total = summary[f"p_{stage}"].sum()
        assert total == pytest.approx(size), stage
        assert summary[f"p_{stage}"].between(0, 1).all()
    ladder = ["p_advance", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    for hi, lo in zip(ladder, ladder[1:]):
        assert (summary[hi] >= summary[lo] - 1e-12).all(), (hi, lo)
    assert (summary["p_group_win"] <= summary["p_advance"]).all()


def test_all_equal_goes_to_lots_et_and_penalties():
    # Every 90-minute match is 1-1: standings are pure lots, every knockout
    # goes to ET/penalties — and the machinery still conserves probability.
    model = FakeGoals({t: 1.0 for t in TEAMS})
    summary, _ = run(model, synthetic_fixtures(), n_sims=256)
    assert summary["p_champion"].sum() == pytest.approx(1.0)
    # No team can dominate a coin-flip tournament.
    assert summary["p_champion"].max() < 0.2
    assert summary["p_advance"].max() <= 1.0


def test_seed_determinism():
    model = FakeGoals(hierarchy(), decisiveness=0.7)
    a, _ = run(model, synthetic_fixtures(), seed=11, n_sims=64)
    b, _ = run(model, synthetic_fixtures(), seed=11, n_sims=64)
    pd.testing.assert_frame_equal(a, b)
    c, _ = run(model, synthetic_fixtures(), seed=12, n_sims=64)
    assert not a.equals(c), "different seeds should differ somewhere"


def test_conditions_on_played_results():
    # Hand South Africa three 9-0 wins: whatever the model thinks of them,
    # they must top group A in every simulation.
    fixtures = synthetic_fixtures()
    strengths = hierarchy()
    strengths["South Africa"] = -99.0  # the model despises them
    sa = fixtures["home_team"].eq("South Africa") | fixtures["away_team"].eq("South Africa")
    for idx in fixtures[sa].index:
        home_is_sa = fixtures.at[idx, "home_team"] == "South Africa"
        fixtures.at[idx, "home_score"] = 9.0 if home_is_sa else 0.0
        fixtures.at[idx, "away_score"] = 0.0 if home_is_sa else 9.0
        fixtures.at[idx, "is_played"] = True
    summary, meta = run(FakeGoals(strengths), fixtures)
    row = summary[summary["team"] == "South Africa"].iloc[0]
    assert row["p_group_win"] == 1.0 and row["p_advance"] == 1.0
    assert meta["n_fixtures_played"] == 3


def test_lookahead_guard():
    model = FakeGoals(hierarchy())
    model.trained_through = pd.Timestamp("2026-06-20")  # has seen fixture days
    with pytest.raises(ValueError, match="lookahead"):
        run(model, synthetic_fixtures())


def test_group_fixture_extraction_dedupes_rematches():
    fixtures = synthetic_fixtures()
    # A knockout rematch of an intra-group pairing later in the tournament
    # must not displace the original group game — and a HISTORICAL World Cup
    # meeting of the same two teams (regression: 17 of them shadowed real
    # 2026 fixtures on first run) must be ignored entirely.
    rematch = fixtures.iloc[[0]].assign(
        date=pd.Timestamp("2026-07-04"), home_score=1.0, away_score=0.0, is_played=True
    )
    historical = fixtures.iloc[[0]].assign(
        date=pd.Timestamp("1998-06-16"), home_score=3.0, away_score=0.0, is_played=True
    )
    extracted = group_stage_fixtures(
        pd.concat([historical, fixtures, rematch]), GROUPS
    )
    assert len(extracted) == 72
    first = extracted[
        (extracted["home_team"] == fixtures.at[0, "home_team"])
        & (extracted["away_team"] == fixtures.at[0, "away_team"])
    ].iloc[0]
    assert first["date"] == fixtures.at[0, "date"]
    assert not first["is_played"]


def test_real_data_fixture_extraction():
    # The real parquet must yield exactly the 72 group fixtures.
    path = "data/processed/matches.parquet"
    try:
        matches = pd.read_parquet(path)
    except (FileNotFoundError, OSError):
        pytest.skip("matches.parquet not built")
    fixtures = group_stage_fixtures(matches, GROUPS)
    assert len(fixtures) == 72
    assert set(fixtures["home_team"]) | set(fixtures["away_team"]) == set(TEAMS)
    # Every fixture sits inside the 2026 group-stage window — no historical
    # World Cup rows shadowing 2026 games.
    assert (fixtures["date"] >= "2026-06-11").all()
    assert (fixtures["date"] <= "2026-06-27").all()
