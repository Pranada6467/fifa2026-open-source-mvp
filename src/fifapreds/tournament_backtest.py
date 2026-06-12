"""E4 — pre-tournament group-qualification backtest (WC 2014/18/22).

The tournament-level question with real statistical power is QUALIFICATION:
"who gets out of the group" is ~32 binary events per World Cup (~96 across
three), against n=3 for the champion. So this module fits each goals model
strictly before each tournament's opening match, Monte-Carlos the GROUP STAGE
ONLY (8 groups of 4, top two advance — the 1998-2022 format), and records
P(advance) per team next to what actually happened. The reliability table
comes from the shared `binary_calibration_table` (T3), Wilson bands included.

Deliberately NOT here (eng review): the old-format knockout bracket. Deep-run
and winner calibration are n=3 — unassessable; the board says so instead of
pretending.

CLI:  .venv/bin/python -m fifapreds.tournament_backtest
writes data/qualification_backtest.parquet (committed; the nightly publisher
passes it through to artifacts/ on a stateless runner).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds.asof import MatchStore
from fifapreds.config import PROJECT_ROOT
from fifapreds.models.base import GoalsModel
from fifapreds.sim.groups import standings

OUT_PARQUET = PROJECT_ROOT / "data" / "qualification_backtest.parquet"

#: Opening-match dates — fits use store.before(opener): strictly pre-tournament.
WC_OPENERS = {
    2014: pd.Timestamp("2014-06-12"),
    2018: pd.Timestamp("2018-06-14"),
    2022: pd.Timestamp("2022-11-20"),
}
N_GROUP_MATCHES = 48          # 8 groups x 6 matches in the 32-team format
QUALIFIERS_PER_GROUP = 2


def wc_matches(store: MatchStore, year: int) -> pd.DataFrame:
    """All played World Cup matches of one edition, kickoff order."""
    m = store.played
    sel = m[(m["tournament"] == "FIFA World Cup") & (m["date"].dt.year == year)]
    if year == 2022:  # Nov-Dec edition; the year filter is already exact
        pass
    return sel.sort_values(["date", "match_id"]).reset_index(drop=True)


def derive_groups(group_matches: pd.DataFrame,
                  n_groups: int = 8) -> dict[str, list[str]]:
    """Recover the groups from the group-stage fixtures themselves: teams are
    in the same group iff they meet in the group stage, so the groups are the
    connected components of the 'played each other' graph. Self-validating:
    anything but `n_groups` components of 4 raises."""
    adjacency: dict[str, set[str]] = {}
    for r in group_matches.itertuples(index=False):
        adjacency.setdefault(r.home_team, set()).add(r.away_team)
        adjacency.setdefault(r.away_team, set()).add(r.home_team)
    seen: set[str] = set()
    components: list[list[str]] = []
    for team in adjacency:
        if team in seen:
            continue
        stack, comp = [team], []
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            comp.append(t)
            stack.extend(adjacency[t] - seen)
        components.append(sorted(comp))
    if len(components) != n_groups or any(len(c) != 4 for c in components):
        raise ValueError(
            f"expected {n_groups} groups of 4, got "
            f"{sorted(len(c) for c in components)} — group-stage slice is wrong"
        )
    components.sort()
    return {f"G{i + 1}": comp for i, comp in enumerate(components)}


def _sample_scores(model: GoalsModel, fixture, n_sims: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n_sims (home_goals, away_goals) draws from the fixture's score grid."""
    grid = model.predict_goals(
        fixture.home_team, fixture.away_team, neutral=bool(fixture.neutral)
    ).grid
    flat_cum = np.cumsum(grid.ravel())
    idx = np.minimum(np.searchsorted(flat_cum, rng.random(n_sims)),
                     grid.size - 1)
    return idx // grid.shape[1], idx % grid.shape[1]


def simulate_qualification(
    model: GoalsModel,
    groups: dict[str, list[str]],
    group_matches: pd.DataFrame,
    *,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """P(finish top {QUALIFIERS_PER_GROUP}) per team, from a seeded group-stage
    Monte Carlo. Reuses the real tiebreaker ladder (`sim.groups.standings`)."""
    rng = np.random.default_rng(seed)
    team_group = {t: g for g, teams in groups.items() for t in teams}
    samples: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = {
        g: [] for g in groups
    }
    for fx in group_matches.itertuples(index=False):
        hg, ag = _sample_scores(model, fx, n_sims, rng)
        samples[team_group[fx.home_team]].append(
            (fx.home_team, fx.away_team, hg, ag))
    advanced = {t: 0 for t in team_group}
    for g, teams in groups.items():
        for s in range(n_sims):
            results = [(h, a, int(hg[s]), int(ag[s]))
                       for h, a, hg, ag in samples[g]]
            order = standings(teams, results, rng)
            for rec in order[:QUALIFIERS_PER_GROUP]:
                advanced[rec.team] += 1
    return {t: c / n_sims for t, c in advanced.items()}


def run(
    models: list[GoalsModel] | None = None,
    *,
    years: tuple[int, ...] = (2014, 2018, 2022),
    n_sims: int = 2000,
    seed: int = 0,
    store: MatchStore | None = None,
    out_path: Path | str | None = OUT_PARQUET,
) -> pd.DataFrame:
    """The E4 readout: per (wc, model, team) qualification probability next to
    what happened. Fits are strictly pre-tournament (the no-leak invariant)."""
    store = store or MatchStore()
    if models is None:
        from fifapreds.models.roster import default_roster

        models = [m for m in default_roster() if isinstance(m, GoalsModel)]

    rows = []
    for year in years:
        opener = WC_OPENERS[year]
        edition = wc_matches(store, year)
        group_stage = edition.head(N_GROUP_MATCHES)
        knockout = edition.iloc[N_GROUP_MATCHES:]
        groups = derive_groups(group_stage)
        actually_advanced = set(knockout["home_team"]) | set(knockout["away_team"])
        train = store.before(opener)
        for proto in models:
            model = proto.fit(train)
            if model.trained_through >= opener:
                raise ValueError(
                    f"leak: {model.model_id} trained through "
                    f"{model.trained_through} but {year} opens {opener}")
            p_advance = simulate_qualification(
                model, groups, group_stage, n_sims=n_sims, seed=seed)
            team_group = {t: g for g, ts in groups.items() for t in ts}
            for team, p in p_advance.items():
                rows.append({
                    "wc": year,
                    "model_id": model.model_id,
                    "team": team,
                    "group": team_group[team],
                    "p_advance": p,
                    "advanced": int(team in actually_advanced),
                    "n_sims": n_sims,
                    "seed": seed,
                    "training_cutoff": model.trained_through.isoformat(),
                })
    out = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out


def main() -> int:
    out = run()
    print(f"qualification backtest -> {OUT_PARQUET}")
    for (year, model_id), grp in out.groupby(["wc", "model_id"]):
        # Brier on the binary qualification claim — quick honesty readout.
        brier = ((grp["p_advance"] - grp["advanced"]) ** 2).mean()
        hits = grp.nlargest(16, "p_advance")["advanced"].sum()
        print(f"  {year} {model_id:<22} qual-Brier {brier:.4f} | "
              f"top-16 by p: {int(hits)}/16 advanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
