"""Seeded tournament Monte Carlo (T11) — per-match grids in, trophy odds out.

`simulate_tournament` replays the whole 2026 tournament `n_sims` times from a
goals-capable model (W/D/L-only models cannot drive this — group tiebreaks
need scorelines):

1. **Group stage.** The 72 group fixtures come from matches.parquet, so
   already-played results are conditioned on as facts and only the remaining
   fixtures are sampled, from each fixture's `ScoreGrid` with its real
   `neutral` flag (hosts keep their fitted edge). Standings + the best-8
   thirds use the FIFA tiebreakers in `sim.groups` (lots are drawn per sim).
2. **Routing.** Thirds land in the bracket via the verified 495-combination
   table (`sim.routing`), memoized per combination.
3. **Knockouts.** Matches 73-104 walk the bracket template. 90-minute scores
   are sampled from the grid; a draw goes to extra time, approximated as two
   Poissons at the grid's marginal goal rates scaled by 30/90; a still-level
   tie is a 50/50 penalty shoot-out. All knockout matches are treated as
   neutral-venue — a documented approximation that slightly undersells host
   nations' deep runs (venue-by-venue host edge is future work).

Determinism: one `numpy` Generator seeded by the caller drives every draw
(scores, lots, ET, penalties), so a (model config, fixtures, seed) triple
always reproduces the same probabilities — the seed lands in the published
meta for provenance, mirroring the predictions log.

No-lookahead: like `loop.predict.log_prediction`, the model's
`trained_through` must predate the earliest *unplayed* fixture being sampled.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from fifapreds.config import PROJECT_ROOT
from fifapreds.models.base import GoalsModel
from fifapreds.sim.groups import rank_thirds, standings
from fifapreds.sim.routing import (
    R32_MATCHES,
    knockout_template,
    r32_slots,
    thirds_assignment,
)

GROUPS_CSV = PROJECT_ROOT / "data" / "raw" / "groups_2026.csv"

#: First 2026 group-stage kickoff. Filters out HISTORICAL World Cup meetings
#: between 2026 same-group opponents (e.g. Brazil-Morocco 1998), which would
#: otherwise masquerade as played 2026 results.
WC2026_START = pd.Timestamp("2026-06-11")

_ET_FACTOR = 30.0 / 90.0  # extra time is a third of regulation

#: stage -> how many teams reach it per simulated tournament (audit invariant)
STAGE_SIZES = {
    "group_win": 12, "advance": 32, "r16": 16, "qf": 8,
    "sf": 4, "final": 2, "champion": 1,
}


@lru_cache(maxsize=1)
def load_groups() -> pd.DataFrame:
    return pd.read_csv(GROUPS_CSV)


@lru_cache(maxsize=512)
def _assignment(combo: str) -> dict[int, str]:
    """Memoized third-place routing — 495 possible keys, few recur per run."""
    return thirds_assignment(list(combo))


def group_stage_fixtures(
    matches: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    start: pd.Timestamp = WC2026_START,
) -> pd.DataFrame:
    """The 72 intra-group WC fixtures from a matches frame (played or not).

    Only rows from `start` on count — earlier World Cups also paired some of
    these teams. Knockout rematches between same-group teams are possible from
    the QF on; keeping only the earliest row per unordered pairing guarantees
    we read the group game, which always precedes any rematch.
    """
    team_group = dict(zip(groups["team"], groups["group"]))
    wc = matches[
        (matches["tournament"] == "FIFA World Cup") & (matches["date"] >= start)
    ].copy()
    same_group = wc["home_team"].map(team_group).notna() & (
        wc["home_team"].map(team_group) == wc["away_team"].map(team_group)
    )
    wc = wc[same_group]
    pair_key = wc.apply(
        lambda r: tuple(sorted((r["home_team"], r["away_team"]))), axis=1
    )
    wc = wc.assign(_pair=pair_key).sort_values("date", kind="stable")
    wc = wc.drop_duplicates("_pair", keep="first").drop(columns="_pair")
    counts = wc["home_team"].map(team_group).value_counts()
    if len(wc) != 72 or not (counts == 6).all():
        raise ValueError(
            f"expected 72 group fixtures (6 per group), found {len(wc)}: "
            f"{counts.to_dict()}"
        )
    return wc.reset_index(drop=True)


def _grid_arrays(model: GoalsModel, home: str, away: str, *, neutral: bool):
    """(cumulative flat probs, n_cols, lambda_home, lambda_away) for sampling."""
    grid = model.predict_goals(home, away, neutral=neutral).grid
    flat = grid.ravel()
    goals = np.arange(grid.shape[0])
    lam_h = float((grid.sum(axis=1) * goals).sum())
    lam_a = float((grid.sum(axis=0) * np.arange(grid.shape[1])).sum())
    return np.cumsum(flat), grid.shape[1], lam_h, lam_a


def simulate_tournament(
    model: GoalsModel,
    *,
    n_sims: int = 10_000,
    seed: int,
    matches: pd.DataFrame | None = None,
    groups: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the tournament `n_sims` times; return (per-team summary, meta).

    `matches`: a matches.parquet-shaped frame holding the 72 group fixtures
    (defaults to the real data via `asof.MatchStore`). Played rows are taken
    as fact; unplayed rows are sampled from the model.
    """
    if model.trained_through is None:
        raise ValueError("model is not fitted")
    if groups is None:
        groups = load_groups()
    if matches is None:
        from fifapreds.asof import MatchStore

        matches = MatchStore().all
    fixtures = group_stage_fixtures(matches, groups)

    unplayed = fixtures[~fixtures["is_played"]]
    if not unplayed.empty and model.trained_through >= unplayed["date"].min():
        raise ValueError(
            f"lookahead: model trained through {model.trained_through} but the "
            f"earliest unplayed fixture kicks off {unplayed['date'].min()}"
        )

    rng = np.random.default_rng(seed)
    teams_by_group: dict[str, list[str]] = {
        g: list(sub["team"]) for g, sub in groups.groupby("group")
    }
    team_group = dict(zip(groups["team"], groups["group"]))

    # ---- group stage: presample every fixture for all sims at once ---------
    # (home, away, home_goals[n_sims], away_goals[n_sims]) per group
    group_samples: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = {
        g: [] for g in teams_by_group
    }
    for fx in fixtures.itertuples(index=False):
        if fx.is_played:
            hg = np.full(n_sims, int(fx.home_score), dtype=np.int64)
            ag = np.full(n_sims, int(fx.away_score), dtype=np.int64)
        else:
            cum, ncols, _, _ = _grid_arrays(
                model, fx.home_team, fx.away_team, neutral=bool(fx.neutral)
            )
            idx = np.minimum(
                np.searchsorted(cum, rng.random(n_sims)), len(cum) - 1
            )
            hg, ag = np.divmod(idx, ncols)
        group_samples[team_group[fx.home_team]].append(
            (fx.home_team, fx.away_team, hg, ag)
        )

    # ---- knockout grids: lazy per ordered pairing, all neutral -------------
    ko_cache: dict[tuple[str, str], tuple[np.ndarray, int, float, float]] = {}

    def play_knockout(home: str, away: str) -> str:
        key = (home, away)
        if key not in ko_cache:
            ko_cache[key] = _grid_arrays(model, home, away, neutral=True)
        cum, ncols, lam_h, lam_a = ko_cache[key]
        idx = min(int(np.searchsorted(cum, rng.random())), len(cum) - 1)
        hg, ag = divmod(idx, ncols)
        if hg != ag:
            return home if hg > ag else away
        et_h = rng.poisson(lam_h * _ET_FACTOR)
        et_a = rng.poisson(lam_a * _ET_FACTOR)
        if et_h != et_a:
            return home if et_h > et_a else away
        return home if rng.random() < 0.5 else away  # penalties: a coin

    # ---- simulate -----------------------------------------------------------
    tally: dict[str, Counter] = {stage: Counter() for stage in STAGE_SIZES}
    slots = r32_slots()
    template = knockout_template()

    for i in range(n_sims):
        winners: dict[str, str] = {}
        runners: dict[str, str] = {}
        thirds = []
        for g, sampled in group_samples.items():
            results = [(h, a, int(hg[i]), int(ag[i])) for h, a, hg, ag in sampled]
            table = standings(teams_by_group[g], results, rng)
            winners[g] = table[0].team
            runners[g] = table[1].team
            thirds.append((g, table[2]))
        ranked = rank_thirds(thirds, rng)
        best_thirds = {g: rec.team for g, rec in ranked[:8]}
        assignment = _assignment("".join(sorted(best_thirds)))

        tally["group_win"].update(winners.values())
        tally["advance"].update(winners.values())
        tally["advance"].update(runners.values())
        tally["advance"].update(best_thirds.values())

        won: dict[int, str] = {}
        lost: dict[int, str] = {}
        for match in R32_MATCHES:
            sides = []
            for kind, group in slots[match]:
                if kind == "winner":
                    sides.append(winners[group])
                elif kind == "runner":
                    sides.append(runners[group])
                else:
                    sides.append(best_thirds[assignment[match]])
            w = play_knockout(sides[0], sides[1])
            won[match] = w
            lost[match] = sides[1] if w == sides[0] else sides[0]
        tally["r16"].update(won[m] for m in R32_MATCHES)

        for match in sorted(template):
            (take_h, src_h), (take_a, src_a) = template[match]
            home = won[src_h] if take_h == "winner" else lost[src_h]
            away = won[src_a] if take_a == "winner" else lost[src_a]
            w = play_knockout(home, away)
            won[match] = w
            lost[match] = away if w == home else home
        tally["qf"].update(won[m] for m in range(89, 97))
        tally["sf"].update(won[m] for m in range(97, 101))
        tally["final"].update(won[m] for m in (101, 102))
        tally["champion"][won[104]] += 1

    # ---- summarize -----------------------------------------------------------
    rows = [
        {
            "team": team,
            "group": team_group[team],
            **{f"p_{stage}": tally[stage][team] / n_sims for stage in STAGE_SIZES},
        }
        for team in groups["team"]
    ]
    summary = (
        pd.DataFrame(rows)
        .sort_values(["p_champion", "p_final", "p_sf", "team"], ascending=[False] * 3 + [True])
        .reset_index(drop=True)
    )

    from fifapreds.loop.predict import code_version

    meta = {
        "model_id": model.model_id,
        "model_version": model.model_version,
        "hyperparams_hash": model.hyperparams_hash,
        "training_cutoff": model.trained_through.isoformat(),
        "code_version": code_version(),
        "seed": seed,
        "n_sims": n_sims,
        "n_fixtures_played": int(fixtures["is_played"].sum()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary, meta
