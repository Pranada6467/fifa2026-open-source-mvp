"""Frozen model-variant roster — the single source of leaderboard entrants.

Why variants are subclasses: the leaderboard keys entrants by
(model_id, hyperparams_hash), so a variant IS its configuration. Giving each
one a distinct `model_id` ClassVar plus frozen constructor defaults makes the
identity unforgeable — you cannot accidentally log an EloDecay prediction
under the baseline's name, and `hyperparams()` (inherited) already reflects
the frozen defaults, so the provenance hash is honest for free.

Freeze discipline: these configs are FROZEN before kickoff. Ratings update
live as results arrive, but hyperparameters are never tuned on 2026 results —
that would contaminate the out-of-sample claim the leaderboard exists to make.
Want a new config? Add a new subclass (new model_id); never edit a frozen one.

The roster and the hypothesis each variant tests:
- BaselineElo / DixonColes — the unmodified references everything must beat.
- EloDecay — does regressing idle teams toward 1500 fix stale ratings for
  nations with sparse schedules?
- EloImportance — does eloratings.net-style match weighting (World Cup counts
  more than a friendly) sharpen ratings where stakes drive effort?
- DixonColesSlowXi — does a flatter time decay (wider effective sample) soften
  DC's documented overconfidence at the extremes (backtest 0-0.1 bin: claimed
  7.7%, happened 17.4% — thin cross-confederation data + sharp xi)?
"""
from __future__ import annotations

from typing import Any, Iterable

from fifapreds.models.base import GoalsModel, Model
from fifapreds.models.dixoncoles import DixonColes
from fifapreds.models.elo import BaselineElo

try:
    from fifapreds.models.hierarchical import HierarchicalPoisson as _HP
    _HAS_PYMC = True
except ImportError:
    _HAS_PYMC = False

# Tournament -> K multiplier for EloImportance, keyed by EXACT martj42
# tournament strings (verified against data/processed/matches.parquet,
# 2026-06-11; match counts in comments — test_roster.py enforces that every
# key exists verbatim in the data). Relative weights follow eloratings.net's
# ladder: World Cup > continental finals > qualifiers > everything else (the
# implicit 1.0) > friendlies, where rotated squads and zero stakes make
# results least informative. Note martj42 quirks: the CONCACAF Gold Cup is
# plain "Gold Cup", and "CONCACAF Championship" is its 1963-89 predecessor.
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    # The World Cup itself.
    "FIFA World Cup": 1.75,                        # 1,036 matches
    # Continental finals (all six confederations).
    "UEFA Euro": 1.5,                              #   388
    "Copa América": 1.5,                           #   869
    "AFC Asian Cup": 1.5,                          #   421
    "African Cup of Nations": 1.5,                 #   845
    "Gold Cup": 1.5,                               #   420
    "CONCACAF Championship": 1.5,                  #   169
    "Oceania Nations Cup": 1.5,                    #   139
    # World Cup + continental qualification: real elimination pressure.
    "FIFA World Cup qualification": 1.25,          # 8,771
    "UEFA Euro qualification": 1.25,               # 2,824
    "Copa América qualification": 1.25,            #     8
    "AFC Asian Cup qualification": 1.25,           #   829
    "African Cup of Nations qualification": 1.25,  # 2,327
    "Gold Cup qualification": 1.25,                #    88
    "CONCACAF Championship qualification": 1.25,   #   151
    "Oceania Nations Cup qualification": 1.25,     #    31
    # Friendlies: experimental lineups, unlimited subs, nothing at stake.
    "Friendly": 0.5,                               # 18,384
}


class EloDecay(BaselineElo):
    """BaselineElo + idle-team regression: ~10%/year toward the initial 1500."""

    model_id = "elo_decay"
    model_version = "1"

    def __init__(self, **overrides: Any):
        overrides.setdefault("decay_rate", 0.1)
        super().__init__(**overrides)


class EloImportance(BaselineElo):
    """BaselineElo + tournament-weighted K (TOURNAMENT_IMPORTANCE above)."""

    model_id = "elo_importance"
    model_version = "1"

    def __init__(self, **overrides: Any):
        # Copy the frozen constant so instances never share mutable state.
        overrides.setdefault("importance", dict(TOURNAMENT_IMPORTANCE))
        super().__init__(**overrides)


class DixonColesSlowXi(DixonColes):
    """DixonColes with flatter time decay (xi 0.0018 -> 0.0005): a wider
    effective sample to soften the extremes-overconfidence quirk."""

    model_id = "dixon_coles_slow_xi"
    model_version = "1"

    def __init__(self, **overrides: Any):
        overrides.setdefault("xi", 0.0005)
        super().__init__(**overrides)


if _HAS_PYMC:
    class HierarchicalPoisson(_HP):
        """Bayesian hierarchical Poisson with confederation partial pooling.

        Hypothesis: borrowing strength across confederation members stabilizes
        predictions for thin cross-confederation matchups where Dixon-Coles is
        documented overconfident (backtest 0-0.1 bin: 7.7% claimed, 17.4% observed)."""

        model_id = "hierarchical_poisson"
        model_version = "1"


def default_roster() -> list[Model]:
    """Fresh, unfitted instances of every frozen leaderboard entrant.

    A new list of new objects on every call — callers fit and mutate their own
    copies, so no state leaks between the live loop, backtests, and tests.
    """
    roster: list[Model] = [
        BaselineElo(),
        DixonColes(),
        EloDecay(),
        EloImportance(),
        DixonColesSlowXi(),
    ]
    if _HAS_PYMC:
        roster.append(HierarchicalPoisson())
    return roster


def goals_models(roster: Iterable[Model]) -> list[GoalsModel]:
    """The simulator-capable subset (Monte Carlo needs scorelines for
    goal-difference / goals-scored tiebreaks; W/D/L-only Elo cannot drive it)."""
    return [m for m in roster if isinstance(m, GoalsModel)]
