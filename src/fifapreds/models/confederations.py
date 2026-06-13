"""Confederation map — static team→confederation for the hierarchical model.

Mirrors the registry pattern: every team must resolve or raise. The map is
loaded once from data/raw/confederations.csv (336 teams, 6 confederations).
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw" / "confederations.csv"


@functools.cache
def _load() -> dict[str, str]:
    df = pd.read_csv(_CSV)
    return dict(zip(df["team"], df["confederation"]))


def team_confederation(team: str) -> str:
    """Return the FIFA confederation for a team name (martj42 canonical)."""
    mapping = _load()
    if team not in mapping:
        raise KeyError(
            f"unknown team {team!r} — not in confederations.csv; "
            "add it via scripts/build_confederations.py"
        )
    return mapping[team]


def all_confederations() -> dict[str, str]:
    """Full team→confederation dict (cached)."""
    return dict(_load())
