"""Canonical team registry.

martj42/international_results naming is canonical — it is our history source, so
ratings and fixtures key off martj42 names. Other providers (e.g. the odds feed)
use slightly different spellings; map them here.

Verified in Block V (V5): all 48 WC2026 teams resolve, only two odds-provider
spellings differ from martj42. Extend `_ALIASES` as new sources are added.
"""
from __future__ import annotations

# alias (as seen in another source) -> canonical martj42 name
_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def canonical(name: str) -> str:
    """Normalise a team name to its canonical martj42 spelling."""
    return _ALIASES.get(name.strip(), name.strip())
