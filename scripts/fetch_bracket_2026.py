"""Fetch the official 2026 World Cup knockout bracket + best-third routing table
from Wikipedia (which mirrors the FIFA-published table) and save normalized data
files. Re-runnable to refresh.

Outputs (data/raw/):
  routing_r32.parquet  — 495 rows: combo_id, groups (8-letter set), and the
                         winner-host -> third-group assignment for each of the
                         8 variable round-of-32 matches.
  bracket_2026.csv     — matches 73-104: match no., home slot, away slot.

Correctness checks run inline:
  - exactly 495 combos; each combo has exactly 8 qualifying groups;
  - each combo assigns its 8 thirds as a PERMUTATION of those groups
    (every qualifying third placed exactly once across the 8 host matches).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Group winner -> round-of-32 match number, for the 8 winners who host a third.
# (Verified against the bracket tables on the same page.)
WINNER_HOST_MATCH = {
    "1E": 74, "1I": 77, "1A": 79, "1L": 80,
    "1D": 81, "1G": 82, "1B": 85, "1K": 87,
}


def fetch_tables() -> list[pd.DataFrame]:
    html = requests.get(URL, headers={"User-Agent": "fifapreds/0.1 (research)"}, timeout=60).text
    return pd.read_html(io.StringIO(html))


def build_routing(t0: pd.DataFrame) -> pd.DataFrame:
    group_cols = list(t0.columns[1:13])          # the 12 group-presence columns
    host_cols = [c for c in t0.columns if str(c).endswith(" vs")]  # '1A vs' ...
    assert len(host_cols) == 8, host_cols
    rows = []
    for _, r in t0.iterrows():
        groups = sorted(str(r[c]).strip() for c in group_cols if pd.notna(r[c]))
        assert len(groups) == 8, (r["No."], groups)
        rec = {"combo_id": int(r["No."]), "groups": "".join(groups)}
        assigned = []
        for c in host_cols:
            host = c.replace(" vs", "").strip()           # '1A'
            third = str(r[c]).strip().lstrip("3")          # '3E' -> 'E'
            rec[host] = third
            rec[f"{host}_match"] = WINNER_HOST_MATCH[host]
            assigned.append(third)
        # PERMUTATION check: the 8 assigned thirds == the 8 qualifying groups.
        assert sorted(assigned) == groups, (rec["combo_id"], sorted(assigned), groups)
        rows.append(rec)
    df = pd.DataFrame(rows)
    assert len(df) == 495, len(df)
    assert df["combo_id"].is_unique and df["combo_id"].min() == 1 and df["combo_id"].max() == 495
    assert df["groups"].is_unique, "duplicate qualifying-group combinations"
    return df


def build_bracket(tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t in tables:
        if t.shape == (1, 3) and str(t.columns[1]).startswith("Match"):
            rows.append({
                "match": int(str(t.columns[1]).split()[1]),
                "home": str(t.columns[0]).strip(),
                "away": str(t.columns[2]).strip(),
            })
    df = pd.DataFrame(rows).sort_values("match").reset_index(drop=True)
    # R32 (73-88) + R16 (89-96) + QF (97-100) + SF (101-102) + 3rd (103) + final (104)
    assert df["match"].min() == 73 and df["match"].max() == 104, df["match"].agg(["min", "max"]).to_dict()
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tables = fetch_tables()
    routing = build_routing(tables[0])
    bracket = build_bracket(tables)
    routing.to_parquet(RAW / "routing_r32.parquet", index=False)
    bracket.to_csv(RAW / "bracket_2026.csv", index=False)
    print(f"routing_r32.parquet: {len(routing)} combos, all checks passed")
    print(f"bracket_2026.csv: {len(bracket)} matches ({bracket.match.min()}-{bracket.match.max()})")
    print("\nsample combo (id=1):")
    print(routing.iloc[0].to_dict())


if __name__ == "__main__":
    main()
