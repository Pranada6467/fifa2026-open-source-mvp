# Block V — Verification Spike findings

Gates the provisional decisions in the plan. Updated as each item resolves.

## V1 — penaltyblog APIs — PASS (pinned 1.11.0)
- `models.DixonColesGoalModel`: **native `neutral_venue`** support (per-match in
  constructor + `predict(..., neutral_venue=...)`). Accepts per-match `weights`.
- `models.dixon_coles_weights(dates, xi=0.0018)`: built-in time decay.
- `DixonColesGoalModel.predict()` → `FootballProbabilityGrid` with `home_win`,
  `draw`, `away_win`, `exact_score`, goal distributions. One call gives both
  `predict_wdl()` and `predict_goals()`. `predict_many()` for batch.
- `implied.ImpliedProbabilities`: odds de-vig (multiple methods).
- `metrics`: `rps_array` (ranked probability score — football-standard, primary),
  `ignorance_score` (log-loss), `multiclass_brier_score`.
- Bonus: `ratings.Elo` + `ratings.PiRatingSystem` (free leaderboard entries),
  `scrapers.Understat` (xG for later), `scrapers.team_mappings` (helps V5).
- **Impact:** neutral-venue handling is upstream, not custom. DC wrapper is thin.
  Custom Elo still written for importance/decay control; pi-ratings added free.

## V2 — martj42 data semantics + freshness — PASS
- Schema: date, home_team, away_team, home_score, away_score, tournament, city,
  country, neutral. `shootouts.csv`: date, home, away, winner, first_shooter.
- **Score semantics: includes extra time, excludes penalties.** Verified:
  2014 final = 1-0 (Götze ET goal counted); 2022 final = 3-3 (AET, pre-shootout).
  - **Cleaning rule:** for the 90-minute goals model, flag matches appearing in
    `shootouts.csv` (definite ET) and exclude/down-weight them. Document residual
    bias from non-shootout ET matches (small; ET is rare).
- **Freshness: data runs to 2026-06-27.** Contains 205 of 2026's warm-up
  friendlies *with scores* and the 72 WC2026 group fixtures as future rows
  (`NaN` scores). No manual friendly entry needed.
  - **Ingest rule:** split `results.csv` into *played* (score not null) and
    *fixtures* (score null); the as-of layer treats fixtures as not-yet-resolved.

## V5 — team-name reconciliation — PASS
- All 48 WC2026 teams present in both martj42 and the odds feed; every team has
  >= 235 historical matches (no thin-data teams). Only 2 spellings differ:
  `USA`→`United States`, `Bosnia & Herzegovina`→`Bosnia and Herzegovina`.
  Captured in `registry.py` (`canonical()`); martj42 naming is canonical.

## V3 — WC2026 group structure + bracket — PASS
- Group labels A–L triple-validated: official final-draw result == fixture-derived
  clusters (martj42) == host anchors (A=Mexico, B=Canada, D=United States).
  Saved `data/raw/groups_2026.csv` via `scripts/build_groups_2026.py` (guards run
  on build). Knockout bracket skeleton (matches 73–104) in `bracket_2026.csv`.
- Still nice-to-have later: venue/kickoff per match (kickoff_ts available from the
  odds payload; venues not yet needed for the model).

## V4 — 2026 best-third routing table — PASS (pending user spot-check)
- Parsed the official 495-row Round-of-32 routing table programmatically (no hand
  transcription) → `data/raw/routing_r32.parquet` via `scripts/fetch_bracket_2026.py`.
- Internal-consistency checks pass: 495 unique combos, each with exactly 8
  qualifying groups, and each combo assigns its 8 thirds as a permutation across
  the 8 host-winner matches (74,77,79,80,81,82,85,87). Host winners: A,B,D,E,G,I,K,L.
- **User-verified (2026-06-10):** combos #1 and #495 confirmed cell-for-cell
  against the official Wikipedia table. Routing data trusted; simulator
  winner-probs may be ungated.

## V6 — odds provider — PASS
- The Odds API key validated (free tier, 500 req/month). Active sport keys:
  `soccer_fifa_world_cup` (h2h) and `soccer_fifa_world_cup_winner` (outrights).
- Snapshotter built (`loop/odds.py`, capture-only, raw payloads → SQLite
  `odds_snapshots`). First snapshot captured: 72 match events + winner market.
- **Bonus:** h2h payload includes each fixture's `commence_time` → explicit
  `kickoff_ts` for the as-of layer (not date-only).
- Cadence: daily-ish (quota-aware); de-vig/blend is Block 2.
