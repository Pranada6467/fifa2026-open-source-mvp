# CLAUDE.md — FIFA 2026 Live Calibration Engine

Read this first. It is the entry point for any new session.

## What this is
A World Cup 2026 prediction system built as a learning project. It rates teams
from historical international results, turns that into per-match and
full-tournament probabilities, and runs as a **live loop** that re-scores its own
predictions and recalibrates after every result. The product is **honest
calibration** ("when I say 70%, it happens ~70% of the time") plus a **technique
leaderboard** ranking models by out-of-sample log-loss / Brier / RPS.

Full context: `docs/PLAN.md` (the approved engineering plan — source of truth),
`docs/DESIGN.md` (the why), `docs/block-v-findings.md` (verified facts).

## Current status (2026-06-10)
**Block V (verification spike): COMPLETE — all 6 items pass, V4 user-verified.**
**Block 0 (calibration spine): in progress.**

Done:
- `ingest.py` (T1) — `data/processed/matches.parquet` from martj42.
- `asof.py` (T2) — point-in-time no-lookahead data access (CRITICAL invariant).
- `loop/odds.py` (T7) — odds snapshotter; first snapshot already captured.
- `config.py`, `db.py`, `registry.py`.
- `models/base.py` (T3) — split `predict_wdl`/`predict_goals` interface; WDL/ScoreGrid
  validating containers; provenance hooks (`hyperparams_hash`, `trained_through`).
- `models/elo.py` (T3) — BaselineElo (Davidson draw model, ν=0.6; neutral/decay/
  importance knobs; incremental `update()` refuses out-of-order results).
- `models/dixoncoles.py` (T4) — penaltyblog wrapper (4y trailing window anchored to
  the frame's max date, xi time decay, ET rows weight-0 by default, thin-window
  doubling fallback; ~6s fit on the real 4y window).
- `db.py` predictions+scores tables, `loop/predict.py`, `loop/score.py` (T5) —
  append-only provenance log (write-time lookahead guard: trained_through must
  predate kickoff), clamped log-loss/Brier/RPS (cross-checked vs penaltyblog;
  ours is natural-log, penaltyblog's ignorance is log2), calibration table,
  `score_pending` (grades resolved rows once; flags integrity violations —
  live rows predicted after kickoff are never scored; backtest rows exempt
  from the wall-clock rule, their claim rests on training_cutoff).
- `backtest.py` (T6) — match-level replay of WC 2014/18/22 through the same
  predict/score path; per-day Elo updates + per-day DC refits; gates: beat
  uniform ln(3), leak canary at log-loss < 0.60.
  CLI: `.venv/bin/python -m fifapreds.backtest` (~10 min; `--elo-only` is fast).
  First readout (2026-06-10, 384 preds, 0 violations): pooled log-loss
  DC 0.973 < Elo 0.982 < uniform 1.099; DC wins 2014+2018, Elo wins 2022;
  calibration max |freq−p| 0.055 (Elo) / 0.099 (DC) in populated bins.
- 51/51 tests passing.

Next, in order (see `docs/PLAN.md` → "Execution blocks"):
- **T8** `app/streamlit_app.py` + `publish/artifacts.py` (read-only viewer).
- Then **Block 1** (simulator: `sim/groups.py`, `sim/routing.py`, `sim/montecarlo.py`)
  and **Block 2** (market-blend, technique leaderboard).

## Setup & run
```bash
# from the project root (note: the path contains a space)
python3 -m venv .venv
.venv/bin/python -m pip install -e .            # editable install
.venv/bin/python -m pip install pytest networkx # dev/data deps

.venv/bin/python -m pytest -q                   # run tests
.venv/bin/python -m fifapreds.ingest            # (re)build matches.parquet
.venv/bin/python -m fifapreds.loop.odds         # capture an odds snapshot (uses quota)

# Reproduce the verified data artifacts:
.venv/bin/python scripts/fetch_bracket_2026.py  # routing_r32.parquet + bracket_2026.csv
.venv/bin/python scripts/build_groups_2026.py   # groups_2026.csv (with guards)
```

## Locked architecture decisions (do not relitigate — see PLAN.md)
- **Storage:** SQLite (`data/fifa2026.db`: predictions, odds_snapshots, rating
  snapshots) + parquet (raw/processed match data). Both gitignored.
- **No lookahead:** all match-history reads go through `asof.MatchStore.before(ts)`,
  which returns only *played* matches strictly before `ts`. Backtest and live loop
  share this path. This is the single most important invariant — never bypass it.
- **Split model interface:** `predict_wdl()` (every model) and `predict_goals()`
  (only goals-capable models: Dixon-Coles). Elo is W/D/L-only and CANNOT drive the
  simulator; the Monte Carlo group sim needs a goals model (for GD/goals tiebreaks).
- **Prediction provenance:** every prediction row stores model_id, model_version,
  code_version (git sha), hyperparams_hash, training_cutoff, odds_snapshot_id, seed,
  predicted_at, kickoff_ts — so the leaderboard is auditable.
- **Leaderboard integrity:** model configs are FROZEN before kickoff. Ratings update
  live; hyperparameters are NOT tuned on 2026 results (that contaminates out-of-sample).
- **Live-loop order:** score old predictions → ingest result → update models → publish.
- **Streamlit is a read-only viewer** over committed precomputed artifacts; never runs
  Monte Carlo on page load.

## Verified facts (from Block V — trust these)
- **penaltyblog 1.11.0** (pinned). `DixonColesGoalModel` has **native `neutral_venue`**
  (constructor per-match + `predict(..., neutral_venue=)`), per-match `weights`, and
  `dixon_coles_weights(dates, xi=0.0018)` for time decay. One `predict()` returns a
  `FootballProbabilityGrid` (W/D/L + `exact_score` grid + goal dists). Metrics:
  `penaltyblog.metrics.rps_array` (primary, football-standard), `ignorance_score`
  (log-loss), `multiclass_brier_score`. Odds de-vig: `penaltyblog.implied`.
  Bonus models: `penaltyblog.ratings.Elo`, `PiRatingSystem` (free leaderboard entries).
- **martj42 data**: scores INCLUDE extra time, EXCLUDE penalties. `ingest.py` flags
  `went_to_et` (via shootouts.csv) so the 90-min goals model can exclude/down-weight
  those. Data is fresh to 2026-06-27 (warm-ups scored; WC group fixtures present as
  unplayed rows). 677 ET matches flagged.
- **Team names**: martj42 is canonical. Only 2 odds-feed aliases (`registry.py`):
  USA→United States, Bosnia & Herzegovina→Bosnia and Herzegovina. All 48 WC teams
  have ≥235 historical matches.
- **Groups A–L**: `data/raw/groups_2026.csv` (triple-validated: official draw ==
  fixture clusters == host anchors A=Mexico/B=Canada/D=United States).
- **Round-of-32 routing**: `data/raw/routing_r32.parquet` (495 combos, permutation-
  checked, user-verified vs official table). 8 winners host thirds: A,B,D,E,G,I,K,L
  → matches 79,85,81,74,82,77,87,80. Bracket skeleton: `data/raw/bracket_2026.csv`.

## Conventions
- Tests are **invariant/golden-first**: no-lookahead audits, known-value metric
  checks, schema/permutation guards — over line-coverage worship. Add tests with the
  code, not after.
- `src/` layout; package is `fifapreds`. `pyproject.toml` sets pytest pythonpath.
- Secrets in `.env` (gitignored); `.env.example` is the committed template.
  `ODDS_API_KEY` is The Odds API (free tier, ~500 req/month; sport keys
  `soccer_fifa_world_cup` h2h + `soccer_fifa_world_cup_winner` outrights).
- Reproducible data builders live in `scripts/`; raw + processed data is gitignored
  (rebuild from source).

## Gotchas
- The project path contains a space (`Vibe_code/fifa preds`) — quote it in shells.
- Odds pulls cost quota; the snapshotter pulls 2 markets/run. Run daily-ish, not
  per-minute.
- `.env` is NOT committed — a fresh clone needs the key re-added to run the odds job.
