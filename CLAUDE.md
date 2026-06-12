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

## Current status (2026-06-11)
**Block V (verification spike): COMPLETE — all 6 items pass, V4 user-verified.**
**Block 0 (calibration spine): COMPLETE (T1–T8).**
**Block 1 (tournament simulator): COMPLETE (T9–T11).**
**Block 2 (technique ladder): COMPLETE (T12–T14). The build is feature-complete;
from here the project is OPERATING the loop, not building it.**
**Phase 2 Wave 1 — E1 automated nightly loop: DONE 2026-06-12** (scheduled
Action commits DB+artifacts; violations gate exits 2 and turns the build red)
**+ T2 fit-failure resilience** (`_fit_roster` drops an entrant on
RuntimeError/ValueError/LinAlgError/PyMC SamplingError instead of crashing).

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
- `loop/predict.py` CLI (T8) — `python -m fifapreds.loop.predict [--days 8|--all]`:
  fits both models, logs live predictions for upcoming WC fixtures; identical
  claims (same match/model/hyperparams/training_cutoff) dedupe to a no-op.
  First live claims logged 2026-06-10: 24 fixtures x 2 models.
- `publish/artifacts.py` (T8) — exports committed `artifacts/` (upcoming,
  leaderboard, calibration, scored parquets + meta.json) from the live +
  backtest DBs; degrades gracefully when a source is missing.
- `app/streamlit_app.py` (T8) — read-only viewer over `artifacts/` only
  (override dir via env `FIFAPREDS_ARTIFACTS`; tested with streamlit AppTest).
  Run: `.venv/bin/python -m streamlit run app/streamlit_app.py`.
- `sim/groups.py` (T9) — FIFA 2026 group tiebreakers (pts/GD/GF, recursive
  head-to-head, seeded drawing of lots; fair play skipped — no card data) +
  cross-group third-place ranking. Pure-python hot-loop code, no pandas.
- `sim/routing.py` (T10) — R32 slots from `bracket_2026.csv` + the verified
  495-combination third-place table; runtime pool checks + the CRITICAL gate
  test re-audits all 495 combos every run. `knockout_template()` for R16→final.
- `sim/montecarlo.py` (T11) — seeded tournament MC over any GoalsModel:
  conditions on played group results (2026 window only — historical WC
  meetings between same-group teams must NOT leak in; regression-tested),
  samples the rest from score grids, walks the bracket; draws → ET as
  Poisson(grid marginals/3) → 50/50 pens; knockouts all neutral-venue
  (documented approximation). 10k sims ≈ 4s after fit.
- `models/market.py` (T12) — h2h snapshot parser (power-method de-vig per
  bookmaker via `penaltyblog.implied`, median across books) + `MarketBlend`
  (default 0.75 market / 0.25 base; per-fixture fallback to base off-coverage;
  `trained_through` delegates to base — odds are pre-match info, audited via
  `odds_snapshot_id` which `predict_upcoming` now auto-attaches).
- `models/roster.py` (T13) — frozen leaderboard entrants as subclasses with
  distinct model_ids: `EloDecay` (decay 0.1/y), `EloImportance` (martj42-exact
  tournament weights — NB the Gold Cup string is plain "Gold Cup"),
  `DixonColesSlowXi` (xi 0.0005, targets the extremes-overconfidence quirk).
  `default_roster()` is THE source of entrants for the loop.
- `orchestrate.py` (T14) — the one-command live loop:
  `python -m fifapreds.orchestrate` = fetch martj42 (certifi SSL, atomic,
  refuses shrinking files; `--no-fetch` to skip) → ingest → score → refit
  roster + MarketBlend → predict (starts TOMORROW: date-only kickoffs make
  same-day claims unscoreable) → 10k-sim MC per goals model
  (`data/tournament_sim.parquet`, seed = UTC YYYYMMDD) → publish, incl.
  `artifacts/tournament.parquet` + a viewer "Tournament odds" section.
  ~25s end-to-end. E2E-tested (incl. tz-aware/naive scoring fix in score.py).
- 107/107 tests passing.
- First tournament odds published 2026-06-11 (DC, 10k sims): Spain 12.0%,
  Argentina 11.8%, England 6.7%; slow-xi flips Argentina/Spain and raises
  Colombia — the leaderboard adjudicates.

Known model quirk (expected, monitored by the calibration loop): Dixon-Coles is
overconfident at the extremes (backtest 0–0.1 bin: claimed 7.7%, happened 17.4%;
e.g. it gives Qatar 2.3% vs Switzerland where Elo says 10%) — cross-confederation
opponent pools are thin and xi-decay sharpens recent form. T13 variants compete
on exactly this.

Live cadence (E1: the loop now runs ITSELF — Phase 2 Wave 1 started 2026-06-12):
- `.github/workflows/nightly.yml` runs orchestrate at 06:30 UTC daily and
  commits the updated `data/fifa2026.db` + `artifacts/` back to the repo.
  **CI is the SOLE writer of the live DB — do NOT run orchestrate locally**
  (it forks the committed DB → binary merge conflict). Local experiments:
  `--db /tmp/scratch.db --no-fetch`. Force a real run via `workflow_dispatch`.
- The Action fails (orchestrate exits 2, nothing committed) when the scorer
  reports violations — a red nightly build means stop and investigate.
- Every day or two, still manual: `git pull` →
  `.venv/bin/python -m fifapreds.loop.odds` → commit + push the DB right away
  (quota-bound, deliberately NOT part of the orchestrator; it writes the DB,
  so don't let it straddle the nightly window).

## Setup & run
```bash
# from the project root (note: the path contains a space)
python3 -m venv .venv
.venv/bin/python -m pip install -e .            # editable install
.venv/bin/python -m pip install pytest networkx # dev/data deps

.venv/bin/python -m pytest -q                   # run tests
.venv/bin/python -m fifapreds.orchestrate       # THE live loop — CI-only now! locally use --db /tmp/scratch.db
.venv/bin/python -m fifapreds.ingest            # (re)build matches.parquet only
.venv/bin/python -m fifapreds.loop.odds         # capture an odds snapshot (uses quota)

# Reproduce the verified data artifacts:
.venv/bin/python scripts/fetch_bracket_2026.py  # routing_r32.parquet + bracket_2026.csv
.venv/bin/python scripts/build_groups_2026.py   # groups_2026.csv (with guards)
```

## Locked architecture decisions (do not relitigate — see PLAN.md)
- **Storage:** SQLite (`data/fifa2026.db`: predictions, odds_snapshots, rating
  snapshots) + parquet (raw/processed match data). The live DB is COMMITTED
  (E1: survives stateless CI runs; CI is its sole writer); `backtest.db`, raw
  CSVs and processed parquet stay gitignored, except the three frozen 2026
  inputs (`groups_2026.csv`, `bracket_2026.csv`, `routing_r32.parquet`).
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
